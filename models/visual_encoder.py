"""Visual encoder using CLIP ViT."""

from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class VisualEncoder(nn.Module):
    """Visual encoder using CLIP's ViT-B/16.

    Supports freezing early layers for transfer learning.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        freeze_layers: int = 6,
        output_tokens: bool = False,
    ):
        """Initialize visual encoder.

        Args:
            model_name: CLIP model name
            pretrained: Pretrained weights ('openai', 'laion2b_s34b_b88k', etc.)
            freeze_layers: Number of layers to freeze (first N transformer blocks)
            output_tokens: Whether to output patch tokens in addition to CLS token
        """
        super().__init__()

        self.model_name = model_name
        self.pretrained = pretrained
        self.freeze_layers = freeze_layers
        self.output_tokens = output_tokens

        try:
            import open_clip
        except ImportError:
            raise ImportError(
                "open_clip_torch is required. Install with: pip install open_clip_torch"
            )

        # Load CLIP model
        self.clip_model, _, _ = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )

        # Extract visual encoder
        self.visual = self.clip_model.visual

        # Get output dimension
        self.output_dim = self.visual.output_dim

        # Freeze specified layers
        self._freeze_layers()

    def _freeze_layers(self) -> None:
        """Freeze first N transformer layers."""
        if self.freeze_layers <= 0:
            return

        # Freeze patch embedding
        if hasattr(self.visual, "patch_embedding"):
            for param in self.visual.patch_embedding.parameters():
                param.requires_grad = False

        if hasattr(self.visual, "class_embedding"):
            self.visual.class_embedding.requires_grad = False

        if hasattr(self.visual, "positional_embedding"):
            self.visual.positional_embedding.requires_grad = False

        # Freeze transformer layers
        if hasattr(self.visual, "transformer"):
            num_layers = len(self.visual.transformer.resblocks)
            for i in range(min(self.freeze_layers, num_layers)):
                block = self.visual.transformer.resblocks[i]
                for param in block.parameters():
                    param.requires_grad = False

        # Freeze ln_pre and ln_post if freezing all layers
        if self.freeze_layers >= num_layers:
            if hasattr(self.visual, "ln_pre"):
                for param in self.visual.ln_pre.parameters():
                    param.requires_grad = False
            if hasattr(self.visual, "ln_post"):
                for param in self.visual.ln_post.parameters():
                    param.requires_grad = False

    def unfreeze_all(self) -> None:
        """Unfreeze all parameters for fine-tuning."""
        for param in self.parameters():
            param.requires_grad = True

    def forward(
        self,
        x: torch.Tensor,
        return_features: bool = False,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input images (B, 3, H, W)
            return_features: Whether to return patch features

        Returns:
            Visual features (B, output_dim) or dict with features
        """
        # Get patch embeddings
        if hasattr(self.visual, "patch_embedding"):
            x = self.visual.patch_embedding(x)  # (B, L, D)
            x = x.flatten(2).transpose(1, 2)  # (B, num_patches, D)

            # Add class token
            if hasattr(self.visual, "class_embedding"):
                batch_size = x.shape[0]
                class_token = self.visual.class_embedding.expand(batch_size, -1, -1)
                x = torch.cat([class_token, x], dim=1)

            # Add positional embedding
            if hasattr(self.visual, "positional_embedding"):
                x = x + self.visual.positional_embedding
        else:
            # Alternative path for different CLIP implementations
            x = self.visual.conv1(x)  # (B, D, H/patch, W/patch)
            x = x.reshape(x.shape[0], x.shape[1], -1)  # (B, D, L)
            x = x.permute(0, 2, 1)  # (B, L, D)

            batch_size = x.shape[0]
            class_token = self.visual.class_embedding.expand(batch_size, -1, -1)
            x = torch.cat([class_token, x], dim=1)

            x = x + self.visual.positional_embedding

        # Layer norm before transformer
        if hasattr(self.visual, "ln_pre"):
            x = self.visual.ln_pre(x)

        # Transformer
        x = x.permute(1, 0, 2)  # (L, B, D) for transformer
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # (B, L, D)

        # Layer norm after transformer
        if hasattr(self.visual, "ln_post"):
            x = self.visual.ln_post(x)

        # Extract class token
        cls_token = x[:, 0]  # (B, D)

        # Project to output dimension
        if hasattr(self.visual, "proj") and self.visual.proj is not None:
            cls_token = cls_token @ self.visual.proj

        if return_features and self.output_tokens:
            # Return both CLS token and patch tokens
            return {
                "cls_token": cls_token,
                "patch_tokens": x[:, 1:],  # Remove CLS token
            }

        return cls_token

    def get_feature_map(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Get spatial feature map for segmentation decoder.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Feature map (B, D, h, w)
        """
        features = self.forward(x, return_features=True)

        if isinstance(features, dict):
            patch_tokens = features["patch_tokens"]  # (B, L, D)
        else:
            # If only CLS token, return empty feature map
            batch_size = x.shape[0]
            return torch.zeros(batch_size, self.output_dim, 1, 1, device=x.device)

        # Reshape to spatial feature map
        batch_size, num_patches, dim = patch_tokens.shape

        # Calculate spatial dimensions
        # For ViT-B/16: input / 16
        h = w = int(num_patches ** 0.5)

        # Handle cases where num_patches is not a perfect square
        if h * w != num_patches:
            # Original image size might be non-square
            # Assume square for now
            h = int(num_patches ** 0.5)
            w = num_patches // h

        feature_map = patch_tokens.transpose(1, 2).reshape(batch_size, dim, h, w)

        return feature_map


class MultiScaleVisualEncoder(nn.Module):
    """Multi-scale visual encoder for better segmentation.

    Extracts features at multiple scales for the lane decoder.
    """

    def __init__(
        self,
        model_name: str = "ViT-B-16",
        pretrained: str = "openai",
        freeze_layers: int = 6,
    ):
        """Initialize multi-scale visual encoder.

        Args:
            model_name: CLIP model name
            pretrained: Pretrained weights
            freeze_layers: Number of layers to freeze
        """
        super().__init__()

        self.base_encoder = VisualEncoder(
            model_name=model_name,
            pretrained=pretrained,
            freeze_layers=freeze_layers,
            output_tokens=True,
        )

        self.output_dim = self.base_encoder.output_dim

    def forward(
        self,
        x: torch.Tensor,
    ) -> dict:
        """Forward pass with multi-scale features.

        Args:
            x: Input images (B, 3, H, W)

        Returns:
            Dictionary with 'cls_token' and 'feature_map'
        """
        # Get features from base encoder
        cls_token = self.base_encoder(x, return_features=False)
        feature_map = self.base_encoder.get_feature_map(x)

        return {
            "cls_token": cls_token,
            "feature_map": feature_map,
        }
