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

        # Load CLIP model with preprocessing
        self.clip_model, _, self.preprocess = open_clip.create_model_and_transforms(
            model_name,
            pretrained=pretrained,
        )

        # Extract visual encoder
        self.visual = self.clip_model.visual

        # Get output dimension
        self.output_dim = self.visual.output_dim

        # Get expected input size from the preprocess transform
        # Usually 224x224 for ViT-B/16
        self.expected_size = self.preprocess.transforms[0].size

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
            x: Input images (B, 3, H, W) - will be resized to CLIP expected size
            return_features: Whether to return patch features

        Returns:
            Visual features (B, output_dim) or dict with features
        """
        # Resize input to CLIP's expected size if needed
        original_size = x.shape[-2:]
        if original_size != self.expected_size:
            x = F.interpolate(x, size=self.expected_size, mode='bilinear', align_corners=False)

        # Use CLIP's built-in encode_image method
        features = self.clip_model.encode_image(x)

        if return_features and self.output_tokens:
            # For feature map, we need to do a manual forward
            # This is more complex, so for now just return CLS token
            return {
                "cls_token": features,
                "patch_tokens": None,
            }

        return features

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
        batch_size = x.shape[0]
        device = x.device
        original_h, original_w = x.shape[2], x.shape[3]

        # Resize to CLIP expected size
        x_resized = F.interpolate(x, size=self.expected_size, mode='bilinear', align_corners=False)

        # Get intermediate features from patch embedding
        if hasattr(self.visual, 'patch_embedding'):
            # Get patch embeddings
            x_feat = self.visual.patch_embedding(x_resized)
            if x_feat.dim() == 4:
                # (B, D, H, W) format - return directly
                # Upsample back to original size
                return F.interpolate(x_feat, size=(original_h, original_w), mode='bilinear', align_corners=False)
            elif x_feat.dim() == 3:
                # (B, L, D) format, need to reshape
                B, L, D = x_feat.shape
                H = W = int(L ** 0.5)
                feat_map = x_feat.transpose(1, 2).reshape(B, D, H, W)
                # Upsample back to original size
                return F.interpolate(feat_map, size=(original_h, original_w), mode='bilinear', align_corners=False)

        # Fallback: return feature map at appropriate size
        target_h = original_h // 16
        target_w = original_w // 16
        return torch.zeros(batch_size, self.output_dim, target_h, target_w, device=device)

        # Reshape to spatial feature map
        batch_size, num_patches, dim = patch_tokens.shape

        # Calculate spatial dimensions
        h = w = int(num_patches ** 0.5)

        # Handle non-square cases
        if h * w != num_patches:
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
