"""Lane segmentation decoder."""

from typing import Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class LaneDecoder(nn.Module):
    """Lightweight decoder for lane segmentation.

    Takes CLIP visual features and outputs binary lane mask.
    """

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dims: List[int] = [256, 128, 64],
        input_size: tuple = (590, 1640),
    ):
        """Initialize lane decoder.

        Args:
            in_dim: Input feature dimension
            hidden_dims: Hidden dimensions for upsampling path
            input_size: Input image size (H, W)
        """
        super().__init__()

        self.in_dim = in_dim
        self.input_size = input_size

        # Calculate feature map size (assuming ViT-B/16)
        self.feature_h = input_size[0] // 16
        self.feature_w = input_size[1] // 16

        # Build decoder
        layers = []
        current_dim = in_dim

        for hidden_dim in hidden_dims:
            # Upsample + conv
            layers.extend([
                nn.ConvTranspose2d(
                    current_dim,
                    hidden_dim,
                    kernel_size=4,
                    stride=2,
                    padding=1,
                ),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU(inplace=True),
            ])
            current_dim = hidden_dim

        self.decoder = nn.Sequential(*layers)

        # Final output layer
        self.output = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], 1, kernel_size=3, padding=1),
            nn.Upsample(size=input_size, mode='bilinear', align_corners=False),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Input features (B, D, h, w) or (B, D)

        Returns:
            Lane mask logits (B, 1, H, W)
        """
        if features.dim() == 2:
            # Reshape CLS token to feature map
            B = features.shape[0]
            features = features.unsqueeze(-1).unsqueeze(-1)
            features = features.expand(B, self.in_dim, self.feature_h, self.feature_w)

        # Decode
        x = self.decoder(features)

        # Output
        x = self.output(x)

        return x


class UNetDecoder(nn.Module):
    """UNet-style decoder with skip connections."""

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dims: List[int] = [256, 128, 64, 32],
        input_size: tuple = (590, 1640),
    ):
        """Initialize UNet decoder.

        Args:
            in_dim: Input feature dimension
            hidden_dims: Hidden dimensions for decoder
            input_size: Input image size (H, W)
        """
        super().__init__()

        self.in_dim = in_dim
        self.input_size = input_size
        self.feature_h = input_size[0] // 16
        self.feature_w = input_size[1] // 16

        # Encoder (for multi-scale features)
        self.encoder_blocks = nn.ModuleList()

        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()

        current_dim = in_dim
        for hidden_dim in hidden_dims:
            # Decoder block: upsample -> conv -> concat -> conv
            self.decoder_blocks.append(
                nn.Sequential(
                    nn.ConvTranspose2d(current_dim, hidden_dim, 4, 2, 1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1),
                    nn.BatchNorm2d(hidden_dim),
                    nn.ReLU(inplace=True),
                )
            )
            current_dim = hidden_dim

        # Output layer
        self.output = nn.Sequential(
            nn.Conv2d(hidden_dims[-1], 1, 3, 1, 1),
            nn.Upsample(size=input_size, mode='bilinear', align_corners=False),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Input features (B, D, h, w) or (B, D)

        Returns:
            Lane mask logits (B, 1, H, W)
        """
        if features.dim() == 2:
            B = features.shape[0]
            features = features.unsqueeze(-1).unsqueeze(-1)
            features = features.expand(B, self.in_dim, self.feature_h, self.feature_w)

        x = features

        # Decode
        for decoder_block in self.decoder_blocks:
            x = decoder_block(x)

        # Output
        x = self.output(x)

        return x


class FPNDecoder(nn.Module):
    """Feature Pyramid Network style decoder."""

    def __init__(
        self,
        in_dim: int = 512,
        fpn_dims: List[int] = [256, 128, 64],
        input_size: tuple = (590, 1640),
    ):
        """Initialize FPN decoder.

        Args:
            in_dim: Input feature dimension
            fpn_dims: FPN output dimensions
            input_size: Input image size (H, W)
        """
        super().__init__()

        self.in_dim = in_dim
        self.input_size = input_size
        self.fpn_dims = fpn_dims

        # Lateral connections
        self.lateral_convs = nn.ModuleList()
        for fpn_dim in fpn_dims:
            self.lateral_convs.append(
                nn.Conv2d(in_dim, fpn_dim, kernel_size=1)
            )

        # Output convolutions
        self.output_convs = nn.ModuleList()
        for fpn_dim in fpn_dims:
            self.output_convs.append(
                nn.Sequential(
                    nn.Conv2d(fpn_dim, fpn_dim, kernel_size=3, padding=1),
                    nn.BatchNorm2d(fpn_dim),
                    nn.ReLU(inplace=True),
                )
            )

        # Final fusion
        total_dim = sum(fpn_dims)
        self.fusion = nn.Sequential(
            nn.Conv2d(total_dim, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, kernel_size=1),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Input features (B, D, h, w) or (B, D)

        Returns:
            Lane mask logits (B, 1, H, W)
        """
        if features.dim() == 2:
            B, D = features.shape
            h = w = int(features.shape[1] ** 0.5)
            features = features.view(B, D, h, w)

        # Build FPN-like features at different scales
        fpn_features = []
        current_size = (features.shape[2], features.shape[3])

        for i, (lateral_conv, output_conv) in enumerate(
            zip(self.lateral_convs, self.output_convs)
        ):
            # Lateral connection
            lateral = lateral_conv(features)

            # Upsample to original feature size
            if i > 0:
                lateral = F.interpolate(
                    lateral,
                    size=current_size,
                    mode='bilinear',
                    align_corners=False,
                )

            # Output conv
            out = output_conv(lateral)
            fpn_features.append(out)

        # Concatenate all features
        x = torch.cat(fpn_features, dim=1)

        # Final fusion and upsampling
        x = self.fusion(x)
        x = F.interpolate(x, size=self.input_size, mode='bilinear', align_corners=False)

        return x


class SimpleUpsampleDecoder(nn.Module):
    """Very simple decoder for ablation."""

    def __init__(
        self,
        in_dim: int = 512,
        input_size: tuple = (590, 1640),
    ):
        """Initialize simple decoder.

        Args:
            in_dim: Input feature dimension
            input_size: Input image size (H, W)
        """
        super().__init__()

        self.decoder = nn.Sequential(
            nn.Linear(in_dim, in_dim * 4),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim * 4, 800 * 320),
        )

        self.input_size = input_size

    def forward(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            features: Input features (B, D)

        Returns:
            Lane mask logits (B, 1, H, W)
        """
        if features.dim() == 3:
            features = features.mean(dim=1)  # Global pool if spatial

        x = self.decoder(features)
        x = x.view(x.shape[0], 1, *self.input_size)

        return x


class LaneDecoderWithCoords(nn.Module):
    """Lane decoder with coordinate regression head.

    Outputs both segmentation mask and lane coordinates.
    """

    def __init__(
        self,
        in_dim: int = 512,
        hidden_dims: List[int] = [256, 128, 64],
        input_size: tuple = (590, 1640),
        n_lanes: int = 4,
        n_points: int = 56,
    ):
        """Initialize lane decoder with coordinate head.

        Args:
            in_dim: Input feature dimension
            hidden_dims: Hidden dimensions for upsampling path
            input_size: Input image size (H, W)
            n_lanes: Maximum number of lanes
            n_points: Number of points per lane
        """
        super().__init__()

        self.in_dim = in_dim
        self.input_size = input_size
        self.n_lanes = n_lanes
        self.n_points = n_points

        # Base decoder for segmentation
        self.base_decoder = LaneDecoder(
            in_dim=in_dim,
            hidden_dims=hidden_dims,
            input_size=input_size,
        )

        # Coordinate regression head
        # Predict (n_lanes * n_points * 2) coordinates
        coord_dim = n_lanes * n_points * 2

        self.coord_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_dim, in_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(in_dim // 2, coord_dim),
        )

    def forward(
        self,
        features: torch.Tensor,
    ) -> dict:
        """Forward pass.

        Args:
            features: Input features (B, D, h, w) or (B, D)

        Returns:
            Dictionary with 'mask' and 'coords'
        """
        # Get segmentation mask
        mask = self.base_decoder(features)

        # Get features for coordinate head
        if features.dim() == 4:
            feat_flat = F.adaptive_avg_pool2d(features, 1).flatten(1)
        else:
            feat_flat = features

        # Predict coordinates
        coords = self.coord_head(feat_flat)  # (B, n_lanes * n_points * 2)

        # Reshape to (B, n_lanes, n_points, 2)
        coords = coords.view(-1, self.n_lanes, self.n_points, 2)

        # Normalize coordinates to image size
        # Apply sigmoid and scale to image dimensions
        coords = torch.sigmoid(coords) * torch.tensor(
            [self.input_size[1], self.input_size[0]],
            device=coords.device
        ).view(1, 1, 1, 2)

        return {
            "mask": mask,
            "coords": coords,
        }
