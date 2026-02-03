"""Projection heads for CLIP features."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class ProjectionHead(nn.Module):
    """Projection head for mapping CLIP features to shared space.

    Implements P_v (visual) and P_l (language) projections from SCC paper.

    Architecture: Linear -> LayerNorm -> ReLU -> Linear
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 512,
        hidden_dim: Optional[int] = None,
        use_bn: bool = False,
    ):
        """Initialize projection head.

        Args:
            in_dim: Input feature dimension
            out_dim: Output projection dimension
            hidden_dim: Hidden layer dimension (default: same as in_dim)
            use_bn: Whether to use BatchNorm instead of LayerNorm
        """
        super().__init__()

        if hidden_dim is None:
            hidden_dim = in_dim

        if use_bn:
            self.projection = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim),
            )
        else:
            self.projection = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, out_dim),
            )

        self.out_dim = out_dim

    def forward(
        self,
        x: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features (B, in_dim)
            normalize: Whether to L2 normalize output

        Returns:
            Projected features (B, out_dim)
        """
        x = self.projection(x)

        if normalize:
            x = F.normalize(x, dim=-1)

        return x


class DualProjectionHead(nn.Module):
    """Dual projection heads for visual and language modalities.

    Maintains separate P_v and P_l projections.
    """

    def __init__(
        self,
        visual_dim: int,
        language_dim: int,
        proj_dim: int = 512,
        hidden_dim: Optional[int] = None,
        shared: bool = False,
    ):
        """Initialize dual projection heads.

        Args:
            visual_dim: Visual feature dimension
            language_dim: Language feature dimension
            proj_dim: Output projection dimension
            hidden_dim: Hidden layer dimension
            shared: Whether to use shared projection (for symmetry)
        """
        super().__init__()

        self.proj_dim = proj_dim

        if hidden_dim is None:
            hidden_dim_v = visual_dim
            hidden_dim_l = language_dim
        else:
            hidden_dim_v = hidden_dim_l = hidden_dim

        if shared:
            # Use same projection for both modalities
            # This assumes visual_dim == language_dim
            self.visual_proj = ProjectionHead(visual_dim, proj_dim, hidden_dim_v)
            self.language_proj = self.visual_proj
        else:
            # Separate projections
            self.visual_proj = ProjectionHead(visual_dim, proj_dim, hidden_dim_v)
            self.language_proj = ProjectionHead(language_dim, proj_dim, hidden_dim_l)

    def forward(
        self,
        visual: Optional[torch.Tensor] = None,
        language: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> dict:
        """Forward pass.

        Args:
            visual: Visual features (B, visual_dim)
            language: Language features (B, language_dim)
            normalize: Whether to L2 normalize outputs

        Returns:
            Dictionary with 'z_v' and/or 'z_l'
        """
        result = {}

        if visual is not None:
            result["z_v"] = self.visual_proj(visual, normalize=normalize)

        if language is not None:
            result["z_l"] = self.language_proj(language, normalize=normalize)

        return result


class AttentionProjection(nn.Module):
    """Attention-based projection for better feature alignment."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 512,
        num_heads: int = 8,
    ):
        """Initialize attention projection.

        Args:
            in_dim: Input feature dimension
            out_dim: Output projection dimension
            num_heads: Number of attention heads
        """
        super().__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            batch_first=True,
        )

        self.projection = nn.Sequential(
            nn.Linear(in_dim, in_dim),
            nn.LayerNorm(in_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_dim, out_dim),
        )

        self.out_dim = out_dim

    def forward(
        self,
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Forward pass with attention.

        Args:
            x: Input features (B, in_dim) or (B, L, in_dim)
            context: Optional context for cross-attention
            normalize: Whether to L2 normalize output

        Returns:
            Projected features (B, out_dim)
        """
        # Add sequence dimension if needed
        squeeze = False
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze = True

        # Self-attention or cross-attention
        if context is not None:
            if context.dim() == 2:
                context = context.unsqueeze(1)
            attended, _ = self.attention(x, context, context)
        else:
            attended, _ = self.attention(x, x, x)

        # Remove sequence dimension
        if squeeze:
            attended = attended.squeeze(1)
        else:
            attended = attended.mean(dim=1)

        # Project
        x = self.projection(attended)

        if normalize:
            x = F.normalize(x, dim=-1)

        return x


class MLPProjection(nn.Module):
    """Simple MLP projection without normalization.

    Alternative projection head for ablation studies.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int = 512,
        num_layers: int = 3,
        use_residual: bool = False,
    ):
        """Initialize MLP projection.

        Args:
            in_dim: Input feature dimension
            out_dim: Output projection dimension
            num_layers: Number of MLP layers
            use_residual: Whether to use residual connections
        """
        super().__init__()

        layers = []
        current_dim = in_dim

        for i in range(num_layers - 1):
            layers.append(nn.Linear(current_dim, out_dim))
            layers.append(nn.ReLU(inplace=True))
            current_dim = out_dim

        layers.append(nn.Linear(current_dim, out_dim))

        self.mlp = nn.Sequential(*layers)

        self.use_residual = use_residual
        self.out_dim = out_dim

        # Residual projection (if dimensions don't match)
        if use_residual and in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
        else:
            self.residual_proj = None

    def forward(
        self,
        x: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input features (B, in_dim)
            normalize: Whether to L2 normalize output

        Returns:
            Projected features (B, out_dim)
        """
        residual = x

        x = self.mlp(x)

        if self.use_residual:
            if self.residual_proj is not None:
                residual = self.residual_proj(residual)
            x = x + residual

        if normalize:
            x = F.normalize(x, dim=-1)

        return x
