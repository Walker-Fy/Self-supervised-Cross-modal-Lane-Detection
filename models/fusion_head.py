"""Fusion head for combining visual and language features."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FusionHead(nn.Module):
    """Combine visual and language features for lane detection.

    Simple linear projection with normalization for fusing
    cross-modal features.
    """

    def __init__(
        self,
        visual_dim: int = 512,
        language_dim: int = 512,
        out_dim: int = 512,
        fusion_type: str = "concat",
    ):
        """Initialize fusion head.

        Args:
            visual_dim: Visual feature dimension
            language_dim: Language feature dimension
            out_dim: Output feature dimension
            fusion_type: Type of fusion ('concat', 'add', 'attention', 'fi lm')
        """
        super().__init__()

        self.fusion_type = fusion_type
        self.visual_dim = visual_dim
        self.language_dim = language_dim
        self.out_dim = out_dim

        if fusion_type == "concat":
            self.fusion = nn.Sequential(
                nn.Linear(visual_dim + language_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.ReLU(inplace=True),
            )
        elif fusion_type == "add":
            # Project both to same dimension and add
            self.visual_proj = nn.Linear(visual_dim, out_dim)
            self.language_proj = nn.Linear(language_dim, out_dim)
            self.norm = nn.LayerNorm(out_dim)
        elif fusion_type == "attention":
            self.attention = nn.MultiheadAttention(
                embed_dim=out_dim,
                num_heads=8,
                batch_first=True,
            )
            self.visual_proj = nn.Linear(visual_dim, out_dim)
            self.language_proj = nn.Linear(language_dim, out_dim)
        elif fusion_type == "film":
            # FiLM: Feature-wise Linear Modulation
            self.film_gen = nn.Sequential(
                nn.Linear(language_dim, visual_dim * 2),
            )
            self.out_proj = nn.Linear(visual_dim, out_dim)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")

    def forward(
        self,
        visual: torch.Tensor,
        language: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Forward pass.

        Args:
            visual: Visual features (B, visual_dim)
            language: Language features (B, language_dim)
            normalize: Whether to L2 normalize output

        Returns:
            Fused features (B, out_dim)
        """
        if self.fusion_type == "concat":
            x = torch.cat([visual, language], dim=-1)
            x = self.fusion(x)

        elif self.fusion_type == "add":
            v = self.visual_proj(visual)
            l = self.language_proj(language)
            x = self.norm(v + l)

        elif self.fusion_type == "attention":
            v = self.visual_proj(visual).unsqueeze(1)  # (B, 1, D)
            l = self.language_proj(language).unsqueeze(1)  # (B, 1, D)

            # Cross-attention
            x, _ = self.attention(v, l, l)
            x = x.squeeze(1)

        elif self.fusion_type == "film":
            # Generate scale and shift from language
            film_params = self.film_gen(language)  # (B, 2 * visual_dim)
            scale, shift = film_params.chunk(2, dim=-1)

            # Modulate visual features
            x = visual * (1 + scale) + shift
            x = self.out_proj(x)

        else:
            raise ValueError(f"Unknown fusion type: {self.fusion_type}")

        if normalize:
            x = F.normalize(x, dim=-1)

        return x


class MultiModalFusion(nn.Module):
    """Multi-modal fusion with gated combination."""

    def __init__(
        self,
        visual_dim: int = 512,
        language_dim: int = 512,
        hidden_dim: int = 256,
    ):
        """Initialize gated fusion.

        Args:
            visual_dim: Visual feature dimension
            language_dim: Language feature dimension
            hidden_dim: Hidden dimension for gate
        """
        super().__init__()

        # Gate network
        self.gate = nn.Sequential(
            nn.Linear(visual_dim + language_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 2),
            nn.Softmax(dim=-1),
        )

        # Projection layers
        self.visual_proj = nn.Linear(visual_dim, visual_dim)
        self.language_proj = nn.Linear(language_dim, visual_dim)

    def forward(
        self,
        visual: torch.Tensor,
        language: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Forward pass with gated fusion.

        Args:
            visual: Visual features (B, visual_dim)
            language: Language features (B, language_dim)
            normalize: Whether to L2 normalize output

        Returns:
            Fused features (B, visual_dim)
        """
        # Compute gate weights
        concat = torch.cat([visual, language], dim=-1)
        gate = self.gate(concat)  # (B, 2)

        # Project features
        v_proj = self.visual_proj(visual)
        l_proj = self.language_proj(language)

        # Gated combination
        x = gate[:, 0:1] * v_proj + gate[:, 1:2] * l_proj

        if normalize:
            x = F.normalize(x, dim=-1)

        return x


class CrossAttention(nn.Module):
    """Cross-modal attention for feature fusion."""

    def __init__(
        self,
        visual_dim: int = 512,
        language_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
    ):
        """Initialize cross-attention.

        Args:
            visual_dim: Visual feature dimension
            language_dim: Language feature dimension
            num_heads: Number of attention heads
            num_layers: Number of attention layers
        """
        super().__init__()

        self.visual_proj = nn.Linear(visual_dim, visual_dim)
        self.language_proj = nn.Linear(language_dim, visual_dim)

        self.layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=visual_dim,
                num_heads=num_heads,
                batch_first=True,
            )
            for _ in range(num_layers)
        ])

        self.norms = nn.ModuleList([
            nn.LayerNorm(visual_dim)
            for _ in range(num_layers)
        ])

        self.out_proj = nn.Linear(visual_dim, visual_dim)

    def forward(
        self,
        visual: torch.Tensor,
        language: torch.Tensor,
        normalize: bool = True,
    ) -> torch.Tensor:
        """Forward pass with cross-attention.

        Args:
            visual: Visual features (B, visual_dim)
            language: Language features (B, language_dim)
            normalize: Whether to L2 normalize output

        Returns:
            Fused features (B, visual_dim)
        """
        v = self.visual_proj(visual).unsqueeze(1)  # (B, 1, D)
        l = self.language_proj(language).unsqueeze(1)  # (B, 1, D)

        for attention, norm in zip(self.layers, self.norms):
            # Self-attention on visual with language as context
            attn_out, _ = attention(v, l, l)
            v = norm(v + attn_out)

        x = v.squeeze(1)
        x = self.out_proj(x)

        if normalize:
            x = F.normalize(x, dim=-1)

        return x
