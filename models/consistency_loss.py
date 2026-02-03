"""Consistency loss functions for SCC training."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class InfoNCELoss(nn.Module):
    """InfoNCE (contrastive) loss for cross-modal learning.

    Bidirectional: visual->text and text->visual
    """

    def __init__(
        self,
        temperature: float = 0.07,
        bidirectional: bool = True,
    ):
        """Initialize InfoNCE loss.

        Args:
            temperature: Temperature parameter for softmax
            bidirectional: Whether to compute bidirectional loss
        """
        super().__init__()

        self.temperature = temperature
        self.bidirectional = bidirectional

    def forward(
        self,
        z_v: torch.Tensor,
        z_l: torch.Tensor,
    ) -> torch.Tensor:
        """Compute InfoNCE loss.

        Args:
            z_v: Visual features (B, D), should be L2 normalized
            z_l: Language features (B, D), should be L2 normalized

        Returns:
            InfoNCE loss value
        """
        batch_size = z_v.shape[0]

        # Compute similarity matrix
        # sim[i, j] = cosine similarity between z_v[i] and z_l[j]
        sim_matrix = torch.matmul(z_v, z_l.T) / self.temperature  # (B, B)

        # Labels are on diagonal (positive pairs)
        labels = torch.arange(batch_size, device=z_v.device)

        # Visual -> Language loss
        loss_v2l = F.cross_entropy(sim_matrix, labels)

        if self.bidirectional:
            # Language -> Visual loss
            loss_l2v = F.cross_entropy(sim_matrix.T, labels)
            loss = (loss_v2l + loss_l2v) / 2
        else:
            loss = loss_v2l

        return loss


class CosineConsistencyLoss(nn.Module):
    """Cosine consistency loss between two augmented views.

    Ensures that features from augmented views of the same image
    remain consistent in the embedding space.
    """

    def __init__(
        self,
        reduction: str = "mean",
    ):
        """Initialize cosine consistency loss.

        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
        """
        super().__init__()

        self.reduction = reduction

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cosine consistency loss.

        Loss = 1 - cosine_similarity(z1, z2)

        Args:
            z1: Features from view 1 (B, D), L2 normalized
            z2: Features from view 2 (B, D), L2 normalized

        Returns:
            Cosine consistency loss
        """
        # Cosine similarity (for normalized vectors, this is just dot product)
        sim = (z1 * z2).sum(dim=-1)  # (B,)

        # Consistency loss
        loss = 1 - sim

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss


class CrossModalConsistencyLoss(nn.Module):
    """Cross-modal consistency loss.

    Ensures that visual features from two views are consistent
    when conditioned on the same text.
    """

    def __init__(
        self,
        temperature: float = 0.07,
    ):
        """Initialize cross-modal consistency loss.

        Args:
            temperature: Temperature parameter
        """
        super().__init__()

        self.temperature = temperature

    def forward(
        self,
        z_v1: torch.Tensor,
        z_v2: torch.Tensor,
        z_l: torch.Tensor,
    ) -> torch.Tensor:
        """Compute cross-modal consistency loss.

        Args:
            z_v1: Visual features from view 1 (B, D)
            z_v2: Visual features from view 2 (B, D)
            z_l: Language features (B, D)

        Returns:
            Cross-modal consistency loss
        """
        batch_size = z_v1.shape[0]

        # Compute similarities
        sim_v1_l = torch.matmul(z_v1, z_l.T) / self.temperature  # (B, B)
        sim_v2_l = torch.matmul(z_v2, z_l.T) / self.temperature  # (B, B)

        # Both views should have similar similarity to text
        consistency_loss = F.mse_loss(sim_v1_l, sim_v2_l)

        return consistency_loss


class NTXentLoss(nn.Module):
    """Normalized Temperature-scaled Cross Entropy Loss.

    Alternative formulation of InfoNCE.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        use_cosine_similarity: bool = True,
    ):
        """Initialize NT-Xent loss.

        Args:
            temperature: Temperature parameter
            use_cosine_similarity: Whether to use cosine similarity
        """
        super().__init__()

        self.temperature = temperature
        self.use_cosine_similarity = use_cosine_similarity

    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """Compute NT-Xent loss.

        Args:
            z1: Features from view 1 (B, D)
            z2: Features from view 2 (B, D)

        Returns:
            NT-Xent loss
        """
        batch_size = z1.shape[0]

        # Concatenate features
        z = torch.cat([z1, z2], dim=0)  # (2B, D)

        if self.use_cosine_similarity:
            # Normalize features
            z = F.normalize(z, dim=-1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(z, z.T) / self.temperature  # (2B, 2B)

        # Remove diagonal (self-similarity)
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        sim_matrix = sim_matrix.masked_fill(mask, -float('inf'))

        # Positive pairs: (i, i+B) and (i+B, i)
        labels = torch.cat([
            torch.arange(batch_size, 2 * batch_size, device=z.device),
            torch.arange(0, batch_size, device=z.device),
        ])

        # Compute loss
        loss = F.cross_entropy(
            sim_matrix.flatten(0),
            labels,
        )

        return loss


class SupConLoss(nn.Module):
    """Supervised Contrastive Loss.

    Extends contrastive loss to use multiple positives per sample.
    """

    def __init__(
        self,
        temperature: float = 0.07,
        base_temperature: float = 0.07,
    ):
        """Initialize supervised contrastive loss.

        Args:
            temperature: Temperature parameter
            base_temperature: Base temperature for scaling
        """
        super().__init__()

        self.temperature = temperature
        self.base_temperature = base_temperature

    def forward(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute supervised contrastive loss.

        Args:
            features: Feature vectors (B, D)
            labels: Ground truth labels (B,)
            mask: Contrastive mask (B, B) where mask[i,j] = 1 if same class

        Returns:
            Supervised contrastive loss
        """
        device = features.device
        batch_size = features.shape[0]

        if labels is not None and mask is not None:
            raise ValueError("Cannot specify both labels and mask")

        if labels is None and mask is None:
            # Without labels, treat as standard contrastive loss
            # Each sample has one positive (its augmented view)
            mask = torch.eye(batch_size, dtype=torch.float32, device=device)
        elif labels is not None:
            # Create mask from labels
            labels = labels.contiguous().view(-1, 1)
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)

        # Compute similarity
        feature_matrix = F.normalize(features, dim=-1)
        anchor_dot_contrast = torch.div(
            torch.matmul(feature_matrix, feature_matrix.T),
            self.temperature,
        )

        # For numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # Mask out self-contrast
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0,
        )
        mask = mask * logits_mask

        # Compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # Compute mean of log-likelihood over positive pairs
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1)

        # Loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
