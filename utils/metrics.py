"""Evaluation metrics for lane detection."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Optional, List


def iou_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Calculate Intersection over Union (IoU).

    Args:
        pred: Predicted mask (B, 1, H, W) or (B, H, W)
        target: Ground truth mask (B, 1, H, W) or (B, H, W)
        eps: Small value for numerical stability

    Returns:
        IoU score
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)

    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) - intersection

    iou = (intersection + eps) / (union + eps)
    return iou.mean()


def dice_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Calculate Dice coefficient.

    Args:
        pred: Predicted mask (B, 1, H, W) or (B, H, W)
        target: Ground truth mask (B, 1, H, W) or (B, H, W)
        eps: Small value for numerical stability

    Returns:
        Dice score
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)

    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    intersection = (pred * target).sum(dim=(2, 3))
    dice = (2 * intersection + eps) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + eps)

    return dice.mean()


def f1_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-7,
) -> torch.Tensor:
    """Calculate F1 score.

    Args:
        pred: Predicted mask (B, 1, H, W) or (B, H, W)
        target: Ground truth mask (B, 1, H, W) or (B, H, W)
        eps: Small value for numerical stability

    Returns:
        F1 score
    """
    return dice_score(pred, target, eps)


def pixel_accuracy(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Calculate pixel-wise accuracy.

    Args:
        pred: Predicted mask (B, 1, H, W) or (B, H, W)
        target: Ground truth mask (B, 1, H, W) or (B, H, W)

    Returns:
        Pixel accuracy
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)

    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    correct = (pred == target).float().sum(dim=(2, 3))
    total = torch.numel(pred[0])

    return (correct / total).mean()


class DiceLoss(nn.Module):
    """Dice loss for segmentation."""

    def __init__(self, eps: float = 1e-7):
        """Initialize Dice loss.

        Args:
            eps: Small value for numerical stability
        """
        super().__init__()
        self.eps = eps

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate Dice loss.

        Args:
            pred: Predicted logits (B, 1, H, W)
            target: Ground truth mask (B, 1, H, W)

        Returns:
            Dice loss
        """
        pred = torch.sigmoid(pred)

        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)

        pred = pred.flatten(1)
        target = target.flatten(1)

        intersection = (pred * target).sum(dim=1)
        union = pred.sum(dim=1) + target.sum(dim=1)

        dice = (2 * intersection + self.eps) / (union + self.eps)
        return 1 - dice.mean()


class BCEDiceLoss(nn.Module):
    """Combined BCE and Dice loss."""

    def __init__(self, bce_weight: float = 0.5, dice_weight: float = 0.5):
        """Initialize BCE + Dice loss.

        Args:
            bce_weight: Weight for BCE loss
            dice_weight: Weight for Dice loss
        """
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Calculate combined loss.

        Args:
            pred: Predicted logits (B, 1, H, W)
            target: Ground truth mask (B, 1, H, W)

        Returns:
            Combined BCE + Dice loss
        """
        bce_loss = self.bce(pred, target)
        dice_loss = self.dice(pred, target)

        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


def calculate_pdr(
    pred: torch.Tensor,
    target: torch.Tensor,
    thresholds: List[float] = [0.05, 0.08],
) -> Dict[str, float]:
    """Calculate Positive Detection Rate (PDR) at different thresholds.

    This is commonly used for CULane evaluation.

    Args:
        pred: Predicted mask (B, 1, H, W)
        target: Ground truth mask (B, 1, H, W)
        thresholds: List of IoU thresholds to evaluate

    Returns:
        Dictionary of threshold -> PDR value
    """
    if pred.dim() == 3:
        pred = pred.unsqueeze(1)
    if target.dim() == 3:
        target = target.unsqueeze(1)

    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    batch_size = pred.shape[0]
    results = {}

    for threshold in thresholds:
        correct = 0
        total = 0

        for i in range(batch_size):
            pred_i = pred[i].squeeze()
            target_i = target[i].squeeze()

            # Skip if no ground truth lane pixels
            if target_i.sum() == 0:
                continue

            intersection = (pred_i * target_i).sum()
            union = pred_i.sum() + target_i.sum() - intersection

            if union > 0:
                iou = (intersection.item() + 1e-7) / (union.item() + 1e-7)
                if iou >= threshold:
                    correct += 1
                total += 1

        pdr = correct / total if total > 0 else 0.0
        results[f"pdr_{int(threshold*100)}"] = pdr

    return results


def calculate_metrics(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> Dict[str, float]:
    """Calculate all metrics.

    Args:
        pred: Predicted mask (B, 1, H, W)
        target: Ground truth mask (B, 1, H, W)

    Returns:
        Dictionary of metric names to values
    """
    metrics = {
        "iou": iou_score(pred, target).item(),
        "dice": dice_score(pred, target).item(),
        "f1": f1_score(pred, target).item(),
        "accuracy": pixel_accuracy(pred, target).item(),
    }

    pdr_metrics = calculate_pdr(pred, target)
    metrics.update(pdr_metrics)

    return metrics


class MetricsTracker:
    """Track and aggregate metrics during training."""

    def __init__(self):
        """Initialize metrics tracker."""
        self.reset()

    def reset(self) -> None:
        """Reset all tracked metrics."""
        self.metrics = {}

    def update(self, metrics: Dict[str, float], batch_size: int = 1) -> None:
        """Update tracked metrics.

        Args:
            metrics: Dictionary of metric names to values
            batch_size: Batch size for averaging
        """
        for name, value in metrics.items():
            if name not in self.metrics:
                self.metrics[name] = {"sum": 0.0, "count": 0}

            self.metrics[name]["sum"] += value * batch_size
            self.metrics[name]["count"] += batch_size

    def get_metrics(self) -> Dict[str, float]:
        """Get average metrics.

        Returns:
            Dictionary of average metric values
        """
        return {
            name: data["sum"] / data["count"]
            for name, data in self.metrics.items()
        }

    def __str__(self) -> str:
        """String representation of metrics."""
        metrics = self.get_metrics()
        parts = [f"{name}: {value:.4f}" for name, value in metrics.items()]
        return ", ".join(parts)


class L1RegressionLoss(nn.Module):
    """L1 loss for lane coordinate regression.

    Used for direct lane coordinate prediction.
    """

    def __init__(
        self,
        reduction: str = "mean",
        normalize_coords: bool = False,
        img_height: int = 590,
        img_width: int = 1640,
    ):
        """Initialize L1 regression loss.

        Args:
            reduction: Reduction method ('mean', 'sum', 'none')
            normalize_coords: Whether to normalize coordinates to [0, 1]
            img_height: Image height for normalization
            img_width: Image width for normalization
        """
        super().__init__()
        self.reduction = reduction
        self.normalize_coords = normalize_coords
        self.img_height = img_height
        self.img_width = img_width

    def forward(
        self,
        pred_lanes: torch.Tensor,
        gt_lanes: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate L1 loss for lane coordinates.

        Args:
            pred_lanes: Predicted lane coordinates (B, max_lanes, n_points, 2)
            gt_lanes: Ground truth lane coordinates (B, max_lanes, n_points, 2)
            mask: Optional mask for valid lanes (B, max_lanes)

        Returns:
            L1 loss value
        """
        # Normalize coordinates if requested
        if self.normalize_coords:
            pred_lanes = pred_lanes / torch.tensor(
                [self.img_width, self.img_height],
                device=pred_lanes.device
            ).view(1, 1, 1, 2)
            gt_lanes = gt_lanes / torch.tensor(
                [self.img_width, self.img_height],
                device=gt_lanes.device
            ).view(1, 1, 1, 2)

        # Calculate L1 loss
        loss = torch.abs(pred_lanes - gt_lanes)

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1).unsqueeze(-1)  # (B, max_lanes, 1, 1)
            loss = loss * mask
            n_valid = mask.sum() + 1e-7
        else:
            n_valid = loss.numel()

        if self.reduction == "mean":
            return loss.sum() / n_valid
        elif self.reduction == "sum":
            return loss.sum()
        else:
            return loss


class SmoothL1RegressionLoss(nn.Module):
    """Smooth L1 loss for lane coordinate regression.

    Less sensitive to outliers than standard L1.
    """

    def __init__(
        self,
        beta: float = 1.0,
        normalize_coords: bool = False,
        img_height: int = 590,
        img_width: int = 1640,
    ):
        """Initialize Smooth L1 regression loss.

        Args:
            beta: Threshold for switching from L1 to L2
            normalize_coords: Whether to normalize coordinates
            img_height: Image height for normalization
            img_width: Image width for normalization
        """
        super().__init__()
        self.beta = beta
        self.normalize_coords = normalize_coords
        self.img_height = img_height
        self.img_width = img_width

    def forward(
        self,
        pred_lanes: torch.Tensor,
        gt_lanes: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Calculate Smooth L1 loss for lane coordinates.

        Args:
            pred_lanes: Predicted lane coordinates (B, max_lanes, n_points, 2)
            gt_lanes: Ground truth lane coordinates (B, max_lanes, n_points, 2)
            mask: Optional mask for valid lanes

        Returns:
            Smooth L1 loss value
        """
        # Normalize coordinates if requested
        if self.normalize_coords:
            pred_lanes = pred_lanes / torch.tensor(
                [self.img_width, self.img_height],
                device=pred_lanes.device
            ).view(1, 1, 1, 2)
            gt_lanes = gt_lanes / torch.tensor(
                [self.img_width, self.img_height],
                device=gt_lanes.device
            ).view(1, 1, 1, 2)

        diff = torch.abs(pred_lanes - gt_lanes)

        # Smooth L1: use quadratic for small differences, linear for large
        loss = torch.where(
            diff < self.beta,
            0.5 * diff ** 2 / self.beta,
            diff - 0.5 * self.beta,
        )

        # Apply mask if provided
        if mask is not None:
            mask = mask.unsqueeze(-1).unsqueeze(-1)
            loss = loss * mask
            n_valid = mask.sum() + 1e-7
        else:
            n_valid = loss.numel()

        return loss.sum() / n_valid


def coordinate_iou(
    pred_lanes: List[np.ndarray],
    gt_lanes: List[np.ndarray],
    img_width: int,
    img_height: int,
    tolerance: float = 0.05,
) -> float:
    """Calculate coordinate-based IoU for lane detection.

    Args:
        pred_lanes: List of predicted lane arrays
        gt_lanes: List of ground truth lane arrays
        img_width: Image width
        img_height: Image height
        tolerance: Distance tolerance for matching (relative to image size)

    Returns:
        IoU score based on coordinate matching
    """
    from .postprocess import LaneEvaluator

    evaluator = LaneEvaluator()

    # Match lanes
    pred_to_gt, _, _ = evaluator.match_lanes(pred_lanes, gt_lanes)

    # Calculate matched lane IoUs
    total_iou = 0.0
    match_count = 0

    for i, gt_idx in enumerate(pred_to_gt):
        if gt_idx is not None:
            iou = evaluator.compute_lane_iou(
                pred_lanes[i], gt_lanes[gt_idx],
                img_width, img_height
            )
            total_iou += iou
            match_count += 1

    # Average over matched pairs
    if match_count > 0:
        return total_iou / match_count

    return 0.0


def lane_point_accuracy(
    pred_lanes: List[np.ndarray],
    gt_lanes: List[np.ndarray],
    x_tolerance: int = 20,
    y_tolerance: int = 5,
) -> Dict[str, float]:
    """Calculate point-level accuracy for lane detection.

    Args:
        pred_lanes: List of predicted lane arrays
        gt_lanes: List of ground truth lane arrays
        x_tolerance: Maximum x distance for match (pixels)
        y_tolerance: Maximum y distance for match (pixels)

    Returns:
        Dictionary with precision, recall, f1
    """
    from .postprocess import LaneEvaluator

    evaluator = LaneEvaluator(x_tolerance=x_tolerance, y_tolerance=y_tolerance)

    # For each frame, evaluate
    total_tp = 0
    total_fp = 0
    total_fn = 0

    # Assume single frame for now
    metrics = evaluator.evaluate_frame(
        pred_lanes, gt_lanes,
        img_width=1640, img_height=590
    )

    return {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
        "tp": metrics["tp"],
        "fp": metrics["fp"],
        "fn": metrics["fn"],
    }
