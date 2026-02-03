"""Training utility functions."""

import os
from typing import Dict, Any, Optional, List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


def create_optimizer(
    model: nn.Module,
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    freeze_bn: bool = True,
) -> torch.optim.Optimizer:
    """Create optimizer with parameter groups.

    Args:
        model: Model to optimize
        lr: Learning rate
        weight_decay: Weight decay
        freeze_bn: Whether to exclude BN from weight decay

    Returns:
        Configured optimizer
    """
    # Separate parameters for weight decay
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]

    if freeze_bn:
        no_decay.extend(["bn", "BatchNorm"])

    optimizer_grouped_parameters = [
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if not any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": weight_decay,
        },
        {
            "params": [
                p
                for n, p in model.named_parameters()
                if any(nd in n for nd in no_decay) and p.requires_grad
            ],
            "weight_decay": 0.0,
        },
    ]

    return AdamW(optimizer_grouped_parameters, lr=lr)


def create_scheduler(
    optimizer: torch.optim.Optimizer,
    epochs: int,
    warmup_epochs: int = 5,
    steps_per_epoch: Optional[int] = None,
    min_lr: float = 1e-6,
) -> torch.optim.lr_scheduler._LRScheduler:
    """Create cosine learning rate scheduler with warmup.

    Args:
        optimizer: Optimizer to schedule
        epochs: Total number of epochs
        warmup_epochs: Number of warmup epochs
        steps_per_epoch: Steps per epoch (for per-step scheduling)
        min_lr: Minimum learning rate

    Returns:
        Learning rate scheduler
    """
    if steps_per_epoch is None:
        # Per-epoch scheduling
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_epochs,
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=epochs - warmup_epochs,
            eta_min=min_lr,
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_epochs],
        )
    else:
        # Per-step scheduling
        total_steps = epochs * steps_per_epoch
        warmup_steps = warmup_epochs * steps_per_epoch

        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            total_iters=warmup_steps,
        )

        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_steps - warmup_steps,
            eta_min=min_lr,
        )

        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps],
        )

    return scheduler


def get_gradient_norm(
    model: nn.Module,
) -> float:
    """Compute total gradient norm.

    Args:
        model: Model to compute gradients for

    Returns:
        Total gradient norm
    """
    total_norm = 0.0

    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2

    return total_norm ** 0.5


def clip_gradients(
    model: nn.Module,
    max_norm: float = 1.0,
) -> float:
    """Clip gradients.

    Args:
        model: Model to clip gradients for
        max_norm: Maximum gradient norm

    Returns:
        Total gradient norm before clipping
    """
    return torch.nn.utils.clip_grad_norm_(
        model.parameters(),
        max_norm,
    )


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    epoch: int,
    metrics: Dict[str, float],
    path: str,
) -> None:
    """Save training checkpoint.

    Args:
        model: Model to save
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch
        metrics: Current metrics
        path: Save path
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, path)


def load_checkpoint(
    model: nn.Module,
    path: str,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler: Optional[Any] = None,
) -> Dict[str, Any]:
    """Load training checkpoint.

    Args:
        model: Model to load weights into
        path: Checkpoint path
        optimizer: Optional optimizer to load state into
        scheduler: Optional scheduler to load state into

    Returns:
        Checkpoint dictionary with epoch and metrics
    """
    checkpoint = torch.load(path, map_location="cpu")

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    if scheduler is not None and "scheduler_state_dict" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "metrics": checkpoint.get("metrics", {}),
    }


def move_to_device(
    batch: Dict[str, Any],
    device: torch.device,
) -> Dict[str, Any]:
    """Move batch tensors to device.

    Args:
        batch: Input batch
        device: Target device

    Returns:
        Batch with tensors moved to device
    """
    result = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            result[key] = value.to(device, non_blocking=True)
        elif isinstance(value, list):
            # Keep lists as-is (e.g., text strings)
            result[key] = value
        else:
            result[key] = value

    return result


class AverageMeter:
    """Compute and store average and current value."""

    def __init__(self):
        """Initialize meter."""
        self.reset()

    def reset(self) -> None:
        """Reset meter."""
        self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0

    def update(
        self,
        val: float,
        n: int = 1,
    ) -> None:
        """Update meter.

        Args:
            val: New value
            n: Number of items
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping based on metric improvement."""

    def __init__(
        self,
        patience: int = 10,
        mode: str = "max",
        min_delta: float = 0.0,
    ):
        """Initialize early stopping.

        Args:
            patience: Number of epochs to wait for improvement
            mode: 'max' or 'min'
            min_delta: Minimum change to qualify as improvement
        """
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(
        self,
        score: float,
    ) -> bool:
        """Check if should stop early.

        Args:
            score: Current metric value

        Returns:
            True if should stop early
        """
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop


def format_time(
    seconds: float,
) -> str:
    """Format time in human-readable format.

    Args:
        seconds: Time in seconds

    Returns:
        Formatted time string
    """
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)

    if hours > 0:
        return f"{hours}h {minutes}m {secs}s"
    elif minutes > 0:
        return f"{minutes}m {secs}s"
    else:
        return f"{secs}s"
