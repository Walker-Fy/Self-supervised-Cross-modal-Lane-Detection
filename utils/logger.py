"""Logging utilities for training."""

import os
import time
from typing import Dict, Any, Optional
import json

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Training logger with TensorBoard support."""

    def __init__(
        self,
        log_dir: str,
        experiment_name: str,
        use_tensorboard: bool = True,
        use_wandb: bool = False,
        wandb_project: Optional[str] = None,
    ):
        """Initialize logger.

        Args:
            log_dir: Directory to save logs
            experiment_name: Name of the experiment
            use_tensorboard: Whether to use TensorBoard
            use_wandb: Whether to use Weights & Biases
            wandb_project: WandB project name
        """
        self.log_dir = log_dir
        self.experiment_name = experiment_name
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb

        os.makedirs(log_dir, exist_ok=True)

        if use_tensorboard:
            tb_dir = os.path.join(log_dir, "tensorboard", experiment_name)
            self.writer = SummaryWriter(tb_dir)
        else:
            self.writer = None

        if use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, name=experiment_name)
                self.wandb = wandb
            except ImportError:
                print("wandb not installed. Skipping wandb logging.")
                self.use_wandb = False
                self.wandb = None
        else:
            self.wandb = None

        self.metrics_history = []
        self.epoch_start_time = None

    def log_metrics(
        self,
        metrics: Dict[str, float],
        step: int,
        prefix: str = "",
    ) -> None:
        """Log metrics to TensorBoard and W&B.

        Args:
            metrics: Dictionary of metric names to values
            step: Current training step
            prefix: Prefix to add to metric names
        """
        for name, value in metrics.items():
            full_name = f"{prefix}/{name}" if prefix else name

            if self.writer is not None:
                self.writer.add_scalar(full_name, value, step)

            if self.use_wandb and self.wandb is not None:
                self.wandb.log({full_name: value}, step=step)

        self.metrics_history.append({"step": step, **metrics})

    def log_image(
        self,
        tag: str,
        image: torch.Tensor,
        step: int,
    ) -> None:
        """Log image to TensorBoard.

        Args:
            tag: Image tag
            image: Image tensor (C, H, W) or (B, C, H, W)
            step: Current training step
        """
        if self.writer is not None:
            self.writer.add_image(tag, image, step)

    def log_images(
        self,
        tag: str,
        images: torch.Tensor,
        step: int,
        max_images: int = 8,
    ) -> None:
        """Log grid of images to TensorBoard.

        Args:
            tag: Image tag
            images: Image tensor (B, C, H, W)
            step: Current training step
            max_images: Maximum number of images to log
        """
        if self.writer is not None:
            grid = images[:max_images]
            self.writer.add_images(tag, grid, step)

    def save_checkpoint(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: Optional[Any],
        epoch: int,
        metrics: Dict[str, float],
        filename: str,
    ) -> str:
        """Save model checkpoint.

        Args:
            model: Model to save
            optimizer: Optimizer state
            scheduler: Learning rate scheduler state
            epoch: Current epoch
            metrics: Current metrics
            filename: Checkpoint filename

        Returns:
            Path to saved checkpoint
        """
        checkpoint_dir = os.path.join(self.log_dir, "checkpoints")
        os.makedirs(checkpoint_dir, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
        }

        if scheduler is not None:
            checkpoint["scheduler_state_dict"] = scheduler.state_dict()

        path = os.path.join(checkpoint_dir, filename)
        torch.save(checkpoint, path)

        return path

    def save_config(self, config: Dict[str, Any]) -> str:
        """Save configuration to JSON file.

        Args:
            config: Configuration dictionary

        Returns:
            Path to saved config
        """
        config_path = os.path.join(self.log_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)

        return config_path

    def start_epoch(self) -> None:
        """Mark start of an epoch for timing."""
        self.epoch_start_time = time.time()

    def end_epoch(self, epoch: int) -> float:
        """Mark end of an epoch and return duration.

        Args:
            epoch: Current epoch number

        Returns:
            Time elapsed for the epoch in seconds
        """
        if self.epoch_start_time is None:
            return 0.0

        elapsed = time.time() - self.epoch_start_time
        return elapsed

    def close(self) -> None:
        """Close logger and save metrics history."""
        if self.writer is not None:
            self.writer.close()

        metrics_path = os.path.join(self.log_dir, "metrics_history.json")
        with open(metrics_path, "w") as f:
            json.dump(self.metrics_history, f, indent=2)

        if self.use_wandb and self.wandb is not None:
            self.wandb.finish()
