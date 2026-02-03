"""Supervised training script for baseline comparison."""

import os
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import (
    VisualEncoder,
    DualProjectionHead,
    LaneDecoder,
)
from utils.metrics import BCEDiceLoss, calculate_metrics, MetricsTracker
from utils.logger import Logger
from .utils_train import (
    create_optimizer,
    create_scheduler,
    clip_gradients,
    move_to_device,
    format_time,
)


class SupervisedModel(nn.Module):
    """Supervised baseline model for lane detection."""

    def __init__(
        self,
        visual_encoder_name: str = "ViT-B-16",
        pretrained: str = "openai",
        freeze_visual_layers: int = 0,
        proj_dim: int = 512,
        input_size: tuple = (800, 320),
    ):
        """Initialize supervised model.

        Args:
            visual_encoder_name: CLIP ViT model name
            pretrained: Pretrained weights
            freeze_visual_layers: Number of visual layers to freeze
            proj_dim: Projection dimension
            input_size: Input image size
        """
        super().__init__()

        self.proj_dim = proj_dim
        self.input_size = input_size

        # Visual encoder
        self.visual_encoder = VisualEncoder(
            model_name=visual_encoder_name,
            pretrained=pretrained,
            freeze_layers=freeze_visual_layers,
            output_tokens=True,
        )

        # Projection head (for potential use)
        self.projection = nn.Sequential(
            nn.Linear(self.visual_encoder.output_dim, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.ReLU(inplace=True),
        )

        # Lane decoder
        self.decoder = LaneDecoder(in_dim=proj_dim, input_size=input_size)

        # Loss function
        self.criterion = BCEDiceLoss()

    def forward(
        self,
        image: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            image: Input image (B, 3, H, W)
            mask: Optional ground truth mask (B, 1, H, W)

        Returns:
            Dictionary with prediction and optional loss
        """
        # Encode visual features
        f_v = self.visual_encoder(image)  # (B, D)

        # Project
        z_v = self.projection(f_v)  # (B, proj_dim)

        # Get feature map for decoder
        feat_map = self.visual_encoder.get_feature_map(image)

        # Decode to lane mask
        pred_mask = self.decoder(feat_map)  # (B, 1, H, W)

        result = {"pred_mask": pred_mask}

        # Compute loss if mask is provided
        if mask is not None:
            loss = self.criterion(pred_mask, mask)
            result["loss"] = loss

        return result

    def predict(self, image: torch.Tensor) -> torch.Tensor:
        """Predict lane mask.

        Args:
            image: Input image (B, 3, H, W)

        Returns:
            Predicted lane mask (B, 1, H, W)
        """
        self.eval()

        with torch.no_grad():
            output = self.forward(image)
            pred_mask = torch.sigmoid(output["pred_mask"])

        return pred_mask


def train_one_epoch(
    model: SupervisedModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    use_amp: bool = True,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: Supervised model
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        use_amp: Whether to use mixed precision
        grad_clip: Gradient clipping norm

    Returns:
        Dictionary of average metrics
    """
    model.train()

    metrics = MetricsTracker()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    pbar = tqdm(dataloader, desc="Training")

    for batch in pbar:
        batch = move_to_device(batch, device)

        view1 = batch["view1"]
        mask = batch.get("mask")

        if mask is None:
            continue

        # Forward pass
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(view1, mask)
                loss = output["loss"]
        else:
            output = model(view1, mask)
            loss = output["loss"]

        # Backward pass
        optimizer.zero_grad()

        if use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            clip_gradients(model, grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            clip_gradients(model, grad_clip)
            optimizer.step()

        # Track metrics
        batch_metrics = {"loss": loss.item()}
        metrics.update(batch_metrics, batch_size=view1.shape[0])

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    return metrics.get_metrics()


@torch.no_grad()
def validate(
    model: SupervisedModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model.

    Args:
        model: Supervised model
        dataloader: Validation dataloader
        device: Device to validate on

    Returns:
        Dictionary of validation metrics
    """
    model.eval()

    metrics = MetricsTracker()
    pbar = tqdm(dataloader, desc="Validation")

    for batch in pbar:
        batch = move_to_device(batch, device)

        view1 = batch["view1"]
        mask = batch.get("mask")

        if mask is None:
            continue

        # Forward pass
        output = model(view1)
        pred_mask = torch.sigmoid(output["pred_mask"])

        # Compute metrics
        batch_metrics = calculate_metrics(pred_mask, mask)
        metrics.update(batch_metrics, batch_size=view1.shape[0])

        pbar.set_postfix({k: f"{v:.4f}" for k, v in batch_metrics.items()})

    return metrics.get_metrics()


def train(
    config: Dict[str, Any],
    logger: Logger,
    resume_from: Optional[str] = None,
) -> None:
    """Main training function.

    Args:
        config: Training configuration
        logger: Logger instance
        resume_from: Optional checkpoint path to resume from
    """
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    model = SupervisedModel(
        visual_encoder_name=config["model"]["visual_encoder"],
        pretrained=config["model"]["pretrained"],
        freeze_visual_layers=config.get("freeze_visual_layers_supervised", 0),
        proj_dim=config["model"]["proj_dim"],
        input_size=tuple(config["data"]["input_size"]),
    ).to(device)

    print(f"Model parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M")

    # Create dataloaders
    from data import create_dataloader

    train_loader = create_dataloader(
        data_root=config["data"]["data_root"],
        dataset=config["data"]["dataset"],
        split="train",
        batch_size=config["training"]["batch_size"],
        input_size=tuple(config["data"]["input_size"]),
        num_workers=config["data"]["num_workers"],
        subset=config.get("subset", None),
    )

    val_loader = create_dataloader(
        data_root=config["data"]["data_root"],
        dataset=config["data"]["dataset"],
        split="val",
        batch_size=config["training"]["batch_size"],
        input_size=tuple(config["data"]["input_size"]),
        num_workers=config["data"]["num_workers"],
    )

    # Create optimizer and scheduler
    optimizer = create_optimizer(
        model,
        lr=config["training"]["lr"],
        weight_decay=config["training"]["weight_decay"],
    )

    scheduler = create_scheduler(
        optimizer,
        epochs=config["training"]["epochs"],
        warmup_epochs=config["training"]["warmup_epochs"],
        steps_per_epoch=len(train_loader),
    )

    # Resume from checkpoint if specified
    start_epoch = 0
    best_iou = 0.0

    if resume_from:
        checkpoint = torch.load(resume_from, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Resumed from epoch {start_epoch}")

    # Save config
    logger.save_config(config)

    # Training loop
    for epoch in range(start_epoch, config["training"]["epochs"]):
        logger.start_epoch()

        print(f"\nEpoch {epoch + 1}/{config['training']['epochs']}")

        # Train
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            use_amp=torch.cuda.is_available(),
            grad_clip=config["training"]["gradient_clip"],
        )

        # Step scheduler
        scheduler.step()

        # Log training metrics
        for name, value in train_metrics.items():
            print(f"  Train {name}: {value:.4f}")

        logger.log_metrics(train_metrics, epoch, prefix="train")

        # Validate
        if (epoch + 1) % config["logging"]["eval_interval"] == 0:
            val_metrics = validate(model, val_loader, device)

            for name, value in val_metrics.items():
                print(f"  Val {name}: {value:.4f}")

            logger.log_metrics(val_metrics, epoch, prefix="val")

            # Save best model
            if val_metrics.get("iou", 0) > best_iou:
                best_iou = val_metrics["iou"]
                logger.save_checkpoint(
                    model,
                    optimizer,
                    scheduler,
                    epoch,
                    val_metrics,
                    "best.pth",
                )
                print(f"  New best IoU: {best_iou:.4f}")

        # Save periodic checkpoint
        if (epoch + 1) % config["logging"]["save_interval"] == 0:
            logger.save_checkpoint(
                model,
                optimizer,
                scheduler,
                epoch,
                train_metrics,
                f"epoch_{epoch + 1}.pth",
            )

        elapsed = logger.end_epoch(epoch)
        print(f"  Time: {format_time(elapsed)}")

    print("\nTraining completed!")
    print(f"Best IoU: {best_iou:.4f}")
