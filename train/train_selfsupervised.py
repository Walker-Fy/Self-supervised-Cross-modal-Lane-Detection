"""Self-supervised training script for SCC Lane Detection."""

import os
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from models import (
    VisualEncoder,
    LanguageEncoder,
    DualProjectionHead,
    LaneDecoder,
    FusionHead,
    InfoNCELoss,
    CosineConsistencyLoss,
)
from utils.metrics import DiceLoss, calculate_metrics, MetricsTracker
from utils.logger import Logger
from .utils_train import (
    create_optimizer,
    create_scheduler,
    clip_gradients,
    move_to_device,
    AverageMeter,
    format_time,
)


class SCCModel(nn.Module):
    """Self-supervised Cross-modal Consistency model for lane detection."""

    def __init__(
        self,
        visual_encoder_name: str = "ViT-B-16",
        pretrained: str = "openai",
        freeze_visual_layers: int = 6,
        freeze_text_encoder: bool = True,
        proj_dim: int = 512,
        input_size: tuple = (800, 320),
        decoder_type: str = "simple",
    ):
        """Initialize SCC model.

        Args:
            visual_encoder_name: CLIP ViT model name
            pretrained: Pretrained weights
            freeze_visual_layers: Number of visual layers to freeze
            freeze_text_encoder: Whether to freeze text encoder
            proj_dim: Projection dimension
            input_size: Input image size
            decoder_type: Type of decoder ('simple', 'unet', 'fpn')
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

        # Language (text) encoder - frozen by default
        self.language_encoder = LanguageEncoder(
            model_name=visual_encoder_name,
            pretrained=pretrained,
            frozen=freeze_text_encoder,
        )

        # Projection heads
        self.projection = DualProjectionHead(
            visual_dim=self.visual_encoder.output_dim,
            language_dim=self.language_encoder.output_dim,
            proj_dim=proj_dim,
        )

        # Fusion head (optional)
        self.fusion = FusionHead(
            visual_dim=proj_dim,
            language_dim=proj_dim,
            out_dim=proj_dim,
            fusion_type="add",
        )

        # Lane decoder
        if decoder_type == "unet":
            from models.lane_decoder import UNetDecoder
            self.decoder = UNetDecoder(in_dim=proj_dim, input_size=input_size)
        elif decoder_type == "fpn":
            from models.lane_decoder import FPNDecoder
            self.decoder = FPNDecoder(in_dim=proj_dim, input_size=input_size)
        else:
            self.decoder = LaneDecoder(in_dim=proj_dim, input_size=input_size)

        # Loss functions
        self.info_nce_loss = InfoNCELoss(temperature=0.07)
        self.cosine_loss = CosineConsistencyLoss()
        self.dice_loss = DiceLoss()

    def forward(
        self,
        view1: torch.Tensor,
        view2: torch.Tensor,
        texts: list,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            view1: First augmented view (B, 3, H, W)
            view2: Second augmented view (B, 3, H, W)
            texts: List of text captions
            mask: Optional ground truth masks (B, 1, H, W)

        Returns:
            Dictionary with losses and predictions
        """
        batch_size = view1.shape[0]

        # Encode visual features
        f_v1 = self.visual_encoder(view1)  # (B, D)
        f_v2 = self.visual_encoder(view2)  # (B, D)

        # Encode text features
        f_l = self.language_encoder.encode_text(texts)  # (B, D)

        # Project to shared space
        proj_v1 = self.projection.visual_proj(f_v1)  # (B, proj_dim)
        proj_v2 = self.projection.visual_proj(f_v2)  # (B, proj_dim)
        proj_l = self.projection.language_proj(f_l)  # (B, proj_dim)

        # Cross-modal InfoNCE loss
        loss_info_nce = self.info_nce_loss(proj_v1, proj_l)

        # Cosine consistency loss
        loss_cosine = self.cosine_loss(proj_v1, proj_v2)

        # Fusion for segmentation
        z_fused = self.fusion(proj_v1, proj_l)

        # Get feature map for decoder
        feat_map = self.visual_encoder.get_feature_map(view1)

        # Decode to lane mask
        pred_mask = self.decoder(feat_map)  # (B, 1, H, W)

        result = {
            "pred_mask": pred_mask,
            "loss_info_nce": loss_info_nce,
            "loss_cosine": loss_cosine,
        }

        # Compute Dice loss if mask is provided
        if mask is not None:
            loss_dice = self.dice_loss(pred_mask, mask)
            result["loss_dice"] = loss_dice

        return result

    def predict(
        self,
        image: torch.Tensor,
        text: str = "a road with 2 lanes",
    ) -> torch.Tensor:
        """Predict lane mask.

        Args:
            image: Input image (B, 3, H, W)
            text: Optional text description

        Returns:
            Predicted lane mask (B, 1, H, W)
        """
        self.eval()

        with torch.no_grad():
            # Encode visual
            f_v = self.visual_encoder(image)

            # Encode text
            f_l = self.language_encoder.encode_text([text] * image.shape[0])

            # Project
            proj_v = self.projection.visual_proj(f_v)
            proj_l = self.projection.language_proj(f_l)

            # Fuse
            z_fused = self.fusion(proj_v, proj_l)

            # Get feature map and decode
            feat_map = self.visual_encoder.get_feature_map(image)
            pred_mask = self.decoder(feat_map)

        return pred_mask


def train_one_epoch(
    model: SCCModel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    lambda_con: float = 0.5,
    lambda_cos: float = 0.1,
    lambda_dice: float = 1.0,
    use_amp: bool = True,
    grad_clip: float = 1.0,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: SCC model
        dataloader: Training dataloader
        optimizer: Optimizer
        device: Device to train on
        lambda_con: InfoNCE loss weight
        lambda_cos: Cosine consistency loss weight
        lambda_dice: Dice loss weight
        use_amp: Whether to use mixed precision
        grad_clip: Gradient clipping norm

    Returns:
        Dictionary of average metrics
    """
    model.train()

    # Unfreeze training parameters
    for param in model.visual_encoder.parameters():
        if param.requires_grad:
            param.requires_grad = True

    # Keep text encoder frozen
    for param in model.language_encoder.parameters():
        param.requires_grad = False

    metrics = MetricsTracker()
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    pbar = tqdm(dataloader, desc="Training")

    for batch in pbar:
        batch = move_to_device(batch, device)

        view1 = batch["view1"]
        view2 = batch["view2"]
        texts = batch["texts"]
        mask = batch.get("mask")

        # Forward pass
        if use_amp:
            with torch.cuda.amp.autocast():
                output = model(view1, view2, texts, mask)

                # Compute total loss
                loss = (
                    lambda_con * output["loss_info_nce"] +
                    lambda_cos * output["loss_cosine"] +
                    lambda_dice * output["loss_dice"]
                )
        else:
            output = model(view1, view2, texts, mask)
            loss = (
                lambda_con * output["loss_info_nce"] +
                lambda_cos * output["loss_cosine"] +
                lambda_dice * output["loss_dice"]
            )

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
        batch_metrics = {
            "loss": loss.item(),
            "loss_info_nce": output["loss_info_nce"].item(),
            "loss_cosine": output["loss_cosine"].item(),
            "loss_dice": output["loss_dice"].item(),
        }
        metrics.update(batch_metrics, batch_size=view1.shape[0])

        pbar.set_postfix({k: f"{v:.4f}" for k, v in batch_metrics.items()})

    return metrics.get_metrics()


@torch.no_grad()
def validate(
    model: SCCModel,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model.

    Args:
        model: SCC model
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
        view2 = batch["view2"]
        texts = batch["texts"]
        mask = batch.get("mask")

        if mask is None:
            continue

        # Forward pass
        output = model(view1, view2, texts, mask)
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
    model = SCCModel(
        visual_encoder_name=config["model"]["visual_encoder"],
        pretrained=config["model"]["pretrained"],
        freeze_visual_layers=config["model"]["freeze_visual_layers"],
        freeze_text_encoder=config["model"]["freeze_text_encoder"],
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
            lambda_con=config["loss"]["lambda_con"],
            lambda_cos=config["loss"]["lambda_cos"],
            lambda_dice=config["loss"]["lambda_dice"],
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
