"""CULane Official Evaluation Script.

This script evaluates lane detection models on the CULane dataset
using the official evaluation protocol and generates predictions
in the official .lines.txt format.
"""

import os
import sys
import argparse
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from train.train_selfsupervised import SCCModel
from data.dataset_loader import create_dataloader
from utils.postprocess import LaneExtractor, LaneEvaluator
from utils.tta import LaneDetectionTTA


def load_model(
    checkpoint_path: str,
    device: torch.device,
    input_size: Tuple[int, int] = (590, 1640),
    decoder_type: str = "simple",
) -> nn.Module:
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        input_size: Input image size
        decoder_type: Type of decoder

    Returns:
        Loaded model
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model config from checkpoint if available
    if "config" in checkpoint:
        config = checkpoint["config"]
        model = SCCModel(
            visual_encoder_name=config.get("model", {}).get("visual_encoder", "ViT-B-16"),
            pretrained=config.get("model", {}).get("pretrained", "openai"),
            freeze_visual_layers=config.get("model", {}).get("freeze_visual_layers", 6),
            freeze_text_encoder=config.get("model", {}).get("freeze_text_encoder", True),
            proj_dim=config.get("model", {}).get("proj_dim", 512),
            input_size=input_size,
            decoder_type=decoder_type,
        )
    else:
        # Default config
        model = SCCModel(
            visual_encoder_name="ViT-B-16",
            pretrained="openai",
            proj_dim=512,
            input_size=input_size,
            decoder_type=decoder_type,
        )

    # Load state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    return model


def generate_predictions(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    output_dir: str,
    input_size: Tuple[int, int] = (590, 1640),
    use_tta: bool = False,
    tta_scales: List[float] = [0.75, 1.0, 1.25],
    n_lanes: int = 4,
    n_points: int = 56,
    confidence_threshold: float = 0.5,
) -> Dict[str, Any]:
    """Generate predictions in CULane format.

    Args:
        model: Trained model
        dataloader: Test dataloader
        device: Device to run inference on
        output_dir: Directory to save predictions
        input_size: Input image size
        use_tta: Whether to use test-time augmentation
        tta_scales: TTA scale factors
        n_lanes: Maximum number of lanes
        n_points: Number of points per lane
        confidence_threshold: Mask confidence threshold

    Returns:
        Dictionary with statistics
    """
    os.makedirs(output_dir, exist_ok=True)

    # Initialize lane extractor
    extractor = LaneExtractor(
        n_lanes=n_lanes,
        n_points=n_points,
        confidence_threshold=confidence_threshold,
        input_size=input_size,
        original_size=input_size,
    )

    # Wrap model with TTA if requested
    if use_tta:
        predictor = LaneDetectionTTA(
            model,
            scales=tta_scales,
            flip=True,
            n_lanes=n_lanes,
            n_points=n_points,
            confidence_threshold=confidence_threshold,
        )
    else:
        predictor = None

    statistics = {
        "total_images": 0,
        "total_lanes": 0,
        "empty_predictions": 0,
        "avg_lanes_per_image": 0,
    }

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Generating predictions"):
            image_paths = batch.get("image_paths", [])
            if not image_paths:
                continue

            # Get images
            if "view1" in batch:
                images = batch["view1"]
            elif "image" in batch:
                images = batch["image"]
            else:
                continue

            images = images.to(device)

            # Get predictions
            if use_tta and predictor is not None:
                results = predictor.predict_batch(
                    [images[i:i+1] for i in range(len(images))],
                    texts=["a road with lanes"] * len(images),
                )

                for i, (img_path, result) in enumerate(zip(image_paths, results)):
                    lanes = result["lanes"]
                    _save_prediction(img_path, lanes, output_dir, extractor)
                    statistics["total_images"] += 1
                    statistics["total_lanes"] += len(lanes)
                    if len(lanes) == 0:
                        statistics["empty_predictions"] += 1

            else:
                # Standard inference
                outputs = model.get_feature_map(images) if hasattr(model, "get_feature_map") else None

                if outputs is None:
                    # Try forward pass
                    output = model(images, images, ["a road with lanes"] * len(images))
                    if isinstance(output, dict):
                        masks = output.get("pred_mask", output.get("mask"))
                    else:
                        masks = output
                else:
                    # Decode features
                    masks = model.decoder(outputs)

                # Apply sigmoid if needed
                if masks.min() < 0:
                    masks = torch.sigmoid(masks)

                # Process each image
                for i, img_path in enumerate(image_paths):
                    mask = masks[i].squeeze().cpu().numpy()

                    # Extract lanes
                    lanes = extractor.process_mask(mask)

                    # Save prediction
                    _save_prediction(img_path, lanes, output_dir, extractor)

                    statistics["total_images"] += 1
                    statistics["total_lanes"] += len(lanes)
                    if len(lanes) == 0:
                        statistics["empty_predictions"] += 1

    # Compute averages
    if statistics["total_images"] > 0:
        statistics["avg_lanes_per_image"] = statistics["total_lanes"] / statistics["total_images"]

    return statistics


def _save_prediction(
    image_path: str,
    lanes: List[np.ndarray],
    output_dir: str,
    extractor: LaneExtractor,
) -> None:
    """Save prediction to .lines.txt file.

    Args:
        image_path: Original image path
        lanes: List of lane arrays
        output_dir: Output directory
        extractor: Lane extractor instance
    """
    # Generate output filename
    img_name = Path(image_path).stem
    output_path = os.path.join(output_dir, f"{img_name}.lines.txt")

    # Save lanes
    extractor.lanes_to_txt(lanes, output_path)


def evaluate_official(
    predictions_dir: str,
    gt_dir: str,
    list_file: Optional[str] = None,
) -> Dict[str, float]:
    """Run official CULane evaluation.

    Args:
        predictions_dir: Directory containing prediction .lines.txt files
        gt_dir: Directory containing ground truth files
        list_file: Optional file listing test images

    Returns:
        Dictionary with evaluation metrics
    """
    # Try to import official evaluation
    try:
        from evaluate import evaluate as culane_evaluate

        return culane_evaluate(predictions_dir, gt_dir)
    except ImportError:
        # Fallback to our own evaluation
        return _evaluate_with_fallback(predictions_dir, gt_dir, list_file)


def _evaluate_with_fallback(
    predictions_dir: str,
    gt_dir: str,
    list_file: Optional[str] = None,
) -> Dict[str, float]:
    """Fallback evaluation when official script is unavailable.

    Args:
        predictions_dir: Directory with predictions
        gt_dir: Directory with ground truth
        list_file: Optional list file

    Returns:
        Dictionary with metrics
    """
    evaluator = LaneEvaluator()

    # Get prediction files
    pred_files = sorted(Path(predictions_dir).glob("*.lines.txt"))

    if not pred_files:
        print(f"Warning: No prediction files found in {predictions_dir}")
        return {"tp": 0, "fp": 0, "fn": 0, "precision": 0, "recall": 0, "f1": 0}

    tp_total = 0
    fp_total = 0
    fn_total = 0

    for pred_file in pred_files:
        # Load predictions
        extractor = LaneExtractor()
        pred_lanes = extractor.lanes_from_txt(str(pred_file))

        # Load ground truth
        gt_file = Path(gt_dir) / pred_file.name
        if not gt_file.exists():
            # Try to find by pattern
            gt_file = None
            for f in Path(gt_dir).glob("*.txt"):
                if pred_file.stem in f.stem:
                    gt_file = f
                    break

        if gt_file is None or not gt_file.exists():
            # All predictions are false positives
            fp_total += len(pred_lanes)
            continue

        gt_lanes = extractor.lanes_from_txt(str(gt_file))

        # Evaluate this frame
        frame_metrics = evaluator.evaluate_frame(
            pred_lanes, gt_lanes,
            img_width=1640, img_height=590
        )

        tp_total += frame_metrics["tp"]
        fp_total += frame_metrics["fp"]
        fn_total += frame_metrics["fn"]

    # Compute overall metrics
    precision = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    recall = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "tp": tp_total,
        "fp": fp_total,
        "fn": fn_total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def evaluate_culane(
    model: nn.Module,
    data_root: str,
    checkpoint_path: str,
    output_dir: str = "./results",
    use_tta: bool = True,
    tta_scales: List[float] = [0.75, 1.0, 1.25],
    batch_size: int = 8,
    num_workers: int = 4,
    input_size: Tuple[int, int] = (590, 1640),
    gt_dir: Optional[str] = None,
    list_file: Optional[str] = None,
) -> Dict[str, Any]:
    """Complete CULane evaluation pipeline.

    Args:
        model: Trained model
        data_root: Root directory of CULane dataset
        checkpoint_path: Path to model checkpoint
        output_dir: Directory to save predictions
        use_tta: Whether to use test-time augmentation
        tta_scales: TTA scale factors
        batch_size: Batch size for inference
        num_workers: Number of data loading workers
        input_size: Input image size
        gt_dir: Ground truth directory for evaluation
        list_file: Optional list file for test split

    Returns:
        Dictionary with generation and evaluation results
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print(f"Loading checkpoint: {checkpoint_path}")
    model = load_model(checkpoint_path, device, input_size)

    # Create dataloader
    print("Creating test dataloader...")
    try:
        test_loader = create_dataloader(
            data_root=data_root,
            dataset="culane",
            split="test",
            batch_size=batch_size,
            input_size=input_size,
            num_workers=num_workers,
            shuffle=False,
        )
    except Exception as e:
        print(f"Warning: Could not create test dataloader: {e}")
        print("Attempting to use val split instead...")
        test_loader = create_dataloader(
            data_root=data_root,
            dataset="culane",
            split="val",
            batch_size=batch_size,
            input_size=input_size,
            num_workers=num_workers,
            shuffle=False,
        )

    # Generate predictions
    print(f"Generating predictions to: {output_dir}")
    statistics = generate_predictions(
        model=model,
        dataloader=test_loader,
        device=device,
        output_dir=output_dir,
        input_size=input_size,
        use_tta=use_tta,
        tta_scales=tta_scales,
    )

    print("\nPrediction Statistics:")
    for key, value in statistics.items():
        print(f"  {key}: {value}")

    # Run evaluation if ground truth is provided
    evaluation_results = None
    if gt_dir is not None:
        print("\nRunning official evaluation...")
        evaluation_results = evaluate_official(output_dir, gt_dir, list_file)

        print("\nEvaluation Results:")
        for key, value in evaluation_results.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")

    return {
        "statistics": statistics,
        "evaluation": evaluation_results,
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="CULane Official Evaluation for SCC Lane Detection"
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="./data/CULane",
        help="Root directory of CULane dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results",
        help="Directory to save predictions",
    )
    parser.add_argument(
        "--gt-dir",
        type=str,
        default=None,
        help="Ground truth directory for evaluation",
    )
    parser.add_argument(
        "--list",
        type=str,
        default=None,
        help="List file for test split",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--input-size",
        type=int,
        nargs=2,
        default=[590, 1640],
        help="Input size (height width)",
    )
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable test-time augmentation",
    )
    parser.add_argument(
        "--tta-scales",
        type=float,
        nargs="+",
        default=[0.75, 1.0, 1.25],
        help="TTA scale factors",
    )
    parser.add_argument(
        "--n-lanes",
        type=int,
        default=4,
        help="Maximum number of lanes",
    )
    parser.add_argument(
        "--n-points",
        type=int,
        default=56,
        help="Number of points per lane",
    )
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold for mask binarization",
    )

    args = parser.parse_args()

    # Create a dummy model for the function signature
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = load_model(
        checkpoint_path=args.checkpoint,
        device=device,
        input_size=tuple(args.input_size),
    )

    # Run evaluation
    results = evaluate_culane(
        model=model,
        data_root=args.data_root,
        checkpoint_path=args.checkpoint,
        output_dir=args.output_dir,
        use_tta=not args.no_tta,
        tta_scales=args.tta_scales,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        input_size=tuple(args.input_size),
        gt_dir=args.gt_dir,
        list_file=args.list,
    )

    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()
