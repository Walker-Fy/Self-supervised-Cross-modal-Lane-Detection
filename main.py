#!/usr/bin/env python3
"""Main entry point for SCC Lane Detection training and evaluation."""

import argparse
import os
import sys
from pathlib import Path

import yaml
import torch

from utils.seed import set_seed
from utils.logger import Logger


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="SCC Lane Detection - Self-supervised Cross-modal Consistency"
    )

    # Basic arguments
    parser.add_argument(
        "--config",
        type=str,
        default="config/default.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--task",
        type=str,
        choices=["train", "train_selfsupervised", "train_supervised", "evaluate",
                 "evaluate_culane", "robustness", "crossdomain", "visualize", "test"],
        default="train_selfsupervised",
        help="Task to perform",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to checkpoint for evaluation/resuming",
    )

    # Data arguments
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Override data root directory",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        choices=["culane", "tusimple", "dummy"],
        default=None,
        help="Override dataset",
    )

    # Training arguments
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Override number of epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Override learning rate",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="GPU IDs to use (comma-separated)",
    )

    # Debug/testing arguments
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (fast iteration)",
    )
    parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Use subset of data for testing",
    )
    parser.add_argument(
        "--use-dummy-data",
        action="store_true",
        help="Use dummy dataset for testing",
    )

    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs",
        help="Output directory for logs and checkpoints",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="scc_lane_detection",
        help="Experiment name for logging",
    )

    # Distributed training
    parser.add_argument(
        "--distributed",
        action="store_true",
        help="Enable distributed training",
    )
    parser.add_argument(
        "--local-rank",
        type=int,
        default=0,
        help="Local rank for distributed training",
    )

    # CULane evaluation arguments
    parser.add_argument(
        "--no-tta",
        action="store_true",
        help="Disable test-time augmentation",
    )
    parser.add_argument(
        "--tta-scales",
        type=float,
        nargs="+",
        default=None,
        help="TTA scale factors (e.g., 0.75 1.0 1.25)",
    )
    parser.add_argument(
        "--prediction-dir",
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

    return parser.parse_args()


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def override_config(config: dict, args: argparse.Namespace) -> dict:
    """Override config with command line arguments.

    Args:
        config: Original config
        args: Command line arguments

    Returns:
        Updated config
    """
    if args.data_root is not None:
        config["data"]["data_root"] = args.data_root

    if args.dataset is not None:
        config["data"]["dataset"] = args.dataset

    if args.batch_size is not None:
        config["training"]["batch_size"] = args.batch_size

    if args.epochs is not None:
        config["training"]["epochs"] = args.epochs

    if args.lr is not None:
        config["training"]["lr"] = args.lr

    if args.subset is not None:
        config["subset"] = args.subset

    if args.use_dummy_data:
        config["data"]["dataset"] = "dummy"
        config["data"]["data_root"] = args.data_root or "./data"

    if args.debug:
        config["training"]["epochs"] = min(config["training"]["epochs"], 2)
        config["training"]["batch_size"] = min(config["training"]["batch_size"], 8)
        config["data"]["num_workers"] = 0
        if args.subset is None:
            config["subset"] = 50

    # Handle TTA arguments
    if args.tta_scales is not None:
        config.setdefault("tta", {})["scales"] = args.tta_scales
    if args.no_tta:
        config.setdefault("tta", {})["enabled"] = False
    if hasattr(args, "prediction_dir"):
        config.setdefault("evaluation", {})["prediction_dir"] = args.prediction_dir
    if hasattr(args, "gt_dir") and args.gt_dir is not None:
        config.setdefault("culane", {})["gt_dir"] = args.gt_dir

    return config


def setup_environment(config: dict, args: argparse.Namespace):
    """Setup training environment.

    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    # Set random seed
    seed = config.get("seed", 42)
    set_seed(seed)

    # Set CUDA devices
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
        device = torch.device("cuda")
        print(f"Using CUDA device(s): {args.gpus}")
    else:
        device = torch.device("cpu")
        print("Using CPU")

    return device


def train_selfsupervised(config: dict, logger: Logger, args: argparse.Namespace):
    """Run self-supervised training.

    Args:
        config: Configuration dictionary
        logger: Logger instance
        args: Command line arguments
    """
    from train.train_selfsupervised import train

    train(config, logger, resume_from=args.checkpoint)


def train_supervised(config: dict, logger: Logger, args: argparse.Namespace):
    """Run supervised training.

    Args:
        config: Configuration dictionary
        logger: Logger instance
        args: Command line arguments
    """
    from train.train_supervised import train

    train(config, logger, resume_from=args.checkpoint)


def evaluate(config: dict, args: argparse.Namespace):
    """Run evaluation.

    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for evaluation")

    from train.train_selfsupervised import SCCModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SCCModel(
        visual_encoder_name=config["model"]["visual_encoder"],
        pretrained=config["model"]["pretrained"],
        proj_dim=config["model"]["proj_dim"],
        input_size=tuple(config["data"]["input_size"]),
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Create dataloader
    from data import create_dataloader

    val_loader = create_dataloader(
        data_root=config["data"]["data_root"],
        dataset=config["data"]["dataset"],
        split="val",
        batch_size=config["training"]["batch_size"],
        input_size=tuple(config["data"]["input_size"]),
        num_workers=0,
    )

    # Evaluate
    from utils.metrics import calculate_metrics, MetricsTracker
    from tqdm import tqdm

    model.eval()
    metrics = MetricsTracker()

    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Evaluating"):
            view1 = batch["view1"].to(device)
            mask = batch.get("mask")

            if mask is None:
                continue

            mask = mask.to(device)

            output = model(view1, view1, batch["texts"], mask)
            pred = torch.sigmoid(output["pred_mask"])

            batch_metrics = calculate_metrics(pred, mask)
            metrics.update(batch_metrics, batch_size=view1.shape[0])

    final_metrics = metrics.get_metrics()

    print("\nEvaluation Results:")
    for name, value in final_metrics.items():
        print(f"  {name}: {value:.4f}")

    return final_metrics


def evaluate_culane(config: dict, args: argparse.Namespace):
    """Run CULane official evaluation.

    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for CULane evaluation")

    from eval.evaluate_culane_official import evaluate_culane as eval_culane
    from train.train_selfsupervised import SCCModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SCCModel(
        visual_encoder_name=config["model"]["visual_encoder"],
        pretrained=config["model"]["pretrained"],
        freeze_visual_layers=config["model"].get("freeze_visual_layers", 6),
        freeze_text_encoder=config["model"].get("freeze_text_encoder", True),
        proj_dim=config["model"]["proj_dim"],
        input_size=tuple(config["data"]["input_size"]),
        decoder_type=config["model"].get("decoder_type", "simple"),
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)

    # Get TTA settings
    use_tta = not args.no_tta if hasattr(args, "no_tta") else True
    tta_scales = getattr(args, "tta_scales", [0.75, 1.0, 1.25])

    # Get output directory
    output_dir = getattr(args, "prediction_dir", "./results")

    # Run evaluation
    results = eval_culane(
        model=model,
        data_root=config["data"]["data_root"],
        checkpoint_path=args.checkpoint,
        output_dir=output_dir,
        use_tta=use_tta,
        tta_scales=tta_scales,
        batch_size=config["training"]["batch_size"],
        num_workers=config["data"]["num_workers"],
        input_size=tuple(config["data"]["input_size"]),
        gt_dir=config.get("culane", {}).get("gt_dir"),
        list_file=config.get("culane_split", {}).get("test_list"),
    )

    return results


def evaluate_robustness(config: dict, args: argparse.Namespace):
    """Run robustness evaluation.

    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for robustness evaluation")

    from eval.evaluate_robustness import evaluate_robustness as eval_robust
    from train.train_selfsupervised import SCCModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SCCModel(
        visual_encoder_name=config["model"]["visual_encoder"],
        pretrained=config["model"]["pretrained"],
        proj_dim=config["model"]["proj_dim"],
        input_size=tuple(config["data"]["input_size"]),
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate
    results = eval_robust(
        model=model,
        data_root=config["data"]["data_root"],
        dataset=config["data"]["dataset"],
        input_size=tuple(config["data"]["input_size"]),
        batch_size=config["training"]["batch_size"],
        device=device,
    )

    # Print results
    from eval.evaluate_robustness import print_robustness_results
    print_robustness_results(results)


def evaluate_crossdomain(config: dict, args: argparse.Namespace):
    """Run cross-domain evaluation.

    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for cross-domain evaluation")

    from eval.evaluate_crossdomain import evaluate_crossdomain as eval_cd
    from train.train_selfsupervised import SCCModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SCCModel(
        visual_encoder_name=config["model"]["visual_encoder"],
        pretrained=config["model"]["pretrained"],
        proj_dim=config["model"]["proj_dim"],
        input_size=tuple(config["data"]["input_size"]),
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate
    results = eval_cd(
        model=model,
        data_roots={
            config["data"]["dataset"]: config["data"]["data_root"],
            "tusimple": "./data/tusimple",
        },
        input_size=tuple(config["data"]["input_size"]),
        batch_size=config["training"]["batch_size"],
        device=device,
    )

    # Print results
    from eval.evaluate_crossdomain import print_crossdomain_results
    print_crossdomain_results(results)


def visualize_features(config: dict, args: argparse.Namespace):
    """Generate feature visualizations.

    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    if args.checkpoint is None:
        raise ValueError("--checkpoint is required for visualization")

    from eval.visualize_features import visualize
    from train.train_selfsupervised import SCCModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model = SCCModel(
        visual_encoder_name=config["model"]["visual_encoder"],
        pretrained=config["model"]["pretrained"],
        proj_dim=config["model"]["proj_dim"],
        input_size=tuple(config["data"]["input_size"]),
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Generate visualizations
    output_dir = os.path.join(args.output_dir, "visualizations")
    visualize(
        model=model,
        data_root=config["data"]["data_root"],
        dataset=config["data"]["dataset"],
        input_size=tuple(config["data"]["input_size"]),
        batch_size=config["training"]["batch_size"],
        output_dir=output_dir,
    )


def run_tests(config: dict, args: argparse.Namespace):
    """Run unit tests.

    Args:
        config: Configuration dictionary
        args: Command line arguments
    """
    import pytest

    # Run tests
    pytest_args = [
        "tests/",
        "-v",
        "-x",
    ]

    if not torch.cuda.is_available():
        pytest_args.extend(["-k", "not (cuda or gpu)"])

    sys.exit(pytest.main(pytest_args))


def main():
    """Main entry point."""
    args = parse_args()

    # Load configuration
    config = load_config(args.config)
    config = override_config(config, args)

    # Setup environment
    setup_environment(config, args)

    # Create logger
    logger = Logger(
        log_dir=os.path.join(args.output_dir, "logs"),
        experiment_name=args.experiment_name,
        use_tensorboard=config["logging"]["use_tensorboard"],
        use_wandb=config["logging"]["use_wandb"],
        wandb_project=config["logging"]["wandb_project"],
    )

    # Execute task
    task = args.task

    if task == "train":
        # Default to self-supervised
        train_selfsupervised(config, logger, args)
    elif task == "train_selfsupervised":
        train_selfsupervised(config, logger, args)
    elif task == "train_supervised":
        train_supervised(config, logger, args)
    elif task == "evaluate":
        evaluate(config, args)
    elif task == "evaluate_culane":
        evaluate_culane(config, args)
    elif task == "robustness":
        evaluate_robustness(config, args)
    elif task == "crossdomain":
        evaluate_crossdomain(config, args)
    elif task == "visualize":
        visualize_features(config, args)
    elif task == "test":
        run_tests(config, args)
    else:
        print(f"Unknown task: {task}")
        sys.exit(1)

    logger.close()


if __name__ == "__main__":
    main()
