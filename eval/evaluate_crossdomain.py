"""Cross-domain evaluation (CULane -> TuSimple/LLAMAS)."""

import os
from typing import Dict, Any, List

import torch
import numpy as np
from tqdm import tqdm

from data import create_dataloader
from utils.metrics import calculate_metrics, MetricsTracker


def evaluate_dataset(
    model,
    data_root: str,
    dataset: str,
    split: str = "test",
    input_size: tuple = (800, 320),
    batch_size: int = 32,
    device: torch.device = None,
) -> Dict[str, float]:
    """Evaluate model on a specific dataset.

    Args:
        model: Trained model
        data_root: Root directory of dataset
        dataset: Dataset name ('culane', 'tusimple', 'llamas')
        split: Data split
        input_size: Input image size
        batch_size: Batch size
        device: Device to evaluate on

    Returns:
        Dictionary of metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval()
    model.to(device)

    # Create dataloader
    dataloader = create_dataloader(
        data_root=data_root,
        dataset=dataset,
        split=split,
        batch_size=batch_size,
        input_size=input_size,
        num_workers=0,
        shuffle=False,
    )

    metrics = MetricsTracker()

    print(f"Evaluating on {dataset}...")

    for batch in tqdm(dataloader, desc=dataset):
        image = batch.get("image", batch.get("view1")).to(device)
        mask = batch.get("mask")

        if mask is None:
            continue

        mask = mask.to(device)

        # Predict
        with torch.no_grad():
            if hasattr(model, "predict"):
                pred = model.predict(image)
            else:
                output = model(image)
                pred = torch.sigmoid(output["pred_mask"])

        # Calculate metrics
        batch_metrics = calculate_metrics(pred, mask)
        metrics.update(batch_metrics, batch_size=image.shape[0])

    return metrics.get_metrics()


def evaluate_crossdomain(
    model,
    source_dataset: str = "culane",
    target_datasets: List[str] = None,
    data_roots: Dict[str, str] = None,
    input_size: tuple = (800, 320),
    batch_size: int = 32,
    device: torch.device = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate cross-domain generalization.

    Args:
        model: Trained model
        source_dataset: Source dataset name
        target_datasets: List of target dataset names
        data_roots: Dictionary mapping dataset names to their root directories
        input_size: Input image size
        batch_size: Batch size
        device: Device to evaluate on

    Returns:
        Dictionary of dataset -> metrics
    """
    if target_datasets is None:
        target_datasets = ["tusimple", "llamas"]

    if data_roots is None:
        data_roots = {
            "culane": "./data/culane",
            "tusimple": "./data/tusimple",
            "llamas": "./data/llamas",
        }

    results = {}

    # Evaluate on source
    print("\nSource Domain Evaluation:")
    source_metrics = evaluate_dataset(
        model=model,
        data_root=data_roots.get(source_dataset, "./data"),
        dataset=source_dataset,
        input_size=input_size,
        batch_size=batch_size,
        device=device,
    )
    results[source_dataset] = source_metrics

    print(f"\n{source_dataset} results:")
    for name, value in source_metrics.items():
        print(f"  {name}: {value:.4f}")

    # Evaluate on targets
    for target in target_datasets:
        if target not in data_roots:
            print(f"\nSkipping {target} (data root not specified)")
            continue

        print(f"\nTarget Domain: {target}")
        target_metrics = evaluate_dataset(
            model=model,
            data_root=data_roots[target],
            dataset=target,
            input_size=input_size,
            batch_size=batch_size,
            device=device,
        )
        results[target] = target_metrics

        print(f"\n{target} results:")
        for name, value in target_metrics.items():
            print(f"  {name}: {value:.4f}")

    return results


def compute_domain_gap(
    source_metrics: Dict[str, float],
    target_metrics: Dict[str, float],
    metric: str = "iou",
) -> float:
    """Compute domain gap as performance drop.

    Args:
        source_metrics: Source domain metrics
        target_metrics: Target domain metrics
        metric: Metric to use for comparison

    Returns:
        Relative performance drop (percentage)
    """
    source_val = source_metrics.get(metric, 0.0)
    target_val = target_metrics.get(metric, 0.0)

    if source_val == 0:
        return 0.0

    return (source_val - target_val) / source_val * 100


def print_crossdomain_results(
    results: Dict[str, Dict[str, float]],
) -> None:
    """Print cross-domain results.

    Args:
        results: Results from evaluate_crossdomain
    """
    print("\n" + "=" * 80)
    print("Cross-Domain Evaluation Results")
    print("=" * 80)

    datasets = list(results.keys())

    if len(datasets) < 2:
        print("Need at least 2 datasets for cross-domain evaluation")
        return

    source = datasets[0]

    # Print table
    print("\n" + "-" * 80)
    print(f"{'Dataset':<15} {'IoU':<10} {'F1':<10} {'Accuracy':<10} {'Drop %':<10}")
    print("-" * 80)

    source_iou = results[source].get("iou", 0.0)

    for dataset in datasets:
        metrics = results[dataset]
        iou = metrics.get("iou", 0.0)
        f1 = metrics.get("f1", 0.0)
        acc = metrics.get("accuracy", 0.0)

        if dataset == source:
            drop = 0.0
        else:
            drop = compute_domain_gap(results[source], metrics, "iou")

        print(f"{dataset:<15} {iou:<10.4f} {f1:<10.4f} {acc:<10.4f} {drop:<10.2f}")

    print("-" * 80)

    # Average domain gap
    if len(datasets) > 1:
        gaps = []
        for dataset in datasets[1:]:
            gap = compute_domain_gap(results[source], results[dataset], "iou")
            gaps.append(gap)

        print(f"\nAverage Domain Gap: {np.mean(gaps):.2f}%")

    print("=" * 80)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--source", type=str, default="culane")
    parser.add_argument("--targets", type=str, nargs="+", default=["tusimple"])
    parser.add_argument("--data_roots", type=str, nargs="+", default=None)
    parser.add_argument("--output", type=str, default="outputs/crossdomain_results.json")

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Parse data roots
    data_roots = {}
    if args.data_roots:
        for i, dataset in enumerate([args.source] + args.targets):
            if i < len(args.data_roots):
                data_roots[dataset] = args.data_roots[i]
    else:
        data_roots[args.source] = config["data"]["data_root"]
        for target in args.targets:
            data_roots[target] = f"./data/{target}"

    # Load model
    from train.train_selfsupervised import SCCModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = SCCModel(
        visual_encoder_name=config["model"]["visual_encoder"],
        pretrained=config["model"]["pretrained"],
        proj_dim=config["model"]["proj_dim"],
        input_size=tuple(config["data"]["input_size"]),
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    # Evaluate
    results = evaluate_crossdomain(
        model=model,
        source_dataset=args.source,
        target_datasets=args.targets,
        data_roots=data_roots,
        input_size=tuple(config["data"]["input_size"]),
        batch_size=config["training"]["batch_size"],
        device=device,
    )

    # Print results
    print_crossdomain_results(results)

    # Save results
    import json

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")
