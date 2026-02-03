"""Robustness evaluation under various perturbations."""

import os
from typing import Dict, Any, List

import torch
import numpy as np
from tqdm import tqdm

from data import create_dataloader
from utils.metrics import calculate_metrics, MetricsTracker


def evaluate_robustness(
    model,
    data_root: str,
    dataset: str = "culane",
    split: str = "test",
    input_size: tuple = (800, 320),
    batch_size: int = 32,
    device: torch.device = None,
    perturbations: List[str] = None,
    severity_levels: List[int] = None,
) -> Dict[str, Dict[str, float]]:
    """Evaluate model robustness under various perturbations.

    Args:
        model: Trained model with predict() method
        data_root: Root directory of dataset
        dataset: Dataset name
        split: Data split
        input_size: Input image size
        batch_size: Batch size
        device: Device to evaluate on
        perturbations: List of perturbation types
        severity_levels: List of severity levels to test

    Returns:
        Dictionary of perturbation -> metrics
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if perturbations is None:
        perturbations = [
            "clean",
            "blur",
            "noise",
            "brightness",
            "darkness",
            "fog",
            "frost",
            "snow",
            "contrast",
        ]

    if severity_levels is None:
        severity_levels = [1, 2, 3, 4, 5]

    model.eval()
    model.to(device)

    results = {}

    print("Evaluating robustness under perturbations...")

    for perturbation in perturbations:
        perturbation_results = {}

        for severity in severity_levels:
            print(f"\nTesting: {perturbation}, severity {severity}")

            # Create dataloader with perturbation
            dataloader = create_dataloader(
                data_root=data_root,
                dataset=dataset,
                split=split,
                batch_size=batch_size,
                input_size=input_size,
                num_workers=0,  # Avoid multiprocessing issues
                shuffle=False,
                robustness=True,
                perturbation=perturbation,
                severity=severity,
            )

            metrics = MetricsTracker()

            for batch in tqdm(dataloader, desc=f"{perturbation}-{severity}"):
                image = batch["image"].to(device)
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

            avg_metrics = metrics.get_metrics()
            key = f"{perturbation}_s{severity}"
            results[key] = avg_metrics

            print(f"  IoU: {avg_metrics['iou']:.4f}, F1: {avg_metrics['f1']:.4f}")

            perturbation_results[f"severity_{severity}"] = avg_metrics

        results[perturbation] = perturbation_results

    return results


def compute_robustness_metrics(
    results: Dict[str, Dict[str, float]],
) -> Dict[str, float]:
    """Compute aggregate robustness metrics.

    Args:
        results: Results from evaluate_robustness

    Returns:
        Dictionary of aggregate metrics
    """
    # Compute mCE (mean Corruption Error) relative to clean
    clean_iou = results.get("clean", {}).get("iou", 0.0)

    if clean_iou == 0:
        return {"mCE": 0.0, "relative_iou_drop": 0.0}

    corruption_errors = []

    for key, metrics in results.items():
        if key == "clean":
            continue

        if isinstance(metrics, dict):
            # Aggregate over severity levels
            for severity_key, severity_metrics in metrics.items():
                if isinstance(severity_metrics, dict) and "iou" in severity_metrics:
                    error = (clean_iou - severity_metrics["iou"]) / clean_iou
                    corruption_errors.append(error)

    if corruption_errors:
        mCE = np.mean(corruption_errors) * 100  # Percentage
        relative_drop = (1 - np.mean([e for e in corruption_errors])) * 100
    else:
        mCE = 0.0
        relative_drop = 0.0

    return {
        "mCE": mCE,
        "relative_iou_drop": relative_drop,
    }


def print_robustness_results(
    results: Dict[str, Dict[str, float]],
) -> None:
    """Print robustness results in a table.

    Args:
        results: Results from evaluate_robustness
    """
    print("\n" + "=" * 80)
    print("Robustness Evaluation Results")
    print("=" * 80)

    # Get clean performance
    clean = results.get("clean", {})
    if clean:
        print(f"\nClean (baseline):")
        print(f"  IoU: {clean.get('iou', 0):.4f}")
        print(f"  F1: {clean.get('f1', 0):.4f}")
        print(f"  Accuracy: {clean.get('accuracy', 0):.4f}")

    # Compute aggregate metrics
    aggregate = compute_robustness_metrics(results)
    print(f"\nAggregate Metrics:")
    print(f"  mCE: {aggregate['mCE']:.2f}%")
    print(f"  Relative IoU Drop: {aggregate['relative_iou_drop']:.2f}%")

    # Table by perturbation
    print("\nPer-Perturbation Results:")
    print("-" * 80)

    perturbations = sorted(set([k.split("_")[0] for k in results.keys() if k != "clean"]))

    for perturbation in perturbations:
        print(f"\n{perturbation}:")
        for severity in [1, 2, 3, 4, 5]:
            key = f"{perturbation}_s{severity}"
            if key in results:
                metrics = results[key]
                print(f"  Severity {severity}: IoU={metrics.get('iou', 0):.4f}, "
                      f"F1={metrics.get('f1', 0):.4f}")

    print("=" * 80)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output", type=str, default="outputs/robustness_results.json")

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

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
    results = evaluate_robustness(
        model=model,
        data_root=args.data_root,
        dataset=config["data"]["dataset"],
        input_size=tuple(config["data"]["input_size"]),
        batch_size=config["training"]["batch_size"],
        device=device,
    )

    # Print results
    print_robustness_results(results)

    # Save results
    import json

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {args.output}")
