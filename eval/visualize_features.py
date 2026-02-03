"""Feature visualization utilities (t-SNE, UMAP)."""

import os
from typing import Dict, Any, List, Optional

import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    from sklearn.manifold import TSNE
    from sklearn.decomposition import PCA
    HAS_SKLEARN = True
except ImportError:
    HAS_SKLEARN = False

try:
    import umap
    HAS_UMAP = True
except ImportError:
    HAS_UMAP = False

from data import create_dataloader
from models import VisualEncoder, LanguageEncoder


def extract_features(
    model,
    dataloader,
    device: torch.device,
    max_samples: int = 1000,
) -> Dict[str, np.ndarray]:
    """Extract visual and text features from model.

    Args:
        model: Trained SCC model
        dataloader: Data loader
        device: Device to run on
        max_samples: Maximum number of samples to extract

    Returns:
        Dictionary with 'visual', 'text', and 'labels' arrays
    """
    model.eval()

    visual_features = []
    text_features = []
    labels = []

    sample_count = 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            if sample_count >= max_samples:
                break

            view1 = batch["view1"].to(device)
            texts = batch["texts"]
            mask = batch.get("mask")

            # Extract visual features
            f_v = model.visual_encoder(view1)
            z_v = model.projection.visual_proj(f_v)

            # Extract text features
            f_l = model.language_encoder.encode_text(texts)
            z_l = model.projection.language_proj(f_l)

            visual_features.append(z_v.cpu().numpy())
            text_features.append(z_l.cpu().numpy())

            # Create labels from lane count in text
            batch_labels = []
            for text in texts:
                if "6" in text or "six" in text.lower():
                    batch_labels.append(6)
                elif "5" in text or "five" in text.lower():
                    batch_labels.append(5)
                elif "4" in text or "four" in text.lower():
                    batch_labels.append(4)
                elif "3" in text or "three" in text.lower():
                    batch_labels.append(3)
                else:
                    batch_labels.append(2)
            labels.append(np.array(batch_labels))

            sample_count += view1.shape[0]

    visual_features = np.vstack(visual_features)[:max_samples]
    text_features = np.vstack(text_features)[:max_samples]
    labels = np.concatenate(labels)[:max_samples]

    return {
        "visual": visual_features,
        "text": text_features,
        "labels": labels,
    }


def plot_tsne(
    features: np.ndarray,
    labels: np.ndarray,
    title: str = "t-SNE Visualization",
    save_path: Optional[str] = None,
    perplexity: int = 30,
    n_iter: int = 1000,
) -> plt.Figure:
    """Plot t-SNE visualization of features.

    Args:
        features: Feature array (N, D)
        labels: Label array (N,)
        title: Plot title
        save_path: Path to save figure
        perplexity: t-SNE perplexity
        n_iter: Number of iterations

    Returns:
        Matplotlib figure
    """
    if not HAS_SKLEARN:
        raise ImportError("scikit-learn is required for t-SNE")

    # Apply t-SNE
    print("Computing t-SNE...")
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=n_iter,
        random_state=42,
    )
    features_2d = tsne.fit_transform(features)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[i]],
            label=f"{label} lanes",
            alpha=0.7,
            s=50,
        )

    ax.set_title(title)
    ax.set_xlabel("t-SNE 1")
    ax.set_ylabel("t-SNE 2")
    ax.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig


def plot_umap(
    features: np.ndarray,
    labels: np.ndarray,
    title: str = "UMAP Visualization",
    save_path: Optional[str] = None,
    n_neighbors: int = 15,
    min_dist: float = 0.1,
) -> plt.Figure:
    """Plot UMAP visualization of features.

    Args:
        features: Feature array (N, D)
        labels: Label array (N,)
        title: Plot title
        save_path: Path to save figure
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter

    Returns:
        Matplotlib figure
    """
    if not HAS_UMAP:
        raise ImportError("umap-learn is required for UMAP")

    # Apply UMAP
    print("Computing UMAP...")
    reducer = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        random_state=42,
    )
    features_2d = reducer.fit_transform(features)

    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))

    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = labels == label
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[i]],
            label=f"{label} lanes",
            alpha=0.7,
            s=50,
        )

    ax.set_title(title)
    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.legend()

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig


def plot_cross_modal(
    visual_features: np.ndarray,
    text_features: np.ndarray,
    labels: np.ndarray,
    title: str = "Cross-Modal Feature Space",
    save_path: Optional[str] = None,
    method: str = "tsne",
) -> plt.Figure:
    """Plot cross-modal feature alignment.

    Args:
        visual_features: Visual features (N, D)
        text_features: Text features (N, D)
        labels: Label array (N,)
        title: Plot title
        save_path: Path to save figure
        method: Dimensionality reduction method ('tsne' or 'umap')

    Returns:
        Matplotlib figure
    """
    # Concatenate features
    all_features = np.vstack([visual_features, text_features])

    # Create modality labels
    modality_labels = np.array(
        ["Visual"] * len(visual_features) + ["Text"] * len(text_features)
    )

    # Reduce dimensionality
    if method.lower() == "tsne":
        if not HAS_SKLEARN:
            raise ImportError("scikit-learn is required for t-SNE")
        print("Computing t-SNE...")
        reducer = TSNE(n_components=2, perplexity=30, random_state=42)
    else:
        if not HAS_UMAP:
            raise ImportError("umap-learn is required for UMAP")
        print("Computing UMAP...")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42)

    features_2d = reducer.fit_transform(all_features)

    # Split back
    visual_2d = features_2d[:len(visual_features)]
    text_2d = features_2d[len(visual_features):]

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Left: by modality
    ax = axes[0]
    ax.scatter(visual_2d[:, 0], visual_2d[:, 1],
               c='blue', label='Visual', alpha=0.5, s=30)
    ax.scatter(text_2d[:, 0], text_2d[:, 1],
               c='red', label='Text', alpha=0.5, s=30)
    ax.set_title("By Modality")
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.legend()

    # Right: by lane count
    ax = axes[1]
    all_labels = np.concatenate([labels, labels])
    unique_labels = np.unique(labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))

    for i, label in enumerate(unique_labels):
        mask = all_labels == label
        ax.scatter(
            features_2d[mask, 0],
            features_2d[mask, 1],
            c=[colors[i]],
            label=f"{label} lanes",
            alpha=0.5,
            s=30,
        )

    ax.set_title("By Lane Count")
    ax.set_xlabel(f"{method.upper()} 1")
    ax.set_ylabel(f"{method.upper()} 2")
    ax.legend()

    fig.suptitle(title)

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"Saved to {save_path}")

    return fig


def visualize(
    model,
    data_root: str,
    dataset: str = "culane",
    split: str = "val",
    input_size: tuple = (800, 320),
    batch_size: int = 32,
    output_dir: str = "outputs/visualizations",
    max_samples: int = 500,
) -> None:
    """Generate all visualizations.

    Args:
        model: Trained model
        data_root: Data root directory
        dataset: Dataset name
        split: Data split
        input_size: Input image size
        batch_size: Batch size
        output_dir: Output directory for visualizations
        max_samples: Maximum samples to visualize
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

    # Extract features
    features = extract_features(model, dataloader, device, max_samples)

    # Visualizations
    if HAS_SKLEARN:
        # t-SNE for visual features
        fig = plot_tsne(
            features["visual"],
            features["labels"],
            title="Visual Features (t-SNE)",
            save_path=os.path.join(output_dir, "tsne_visual.png"),
        )
        plt.close(fig)

        # t-SNE for text features
        fig = plot_tsne(
            features["text"],
            features["labels"],
            title="Text Features (t-SNE)",
            save_path=os.path.join(output_dir, "tsne_text.png"),
        )
        plt.close(fig)

        # Cross-modal t-SNE
        fig = plot_cross_modal(
            features["visual"],
            features["text"],
            features["labels"],
            title="Cross-Modal Alignment (t-SNE)",
            save_path=os.path.join(output_dir, "cross_modal_tsne.png"),
            method="tsne",
        )
        plt.close(fig)

    if HAS_UMAP:
        # UMAP for visual features
        fig = plot_umap(
            features["visual"],
            features["labels"],
            title="Visual Features (UMAP)",
            save_path=os.path.join(output_dir, "umap_visual.png"),
        )
        plt.close(fig)

        # Cross-modal UMAP
        fig = plot_cross_modal(
            features["visual"],
            features["text"],
            features["labels"],
            title="Cross-Modal Alignment (UMAP)",
            save_path=os.path.join(output_dir, "cross_modal_umap.png"),
            method="umap",
        )
        plt.close(fig)


if __name__ == "__main__":
    import argparse
    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="config/default.yaml")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--output_dir", type=str, default="outputs/visualizations")
    parser.add_argument("--max_samples", type=int, default=500)

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

    # Generate visualizations
    visualize(
        model=model,
        data_root=args.data_root,
        dataset=config["data"]["dataset"],
        input_size=tuple(config["data"]["input_size"]),
        batch_size=config["training"]["batch_size"],
        output_dir=args.output_dir,
        max_samples=args.max_samples,
    )

    print(f"\nVisualizations saved to {args.output_dir}")
