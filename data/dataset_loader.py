"""Dataset loader for lane detection."""

import os
from typing import Dict, Any, Optional, List, Tuple, Callable
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import cv2
from PIL import Image

from .data_augment import DataAugmentation, RobustnessAugmentation
from .text_generator import TextGenerator


class LaneDataset(Dataset):
    """Base dataset class for lane detection.

    Supports CULane and TuSimple formats with two-view augmentation
    for self-supervised cross-modal consistency training.
    """

    def __init__(
        self,
        data_root: str,
        dataset: str = "culane",
        split: str = "train",
        transform: Optional[Callable] = None,
        input_size: Tuple[int, int] = (590, 1640),
        generate_text: bool = True,
        text_generator: Optional[TextGenerator] = None,
    ):
        """Initialize lane dataset.

        Args:
            data_root: Root directory of dataset
            dataset: Dataset type ('culane', 'tusimple', or 'dummy')
            split: Data split ('train', 'val', 'test')
            transform: Optional augmentation transform
            input_size: Input image size (H, W)
            generate_text: Whether to generate text captions
            text_generator: Optional text generator instance
        """
        self.data_root = Path(data_root)
        self.dataset = dataset.lower()
        self.split = split
        self.input_size = input_size
        self.generate_text = generate_text

        if transform is None:
            self.transform = DataAugmentation(input_size=input_size)
        else:
            self.transform = transform

        if text_generator is None:
            self.text_generator = TextGenerator()
        else:
            self.text_generator = text_generator

        # Load file list
        self.samples = self._load_file_list()

        print(f"Loaded {len(self.samples)} samples for {dataset} {split}")

    def _load_file_list(self) -> List[Dict[str, Any]]:
        """Load list of image/mask pairs.

        Returns:
            List of sample dictionaries
        """
        samples = []

        if self.dataset == "dummy":
            samples = self._load_dummy()
        elif self.dataset == "culane":
            samples = self._load_culane()
        elif self.dataset == "tusimple":
            samples = self._load_tusimple()
        else:
            raise ValueError(f"Unknown dataset: {self.dataset}")

        return samples

    def _load_dummy(self) -> List[Dict[str, Any]]:
        """Load dummy dataset for testing.

        Returns:
            List of sample dictionaries
        """
        dummy_dir = self.data_root / "dummy"
        if not dummy_dir.exists():
            return []

        image_dir = dummy_dir / "images"
        mask_dir = dummy_dir / "masks"

        if not image_dir.exists():
            return []

        samples = []
        for img_path in sorted(image_dir.glob("*.jpg")):
            mask_path = mask_dir / img_path.name
            samples.append({
                "image": str(img_path),
                "mask": str(mask_path) if mask_path.exists() else None,
            })

        return samples

    def _load_culane(self) -> List[Dict[str, Any]]:
        """Load CULane dataset file list.

        Returns:
            List of sample dictionaries
        """
        # CULane directory structure:
        # data_root/
        #   driver_23_30frame/
        #     05240723_05241333/
        #       0.jpg
        #   lanes/
        #     driver_23_30frame/05240723_05241333/0.lines.txt

        list_file = self.data_root / "list" / f"{self.split}.txt"

        if not list_file.exists():
            # Try to find images directly
            return self._scan_culane_images()

        samples = []
        with open(list_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                # CULane lines: image_path (absolute path like /driver_xxx/...)
                # Remove leading slash if present and prepend data_root
                img_path = line.lstrip('/')  # Remove leading slash

                # Construct full path
                full_path = self.data_root / img_path

                # Check if file exists
                if not os.path.exists(full_path):
                    # Try with the original path
                    full_path = "/" + img_path
                    if not os.path.exists(full_path):
                        continue

                img_path = str(full_path)

                # Derive mask path (for training data, masks might be in separate directory)
                mask_path = None
                # Try different mask locations for CULane
                for mask_dir in ["laneseg_label_w16", "laneseg_label_w16_test"]:
                    # Try replacing driver_ path to find mask
                    test_mask = img_path
                    for driver_prefix in ["driver_23_30frame", "driver_37_30frame",
                                         "driver_100_30frame", "driver_161_90frame",
                                         "driver_182_30frame", "driver_193_90frame"]:
                        if driver_prefix in test_mask:
                            test_mask = test_mask.replace(driver_prefix, f"{mask_dir}/{driver_prefix}")
                            break

                    # CULane masks are PNG images, not .lines.txt
                    test_mask_png = test_mask.replace(".jpg", ".png")
                    if os.path.exists(test_mask_png):
                        mask_path = test_mask_png
                        break
                    # Also try .lines.txt format (for some variants)
                    test_mask_txt = test_mask.replace(".jpg", ".lines.txt")
                    if os.path.exists(test_mask_txt):
                        mask_path = test_mask_txt
                        break

                samples.append({
                    "image": img_path,
                    "mask": mask_path,
                })

        return samples

    def _scan_culane_images(self) -> List[Dict[str, Any]]:
        """Scan CULane directory for images.

        Returns:
            List of sample dictionaries
        """
        samples = []

        # All possible driver directories in CULane
        driver_dirs = [
            "driver_23_30frame", "driver_37_30frame", "driver_100_30frame",
            "driver_161_90frame", "driver_182_30frame", "driver_193_90frame",
            "05081544_0305",
        ]

        for split_dir in driver_dirs:
            img_root = self.data_root / split_dir

            # Check if it's a symlink and follow it
            if img_root.is_symlink():
                img_root = Path(os.path.realpath(img_root))

            if not img_root.exists():
                continue

            for img_path in img_root.rglob("*.jpg"):
                # Try multiple mask locations
                mask_path = None
                for mask_dir in ["laneseg_label_w16", "lanes"]:
                    # Try different mask path patterns
                    test_mask = str(img_path).replace(
                        f"/{split_dir}/", f"/{mask_dir}/{split_dir}/"
                    ).replace(".jpg", ".lines.txt")

                    if os.path.exists(test_mask):
                        mask_path = test_mask
                        break

                samples.append({
                    "image": str(img_path),
                    "mask": mask_path,
                })

        return samples

    def _load_tusimple(self) -> List[Dict[str, Any]]:
        """Load TuSimple dataset file list.

        Returns:
            List of sample dictionaries
        """
        # TuSimple directory structure:
        # data_root/
        #   clips/0313-1/60/1.jpg
        #   lanes/0313-1/60/1.lines.txt

        samples = []

        for clip_dir in (self.data_root / "clips").iterdir():
            if not clip_dir.is_dir():
                continue

            for frame_dir in clip_dir.iterdir():
                if not frame_dir.is_dir():
                    continue

                for img_path in frame_dir.glob("*.jpg"):
                    rel_path = img_path.relative_to(self.data_root / "clips")
                    mask_path = str(self.data_root / "lanes" / rel_path).replace(
                        ".jpg", ".lines.txt"
                    )

                    samples.append({
                        "image": str(img_path),
                        "mask": mask_path if os.path.exists(mask_path) else None,
                    })

        return samples

    def _load_image(self, path: str) -> np.ndarray:
        """Load image from file.

        Args:
            path: Image file path

        Returns:
            Image as numpy array (H, W, 3)
        """
        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Failed to load image: {path}")

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def _load_mask(self, path: Optional[str]) -> Optional[np.ndarray]:
        """Load lane mask from file.

        Args:
            path: Mask file path (can be image or text file with coordinates)

        Returns:
            Mask as numpy array (H, W) or None
        """
        if path is None or not os.path.exists(path):
            return None

        # Try loading as image first
        if path.endswith((".png", ".jpg")):
            mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if mask is not None:
                return mask

        # Try loading as text file with lane coordinates
        try:
            mask = np.zeros(self.input_size, dtype=np.uint8)

            with open(path, "r") as f:
                for line in f:
                    coords = line.strip().split()
                    if len(coords) < 4:
                        continue

                    # Parse coordinates: y1 x1 y2 x2 y3 x3 ...
                    points = []
                    for i in range(0, len(coords), 2):
                        x, y = float(coords[i + 1]), float(coords[i])
                        points.append([x, y])

                    if len(points) >= 2:
                        points = np.array(points, dtype=np.int32)
                        # Scale coordinates to input size
                        # CULane uses 1640x590, coordinates may be normalized or absolute
                        if self.input_size == (590, 1640) or self.input_size == (590, 1640):
                            # For CULane official size, assume coordinates are already correct
                            # or scale from standard reference size
                            if points[:, 0].max() > 1640 or points[:, 1].max() > 590:
                                # Coordinates might be normalized or from different reference
                                # Try scaling from 1280x720 (common reference)
                                points[:, 0] = points[:, 0] * self.input_size[1] / 1280  # Scale x
                                points[:, 1] = points[:, 1] * self.input_size[0] / 720  # Scale y
                        else:
                            # Scale for other input sizes
                            points[:, 0] = points[:, 0] * self.input_size[1] / 1280  # Scale x
                            points[:, 1] = points[:, 1] * self.input_size[0] / 720  # Scale y
                        cv2.polylines(mask, [points], False, 255, thickness=5)

            return mask
        except Exception:
            return None

    def _generate_text_from_mask(self, mask: np.ndarray) -> str:
        """Generate text caption from mask.

        Args:
            mask: Binary mask (H, W)

        Returns:
            Generated caption
        """
        return self.text_generator.generate_from_mask(mask)

    def __len__(self) -> int:
        """Get dataset length.

        Returns:
            Number of samples
        """
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample.

        Args:
            idx: Sample index

        Returns:
            Dictionary with 'view1', 'view2', 'text', and optionally 'mask'
        """
        sample = self.samples[idx]

        # Load image
        image = self._load_image(sample["image"])

        # Load mask if available
        mask = None
        if sample.get("mask"):
            mask = self._load_mask(sample["mask"])

        # Apply augmentation (creates two views)
        augmented = self.transform(image, mask)

        result = {
            "view1": augmented["view1"],
            "view2": augmented["view2"],
            "image_path": sample["image"],
        }

        if mask is not None:
            result["mask"] = augmented["mask"]
        else:
            # Generate pseudo mask from image for self-supervised learning
            if "mask" in augmented:
                result["mask"] = augmented["mask"]

        # Generate text caption
        if self.generate_text:
            if mask is not None and "mask" in augmented:
                mask_np = augmented["mask"].squeeze().numpy()
                text = self._generate_text_from_mask(mask_np)
            else:
                # Default caption
                text = self.text_generator.generate(2, "solid")
            result["text"] = text

        return result


class RobustnessDataset(Dataset):
    """Dataset for robustness evaluation with specific perturbations."""

    def __init__(
        self,
        data_root: str,
        dataset: str = "culane",
        split: str = "test",
        perturbation: str = "clean",
        severity: int = 1,
        input_size: Tuple[int, int] = (590, 1640),
    ):
        """Initialize robustness dataset.

        Args:
            data_root: Root directory of dataset
            dataset: Dataset type
            split: Data split
            perturbation: Type of perturbation to apply
            severity: Severity level (1-5)
            input_size: Input image size (H, W)
        """
        self.dataset = LaneDataset(
            data_root=data_root,
            dataset=dataset,
            split=split,
            transform=None,
            input_size=input_size,
            generate_text=False,
        )

        self.perturbation = perturbation
        self.severity = severity
        self.augmentation = RobustnessAugmentation(input_size=input_size)

    def __len__(self) -> int:
        """Get dataset length."""
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample with perturbation applied.

        Args:
            idx: Sample index

        Returns:
            Dictionary with 'image' and 'mask'
        """
        sample = self.dataset.samples[idx]

        # Load image
        image = self.dataset._load_image(sample["image"])

        # Load mask
        mask = None
        if sample.get("mask"):
            mask = self.dataset._load_mask(sample["mask"])

        # Apply perturbation
        image_tensor = self.augmentation(image, self.perturbation, self.severity)

        result = {
            "image": image_tensor,
            "image_path": sample["image"],
        }

        if mask is not None:
            # Resize mask to match image
            mask_resized = cv2.resize(
                mask,
                (self.augmentation.input_size[1], self.augmentation.input_size[0]),
                interpolation=cv2.INTER_NEAREST,
            )
            mask_tensor = torch.from_numpy(mask_resized).unsqueeze(0).float() / 255.0
            result["mask"] = mask_tensor

        return result


def create_dataloader(
    data_root: str,
    dataset: str = "culane",
    split: str = "train",
    batch_size: int = 32,
    input_size: Tuple[int, int] = (590, 1640),
    num_workers: int = 4,
    pin_memory: bool = True,
    shuffle: Optional[bool] = None,
    subset: Optional[int] = None,
    robustness: bool = False,
    perturbation: str = "clean",
    severity: int = 1,
) -> DataLoader:
    """Create a DataLoader for lane detection.

    Args:
        data_root: Root directory of dataset
        dataset: Dataset type ('culane', 'tusimple', 'dummy')
        split: Data split ('train', 'val', 'test')
        batch_size: Batch size
        input_size: Input image size (H, W)
        num_workers: Number of data loading workers
        pin_memory: Whether to pin memory for GPU transfer
        shuffle: Whether to shuffle (default: True for train, False for others)
        subset: If specified, only use first N samples
        robustness: Whether to use robustness perturbations
        perturbation: Type of perturbation for robustness testing
        severity: Severity level for perturbation

    Returns:
        Configured DataLoader
    """
    if shuffle is None:
        shuffle = (split == "train")

    if robustness:
        dataset = RobustnessDataset(
            data_root=data_root,
            dataset=dataset,
            split=split,
            perturbation=perturbation,
            severity=severity,
            input_size=input_size,
        )
    else:
        dataset = LaneDataset(
            data_root=data_root,
            dataset=dataset,
            split=split,
            input_size=input_size,
        )

    # Use subset if specified
    if subset is not None and subset < len(dataset):
        import torch.utils.data as data_utils
        dataset = data_utils.Subset(dataset, range(subset))

    def collate_fn(batch):
        """Custom collate function for variable-length batch."""
        view1 = torch.stack([item["view1"] for item in batch])
        view2 = torch.stack([item["view2"] for item in batch])
        texts = [item.get("text", "a road with 2 lanes") for item in batch]
        image_paths = [item["image_path"] for item in batch]

        result = {
            "view1": view1,
            "view2": view2,
            "texts": texts,
            "image_paths": image_paths,
        }

        # Include masks if available
        if "mask" in batch[0]:
            masks = torch.stack([item["mask"] for item in batch])
            result["mask"] = masks

        return result

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        drop_last=(split == "train"),
    )
