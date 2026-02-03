"""Tests for data loading functionality."""

import pytest
import torch
import numpy as np
from PIL import Image
import tempfile
import shutil
from pathlib import Path


@pytest.fixture
def temp_data_dir():
    """Create temporary data directory with dummy images."""
    temp_dir = tempfile.mkdtemp()

    # Create directory structure
    images_dir = Path(temp_dir) / "dummy" / "images"
    masks_dir = Path(temp_dir) / "dummy" / "masks"
    images_dir.mkdir(parents=True)
    masks_dir.mkdir(parents=True)

    # Create dummy images and masks
    for i in range(10):
        # Create random image
        img = Image.fromarray(np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8))
        img.save(images_dir / f"{i:05d}.jpg")

        # Create random mask
        mask = Image.fromarray(np.random.randint(0, 2, (720, 1280), dtype=np.uint8) * 255)
        mask.save(masks_dir / f"{i:05d}.png")

    yield temp_dir

    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    return {
        "view1": torch.randn(4, 3, 320, 800),
        "view2": torch.randn(4, 3, 320, 800),
        "mask": torch.randint(0, 2, (4, 1, 320, 800)).float(),
        "texts": ["a road with 2 lanes"] * 4,
    }


class TestDataAugmentation:
    """Tests for data augmentation."""

    def test_augmentation_output_shape(self, temp_data_dir):
        """Test that augmentation produces correct output shapes."""
        from data.data_augment import DataAugmentation

        # Create sample image and mask
        image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        mask = np.random.randint(0, 2, (720, 1280), dtype=np.uint8)

        # Create augmentation
        aug = DataAugmentation(input_size=(800, 320))

        # Apply
        result = aug(image, mask)

        assert "view1" in result
        assert "view2" in result
        assert "mask" in result

        assert result["view1"].shape == (3, 800, 320)
        assert result["view2"].shape == (3, 800, 320)
        assert result["mask"].shape == (1, 800, 320)

    def test_augmentation_without_mask(self, temp_data_dir):
        """Test augmentation works without mask."""
        from data.data_augment import DataAugmentation

        image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        aug = DataAugmentation(input_size=(800, 320))

        result = aug(image, None)

        assert "view1" in result
        assert "view2" in result
        assert "mask" in result  # Mask should still be created

    def test_robustness_augmentation(self, temp_data_dir):
        """Test robustness augmentation."""
        from data.data_augment import RobustnessAugmentation

        image = np.random.randint(0, 255, (720, 1280, 3), dtype=np.uint8)
        aug = RobustnessAugmentation(input_size=(800, 320))

        perturbations = ["clean", "blur", "noise", "brightness"]

        for pert in perturbations:
            result = aug(image, pert, severity=1)
            assert result.shape == (3, 800, 320)


class TestTextGenerator:
    """Tests for text generation."""

    def test_generate_caption(self):
        """Test caption generation."""
        from data.text_generator import TextGenerator

        gen = TextGenerator()
        caption = gen.generate(lane_count=3, line_type="dashed")

        assert "3" in caption or "three" in caption.lower()
        assert "dashed" in caption.lower()

    def test_estimate_lane_count(self):
        """Test lane count estimation."""
        from data.text_generator import TextGenerator

        gen = TextGenerator()

        # Create mask with vertical stripes (simulating lanes)
        mask = np.zeros((100, 100))
        mask[:, 20:30] = 1
        mask[:, 50:60] = 1
        mask[:, 70:80] = 1

        count = gen._estimate_lane_count(mask)
        assert 2 <= count <= 6

    def test_extract_lane_info(self):
        """Test lane info extraction from text."""
        from data.text_generator import TextGenerator

        gen = TextGenerator()

        text = "a 4-lane road with solid center line"
        info = gen.extract_lane_info(text)

        assert info["lane_count"] == 4
        assert info["line_type"] == "solid"


class TestDataset:
    """Tests for dataset loading."""

    def test_dataset_length(self, temp_data_dir):
        """Test dataset returns correct length."""
        from data.dataset_loader import LaneDataset

        dataset = LaneDataset(
            data_root=temp_data_dir,
            dataset="dummy",
            split="train",
            generate_text=False,
        )

        assert len(dataset) == 10

    def test_dataset_getitem(self, temp_data_dir):
        """Test dataset item retrieval."""
        from data.dataset_loader import LaneDataset

        dataset = LaneDataset(
            data_root=temp_data_dir,
            dataset="dummy",
            split="train",
            generate_text=True,
        )

        sample = dataset[0]

        assert "view1" in sample
        assert "view2" in sample
        assert "text" in sample
        assert "image_path" in sample

        assert sample["view1"].shape == (3, 800, 320)
        assert sample["view2"].shape == (3, 800, 320)

    def test_dataset_subset(self, temp_data_dir):
        """Test dataset subset creation."""
        from data.dataset_loader import LaneDataset
        import torch.utils.data as data_utils

        dataset = LaneDataset(
            data_root=temp_data_dir,
            dataset="dummy",
            split="train",
            generate_text=False,
        )

        subset = data_utils.Subset(dataset, range(5))
        assert len(subset) == 5


class TestDataLoader:
    """Tests for dataloader creation."""

    def test_dataloader_creation(self, temp_data_dir):
        """Test dataloader can be created."""
        from data.dataset_loader import create_dataloader

        loader = create_dataloader(
            data_root=temp_data_dir,
            dataset="dummy",
            split="train",
            batch_size=4,
            num_workers=0,
            subset=10,
        )

        assert loader is not None
        assert loader.batch_size == 4

    def test_dataloader_iteration(self, temp_data_dir):
        """Test dataloader can iterate."""
        from data.dataset_loader import create_dataloader

        loader = create_dataloader(
            data_root=temp_data_dir,
            dataset="dummy",
            split="train",
            batch_size=4,
            num_workers=0,
            subset=10,
        )

        batch = next(iter(loader))

        assert "view1" in batch
        assert "view2" in batch
        assert "texts" in batch

        assert batch["view1"].shape[0] <= 4  # Allow for smaller last batch
        assert batch["view1"].shape[1:] == (3, 800, 320)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
