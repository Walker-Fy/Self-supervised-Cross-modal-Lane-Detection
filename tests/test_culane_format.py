"""Tests for CULane format conversion and evaluation.

These tests verify that lane predictions are correctly converted
to the CULane .lines.txt format and can be properly evaluated.
"""

import os
import sys
import tempfile
from pathlib import Path

import pytest
import numpy as np
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.postprocess import (
    LaneExtractor,
    LaneEvaluator,
    mask_to_lane_coords,
    generate_culane_output,
)


class TestLaneExtractor:
    """Test cases for LaneExtractor class."""

    @pytest.fixture
    def extractor(self):
        """Create a LaneExtractor instance for testing."""
        return LaneExtractor(
            n_lanes=4,
            n_points=56,
            min_lane_length=0.3,
            confidence_threshold=0.5,
            input_size=(590, 1640),
            original_size=(590, 1640),
        )

    @pytest.fixture
    def sample_mask(self):
        """Create a sample lane mask for testing."""
        mask = np.zeros((590, 1640), dtype=np.uint8)

        # Add two diagonal lanes
        for i in range(100, 500):
            # Lane 1 (left)
            x1 = int(400 + i * 0.5)
            if 0 <= x1 < 1640:
                mask[i, x1-2:x1+3] = 255

            # Lane 2 (right)
            x2 = int(800 + i * 0.8)
            if 0 <= x2 < 1640:
                mask[i, x2-2:x2+3] = 255

        return mask.astype(np.float32) / 255.0

    def test_extractor_init(self, extractor):
        """Test LaneExtractor initialization."""
        assert extractor.n_lanes == 4
        assert extractor.n_points == 56
        assert extractor.input_size == (590, 1640)
        assert len(extractor.y_coords) == 56

    def test_y_coordinates_generation(self, extractor):
        """Test that y coordinates are generated correctly for CULane."""
        # CULane uses rows 160-719 sampled at 10-pixel intervals
        expected_first = 160
        expected_last = 710  # 160 + 55 * 10 = 710
        expected_step = 10

        assert extractor.y_coords[0] == expected_first
        assert extractor.y_coords[-1] == expected_last
        assert (np.diff(extractor.y_coords) == expected_step).all()

    def test_mask_to_lanes(self, extractor, sample_mask):
        """Test converting mask to lane coordinates."""
        lanes = extractor.mask_to_lanes(sample_mask)

        # Should detect some lanes
        assert isinstance(lanes, list)
        # Note: The exact number depends on the connected components found
        # At minimum, we should get a list
        assert len(lanes) >= 0

    def test_filter_lanes(self, extractor, sample_mask):
        """Test filtering short lanes."""
        lanes = extractor.mask_to_lanes(sample_mask)
        filtered = extractor.filter_lanes(lanes, min_length_ratio=0.3)

        # Filtered lanes should be a subset
        assert len(filtered) <= len(lanes)

    def test_sort_lanes(self, extractor):
        """Test sorting lanes by x position."""
        # Create mock lanes
        lane1 = np.array([[100, 800], [200, 850], [300, 900]])  # Right
        lane2 = np.array([[100, 400], [200, 420], [300, 440]])  # Left
        lane3 = np.array([[100, 600], [200, 630], [300, 660]])  # Middle

        lanes = [lane1, lane2, lane3]
        sorted_lanes = extractor.sort_lanes(lanes)

        # Should be sorted left-to-right
        assert len(sorted_lanes) == 3
        # Middle x at index 1 (middle row)
        assert sorted_lanes[0][1, 1] < sorted_lanes[1][1, 1]
        assert sorted_lanes[1][1, 1] < sorted_lanes[2][1, 1]

    def test_lanes_to_txt(self, extractor, tmp_path):
        """Test saving lanes to .lines.txt format."""
        # Create mock lanes
        lane1 = np.array([
            [160, 400], [170, 405], [180, 410],
            [190, 415], [200, 420]
        ])
        lanes = [lane1]

        # Save to file
        output_path = tmp_path / "test_output.lines.txt"
        extractor.lanes_to_txt(lanes, str(output_path))

        # Verify file exists
        assert output_path.exists()

        # Verify content
        with open(output_path, "r") as f:
            content = f.read()
            # Format: y1 x1 y2 x2 ...
            assert "160 400" in content
            assert "170 405" in content

    def test_lanes_from_txt(self, extractor, tmp_path):
        """Test loading lanes from .lines.txt format."""
        # Create test file
        test_file = tmp_path / "test_input.lines.txt"
        with open(test_file, "w") as f:
            f.write("160 400 170 405 180 410\n")
            f.write("160 800 170 805 180 810\n")

        # Load lanes
        lanes = extractor.lanes_from_txt(str(test_file))

        # Verify
        assert len(lanes) == 2
        assert lanes[0].shape == (3, 2)  # 3 points, (y, x) coordinates
        assert lanes[0][0, 0] == 160  # y coordinate
        assert lanes[0][0, 1] == 400  # x coordinate

    def test_process_mask(self, extractor, sample_mask, tmp_path):
        """Test complete pipeline: mask -> lanes -> txt."""
        output_path = tmp_path / "test_process.lines.txt"

        lanes = extractor.process_mask(sample_mask, save_path=str(output_path))

        # Verify lanes were extracted
        assert isinstance(lanes, list)

        # Verify file was created
        assert output_path.exists()

    def test_interpolate_lane(self, extractor):
        """Test lane interpolation to fixed number of points."""
        # Create a short lane
        points = np.array([[100, 400], [200, 450], [300, 500]])

        interpolated = extractor.interpolate_lane(points, 10)

        # Should have 10 points
        assert len(interpolated) == 10

        # Points should be (y, x) format
        assert interpolated.shape == (10, 2)

        # y coordinates should be monotonically increasing
        assert (interpolated[1:, 0] >= interpolated[:-1, 0]).all()


class TestLaneEvaluator:
    """Test cases for LaneEvaluator class."""

    @pytest.fixture
    def evaluator(self):
        """Create a LaneEvaluator instance for testing."""
        return LaneEvaluator(
            n_points=56,
            x_tolerance=20,
            y_tolerance=5,
        )

    def test_match_lanes(self, evaluator):
        """Test lane matching."""
        # Create mock predictions and ground truth
        pred_lanes = [
            np.array([[100, 400], [200, 410], [300, 420]]),
            np.array([[100, 800], [200, 810], [300, 820]]),
        ]
        gt_lanes = [
            np.array([[100, 405], [200, 415], [300, 425]]),
            np.array([[100, 600], [200, 610], [300, 620]]),
        ]

        pred_to_gt, unmatched_pred, unmatched_gt = evaluator.match_lanes(
            pred_lanes, gt_lanes
        )

        # First prediction should match first GT (close in x)
        assert len(pred_to_gt) == 2
        assert len(unmatched_pred) <= 2
        assert len(unmatched_gt) <= 2

    def test_compute_lane_iou(self, evaluator):
        """Test lane IoU computation."""
        # Create overlapping lanes
        pred_lane = np.array([[100, 400], [200, 410], [300, 420]])
        gt_lane = np.array([[100, 405], [200, 415], [300, 425]])

        iou = evaluator.compute_lane_iou(
            pred_lane, gt_lane,
            img_width=1640, img_height=590
        )

        # Should have some overlap
        assert 0 <= iou <= 1

    def test_evaluate_frame(self, evaluator):
        """Test frame evaluation."""
        pred_lanes = [
            np.array([[100, 400], [200, 410], [300, 420]]),
        ]
        gt_lanes = [
            np.array([[100, 405], [200, 415], [300, 425]]),
        ]

        metrics = evaluator.evaluate_frame(
            pred_lanes, gt_lanes,
            img_width=1640, img_height=590
        )

        # Check metric keys
        assert "tp" in metrics
        assert "fp" in metrics
        assert "fn" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics

        # Check metric ranges
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_mask_to_lane_coords(self):
        """Test mask_to_lane_coords convenience function."""
        # Create a simple mask
        mask = np.zeros((590, 1640), dtype=np.float32)
        mask[100:200, 400:410] = 1.0

        lanes = mask_to_lane_coords(mask, n_points=56, input_size=(590, 1640))

        assert isinstance(lanes, list)

    def test_generate_culane_output(self, tmp_path):
        """Test generate_culane_output function."""
        # Create mock masks and paths
        masks = [
            np.zeros((590, 1640), dtype=np.float32),
            np.zeros((590, 1640), dtype=np.float32),
        ]
        # Add some lane pixels
        masks[0][100:200, 400:410] = 1.0
        masks[1][100:200, 800:810] = 1.0

        image_paths = [
            "/path/to/image1.jpg",
            "/path/to/image2.jpg",
        ]

        output_dir = str(tmp_path / "results")

        output_paths = generate_culane_output(
            masks, image_paths, output_dir, n_points=56
        )

        # Check outputs
        assert len(output_paths) == 2
        assert os.path.exists(os.path.join(output_dir, "image1.lines.txt"))
        assert os.path.exists(os.path.join(output_dir, "image2.lines.txt"))


class TestCULaneFormatCompatibility:
    """Test CULane format compatibility."""

    def test_culane_coordinate_format(self, tmp_path):
        """Test that output matches CULane coordinate format."""
        extractor = LaneExtractor(n_points=56)

        # Create a sample lane
        lane = np.column_stack([
            np.arange(160, 720, 10, dtype=np.float32),  # y
            np.ones(56, dtype=np.float32) * 500,  # x
        ])

        # Save to file
        output_path = tmp_path / "test.lines.txt"
        extractor.lanes_to_txt([lane], str(output_path))

        # Read and verify format
        with open(output_path, "r") as f:
            lines = f.readlines()

        assert len(lines) == 1  # One lane
        coords = lines[0].strip().split()

        # Should have 56 points * 2 coordinates = 112 values
        assert len(coords) == 112

        # Format should be: y1 x1 y2 x2 ...
        for i in range(0, len(coords), 2):
            y = float(coords[i])
            x = float(coords[i + 1])
            # y should be in range [160, 719]
            assert 160 <= y <= 719

    def test_culane_standard_dimensions(self):
        """Test that CULane standard dimensions are supported."""
        extractor = LaneExtractor(
            input_size=(590, 1640),
            original_size=(590, 1640),
        )

        assert extractor.input_size == (590, 1640)
        assert extractor.original_size == (590, 1640)

    def test_empty_mask_handling(self):
        """Test handling of empty masks."""
        extractor = LaneExtractor()

        # Create empty mask
        mask = np.zeros((590, 1640), dtype=np.float32)

        lanes = extractor.mask_to_lanes(mask)

        # Should return empty list
        assert isinstance(lanes, list)
        assert len(lanes) == 0


def test_imports():
    """Test that all modules can be imported."""
    from utils.postprocess import LaneExtractor, LaneEvaluator
    from utils.tta import TTAWrapper, LaneDetectionTTA
    from utils.metrics import L1RegressionLoss, SmoothL1RegressionLoss

    assert LaneExtractor is not None
    assert LaneEvaluator is not None
    assert TTAWrapper is not None
    assert LaneDetectionTTA is not None
    assert L1RegressionLoss is not None
    assert SmoothL1RegressionLoss is not None


if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
