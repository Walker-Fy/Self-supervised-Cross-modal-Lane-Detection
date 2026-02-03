"""Post-processing utilities for lane detection.

This module provides functions for converting segmentation masks to
CULane-compatible lane coordinate files (.lines.txt).
"""

import os
from typing import List, Tuple, Optional, Dict, Any
from pathlib import Path

import numpy as np
import cv2
from scipy import ndimage
from scipy.interpolate import interp1d


class LaneExtractor:
    """Extract lane coordinates from segmentation masks.

    Converts binary lane segmentation masks into the CULane format
    with lane coordinates as (y, x) point pairs.
    """

    def __init__(
        self,
        n_lanes: int = 4,
        n_points: int = 56,
        min_lane_length: float = 0.3,
        confidence_threshold: float = 0.5,
        input_size: Tuple[int, int] = (590, 1640),
        original_size: Tuple[int, int] = (590, 1640),
    ):
        """Initialize lane extractor.

        Args:
            n_lanes: Maximum number of lanes to extract
            n_points: Number of points per lane (CULane standard: 56)
            min_lane_length: Minimum lane length as ratio of image height
            confidence_threshold: Mask confidence threshold
            input_size: Model input size (H, W)
            original_size: Original image size (H, W) for coordinate scaling
        """
        self.n_lanes = n_lanes
        self.n_points = n_points
        self.min_lane_length = min_lane_length
        self.confidence_threshold = confidence_threshold
        self.input_size = input_size
        self.original_size = original_size

        # Generate fixed y coordinates for CULane format
        # CULane uses rows 160-719 (560 rows) sampled at 10-pixel intervals
        self.y_coords = np.arange(160, 720, 10, dtype=np.float32)

    def mask_to_lanes(
        self,
        mask: np.ndarray,
    ) -> List[np.ndarray]:
        """Extract lane coordinates from segmentation mask.

        Args:
            mask: Binary segmentation mask (H, W) with values in [0, 1]

        Returns:
            List of lane arrays, each with shape (N, 2) containing (y, x) coordinates
        """
        # Threshold mask
        binary_mask = (mask > self.confidence_threshold).astype(np.uint8)

        # Find connected components (potential lanes)
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            binary_mask, connectivity=8
        )

        lanes = []

        # Skip background (label 0)
        for label in range(1, num_labels):
            # Create mask for this component
            component_mask = (labels == label).astype(np.uint8)

            # Filter by area
            area = stats[label, cv2.CC_STAT_AREA]
            if area < self.min_lane_length * mask.shape[0] * mask.shape[1] * 0.01:
                continue

            # Extract lane coordinates
            lane_points = self._extract_lane_points(component_mask)

            if lane_points is not None and len(lane_points) >= 2:
                # Interpolate to fixed number of points
                lane_interp = self.interpolate_lane(lane_points, self.n_points)
                lanes.append(lane_interp)

        # Sort lanes by x position (left to right)
        lanes = self.sort_lanes(lanes)

        return lanes

    def _extract_lane_points(
        self,
        mask: np.ndarray,
    ) -> Optional[np.ndarray]:
        """Extract points from a single lane mask.

        Args:
            mask: Binary mask for a single lane

        Returns:
            Array of (y, x) coordinates or None
        """
        # Find contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            return None

        # Get the largest contour
        contour = max(contours, key=cv2.contourArea)

        if len(contour) < 2:
            return None

        # Reshape contour to (N, 2) and convert to (y, x)
        points = contour.reshape(-1, 2)
        points = points[:, [1, 0]]  # Convert (x, y) to (y, x)

        return points

    def interpolate_lane(
        self,
        points: np.ndarray,
        n_points: int,
    ) -> np.ndarray:
        """Interpolate lane to fixed number of points.

        Args:
            points: Lane points (N, 2) with (y, x) coordinates
            n_points: Target number of points

        Returns:
            Interpolated lane points (n_points, 2)
        """
        if len(points) < 2:
            # Return points along left edge
            x_coords = np.zeros(n_points)
            return np.column_stack([self.y_coords[:n_points], x_coords])

        # Sort points by y coordinate
        points = points[points[:, 0].argsort()]

        # Remove duplicate y values
        unique_y, unique_indices = np.unique(points[:, 0], return_index=True)
        if len(unique_y) < 2:
            x_coords = np.zeros(n_points)
            return np.column_stack([self.y_coords[:n_points], x_coords])

        unique_points = points[unique_indices]

        try:
            # Create interpolation function
            # Use linear interpolation with extrapolation
            interp_func = interp1d(
                unique_points[:, 0],
                unique_points[:, 1],
                kind='linear',
                bounds_error=False,
                fill_value='extrapolate'
            )

            # Interpolate at fixed y coordinates
            # Handle case where we have fewer y_coords than n_points
            n_interp = min(n_points, len(self.y_coords))
            y_interp = self.y_coords[:n_interp]

            x_interp = interp_func(y_interp)

            # Clamp x coordinates to valid range
            x_interp = np.clip(x_interp, 0, self.input_size[1] - 1)

            # Combine into lane coordinates
            lane = np.column_stack([y_interp, x_interp])

            # Pad with edge values if needed
            if len(lane) < n_points:
                pad_size = n_points - len(lane)
                last_y = lane[-1, 0] if len(lane) > 0 else self.y_coords[-1]
                last_x = lane[-1, 1] if len(lane) > 0 else 0
                padding = np.tile([last_y + 10, last_x], (pad_size, 1))
                lane = np.vstack([lane, padding])

        except Exception:
            # Fallback: return simple lane along detected x position
            mean_x = np.mean(points[:, 1]) if len(points) > 0 else 0
            x_coords = np.full(n_points, mean_x)
            lane = np.column_stack([self.y_coords[:n_points], x_coords])

        return lane

    def filter_lanes(
        self,
        lanes: List[np.ndarray],
        min_length_ratio: float = 0.3,
    ) -> List[np.ndarray]:
        """Filter out spurious short lanes.

        Args:
            lanes: List of lane arrays
            min_length_ratio: Minimum length as ratio of image height

        Returns:
            Filtered list of lanes
        """
        filtered = []

        min_length = self.input_size[0] * min_length_ratio

        for lane in lanes:
            if len(lane) < 2:
                continue

            # Calculate lane length (y-span)
            y_span = lane[:, 0].max() - lane[:, 0].min()

            if y_span >= min_length:
                filtered.append(lane)

        return filtered

    def sort_lanes(
        self,
        lanes: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Sort lanes by x position (left to right).

        Args:
            lanes: List of lane arrays

        Returns:
            Sorted list of lanes
        """
        if not lanes:
            return lanes

        # Sort by mean x position
        lane_x_positions = []

        for lane in lanes:
            # Use mean x position at middle rows
            mid_idx = len(lane) // 2
            x_pos = lane[mid_idx, 1] if mid_idx < len(lane) else lane[:, 1].mean()
            lane_x_positions.append(x_pos)

        sorted_indices = np.argsort(lane_x_positions)
        return [lanes[i] for i in sorted_indices]

    def lanes_to_txt(
        self,
        lanes: List[np.ndarray],
        save_path: str,
    ) -> None:
        """Save lanes to CULane format .lines.txt file.

        Format: Each line contains "y1 x1 y2 x2 y3 x3 ..." for one lane.

        Args:
            lanes: List of lane arrays with (y, x) coordinates
            save_path: Path to save the .lines.txt file
        """
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        with open(save_path, 'w') as f:
            for lane in lanes:
                # Flatten lane to "y1 x1 y2 x2 ..." format
                coords = lane.flatten().astype(str)
                line = ' '.join(coords)
                f.write(line + '\n')

    def lanes_from_txt(
        self,
        txt_path: str,
    ) -> List[np.ndarray]:
        """Load lanes from CULane format .lines.txt file.

        Args:
            txt_path: Path to the .lines.txt file

        Returns:
            List of lane arrays with (y, x) coordinates
        """
        lanes = []

        if not os.path.exists(txt_path):
            return lanes

        with open(txt_path, 'r') as f:
            for line in f:
                coords = line.strip().split()
                if len(coords) < 4:
                    continue

                # Parse "y1 x1 y2 x2 ..." format
                points = []
                for i in range(0, len(coords), 2):
                    y = float(coords[i])
                    x = float(coords[i + 1])
                    points.append([y, x])

                if points:
                    lanes.append(np.array(points))

        return lanes

    def process_mask(
        self,
        mask: np.ndarray,
        save_path: Optional[str] = None,
    ) -> List[np.ndarray]:
        """Complete pipeline: mask -> lanes -> optionally save to txt.

        Args:
            mask: Binary segmentation mask (H, W)
            save_path: Optional path to save .lines.txt file

        Returns:
            List of lane arrays
        """
        # Extract lanes
        lanes = self.mask_to_lanes(mask)

        # Filter lanes
        lanes = self.filter_lanes(lanes)

        # Sort lanes
        lanes = self.sort_lanes(lanes)

        # Save to file if path provided
        if save_path:
            self.lanes_to_txt(lanes, save_path)

        return lanes


def mask_to_lane_coords(
    mask: np.ndarray,
    n_points: int = 56,
    input_size: Tuple[int, int] = (590, 1640),
) -> List[np.ndarray]:
    """Convenience function to convert mask to lane coordinates.

    Args:
        mask: Binary segmentation mask (H, W)
        n_points: Number of points per lane
        input_size: Input size (H, W)

    Returns:
        List of lane arrays with (y, x) coordinates
    """
    extractor = LaneExtractor(
        n_lanes=4,
        n_points=n_points,
        input_size=input_size,
        original_size=input_size,
    )
    return extractor.process_mask(mask)


def generate_culane_output(
    masks: List[np.ndarray],
    image_paths: List[str],
    output_dir: str,
    n_points: int = 56,
) -> Dict[str, str]:
    """Generate CULane format output files from a batch of masks.

    Args:
        masks: List of segmentation masks
        image_paths: List of corresponding image paths
        output_dir: Directory to save .lines.txt files
        n_points: Number of points per lane

    Returns:
        Dictionary mapping image paths to output file paths
    """
    os.makedirs(output_dir, exist_ok=True)

    extractor = LaneExtractor(n_points=n_points)
    output_paths = {}

    for mask, img_path in zip(masks, image_paths):
        # Generate output filename
        img_name = Path(img_path).stem
        output_path = os.path.join(output_dir, f"{img_name}.lines.txt")

        # Process mask
        extractor.process_mask(mask, save_path=output_path)

        output_paths[img_path] = output_path

    return output_paths


class LaneEvaluator:
    """Evaluate lane predictions against CULane ground truth."""

    def __init__(
        self,
        n_points: int = 56,
        x_tolerance: int = 20,  # Pixels
        y_tolerance: int = 5,  # Pixels
    ):
        """Initialize lane evaluator.

        Args:
            n_points: Number of points per lane
            x_tolerance: Maximum x distance for match (pixels)
            y_tolerance: Maximum y distance for match (pixels)
        """
        self.n_points = n_points
        self.x_tolerance = x_tolerance
        self.y_tolerance = y_tolerance

    def match_lanes(
        self,
        pred_lanes: List[np.ndarray],
        gt_lanes: List[np.ndarray],
    ) -> Tuple[List[Optional[int]], List[int], List[int]]:
        """Match predicted lanes to ground truth lanes.

        Args:
            pred_lanes: List of predicted lane arrays
            gt_lanes: List of ground truth lane arrays

        Returns:
            Tuple of (pred_to_gt, unmatched_pred, unmatched_gt)
            - pred_to_gt: List mapping pred index to gt index (or None)
            - unmatched_pred: Indices of unmatched predictions
            - unmatched_gt: Indices of unmatched ground truth
        """
        if not pred_lanes:
            return [], [], list(range(len(gt_lanes)))

        if not gt_lanes:
            return [None] * len(pred_lanes), list(range(len(pred_lanes))), []

        pred_to_gt = [None] * len(pred_lanes)
        matched_gt = set()

        # Simple matching: compare mean x positions
        pred_x = []
        gt_x = []

        for lane in pred_lanes:
            if len(lane) > 0:
                pred_x.append(np.mean(lane[:, 1]))
            else:
                pred_x.append(float('inf'))

        for lane in gt_lanes:
            if len(lane) > 0:
                gt_x.append(np.mean(lane[:, 1]))
            else:
                gt_x.append(float('inf'))

        # Match each prediction to closest GT
        for i, px in enumerate(pred_x):
            if px == float('inf'):
                continue

            best_j = None
            best_dist = float('inf')

            for j, gx in enumerate(gt_x):
                if j in matched_gt:
                    continue

                dist = abs(px - gx)
                if dist < best_dist and dist < self.x_tolerance * 2:
                    best_dist = dist
                    best_j = j

            if best_j is not None:
                pred_to_gt[i] = best_j
                matched_gt.add(best_j)

        # Find unmatched
        unmatched_pred = [i for i, gt_idx in enumerate(pred_to_gt) if gt_idx is None]
        unmatched_gt = [j for j in range(len(gt_lanes)) if j not in matched_gt]

        return pred_to_gt, unmatched_pred, unmatched_gt

    def compute_lane_iou(
        self,
        pred_lane: np.ndarray,
        gt_lane: np.ndarray,
        img_width: int,
        img_height: int,
    ) -> float:
        """Compute IoU between predicted and ground truth lane.

        Args:
            pred_lane: Predicted lane coordinates (N, 2)
            gt_lane: Ground truth lane coordinates (M, 2)
            img_width: Image width
            img_height: Image height

        Returns:
            IoU score
        """
        # Convert lanes to masks
        pred_mask = self._lane_to_mask(pred_lane, img_width, img_height)
        gt_mask = self._lane_to_mask(gt_lane, img_width, img_height)

        # Compute IoU
        intersection = np.logical_and(pred_mask, gt_mask).sum()
        union = np.logical_or(pred_mask, gt_mask).sum()

        if union == 0:
            return 0.0

        return intersection / union

    def _lane_to_mask(
        self,
        lane: np.ndarray,
        width: int,
        height: int,
        thickness: int = 5,
    ) -> np.ndarray:
        """Convert lane coordinates to binary mask.

        Args:
            lane: Lane coordinates (N, 2) with (y, x)
            width: Image width
            height: Image height
            thickness: Line thickness

        Returns:
            Binary mask
        """
        mask = np.zeros((height, width), dtype=np.uint8)

        if len(lane) < 2:
            return mask

        # Convert (y, x) to (x, y) for OpenCV
        points = lane[:, [1, 0]].astype(np.int32)
        points = np.clip(points, [0, 0], [width - 1, height - 1])

        cv2.polylines(mask, [points], False, 1, thickness=thickness)

        return mask

    def evaluate_frame(
        self,
        pred_lanes: List[np.ndarray],
        gt_lanes: List[np.ndarray],
        img_width: int,
        img_height: int,
    ) -> Dict[str, Any]:
        """Evaluate prediction for a single frame.

        Args:
            pred_lanes: List of predicted lanes
            gt_lanes: List of ground truth lanes
            img_width: Image width
            img_height: Image height

        Returns:
            Dictionary with evaluation metrics
        """
        # Match lanes
        pred_to_gt, unmatched_pred, unmatched_gt = self.match_lanes(
            pred_lanes, gt_lanes
        )

        # Compute metrics
        tp = 0  # True positives (matched lanes with good IoU)
        fp = len(unmatched_pred)  # False positives
        fn = len(unmatched_gt)  # False negatives

        for i, gt_idx in enumerate(pred_to_gt):
            if gt_idx is not None:
                iou = self.compute_lane_iou(
                    pred_lanes[i], gt_lanes[gt_idx],
                    img_width, img_height
                )
                if iou >= 0.5:  # Standard IoU threshold
                    tp += 1
                else:
                    fp += 1

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
