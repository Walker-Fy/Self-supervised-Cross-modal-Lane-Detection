"""Text generation for cross-modal learning."""

import re
from typing import List, Dict, Any, Optional
import numpy as np


class TextGenerator:
    """Generate pseudo captions for lane images.

    Creates text descriptions based on lane configurations to enable
    cross-modal learning with CLIP.
    """

    # Templates for describing lanes
    LANE_COUNT_TEMPLATES = [
        "a road with {count} lane(s)",
        "a {count}-lane road",
        "a highway showing {count} lane(s)",
        "a street with {count} lane(s) visible",
    ]

    LINE_TYPE_TEMPLATES = [
        "a {line_type} center line",
        "with {line_type} road markings",
        "featuring {line_type} lane markings",
    ]

    COMBINED_TEMPLATES = [
        "a {count}-lane road with {line_type} center line",
        "a road with {count} lane(s) and {line_type} markings",
        "a highway scene showing {count} lane(s) with {line_type} center line",
    ]

    # Line type descriptions
    LINE_TYPES = {
        "solid": ["solid", "continuous", "unbroken"],
        "dashed": ["dashed", "dotted", "broken", "segmented"],
        "double": ["double", "dual"],
        "mixed": ["mixed", "combination"],
    }

    def __init__(self, use_templates: bool = True):
        """Initialize text generator.

        Args:
            use_templates: Whether to use multiple template variations
        """
        self.use_templates = use_templates
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for text extraction."""
        self.lane_count_pattern = re.compile(
            r'(\d+)\s*lane', re.IGNORECASE
        )
        self.line_type_patterns = {
            "solid": re.compile(r'\b(solid|continuous|unbroken)\b', re.IGNORECASE),
            "dashed": re.compile(r'\b(dashed|dotted|broken|segmented)\b', re.IGNORECASE),
            "double": re.compile(r'\b(double|dual)\b', re.IGNORECASE),
        }

    def generate_from_mask(
        self,
        mask: np.ndarray,
    ) -> str:
        """Generate caption from lane mask.

        Args:
            mask: Binary lane mask (H, W)

        Returns:
            Generated text caption
        """
        lane_count = self._estimate_lane_count(mask)
        line_type = self._estimate_line_type(mask)

        return self.generate(lane_count, line_type)

    def _estimate_lane_count(self, mask: np.ndarray) -> int:
        """Estimate number of lanes from mask.

        Args:
            mask: Binary lane mask (H, W)

        Returns:
            Estimated lane count
        """
        # Simple approach: count connected components in lower half
        h, w = mask.shape
        lower_region = mask[h // 2:, :]

        # Use horizontal projection profile
        h_proj = lower_region.sum(axis=0)

        # Find peaks
        from scipy import signal
        peaks, _ = signal.find_peaks(h_proj, distance=w // 10)

        # Estimate lane count (2 lanes per detected peak roughly)
        lane_count = max(2, min(len(peaks), 6))

        return lane_count

    def _estimate_line_type(self, mask: np.ndarray) -> str:
        """Estimate center line type from mask.

        Args:
            mask: Binary lane mask (H, W)

        Returns:
            Line type ('solid', 'dashed', or 'double')
        """
        h, w = mask.shape

        # Sample center column region
        center_region = mask[:, w // 2 - 10 : w // 2 + 10]

        if center_region.sum() == 0:
            # Sample wider center region
            center_region = mask[:, w // 3 : 2 * w // 3]

        # Vertical projection
        v_proj = center_region.sum(axis=1)

        # Check for gaps (dashed lines)
        threshold = v_proj.max() * 0.1
        above_thresh = v_proj > threshold

        # Count transitions
        transitions = 0
        for i in range(1, len(above_thresh)):
            if above_thresh[i] != above_thresh[i - 1]:
                transitions += 1

        # Heuristic: dashed lines have many transitions
        if transitions > 10:
            return "dashed"
        elif transitions > 4:
            return "mixed"
        else:
            return "solid"

    def generate(
        self,
        lane_count: int,
        line_type: str = "solid",
    ) -> str:
        """Generate caption from lane parameters.

        Args:
            lane_count: Number of lanes
            line_type: Type of center line ('solid', 'dashed', 'double', 'mixed')

        Returns:
            Generated text caption
        """
        # Get random line type descriptor
        line_type_desc = line_type
        if line_type in self.LINE_TYPES and self.use_templates:
            descriptors = self.LINE_TYPES[line_type]
            line_type_desc = np.random.choice(descriptors)

        # Choose template
        if np.random.random() < 0.5:
            template = np.random.choice(self.COMBINED_TEMPLATES)
            caption = template.format(
                count=lane_count,
                line_type=line_type_desc,
            )
        else:
            lane_template = np.random.choice(self.LANE_COUNT_TEMPLATES)
            line_template = np.random.choice(self.LINE_TYPE_TEMPLATES)

            caption = f"{lane_template.format(count=lane_count)}, {line_template.format(line_type=line_type_desc)}"

        # Post-process: handle pluralization
        caption = self._fix_pluralization(caption, lane_count)

        return caption

    def _fix_pluralization(self, text: str, count: int) -> str:
        """Fix pluralization in generated text.

        Args:
            text: Input text
            count: Number for pluralization

        Returns:
            Text with corrected pluralization
        """
        if count == 1:
            text = text.replace("lane(s)", "lane")
            text = text.replace("1-lanes", "1-lane")
        else:
            text = text.replace("lane(s)", "lanes")

        return text

    def generate_batch(
        self,
        lane_counts: List[int],
        line_types: Optional[List[str]] = None,
    ) -> List[str]:
        """Generate multiple captions.

        Args:
            lane_counts: List of lane counts
            line_types: List of line types (defaults to all 'solid')

        Returns:
            List of generated captions
        """
        if line_types is None:
            line_types = ["solid"] * len(lane_counts)

        captions = []
        for count, line_type in zip(lane_counts, line_types):
            captions.append(self.generate(count, line_type))

        return captions

    def extract_lane_info(
        self,
        text: str,
    ) -> Dict[str, Any]:
        """Extract lane information from text caption.

        Args:
            text: Input text caption

        Returns:
            Dictionary with 'lane_count' and 'line_type'
        """
        info = {
            "lane_count": 2,  # Default
            "line_type": "solid",  # Default
        }

        # Extract lane count
        match = self.lane_count_pattern.search(text)
        if match:
            info["lane_count"] = int(match.group(1))

        # Extract line type
        for line_type, pattern in self.line_type_patterns.items():
            if pattern.search(text):
                info["line_type"] = line_type
                break

        return info

    def get_all_prompts(
        self,
        max_lanes: int = 6,
    ) -> List[str]:
        """Get all possible prompts for the dataset.

        Args:
            max_lanes: Maximum number of lanes

        Returns:
            List of all possible prompts
        """
        prompts = []

        for count in range(2, max_lanes + 1):
            for line_type in ["solid", "dashed", "double", "mixed"]:
                prompts.append(self.generate(count, line_type))

        return prompts

    def encode_text(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 77,
    ) -> Dict[str, Any]:
        """Tokenize texts for CLIP text encoder.

        Args:
            texts: List of text strings
            tokenizer: CLIP tokenizer
            max_length: Maximum sequence length

        Returns:
            Dictionary with tokenized texts
        """
        return tokenizer(texts, max_length=max_length)
