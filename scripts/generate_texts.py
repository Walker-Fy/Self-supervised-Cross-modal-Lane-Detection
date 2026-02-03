"""Generate text captions for lane dataset."""

import os
import argparse
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image

from data.text_generator import TextGenerator


def generate_captions_for_images(
    image_dir: str,
    mask_dir: str,
    output_file: str,
    text_generator: TextGenerator = None,
) -> List[str]:
    """Generate captions for a dataset.

    Args:
        image_dir: Directory containing images
        mask_dir: Directory containing masks
        output_file: Output text file path
        text_generator: Text generator instance

    Returns:
        List of generated captions
    """
    if text_generator is None:
        text_generator = TextGenerator()

    image_dir = Path(image_dir)
    mask_dir = Path(mask_dir)

    if not mask_dir.exists():
        print(f"Mask directory {mask_dir} not found. Using image directory.")
        mask_dir = image_dir

    captions = []
    mask_files = sorted(mask_dir.glob("*.png")) + sorted(mask_dir.glob("*.jpg"))

    print(f"Found {len(mask_files)} mask files")

    for mask_path in mask_files:
        # Load mask
        mask = np.array(Image.open(mask_path))

        # Convert to binary if needed
        if mask.ndim == 3:
            mask = mask[:, :, 0]

        mask = (mask > 127).astype(np.uint8) * 255

        # Generate caption
        caption = text_generator.generate_from_mask(mask)

        # Use image stem as ID
        image_id = mask_path.stem
        captions.append(f"{image_id}\t{caption}")

    # Save captions
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("\n".join(captions))

    print(f"Saved {len(captions)} captions to {output_file}")

    return captions


def generate_all_prompts(
    output_file: str,
    max_lanes: int = 6,
) -> None:
    """Generate all possible lane prompts.

    Args:
        output_file: Output file path
        max_lanes: Maximum number of lanes
    """
    text_generator = TextGenerator()
    prompts = text_generator.get_all_prompts(max_lanes=max_lanes)

    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        f.write("\n".join(prompts))

    print(f"Saved {len(prompts)} prompts to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--image-dir",
        type=str,
        default="./data/dummy/images",
        help="Image directory",
    )
    parser.add_argument(
        "--mask-dir",
        type=str,
        default="./data/dummy/masks",
        help="Mask directory",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./data/dummy/captions.txt",
        help="Output caption file",
    )
    parser.add_argument(
        "--all-prompts",
        action="store_true",
        help="Generate all possible prompts",
    )
    parser.add_argument(
        "--max-lanes",
        type=int,
        default=6,
        help="Maximum number of lanes for prompt generation",
    )

    args = parser.parse_args()

    if args.all_prompts:
        generate_all_prompts(args.output, args.max_lanes)
    else:
        generate_captions_for_images(
            args.image_dir,
            args.mask_dir,
            args.output,
        )


if __name__ == "__main__":
    main()
