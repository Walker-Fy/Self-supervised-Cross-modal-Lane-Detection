"""Create dummy lane dataset for testing.

Generates synthetic lane images and masks for development/testing
without requiring real datasets.
"""

import os
import argparse
from pathlib import Path
from typing import Tuple, List

import numpy as np
from PIL import Image, ImageDraw


class DummyLaneGenerator:
    """Generate synthetic lane images and masks."""

    def __init__(
        self,
        output_dir: str,
        image_size: Tuple[int, int] = (1280, 720),
        lane_width: int = 10,
    ):
        """Initialize generator.

        Args:
            output_dir: Output directory
            image_size: Image size (width, height)
            lane_width: Width of lane markings in pixels
        """
        self.output_dir = Path(output_dir)
        self.image_size = image_size
        self.lane_width = lane_width

        # Create directories
        self.images_dir = self.output_dir / "dummy" / "images"
        self.masks_dir = self.output_dir / "dummy" / "masks"

        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)

    def generate_background(self) -> np.ndarray:
        """Generate road background.

        Returns:
            Background image (H, W, 3) as uint8
        """
        h, w = self.image_size[1], self.image_size[0]

        # Create gradient sky (ensure uint8)
        sky = np.linspace((135, 206, 235), (200, 230, 255), h // 3)
        sky = np.tile(sky[:, np.newaxis, :], (1, w, 1))
        sky = sky.astype(np.uint8)  # Convert to uint8

        # Create road surface
        road_color = np.array([50, 50, 50])
        road = np.full((2 * h // 3, w, 3), road_color, dtype=np.uint8)

        # Add some texture variation
        noise = np.random.randint(-10, 10, road.shape, dtype=np.int16)
        road = np.clip(road.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        # Combine (both are now uint8)
        image = np.vstack([sky, road])

        return image

    def draw_lane(
        self,
        image: np.ndarray,
        start_point: Tuple[float, float],
        end_point: Tuple[float, float],
        color: Tuple[int, int, int] = (255, 255, 255),
        dashed: bool = False,
    ) -> None:
        """Draw a lane on the image.

        Args:
            image: Image to draw on
            start_point: Start point (x, y) as fractions of image size
            end_point: End point (x, y) as fractions of image size
            color: Lane color
            dashed: Whether to draw dashed line
        """
        h, w = image.shape[:2]

        # Convert to pixel coordinates
        x1, y1 = int(start_point[0] * w), int(start_point[1] * h)
        x2, y2 = int(end_point[0] * w), int(end_point[1] * h)

        if dashed:
            # Draw dashed line
            dash_length = 30
            gap_length = 20
            total_length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            num_dashes = int(total_length / (dash_length + gap_length))

            dx = (x2 - x1) / num_dashes
            dy = (y2 - y1) / num_dashes

            for i in range(num_dashes):
                sx = x1 + i * (dx + dx * gap_length / dash_length)
                sy = y1 + i * (dy + dy * gap_length / dash_length)
                ex = sx + dx
                ey = sy + dy

                # Draw thick line segment
                for offset in range(-self.lane_width // 2, self.lane_width // 2 + 1):
                    ImageDraw.Draw(
                        Image.fromarray(image)
                    ).line(
                        [(sx, sy + offset), (ex, ey + offset)],
                        fill=color,
                        width=2,
                    )
        else:
            # Draw solid line
            for offset in range(-self.lane_width // 2, self.lane_width // 2 + 1):
                ImageDraw.Draw(
                    Image.fromarray(image)
                ).line(
                    [(x1, y1 + offset), (x2, y2 + offset)],
                    fill=color,
                    width=2,
                )

    def generate_sample(
        self,
        idx: int,
    ) -> Tuple[str, str]:
        """Generate a single sample.

        Args:
            idx: Sample index

        Returns:
            Tuple of (image_path, mask_path)
        """
        # Random parameters
        num_lanes = np.random.randint(2, 5)
        has_center_line = np.random.random() > 0.3
        center_dashed = np.random.random() > 0.5

        # Generate background
        image = self.generate_background()
        mask = np.zeros(self.image_size[::-1], dtype=np.uint8)

        # Draw lanes
        h, w = self.image_size[1], self.image_size[0]

        # Perspective effect: lanes converge towards horizon
        horizon_y = 0.35  # Horizon at 35% of image height

        # Left lanes
        for i in range(num_lanes // 2):
            x_bottom = 0.1 + i * 0.15 + np.random.uniform(-0.02, 0.02)
            x_top = 0.35 + i * 0.05 + np.random.uniform(-0.01, 0.01)

            self.draw_lane(
                image,
                (x_bottom, 1.0),
                (x_top, horizon_y),
                color=(255, 255, 255),
                dashed=True,
            )

            # Draw on mask
            draw = ImageDraw.Draw(Image.fromarray(mask))
            for offset in range(-self.lane_width // 2, self.lane_width // 2 + 1):
                draw.line(
                    [(int(x_bottom * w), int(h) + offset),
                     (int(x_top * w), int(horizon_y * h) + offset)],
                    fill=255,
                    width=2,
                )

        # Right lanes
        for i in range(num_lanes // 2):
            x_bottom = 0.9 - i * 0.15 + np.random.uniform(-0.02, 0.02)
            x_top = 0.65 - i * 0.05 + np.random.uniform(-0.01, 0.01)

            self.draw_lane(
                image,
                (x_bottom, 1.0),
                (x_top, horizon_y),
                color=(255, 255, 255),
                dashed=True,
            )

            # Draw on mask
            draw = ImageDraw.Draw(Image.fromarray(mask))
            for offset in range(-self.lane_width // 2, self.lane_width // 2 + 1):
                draw.line(
                    [(int(x_bottom * w), int(h) + offset),
                     (int(x_top * w), int(horizon_y * h) + offset)],
                    fill=255,
                    width=2,
                )

        # Center line
        if has_center_line:
            x_bottom = 0.5 + np.random.uniform(-0.02, 0.02)
            x_top = 0.5 + np.random.uniform(-0.01, 0.01)

            color = (255, 255, 0) if center_dashed else (255, 255, 255)

            self.draw_lane(
                image,
                (x_bottom, 1.0),
                (x_top, horizon_y),
                color=color,
                dashed=center_dashed,
            )

            # Draw on mask
            draw = ImageDraw.Draw(Image.fromarray(mask))
            for offset in range(-self.lane_width // 2 - 2, self.lane_width // 2 + 3):
                draw.line(
                    [(int(x_bottom * w), int(h) + offset),
                     (int(x_top * w), int(horizon_y * h) + offset)],
                    fill=255,
                    width=2,
                )

        # Save images
        image_path = self.images_dir / f"{idx:05d}.jpg"
        mask_path = self.masks_dir / f"{idx:05d}.png"

        Image.fromarray(image).save(image_path)
        Image.fromarray(mask).save(mask_path)

        return str(image_path), str(mask_path)

    def generate(
        self,
        num_samples: int = 100,
    ) -> List[Tuple[str, str]]:
        """Generate multiple samples.

        Args:
            num_samples: Number of samples to generate

        Returns:
            List of (image_path, mask_path) tuples
        """
        samples = []

        print(f"Generating {num_samples} dummy samples...")

        for i in range(num_samples):
            image_path, mask_path = self.generate_sample(i)
            samples.append((image_path, mask_path))

            if (i + 1) % 10 == 0:
                print(f"  Generated {i + 1}/{num_samples}")

        print(f"Done! Saved to {self.output_dir}")

        return samples


def generate_texts(
    data_root: str,
    output_file: str = None,
) -> None:
    """Generate text captions for dummy dataset.

    Args:
        data_root: Data root directory
        output_file: Output file path
    """
    from data.text_generator import TextGenerator

    mask_dir = Path(data_root) / "dummy" / "masks"

    if output_file is None:
        output_file = Path(data_root) / "dummy" / "captions.txt"

    generator = TextGenerator()

    captions = []

    for mask_path in sorted(mask_dir.glob("*.png")):
        mask = np.array(Image.open(mask_path))
        caption = generator.generate_from_mask(mask)
        captions.append(f"{mask_path.stem}\t{caption}")

    with open(output_file, "w") as f:
        f.write("\n".join(captions))

    print(f"Saved {len(captions)} captions to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=str,
        default="./data",
        help="Output directory",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to generate",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[1280, 720],
        help="Image size (width height)",
    )
    parser.add_argument(
        "--generate-texts",
        action="store_true",
        help="Generate text captions",
    )

    args = parser.parse_args()

    # Create dummy dataset
    generator = DummyLaneGenerator(
        output_dir=args.output,
        image_size=tuple(args.image_size),
    )

    generator.generate(num_samples=args.num_samples)

    # Generate texts if requested
    if args.generate_texts:
        generate_texts(args.output)


if __name__ == "__main__":
    main()
