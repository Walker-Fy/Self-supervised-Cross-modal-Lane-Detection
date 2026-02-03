"""Apply perturbations to images for robustness testing."""

import argparse
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import cv2


class ImagePerturbation:
    """Apply various perturbations to images."""

    @staticmethod
    def gaussian_blur(
        image: Image.Image,
        kernel_size: int = 5,
    ) -> Image.Image:
        """Apply Gaussian blur.

        Args:
            image: Input image
            kernel_size: Blur kernel size (must be odd)

        Returns:
            Blurred image
        """
        if kernel_size % 2 == 0:
            kernel_size += 1
        return image.filter(ImageFilter.GaussianBlur(radius=kernel_size // 2))

    @staticmethod
    def motion_blur(
        image: Image.Image,
        kernel_size: int = 5,
    ) -> Image.Image:
        """Apply motion blur.

        Args:
            image: Input image
            kernel_size: Blur kernel size

        Returns:
            Blurred image
        """
        # Convert to numpy
        img = np.array(image)

        # Create motion blur kernel
        kernel = np.zeros((kernel_size, kernel_size))
        kernel[int((kernel_size - 1) / 2), :] = np.ones(kernel_size)
        kernel = kernel / kernel_size

        # Apply
        blurred = cv2.filter2D(img, -1, kernel)

        return Image.fromarray(blurred)

    @staticmethod
    def gaussian_noise(
        image: Image.Image,
        sigma: float = 0.05,
    ) -> Image.Image:
        """Add Gaussian noise.

        Args:
            image: Input image
            sigma: Noise standard deviation (0-1)

        Returns:
            Noisy image
        """
        img = np.array(image).astype(float) / 255.0

        noise = np.random.normal(0, sigma, img.shape)
        noisy = np.clip(img + noise, 0, 1)

        return Image.fromarray((noisy * 255).astype(np.uint8))

    @staticmethod
    def brightness(
        image: Image.Image,
        factor: float = 1.2,
    ) -> Image.Image:
        """Adjust brightness.

        Args:
            image: Input image
            factor: Brightness factor (>1 brighter, <1 darker)

        Returns:
            Adjusted image
        """
        enhancer = ImageEnhance.Brightness(image)
        return enhancer.enhance(factor)

    @staticmethod
    def contrast(
        image: Image.Image,
        factor: float = 1.2,
    ) -> Image.Image:
        """Adjust contrast.

        Args:
            image: Input image
            factor: Contrast factor (>1 more contrast, <1 less)

        Returns:
            Adjusted image
        """
        enhancer = ImageEnhance.Contrast(image)
        return enhancer.enhance(factor)

    @staticmethod
    def defocus_blur(
        image: Image.Image,
        kernel_size: int = 5,
    ) -> Image.Image:
        """Apply defocus blur.

        Args:
            image: Input image
            kernel_size: Blur kernel size

        Returns:
            Blurred image
        """
        # Create disk kernel
        kernel = np.zeros((kernel_size, kernel_size))
        center = kernel_size // 2
        radius = center

        for i in range(kernel_size):
            for j in range(kernel_size):
                if (i - center) ** 2 + (j - center) ** 2 <= radius ** 2:
                    kernel[i, j] = 1

        kernel = kernel / kernel.sum()

        img = np.array(image)
        blurred = cv2.filter2D(img, -1, kernel)

        return Image.fromarray(blurred)

    @staticmethod
    def frost(
        image: Image.Image,
        intensity: float = 0.5,
    ) -> Image.Image:
        """Simulate frost effect.

        Args:
            image: Input image
            intensity: Frost intensity (0-1)

        Returns:
            Frosted image
        """
        img = np.array(image).astype(float)

        # Create frost overlay
        frost = np.random.randn(*img.shape) * 30 * intensity
        frost = frost + 200

        # Blend
        result = img * (1 - intensity * 0.5) + frost * intensity * 0.5

        return Image.fromarray(np.clip(result, 0, 255).astype(np.uint8))

    @staticmethod
    def snow(
        image: Image.Image,
        intensity: float = 0.3,
    ) -> Image.Image:
        """Simulate snow effect.

        Args:
            image: Input image
            intensity: Snow intensity (0-1)

        Returns:
            Snowy image
        """
        img = np.array(image)

        # Add snow flakes
        h, w = img.shape[:2]
        num_flakes = int(h * w * intensity * 0.01)

        for _ in range(num_flakes):
            x = np.random.randint(0, w)
            y = np.random.randint(0, h)
            radius = np.random.randint(1, 3)

            # Draw white circle
            cv2.circle(img, (x, y), radius, (255, 255, 255), -1)

        # Add slight blue tint
        img = img.astype(float)
        img[:, :, 2] = np.clip(img[:, :, 2] * 1.1, 0, 255)

        return Image.fromarray(img.astype(np.uint8))

    @staticmethod
    def fog(
        image: Image.Image,
        intensity: float = 0.5,
    ) -> Image.Image:
        """Simulate fog effect.

        Args:
            image: Input image
            intensity: Fog intensity (0-1)

        Returns:
            Foggy image
        """
        img = np.array(image).astype(float)

        # Create fog layer
        fog = np.ones_like(img) * 200

        # Blend with depth-dependent alpha
        h, w = img.shape[:2]
        alpha = np.linspace(0, intensity, h).reshape(h, 1, 1)

        result = img * (1 - alpha) + fog * alpha

        return Image.fromarray(result.astype(np.uint8))

    @staticmethod
    def zoom_blur(
        image: Image.Image,
        factor: float = 1.2,
    ) -> Image.Image:
        """Apply zoom blur.

        Args:
            image: Input image
            factor: Zoom factor

        Returns:
            Blurred image
        """
        img = np.array(image)
        h, w = img.shape[:2]

        # Create zoomed versions
        zoomed = []
        for scale in np.linspace(1, factor, 5):
            new_h, new_w = int(h / scale), int(w / scale)
            resized = cv2.resize(img, (new_w, new_h))

            # Pad back to original size
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2

            padded = cv2.copyMakeBorder(
                resized,
                pad_h, h - new_h - pad_h,
                pad_w, w - new_w - pad_w,
                cv2.BORDER_CONSTANT,
                value=(0, 0, 0),
            )

            zoomed.append(padded)

        # Average
        result = np.mean(zoomed, axis=0).astype(np.uint8)

        return Image.fromarray(result)


def perturb_directory(
    input_dir: str,
    output_dir: str,
    perturbation: str,
    severity: int = 1,
) -> None:
    """Apply perturbation to all images in directory.

    Args:
        input_dir: Input image directory
        output_dir: Output directory
        perturbation: Type of perturbation
        severity: Severity level (1-5)
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir) / perturbation / f"severity_{severity}"

    output_path.mkdir(parents=True, exist_ok=True)

    # Map perturbation names to methods
    perturb_fn = {
        "blur": ImagePerturbation.gaussian_blur,
        "motion_blur": ImagePerturbation.motion_blur,
        "defocus_blur": ImagePerturbation.defocus_blur,
        "zoom_blur": ImagePerturbation.zoom_blur,
        "noise": ImagePerturbation.gaussian_noise,
        "brightness": ImagePerturbation.brightness,
        "darkness": ImagePerturbation.brightness,
        "contrast": ImagePerturbation.contrast,
        "frost": ImagePerturbation.frost,
        "snow": ImagePerturbation.snow,
        "fog": ImagePerturbation.fog,
    }

    if perturbation not in perturb_fn:
        print(f"Unknown perturbation: {perturbation}")
        print(f"Available: {list(perturb_fn.keys())}")
        return

    # Get all images
    image_files = list(input_path.glob("*.jpg")) + list(input_path.glob("*.png"))

    print(f"Found {len(image_files)} images")
    print(f"Applying {perturbation} (severity {severity})...")

    # Apply perturbation based on severity
    severity_params = {
        "blur": lambda s: {"kernel_size": 3 + s * 2},
        "motion_blur": lambda s: {"kernel_size": 3 + s * 2},
        "defocus_blur": lambda s: {"kernel_size": 3 + s * 2},
        "zoom_blur": lambda s: {"factor": 1.1 + s * 0.1},
        "noise": lambda s: {"sigma": 0.01 + s * 0.01},
        "brightness": lambda s: {"factor": 1.0 + s * 0.1},
        "darkness": lambda s: {"factor": 1.0 - s * 0.1},
        "contrast": lambda s: {"factor": 1.0 + s * 0.1},
        "frost": lambda s: {"intensity": 0.1 + s * 0.15},
        "snow": lambda s: {"intensity": 0.1 + s * 0.1},
        "fog": lambda s: {"intensity": 0.1 + s * 0.15},
    }

    params = severity_params[perturbation](severity)
    fn = perturb_fn[perturbation]

    for i, image_file in enumerate(image_files):
        # Load image
        image = Image.open(image_file).convert("RGB")

        # Apply perturbation
        perturbed = fn(image, **params)

        # Save
        output_file = output_path / image_file.name
        perturbed.save(output_file)

        if (i + 1) % 100 == 0:
            print(f"  Processed {i + 1}/{len(image_files)}")

    print(f"Saved perturbed images to {output_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, default="./data/perturbed")
    parser.add_argument(
        "--perturbation",
        type=str,
        choices=[
            "blur", "motion_blur", "defocus_blur", "zoom_blur",
            "noise", "brightness", "darkness", "contrast",
            "frost", "snow", "fog",
        ],
        required=True,
    )
    parser.add_argument("--severity", type=int, default=1, choices=[1, 2, 3, 4, 5])

    args = parser.parse_args()

    perturb_directory(
        args.input_dir,
        args.output_dir,
        args.perturbation,
        args.severity,
    )


if __name__ == "__main__":
    main()
