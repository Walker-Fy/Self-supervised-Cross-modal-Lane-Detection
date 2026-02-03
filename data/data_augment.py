"""Data augmentation for lane detection."""

import random
from typing import Dict, Any, Optional, Tuple, List

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import torch


class DataAugmentation:
    """Data augmentation pipeline for SCC training.

    Creates two augmented views (x, x') from the same image for
    cross-modal consistency training.
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (590, 1640),
        gaussian_blur: Tuple[int, int] = (3, 7),
        gaussian_noise: Tuple[float, float] = (0.01, 0.06),
        brightness_contrast: float = 0.25,
        horizontal_flip: float = 0.0,
        rotate: float = 0.0,
    ):
        """Initialize augmentation pipeline.

        Args:
            input_size: Target size (height, width)
            gaussian_blur: Kernel size range for Gaussian blur
            gaussian_noise: Sigma range for Gaussian noise
            brightness_contrast: Max brightness/contrast variation
            horizontal_flip: Probability of horizontal flip
            rotate: Max rotation degrees
        """
        self.input_size = input_size
        self.gaussian_blur = gaussian_blur
        self.gaussian_noise = gaussian_noise
        self.brightness_contrast = brightness_contrast
        self.horizontal_flip = horizontal_flip
        self.rotate = rotate

        # Base augmentation pipeline (common to both views)
        self.base_transform = A.Compose([
            A.Resize(height=input_size[0], width=input_size[1]),
        ], is_check_shapes=False)

        # Strong augmentation for view 1
        self.strong_transform_1 = A.Compose([
            A.GaussianBlur(blur_limit=gaussian_blur, p=0.5),
            A.GaussNoise(var_limit=(gaussian_noise[0] * 255, gaussian_noise[1] * 255), p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=brightness_contrast,
                contrast_limit=brightness_contrast,
                p=0.5
            ),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), p=0.3),
        ], is_check_shapes=False)

        # Strong augmentation for view 2 (different parameters)
        self.strong_transform_2 = A.Compose([
            A.OneOf([
                A.GaussianBlur(blur_limit=gaussian_blur, p=1.0),
                A.MotionBlur(blur_limit=gaussian_blur, p=1.0),
            ], p=0.4),
            A.GaussNoise(var_limit=(gaussian_noise[0] * 255, gaussian_noise[1] * 255), p=0.4),
            A.RandomBrightnessContrast(
                brightness_limit=brightness_contrast,
                contrast_limit=brightness_contrast,
                p=0.4
            ),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(4, 4), p=0.3),
        ], is_check_shapes=False)

        # Geometric augmentation (applied consistently to both views)
        self.geometric_transform = A.Compose([
            A.HorizontalFlip(p=horizontal_flip),
            A.Rotate(limit=rotate, border_mode=cv2.BORDER_CONSTANT, value=0, p=0.3),
        ], is_check_shapes=False)

        # Normalization for model input
        self.normalize = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], is_check_shapes=False)

        # Transform for mask only (no normalization)
        self.mask_transform = A.Compose([
            ToTensorV2(),
        ], is_check_shapes=False)

    def __call__(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Apply augmentation to create two views.

        Args:
            image: Input image (H, W, 3)
            mask: Optional ground truth mask (H, W)

        Returns:
            Dictionary with 'view1', 'view2', 'mask' (if provided)
        """
        # Apply geometric transform consistently
        if mask is not None:
            augmented = self.geometric_transform(image=image, mask=mask)
            image_geo, mask_geo = augmented["image"], augmented["mask"]
        else:
            augmented = self.geometric_transform(image=image)
            image_geo = augmented["image"]

        # Resize base
        image_resized = self.base_transform(image=image_geo)["image"]
        mask_resized = self.base_transform(image=mask_geo)["mask"] if mask is not None else None

        # Create two augmented views
        view1 = self.strong_transform_1(image=image_resized.copy())["image"]
        view2 = self.strong_transform_2(image=image_resized.copy())["image"]

        # Normalize and convert to tensor
        view1_tensor = self.normalize(image=view1)["image"]
        view2_tensor = self.normalize(image=view2)["image"]

        result = {
            "view1": view1_tensor,
            "view2": view2_tensor,
        }

        if mask is not None:
            mask_tensor = (self.mask_transform(image=mask_resized)["image"] > 0.5).float()
            result["mask"] = mask_tensor

        return result

    def preprocess_eval(
        self,
        image: np.ndarray,
        mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Preprocess for evaluation (minimal augmentation).

        Args:
            image: Input image (H, W, 3)
            mask: Optional ground truth mask (H, W)

        Returns:
            Dictionary with 'image' and 'mask' (if provided)
        """
        # Resize
        image_resized = self.base_transform(image=image)["image"]

        # Normalize
        image_tensor = self.normalize(image=image_resized)["image"]

        result = {"image": image_tensor}

        if mask is not None:
            mask_resized = self.base_transform(image=mask)["mask"]
            mask_tensor = (self.mask_transform(image=mask_resized)["image"] > 0.5).float()
            result["mask"] = mask_tensor

        return result


class RobustnessAugmentation:
    """Augmentation for robustness evaluation.

    Applies specific perturbations for testing model robustness.
    """

    def __init__(
        self,
        input_size: Tuple[int, int] = (590, 1640),
    ):
        """Initialize robustness augmentation.

        Args:
            input_size: Target size (height, width)
        """
        self.input_size = input_size

        self.base_transform = A.Compose([
            A.Resize(height=input_size[0], width=input_size[1]),
        ], is_check_shapes=False)

        self.normalize = A.Compose([
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ], is_check_shapes=False)

    def __call__(
        self,
        image: np.ndarray,
        perturbation: str = "clean",
        severity: int = 1,
    ) -> torch.Tensor:
        """Apply perturbation for robustness testing.

        Args:
            image: Input image (H, W, 3)
            perturbation: Type of perturbation ('clean', 'blur', 'noise',
                         'brightness', 'darkness', 'fog', 'frost', 'snow')
            severity: Severity level (1-5)

        Returns:
            Perturbed image tensor
        """
        # Resize first
        image_resized = self.base_transform(image=image)["image"]

        if perturbation == "clean":
            pass
        elif perturbation == "blur":
            kernel = 3 + 2 * severity
            transform = A.GaussianBlur(blur_limit=(kernel, kernel), p=1.0)
            image_resized = transform(image=image_resized)["image"]
        elif perturbation == "noise":
            sigma = 0.01 + 0.01 * severity
            transform = A.GaussNoise(var_limit=(sigma * 255, sigma * 255), p=1.0)
            image_resized = transform(image=image_resized)["image"]
        elif perturbation == "brightness":
            limit = 0.1 + 0.05 * severity
            transform = A.RandomBrightness(brightness_limit=(limit, limit), p=1.0)
            image_resized = transform(image=image_resized)["image"]
        elif perturbation == "darkness":
            limit = -0.1 - 0.05 * severity
            transform = A.RandomBrightness(brightness_limit=(limit, limit), p=1.0)
            image_resized = transform(image=image_resized)["image"]
        elif perturbation == "fog":
            transform = A.RandomFog(fog_coef_lower=0.1 + 0.03 * severity,
                                   fog_coef_upper=0.15 + 0.03 * severity, p=1.0)
            image_resized = transform(image=image_resized)["image"]
        elif perturbation == "frost":
            transform = A.RandomFrost(alpha_offset=(0.1 + 0.02 * severity,
                                                   0.15 + 0.02 * severity), p=1.0)
            image_resized = transform(image=image_resized)["image"]
        elif perturbation == "snow":
            transform = A.RandomSnow(snow_point_lower=0.1 + 0.05 * severity,
                                   snow_point_upper=0.15 + 0.05 * severity,
                                   brightness_coeff=1.5 + 0.1 * severity, p=1.0)
            image_resized = transform(image=image_resized)["image"]
        elif perturbation == "contrast":
            limit = 0.1 + 0.05 * severity
            transform = A.RandomContrast(contrast_limit=(limit, limit), p=1.0)
            image_resized = transform(image=image_resized)["image"]
        elif perturbation == "defocus_blur":
            kernel = 3 + 2 * severity
            transform = A.Blur(blur_limit=(kernel, kernel), p=1.0)
            image_resized = transform(image=image_resized)["image"]
        elif perturbation == "motion_blur":
            kernel = 3 + 2 * severity
            transform = A.MotionBlur(blur_limit=(kernel, kernel), p=1.0)
            image_resized = transform(image=image_resized)["image"]
        elif perturbation == "zoom_blur":
            transform = A.ZoomBlur(max_factor=1.0 + 0.1 * severity, p=1.0)
            image_resized = transform(image=image_resized)["image"]
        else:
            raise ValueError(f"Unknown perturbation: {perturbation}")

        return self.normalize(image=image_resized)["image"]


def get_test_augmentations(
    input_size: Tuple[int, int] = (590, 1640),
) -> List[str]:
    """Get list of available test augmentations.

    Args:
        input_size: Target size (height, width)

    Returns:
        List of perturbation names
    """
    return [
        "clean",
        "blur",
        "noise",
        "brightness",
        "darkness",
        "fog",
        "frost",
        "snow",
        "contrast",
        "defocus_blur",
        "motion_blur",
        "zoom_blur",
    ]
