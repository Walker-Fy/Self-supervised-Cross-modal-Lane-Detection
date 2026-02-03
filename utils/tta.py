"""Test-Time Augmentation (TTA) utilities for lane detection.

This module provides TTA wrappers for improving inference results
through multi-scale and flip augmentation.
"""

from typing import List, Optional, Callable, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F


class TTAWrapper:
    """Test-Time Augmentation wrapper for lane detection models.

    Applies multi-scale and horizontal flip augmentation during inference,
    then aggregates predictions for more robust results.
    """

    def __init__(
        self,
        model: nn.Module,
        scales: List[float] = [0.75, 1.0, 1.25],
        flip: bool = True,
        aggregate: str = "mean",
        merge_fn: Optional[Callable] = None,
        input_size: tuple = (590, 1640),
    ):
        """Initialize TTA wrapper.

        Args:
            model: Lane detection model
            scales: List of scale factors for multi-scale testing
            flip: Whether to use horizontal flip TTA
            aggregate: Aggregation method ('mean', 'max', 'vote', or 'gmean')
            merge_fn: Optional custom merge function
            input_size: Original input size (H, W)
        """
        self.model = model
        self.scales = scales
        self.flip = flip
        self.aggregate = aggregate
        self.merge_fn = merge_fn
        self.input_size = input_size

        # Set model to eval mode
        self.model.eval()

    def predict(
        self,
        image: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Predict with TTA.

        Args:
            image: Input image (B, 3, H, W)
            **kwargs: Additional arguments passed to model

        Returns:
            Aggregated prediction (B, 1, H, W)
        """
        predictions = []

        with torch.no_grad():
            for scale in self.scales:
                # Scale image
                scaled = self._scale_image(image, scale)

                # Forward pass
                if hasattr(self.model, 'forward'):
                    # Try to get prediction directly
                    try:
                        output = self._get_model_output(scaled, **kwargs)
                        # Scale back to original size
                        output = self._scale_output(output, scale)
                        predictions.append(output)
                    except Exception:
                        # Fallback to standard forward
                        output = self.model(scaled, **kwargs)
                        if isinstance(output, dict):
                            output = output.get("pred_mask", output.get("mask", list(output.values())[0]))
                        output = self._scale_output(output, scale)
                        predictions.append(output)

                # Flip TTA
                if self.flip:
                    flipped = torch.flip(scaled, dims=[-1])

                    try:
                        output = self._get_model_output(flipped, **kwargs)
                        output = self._scale_output(output, scale)
                        # Flip back
                        output = torch.flip(output, dims=[-1])
                        predictions.append(output)
                    except Exception:
                        output = self.model(flipped, **kwargs)
                        if isinstance(output, dict):
                            output = output.get("pred_mask", output.get("mask", list(output.values())[0]))
                        output = self._scale_output(output, scale)
                        output = torch.flip(output, dims=[-1])
                        predictions.append(output)

        # Aggregate predictions
        if self.merge_fn is not None:
            return self.merge_fn(predictions)

        return self._aggregate(predictions)

    def _get_model_output(
        self,
        image: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        """Get model output, handling different return types.

        Args:
            image: Input image
            **kwargs: Additional arguments

        Returns:
            Model output tensor
        """
        output = self.model(image, **kwargs)

        # Handle different output formats
        if isinstance(output, dict):
            # Try common keys
            for key in ["pred_mask", "mask", "output", "logits", "prediction"]:
                if key in output:
                    return output[key]
            # Last resort: first tensor value
            for v in output.values():
                if isinstance(v, torch.Tensor):
                    return v
        elif isinstance(output, (list, tuple)):
            return output[0]

        return output

    def _scale_image(
        self,
        image: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Scale image by factor.

        Args:
            image: Input image (B, C, H, W)
            scale: Scale factor

        Returns:
            Scaled image
        """
        if scale == 1.0:
            return image

        _, _, h, w = image.shape
        new_h = int(h * scale)
        new_w = int(w * scale)

        return F.interpolate(
            image,
            size=(new_h, new_w),
            mode='bilinear',
            align_corners=False,
        )

    def _scale_output(
        self,
        output: torch.Tensor,
        scale: float,
    ) -> torch.Tensor:
        """Scale output back to original size.

        Args:
            output: Model output (B, C, H', W')
            scale: Scale factor that was applied

        Returns:
            Scaled output (B, C, H, W)
        """
        if scale == 1.0:
            return output

        _, _, h_out, w_out = output.shape
        target_h = int(h_out / scale)
        target_w = int(w_out / scale)

        return F.interpolate(
            output,
            size=(target_h, target_w),
            mode='bilinear',
            align_corners=False,
        )

    def _aggregate(
        self,
        predictions: List[torch.Tensor],
    ) -> torch.Tensor:
        """Aggregate multiple predictions.

        Args:
            predictions: List of prediction tensors

        Returns:
            Aggregated prediction
        """
        if not predictions:
            raise ValueError("No predictions to aggregate")

        if len(predictions) == 1:
            return predictions[0]

        stacked = torch.stack(predictions, dim=0)

        if self.aggregate == "mean":
            return stacked.mean(dim=0)
        elif self.aggregate == "max":
            return stacked.max(dim=0)[0]
        elif self.aggregate == "gmean":
            # Geometric mean (exp(mean(log))))
            return torch.exp(torch.log(stacked.abs() + 1e-7).mean(dim=0)) * torch.sign(stacked).mean(dim=0)
        elif self.aggregate == "vote":
            # Majority voting (binarize first)
            binary = (stacked > 0).float()
            return binary.mean(dim=0)
        else:
            return stacked.mean(dim=0)


class MultiScaleTTA:
    """Multi-scale TTA for lane detection.

    Specialized TTA that handles different scales more efficiently
    by pre-computing feature pyramids when possible.
    """

    def __init__(
        self,
        model: nn.Module,
        scales: List[float] = [0.8, 1.0, 1.2],
        flip: bool = True,
        input_size: tuple = (590, 1640),
    ):
        """Initialize multi-scale TTA.

        Args:
            model: Lane detection model
            scales: Scale factors for multi-scale testing
            flip: Whether to use flip augmentation
            input_size: Input size (H, W)
        """
        self.model = model
        self.scales = scales
        self.flip = flip
        self.input_size = input_size
        self.model.eval()

    @torch.no_grad()
    def predict(
        self,
        image: torch.Tensor,
        text: Optional[str] = None,
    ) -> torch.Tensor:
        """Predict with multi-scale TTA.

        Args:
            image: Input image (B, 3, H, W)
            text: Optional text caption

        Returns:
            Aggregated lane mask (B, 1, H, W)
        """
        predictions = []

        for scale in self.scales:
            # Resize image
            _, _, h, w = image.shape
            scaled_h, scaled_w = int(h * scale), int(w * scale)
            scaled = F.interpolate(image, size=(scaled_h, scaled_w), mode='bilinear', align_corners=False)

            # Predict
            if text is not None and hasattr(self.model, 'predict'):
                pred = self.model.predict(scaled, text)
            else:
                pred = self._forward_model(scaled, text)

            # Resize back
            pred = F.interpolate(pred, size=(h, w), mode='bilinear', align_corners=False)
            predictions.append(pred)

            # Flip prediction
            if self.flip:
                flipped = torch.flip(scaled, dims=[-1])
                pred_flip = self._forward_model(flipped, text)
                pred_flip = F.interpolate(pred_flip, size=(h, w), mode='bilinear', align_corners=False)
                pred_flip = torch.flip(pred_flip, dims=[-1])
                predictions.append(pred_flip)

        # Average predictions
        return torch.stack(predictions).mean(dim=0)

    def _forward_model(
        self,
        image: torch.Tensor,
        text: Optional[str] = None,
    ) -> torch.Tensor:
        """Forward pass through model.

        Args:
            image: Input image
            text: Optional text

        Returns:
            Model prediction
        """
        if hasattr(self.model, 'predict'):
            default_text = text or "a road with lanes"
            return self.model.predict(image, default_text)
        else:
            output = self.model(image)
            if isinstance(output, dict):
                return output.get("pred_mask", output.get("mask", list(output.values())[0]))
            return output


class LaneDetectionTTA:
    """Specialized TTA for lane detection with coordinate output.

    This class applies TTA and returns lane coordinates in CULane format.
    """

    def __init__(
        self,
        model: nn.Module,
        scales: List[float] = [0.75, 1.0, 1.25],
        flip: bool = True,
        n_lanes: int = 4,
        n_points: int = 56,
        confidence_threshold: float = 0.5,
    ):
        """Initialize lane detection TTA.

        Args:
            model: Lane detection model
            scales: Scale factors for TTA
            flip: Whether to use flip TTA
            n_lanes: Maximum number of lanes
            n_points: Points per lane
            confidence_threshold: Mask threshold
        """
        self.model = model
        self.tta = MultiScaleTTA(model, scales=scales, flip=flip)

        # Import lane extractor
        from .postprocess import LaneExtractor
        self.extractor = LaneExtractor(
            n_lanes=n_lanes,
            n_points=n_points,
            confidence_threshold=confidence_threshold,
        )

    @torch.no_grad()
    def predict_lanes(
        self,
        image: torch.Tensor,
        text: Optional[str] = None,
        return_mask: bool = False,
    ) -> Dict[str, Any]:
        """Predict lanes with TTA.

        Args:
            image: Input image (B, 3, H, W)
            text: Optional text caption
            return_mask: Whether to also return the mask

        Returns:
            Dictionary with 'lanes' and optionally 'mask'
        """
        # Get TTA prediction
        mask = self.tta.predict(image, text)

        # Apply sigmoid if needed
        if mask.min() < 0 and mask.max() > 0:
            mask = torch.sigmoid(mask)

        # Convert to numpy
        mask_np = mask.squeeze().cpu().numpy()

        # Extract lanes
        lanes = self.extractor.process_mask(mask_np)

        result = {"lanes": lanes}

        if return_mask:
            result["mask"] = mask_np

        return result

    def predict_batch(
        self,
        images: List[torch.Tensor],
        texts: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """Predict lanes for a batch of images.

        Args:
            images: List of input images
            texts: Optional list of text captions

        Returns:
            List of prediction dictionaries
        """
        if texts is None:
            texts = ["a road with lanes"] * len(images)

        results = []

        for img, txt in zip(images, texts):
            result = self.predict_lanes(img, txt)
            results.append(result)

        return results


def apply_tta_to_model(
    model: nn.Module,
    scales: List[float] = [0.75, 1.0, 1.25],
    flip: bool = True,
    input_size: tuple = (590, 1640),
) -> TTAWrapper:
    """Convenience function to wrap a model with TTA.

    Args:
        model: Model to wrap
        scales: Scale factors
        flip: Whether to use flip
        input_size: Input size

    Returns:
        TTA-wrapped model
    """
    return TTAWrapper(
        model=model,
        scales=scales,
        flip=flip,
        input_size=input_size,
    )
