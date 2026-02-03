"""Tests for model forward pass."""

import pytest
import torch
import numpy as np


@pytest.fixture
def sample_batch():
    """Create a sample batch for testing."""
    return {
        "view1": torch.randn(4, 3, 320, 800),
        "view2": torch.randn(4, 3, 320, 800),
        "mask": torch.randint(0, 2, (4, 1, 320, 800)).float(),
        "texts": ["a road with 2 lanes"] * 4,
    }


@pytest.fixture
def device():
    """Get device for testing."""
    return torch.device("cpu")  # Always use CPU for tests


class TestVisualEncoder:
    """Tests for visual encoder."""

    def test_visual_encoder_init(self, device):
        """Test visual encoder initialization."""
        from models.visual_encoder import VisualEncoder

        encoder = VisualEncoder(
            model_name="ViT-B-16",
            pretrained="openai",
            freeze_layers=6,
        )

        assert encoder is not None
        assert encoder.output_dim > 0

    @pytest.mark.skipif(
        not torch.cuda.is_available() or
        True,  # Skip due to large download
        reason="Requires CLIP model download"
    )
    def test_visual_encoder_forward(self, device):
        """Test visual encoder forward pass."""
        from models.visual_encoder import VisualEncoder

        encoder = VisualEncoder(
            model_name="ViT-B-16",
            pretrained="openai",
            freeze_layers=6,
        ).to(device)

        x = torch.randn(2, 3, 320, 800).to(device)

        with torch.no_grad():
            out = encoder(x)

        assert out.shape[0] == 2
        assert out.dim() == 2

    @pytest.mark.skipif(
        not torch.cuda.is_available() or
        True,  # Skip due to large download
        reason="Requires CLIP model download"
    )
    def test_visual_encoder_feature_map(self, device):
        """Test visual encoder feature map extraction."""
        from models.visual_encoder import VisualEncoder

        encoder = VisualEncoder(
            model_name="ViT-B-16",
            pretrained="openai",
            freeze_layers=6,
        ).to(device)

        x = torch.randn(2, 3, 320, 800).to(device)

        with torch.no_grad():
            feat_map = encoder.get_feature_map(x)

        assert feat_map.shape[0] == 2
        assert feat_map.dim() == 4  # (B, C, H, W)


class TestLanguageEncoder:
    """Tests for language encoder."""

    def test_language_encoder_init(self):
        """Test language encoder initialization."""
        try:
            from models.language_encoder import LanguageEncoder

            encoder = LanguageEncoder(
                model_name="ViT-B-16",
                pretrained="openai",
                frozen=True,
            )

            assert encoder is not None
            assert encoder.output_dim > 0
        except ImportError:
            pytest.skip("open_clip_torch not available")

    def test_language_encoder_tokenize(self):
        """Test text tokenization."""
        try:
            from models.language_encoder import LanguageEncoder

            encoder = LanguageEncoder(
                model_name="ViT-B-16",
                pretrained="openai",
            )

            tokens = encoder.tokenize(["a road with 2 lanes"])

            assert tokens.dim() == 2
        except ImportError:
            pytest.skip("open_clip_torch not available")


class TestProjectionHead:
    """Tests for projection head."""

    def test_projection_head(self):
        """Test projection head forward pass."""
        from models.projection import ProjectionHead

        proj = ProjectionHead(in_dim=512, out_dim=256)

        x = torch.randn(4, 512)
        out = proj(x)

        assert out.shape == (4, 256)

    def test_projection_normalize(self):
        """Test projection normalization."""
        from models.projection import ProjectionHead

        proj = ProjectionHead(in_dim=512, out_dim=256)

        x = torch.randn(4, 512)
        out = proj(x, normalize=True)

        # Check L2 norm is approximately 1
        norms = torch.norm(out, dim=-1)
        assert torch.allclose(norms, torch.ones_like(norms), atol=1e-5)


class TestLaneDecoder:
    """Tests for lane decoder."""

    def test_lane_decoder_forward(self):
        """Test lane decoder forward pass."""
        from models.lane_decoder import LaneDecoder

        decoder = LaneDecoder(in_dim=512, input_size=(800, 320))

        # Test with CLS token
        x = torch.randn(4, 512)
        out = decoder(x)

        assert out.shape == (4, 1, 800, 320)

    def test_lane_decoder_with_feature_map(self):
        """Test lane decoder with feature map input."""
        from models.lane_decoder import LaneDecoder

        decoder = LaneDecoder(in_dim=512, input_size=(800, 320))

        # Test with feature map
        x = torch.randn(4, 512, 20, 40)
        out = decoder(x)

        assert out.shape == (4, 1, 800, 320)

    def test_unet_decoder(self):
        """Test UNet decoder."""
        from models.lane_decoder import UNetDecoder

        decoder = UNetDecoder(in_dim=512, input_size=(800, 320))

        x = torch.randn(4, 512)
        out = decoder(x)

        assert out.shape == (4, 1, 800, 320)

    def test_fpn_decoder(self):
        """Test FPN decoder."""
        from models.lane_decoder import FPNDecoder

        decoder = FPNDecoder(in_dim=512, input_size=(800, 320))

        x = torch.randn(4, 512, 20, 40)
        out = decoder(x)

        assert out.shape == (4, 1, 800, 320)


class TestFusionHead:
    """Tests for fusion head."""

    def test_fusion_concat(self):
        """Test concat fusion."""
        from models.fusion_head import FusionHead

        fusion = FusionHead(
            visual_dim=512,
            language_dim=512,
            out_dim=512,
            fusion_type="concat",
        )

        visual = torch.randn(4, 512)
        language = torch.randn(4, 512)

        out = fusion(visual, language)

        assert out.shape == (4, 512)

    def test_fusion_add(self):
        """Test add fusion."""
        from models.fusion_head import FusionHead

        fusion = FusionHead(
            visual_dim=512,
            language_dim=512,
            out_dim=512,
            fusion_type="add",
        )

        visual = torch.randn(4, 512)
        language = torch.randn(4, 512)

        out = fusion(visual, language)

        assert out.shape == (4, 512)

    def test_fusion_attention(self):
        """Test attention fusion."""
        from models.fusion_head import FusionHead

        fusion = FusionHead(
            visual_dim=512,
            language_dim=512,
            out_dim=512,
            fusion_type="attention",
        )

        visual = torch.randn(4, 512)
        language = torch.randn(4, 512)

        out = fusion(visual, language)

        assert out.shape == (4, 512)


class TestConsistencyLoss:
    """Tests for consistency loss functions."""

    def test_info_nce_loss(self):
        """Test InfoNCE loss."""
        from models.consistency_loss import InfoNCELoss

        loss_fn = InfoNCELoss(temperature=0.07)

        z_v = torch.randn(4, 512)
        z_l = torch.randn(4, 512)

        # Normalize
        z_v = torch.nn.functional.normalize(z_v, dim=-1)
        z_l = torch.nn.functional.normalize(z_l, dim=-1)

        loss = loss_fn(z_v, z_l)

        assert loss.dim() == 0  # Scalar
        assert loss.item() > 0

    def test_cosine_consistency_loss(self):
        """Test cosine consistency loss."""
        from models.consistency_loss import CosineConsistencyLoss

        loss_fn = CosineConsistencyLoss()

        z1 = torch.randn(4, 512)
        z2 = torch.randn(4, 512)

        # Normalize
        z1 = torch.nn.functional.normalize(z1, dim=-1)
        z2 = torch.nn.functional.normalize(z2, dim=-1)

        loss = loss_fn(z1, z2)

        assert loss.dim() == 0
        assert 0 <= loss.item() <= 2  # Cosine distance range


class TestDiceLoss:
    """Tests for Dice loss."""

    def test_dice_loss(self):
        """Test Dice loss computation."""
        from utils.metrics import DiceLoss

        loss_fn = DiceLoss()

        pred = torch.randn(2, 1, 100, 100)
        target = torch.randint(0, 2, (2, 1, 100, 100)).float()

        loss = loss_fn(pred, target)

        assert loss.dim() == 0
        assert 0 <= loss.item() <= 1

    def test_dice_perfect_prediction(self):
        """Test Dice loss with perfect prediction."""
        from utils.metrics import DiceLoss

        loss_fn = DiceLoss()

        pred = torch.ones(2, 1, 100, 100) * 10  # High logits -> sigmoid -> 1
        target = torch.ones(2, 1, 100, 100)

        loss = loss_fn(pred, target)

        assert loss.item() < 0.1  # Should be close to 0


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_iou_score(self):
        """Test IoU computation."""
        from utils.metrics import iou_score

        pred = torch.ones(2, 1, 100, 100)
        target = torch.ones(2, 1, 100, 100)

        iou = iou_score(pred, target)

        assert iou.item() == pytest.approx(1.0, abs=1e-3)

    def test_dice_score(self):
        """Test Dice computation."""
        from utils.metrics import dice_score

        pred = torch.ones(2, 1, 100, 100)
        target = torch.ones(2, 1, 100, 100)

        dice = dice_score(pred, target)

        assert dice.item() == pytest.approx(1.0, abs=1e-3)

    def test_pixel_accuracy(self):
        """Test pixel accuracy."""
        from utils.metrics import pixel_accuracy

        pred = torch.ones(2, 1, 100, 100)
        target = torch.ones(2, 1, 100, 100)

        acc = pixel_accuracy(pred, target)

        assert acc.item() == pytest.approx(1.0, abs=1e-3)

    def test_calculate_metrics(self):
        """Test full metrics calculation."""
        from utils.metrics import calculate_metrics

        pred = torch.ones(2, 1, 100, 100)
        target = torch.ones(2, 1, 100, 100)

        metrics = calculate_metrics(pred, target)

        assert "iou" in metrics
        assert "dice" in metrics
        assert "f1" in metrics
        assert "accuracy" in metrics

        assert metrics["iou"] == pytest.approx(1.0, abs=1e-3)


class TestMetricsTracker:
    """Tests for metrics tracker."""

    def test_metrics_tracker(self):
        """Test metrics tracking."""
        from utils.metrics import MetricsTracker

        tracker = MetricsTracker()

        tracker.update({"loss": 1.0}, batch_size=4)
        tracker.update({"loss": 2.0}, batch_size=4)

        metrics = tracker.get_metrics()

        assert metrics["loss"] == 1.5  # Average


if __name__ == "__main__":
    # Run tests on CPU only (no CLIP download)
    pytest.main([__file__, "-v", "-m", "not skip"])
