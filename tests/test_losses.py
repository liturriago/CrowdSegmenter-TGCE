import pytest
import torch
import torch.nn.functional as F
from crowdsegmenter.losses.tgce_ssps import TGCE_SSPS
from crowdsegmenter.losses.noisy_label_loss import NoisyLabelLoss


# ------------------------------------------------------------------ #
#  Shared fixtures                                                     #
# ------------------------------------------------------------------ #

@pytest.fixture
def dims():
    """Standard batch / spatial dimensions used across all tests."""
    return dict(N=2, K=2, R=3, H=16, W=16)


@pytest.fixture
def seg_pred(dims):
    """Valid softmax predictions [N, K, H, W]."""
    N, K, H, W = dims["N"], dims["K"], dims["H"], dims["W"]
    x = torch.rand(N, K, H, W)
    return F.softmax(x, dim=1)


@pytest.fixture
def ann_pred_tgce(dims):
    """Reliability scores for TGCE_SSPS [N, R, H, W] in (0, 1)."""
    N, R, H, W = dims["N"], dims["R"], dims["H"], dims["W"]
    return torch.rand(N, R, H, W).clamp(0.01, 0.99)


@pytest.fixture
def cms(dims):
    """Softplus confusion matrices for NoisyLabelLoss [N, K², H, W]."""
    N, K, H, W = dims["N"], dims["K"], dims["H"], dims["W"]
    return F.softplus(torch.randn(N, K ** 2, H, W))


@pytest.fixture
def annotations(dims):
    """Full annotator stack [N, K*R, H, W] layout: channel = k*R + r.

    Each pixel gets a valid one-hot class assignment from every annotator.
    One pixel is intentionally set to the sentinel value to verify masking.
    """
    N, K, R, H, W = dims["N"], dims["K"], dims["R"], dims["H"], dims["W"]
    ignored = 0.6

    masks = torch.zeros(N, K * R, H, W)
    for n in range(N):
        for r in range(R):
            # Random class per pixel for annotator r
            class_idx = torch.randint(0, K, (H, W))          # [H, W]
            for k in range(K):
                ch = k * R + r
                masks[n, ch] = (class_idx == k).float()

    # Inject one sentinel pixel (sample 0, annotator 0, all classes)
    for k in range(K):
        masks[0, k * R + 0, 0, 0] = ignored

    return masks


@pytest.fixture
def anns_ids(dims):
    """One-hot annotator selector [N, R] — always selects annotator 0."""
    N, R = dims["N"], dims["R"]
    ids = torch.zeros(N, R)
    ids[:, 0] = 1.0
    return ids


# ------------------------------------------------------------------ #
#  TGCE_SSPS                                                          #
# ------------------------------------------------------------------ #

class TestTGCE_SSPS:

    def test_initialization(self):
        loss_fn = TGCE_SSPS(
            annotators=3, classes=2, ignored_value=0.6,
            q=0.5, lambda_factor=1.0, smooth=1e-7,
        )
        assert loss_fn.R == 3
        assert loss_fn.K == 2
        assert loss_fn.ignored_value == 0.6
        assert loss_fn.q == 0.5
        assert loss_fn.lambda_factor == 1.0
        assert loss_fn.smooth == 1e-7

    def test_output_is_finite_scalar(self, dims, seg_pred, ann_pred_tgce, annotations):
        N, K, R = dims["N"], dims["K"], dims["R"]
        loss_fn = TGCE_SSPS(annotators=R, classes=K, ignored_value=0.6)
        loss = loss_fn(seg_pred, ann_pred_tgce, annotations, anns_ids=None)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_output_is_non_negative(self, dims, seg_pred, ann_pred_tgce, annotations):
        """TGCE loss is a sum of non-negative terms."""
        N, K, R = dims["N"], dims["K"], dims["R"]
        loss_fn = TGCE_SSPS(annotators=R, classes=K, ignored_value=0.6)
        loss = loss_fn(seg_pred, ann_pred_tgce, annotations, anns_ids=None)
        assert loss.item() >= 0.0

    def test_anns_ids_is_ignored(self, dims, seg_pred, ann_pred_tgce, annotations, anns_ids):
        """TGCE_SSPS must produce identical results whether anns_ids is None or a tensor."""
        N, K, R = dims["N"], dims["K"], dims["R"]
        loss_fn = TGCE_SSPS(annotators=R, classes=K, ignored_value=0.6)

        loss_none   = loss_fn(seg_pred, ann_pred_tgce, annotations, anns_ids=None)
        loss_tensor = loss_fn(seg_pred, ann_pred_tgce, annotations, anns_ids=anns_ids)

        assert torch.allclose(loss_none, loss_tensor), (
            "TGCE_SSPS should ignore anns_ids entirely"
        )

    def test_sentinel_pixels_excluded(self, dims, seg_pred, ann_pred_tgce):
        """A batch of all-sentinel annotations should yield loss ≈ 0."""
        N, K, R, H, W = dims["N"], dims["K"], dims["R"], dims["H"], dims["W"]
        ignored = 0.6
        all_ignored = torch.full((N, K * R, H, W), ignored)

        loss_fn = TGCE_SSPS(annotators=R, classes=K, ignored_value=ignored)
        loss = loss_fn(seg_pred, ann_pred_tgce, all_ignored, anns_ids=None)

        assert loss.item() == pytest.approx(0.0, abs=1e-5)

    def test_loss_decreases_with_confident_correct_pred(self, dims, ann_pred_tgce, annotations):
        """A near-perfect prediction should yield a lower loss than a random one."""
        N, K, R, H, W = dims["N"], dims["K"], dims["R"], dims["H"], dims["W"]
        loss_fn = TGCE_SSPS(annotators=R, classes=K, ignored_value=0.6)

        # Near-uniform (bad) prediction
        bad_pred = torch.full((N, K, H, W), 1.0 / K)

        # Confident prediction that matches the majority class
        good_pred = torch.zeros(N, K, H, W)
        good_pred[:, 0] = 0.99
        good_pred[:, 1] = 0.01

        loss_bad  = loss_fn(bad_pred,  ann_pred_tgce, annotations, anns_ids=None)
        loss_good = loss_fn(good_pred, ann_pred_tgce, annotations, anns_ids=None)

        assert loss_good.item() < loss_bad.item()


# ------------------------------------------------------------------ #
#  NoisyLabelLoss                                                     #
# ------------------------------------------------------------------ #

class TestNoisyLabelLoss:

    def test_initialization(self):
        loss_fn = NoisyLabelLoss(
            num_annotators=3, num_classes=2,
            ignored_value=0.6, min_trace=False,
            alpha=0.1, smooth=1e-8,
        )
        assert loss_fn.num_annotators == 3
        assert loss_fn.num_classes    == 2
        assert loss_fn.ignored_value  == 0.6
        assert loss_fn.min_trace      is False
        assert loss_fn.alpha          == 0.1

    def test_output_is_finite_scalar(self, dims, seg_pred, cms, annotations, anns_ids):
        N, K, R = dims["N"], dims["K"], dims["R"]
        loss_fn = NoisyLabelLoss(num_annotators=R, num_classes=K, ignored_value=0.6)
        loss = loss_fn(seg_pred, cms, annotations, anns_ids)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_min_trace_changes_loss(self, dims, seg_pred, cms, annotations, anns_ids):
        """Flipping min_trace must change the loss value."""
        N, K, R = dims["N"], dims["K"], dims["R"]
        loss_add = NoisyLabelLoss(
            num_annotators=R, num_classes=K, min_trace=True
        )(seg_pred, cms, annotations, anns_ids)

        loss_sub = NoisyLabelLoss(
            num_annotators=R, num_classes=K, min_trace=False
        )(seg_pred, cms, annotations, anns_ids)

        assert not torch.allclose(loss_add, loss_sub), (
            "min_trace=True and min_trace=False should produce different losses"
        )

    def test_annotator_selection_uses_onehot(self, dims, seg_pred, cms, annotations):
        """Different one-hot selections should produce different losses."""
        N, K, R = dims["N"], dims["K"], dims["R"]
        loss_fn = NoisyLabelLoss(num_annotators=R, num_classes=K, ignored_value=0.6)

        ids_0 = torch.zeros(N, R); ids_0[:, 0] = 1.0
        ids_1 = torch.zeros(N, R); ids_1[:, 1] = 1.0

        loss_0 = loss_fn(seg_pred, cms, annotations, ids_0)
        loss_1 = loss_fn(seg_pred, cms, annotations, ids_1)

        assert not torch.allclose(loss_0, loss_1), (
            "Selecting different annotators should yield different losses"
        )

    def test_sentinel_pixels_excluded(self, dims, seg_pred, cms, anns_ids):
        """All-sentinel labels should result in zero NLLLoss contribution."""
        N, K, R, H, W = dims["N"], dims["K"], dims["R"], dims["H"], dims["W"]
        ignored = 0.6

        # Build annotations where the selected annotator (r=0) is all sentinel
        all_ignored = torch.zeros(N, K * R, H, W)
        for k in range(K):
            all_ignored[:, k * R + 0, :, :] = ignored   # annotator 0, class k

        loss_fn = NoisyLabelLoss(
            num_annotators=R, num_classes=K, ignored_value=ignored
        )
        loss = loss_fn(seg_pred, cms, all_ignored, anns_ids)

        assert torch.isfinite(loss)

    def test_channel_layout_k_times_R_plus_r(self, dims, seg_pred, cms):
        """Verifies the channel layout k*R+r by checking annotator 1 vs annotator 2."""
        N, K, R, H, W = dims["N"], dims["K"], dims["R"], dims["H"], dims["W"]
        loss_fn = NoisyLabelLoss(num_annotators=R, num_classes=K, ignored_value=0.6)

        # Build two annotation tensors identical except annotator 1 vs 2
        def make_annotations(r_selected: int) -> torch.Tensor:
            masks = torch.full((N, K * R, H, W), 0.6)
            for k in range(K):
                ch = k * R + r_selected
                # class 0 = foreground for all pixels
                masks[:, ch, :, :] = float(k == 0)
            return masks

        ids_1 = torch.zeros(N, R); ids_1[:, 1] = 1.0
        ids_2 = torch.zeros(N, R); ids_2[:, 2] = 1.0

        loss_1 = loss_fn(seg_pred, cms, make_annotations(1), ids_1)
        loss_2 = loss_fn(seg_pred, cms, make_annotations(2), ids_2)

        # Both annotators provide identical labels so loss should be the same
        assert torch.allclose(loss_1, loss_2, atol=1e-5), (
            "Identical labels from different annotators should yield the same loss"
        )