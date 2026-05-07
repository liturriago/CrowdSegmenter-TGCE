import pytest
import torch
import torch.nn.functional as F
from crowdsegmenter.losses.tgce_ssps import TGCE_SSPS
from crowdsegmenter.losses.noisy_label import NoisyLabelLoss

# --- Constants for Testing ---
B, C, A, H, W = 2, 3, 2, 8, 8  # Batch, Classes, Annotators, Height, Width

@pytest.fixture
def noisy_label_inputs():
    """Generates valid inputs for NoisyLabelLoss."""
    # Predictions (Softmaxed)
    pred = F.softmax(torch.randn(B, C, H, W), dim=1)
    
    # Confusion Matrices (Softplus-ed)
    cms = F.softplus(torch.randn(B, C**2, H, W))
    
    # Annotations: layout channel k * A + r
    annotations = torch.randn(B, C * A, H, W)
    
    # Annotator IDs: One-hot [B, A]
    anns_ids = torch.zeros(B, A)
    anns_ids[0, 0] = 1 # Sample 0 uses annotator 0
    anns_ids[1, 1] = 1 # Sample 1 uses annotator 1
    
    return pred, cms, annotations, anns_ids

@pytest.fixture
def tgce_inputs():
    """Generates valid inputs for TGCE_SSPS."""
    seg_pred = F.softmax(torch.randn(B, C, H, W), dim=1)
    ann_pred = torch.sigmoid(torch.randn(B, A, H, W)) # Reliability weights [0, 1]
    annotations = torch.randn(B, C * A, H, W)
    return seg_pred, ann_pred, annotations

# --- Tests for NoisyLabelLoss ---

def test_noisy_label_forward_shape(noisy_label_inputs):
    """Checks if the loss returns a scalar tensor."""
    criterion = NoisyLabelLoss(num_annotators=A, num_classes=C)
    loss = criterion(*noisy_label_inputs)
    assert loss.dim() == 0
    assert not torch.isnan(loss)

def test_noisy_label_ignored_value(noisy_label_inputs):
    """Verifies that ignored_value pixels are excluded (loss changes if data changes outside ignore)."""
    pred, cms, annotations, anns_ids = noisy_label_inputs
    criterion = NoisyLabelLoss(num_annotators=A, num_classes=C, ignored_value=0.6)
    
    # Set a specific pixel to ignored_value across all channels for that pixel
    annotations[:, :, 0, 0] = 0.6
    loss1 = criterion(pred, cms, annotations, anns_ids)
    
    # Change the value of an ignored pixel - loss should NOT change
    annotations[:, :, 0, 0] = 0.6 # Ensure it's still ignored
    # We modify the 'pred' at the ignored location
    pred_mod = pred.clone()
    pred_mod[:, :, 0, 0] = 0.5 
    loss2 = criterion(pred_mod, cms, annotations, anns_ids)
    
    torch.testing.assert_close(loss1, loss2)

def test_noisy_label_backward(noisy_label_inputs):
    """Checks if gradients propagate to predictions and confusion matrices."""
    pred, cms, annotations, anns_ids = noisy_label_inputs
    pred.requires_grad_(True)
    cms.requires_grad_(True)
    
    criterion = NoisyLabelLoss(num_annotators=A, num_classes=C)
    loss = criterion(pred, cms, annotations, anns_ids)
    loss.backward()
    
    assert pred.grad is not None
    assert cms.grad is not None

def test_noisy_label_min_trace_toggle(noisy_label_inputs):
    """Ensures min_trace changes the loss value (regularization sign)."""
    criterion_plus = NoisyLabelLoss(num_annotators=A, num_classes=C, min_trace=True, alpha=1.0)
    criterion_minus = NoisyLabelLoss(num_annotators=A, num_classes=C, min_trace=False, alpha=1.0)
    
    loss_plus = criterion_plus(*noisy_label_inputs)
    loss_minus = criterion_minus(*noisy_label_inputs)
    
    assert loss_plus > loss_minus

# --- Tests for TGCE_SSPS ---

def test_tgce_forward_shape(tgce_inputs):
    """Checks if TGCE returns a scalar."""
    criterion = TGCE_SSPS(annotators=A, classes=C)
    loss = criterion(*tgce_inputs)
    assert loss.dim() == 0
    assert not torch.isnan(loss)

def test_tgce_ignored_value(tgce_inputs):
    """Verifies pixels with ignored_value are masked in TGCE."""
    seg_pred, ann_pred, annotations = tgce_inputs
    criterion = TGCE_SSPS(annotators=A, classes=C, ignored_value=99.0)
    
    # Set a pixel to ignore
    annotations[:, :, 0, 0] = 99.0
    loss1 = criterion(seg_pred, ann_pred, annotations)
    
    # Modify values at the ignored location
    seg_pred_mod = seg_pred.clone()
    seg_pred_mod[:, :, 0, 0] = 0.1
    loss2 = criterion(seg_pred_mod, ann_pred, annotations)
    
    torch.testing.assert_close(loss1, loss2)

def test_tgce_backward(tgce_inputs):
    """Checks gradient flow for both segmentation and reliability heads."""
    seg_pred, ann_pred, annotations = tgce_inputs
    seg_pred.requires_grad_(True)
    ann_pred.requires_grad_(True)
    
    criterion = TGCE_SSPS(annotators=A, classes=C)
    loss = criterion(seg_pred, ann_pred, annotations)
    loss.backward()
    
    assert seg_pred.grad is not None
    assert ann_pred.grad is not None

@pytest.mark.parametrize("q", [0.1, 0.5, 0.9])
def test_tgce_q_parameter(tgce_inputs, q):
    """Checks if the loss works with different truncation parameters."""
    criterion = TGCE_SSPS(annotators=A, classes=C, q=q)
    loss = criterion(*tgce_inputs)
    assert not torch.isnan(loss)

def test_tgce_reliability_weighting(tgce_inputs):
    """Ensures loss responds to changes in reliability (ann_pred)."""
    seg_pred, ann_pred, annotations = tgce_inputs
    criterion = TGCE_SSPS(annotators=A, classes=C)
    
    # High reliability
    ann_pred_high = torch.ones_like(ann_pred)
    loss_high = criterion(seg_pred, ann_pred_high, annotations)
    
    # Low reliability
    ann_pred_low = torch.zeros_like(ann_pred)
    loss_low = criterion(seg_pred, ann_pred_low, annotations)
    
    assert loss_high != loss_low