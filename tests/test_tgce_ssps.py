import pytest
import torch
from crowdsegmenter.losses.tgce_ssps import TGCE_SSPS

def test_tgce_ssps_initialization():
    loss_fn = TGCE_SSPS(annotators=3, classes=2, ignore_value=0.6, q=0.5, lambda_factor=1.0)
    assert loss_fn.R == 3
    assert loss_fn.K == 2
    assert loss_fn.ignore_value == 0.6
    assert loss_fn.q == 0.5
    assert loss_fn.lambda_factor == 1.0

def test_tgce_ssps_forward():
    # Setup dummy data
    N, K, R, H, W = 2, 2, 3, 16, 16
    
    # seg_pred: (N, K, H, W) probabilities
    seg_pred = torch.rand((N, K, H, W))
    seg_pred = seg_pred / seg_pred.sum(dim=1, keepdim=True) # Normalize classes
    
    # ann_pred: (N, R, H, W) reliability scores [0, 1]
    ann_pred = torch.rand((N, R, H, W))
    
    # annotations: (N, K * R, H, W) multi-hot or probabilities.
    # To keep it simple, we simulate one-hot encoded classes for each annotator.
    # Each annotator provides a K-dimensional vector for each pixel.
    annotations = torch.zeros((N, K, R, H, W))
    # Randomly assign a class for each pixel from each annotator
    for n in range(N):
        for r in range(R):
            for h in range(H):
                for w in range(W):
                    c = torch.randint(0, K, (1,)).item()
                    annotations[n, c, r, h, w] = 1.0
                    
    # Simulate ignore value
    annotations[0, :, 0, 0, 0] = 0.6
    
    # Reshape annotations to match expected input format
    annotations_input = annotations.reshape(N, K * R, H, W)
    
    loss_fn = TGCE_SSPS(annotators=R, classes=K, ignore_value=0.6)
    loss = loss_fn(seg_pred, ann_pred, annotations_input)
    
    assert isinstance(loss, torch.Tensor)
    assert loss.dim() == 0  # It should be a scalar
    assert torch.isfinite(loss)
    assert loss.item() >= 0 # Loss should be non-negative
