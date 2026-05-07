import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple

class NoisyLabelLoss(nn.Module):
    """
    Noisy Label Loss for crowdsourcing methods, using the Cross-Entropy (CE) configuration.

    This loss models annotator reliability using confusion matrices.

    Args:
        ignored_value (float): Value in annotations to ignore.
        min_trace (bool): Whether to add (True) or subtract (False) the trace regularization.
        alpha (float): Scaling factor for the trace regularization.
        eps (float): Small constant to prevent log(0).
    """

    def __init__(
        self,
        ignored_value: float = 0.6,
        min_trace: bool = False,
        alpha: float = 0.1,
        smooth: float = 1e-8,
    ) -> None:
        super().__init__()
        self.ignore_value = ignore_value
        self.min_trace = min_trace
        self.alpha = alpha
        self.smooth = smooth
        
        # CE loss criterion
        self.nll_loss = nn.NLLLoss(reduction='mean', ignore_index=-100)

    def forward(
        self, 
        pred: Tensor, 
        cms: Tensor, 
        labels: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass to compute Noisy Label Loss.

        Args:
            pred (Tensor): Network predictions with shape (B, C, H, W) after Softmax.
            cms (Tensor): Confusion matrices for each pixel with shape (B, C**2, H, W) after Softplus.
            labels (Tensor): Ground truth labels with shape (B, C, H, W) one-hot.

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Total loss, CE loss, and trace regularization term.
        """
        b, c, h, w = pred.size()

        # 1. Build valid-pixel mask — works correctly with float sentinel
        valid_mask = (labels != self.ignore_value).all(dim=1)  # [B, H, W]

        # 2. One-hot → integer indices for NLLLoss
        labels_int = labels.argmax(dim=1).long()               # [B, H, W]
        labels_int[~valid_mask] = -100                         # NLLLoss native ignore

        # 3. Reshape pred → [B*H*W, C, 1]
        pred_flat = (pred.view(b, c, h * w)
                        .permute(0, 2, 1)
                        .contiguous()
                        .view(b * h * w, c, 1))

        # 4. Reshape and row-normalize cms → [B*H*W, C, C]
        cm = (cms.view(b, c ** 2, h * w)
                .permute(0, 2, 1)
                .contiguous()
                .view(b * h * w, c, c))
        cm = cm / (cm.sum(dim=1, keepdim=True) + self.smooth)

        # 5. Apply confusion matrix → [B, C, H, W]
        pred_noisy = torch.bmm(cm, pred_flat)
        pred_noisy = (pred_noisy.view(b, h * w, c)
                                .permute(0, 2, 1)
                                .contiguous()
                                .view(b, c, h, w))

        # 6. NLLLoss with native ignore
        loss_ce = self.nll_loss(torch.log(pred_noisy + self.smooth), labels_int)

        # 7. Trace regularization
        regularisation = (
            self.alpha
            * torch.trace(cm.sum(dim=0).T).sum()
            / (b * h * w)
        )

        loss = loss_ce + regularisation if self.min_trace else loss_ce - regularisation
        return loss, loss_ce, regularisation
