import torch
import torch.nn as nn
from torch import Tensor
from typing import Tuple


class NoisyLabelLoss(nn.Module):
    """
    Noisy Label Loss for crowdsourcing methods, using the Cross-Entropy (CE) configuration.

    This loss models annotator reliability using confusion matrices.

    Args:
        ignore_index (int): Value in annotations to ignore.
        min_trace (bool): Whether to add (True) or subtract (False) the trace regularization.
        alpha (float): Scaling factor for the trace regularization.
        eps (float): Small constant to prevent log(0).
    """

    def __init__(
        self,
        ignore_index: int = 255,
        min_trace: bool = False,
        alpha: float = 0.1,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        self.ignore_index = ignore_index
        self.min_trace = min_trace
        self.alpha = alpha
        self.eps = eps
        
        # CE loss criterion
        self.nll_loss = nn.NLLLoss(reduction='mean', ignore_index=self.ignore_index)

    def forward(
        self, 
        pred: Tensor, 
        cms: Tensor, 
        labels: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Forward pass to compute Noisy Label Loss.

        Args:
            pred (Tensor): Network predictions with shape (B, C, H, W).
            cms (Tensor): Confusion matrices for each pixel with shape (B, C**2, H, W).
            labels (Tensor): Ground truth labels with shape (B, H, W).

        Returns:
            Tuple[Tensor, Tensor, Tensor]: Total loss, CE loss, and trace regularization term.
        """
        b, c, h, w = pred.size()

        # Normalize and reshape predictions -> (B*H*W, C, 1)
        pred_norm = pred.view(b, c, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c, 1)
        
        # Reshape and normalize confusion matrices -> (B*H*W, C, C)
        cm = cms.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)
        cm = cm / cm.sum(1, keepdim=True)

        # Compute noisy predictions -> (B, C, H, W)
        pred_noisy = torch.bmm(cm, pred_norm).view(b * h * w, c)
        pred_noisy = pred_noisy.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
        
        # Calculate Cross-Entropy Loss
        loss_ce = self.nll_loss(torch.log(pred_noisy + self.eps), labels.view(b, h, w).long())

        # Regularization term (Trace)
        regularisation = torch.trace(torch.transpose(torch.sum(cm, dim=0), 0, 1)).sum() / (b * h * w)
        regularisation = self.alpha * regularisation

        # Total Loss
        if self.min_trace:
            loss = loss_ce + regularisation
        else:
            loss = loss_ce - regularisation

        return loss, loss_ce, regularisation
