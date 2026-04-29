import torch
import torch.nn as nn
from torch import Tensor


class TGCE_SSPS(nn.Module):
    """
    Truncated Generalized Cross-Entropy for Semantice Segmentation Pixel 
    Selection (TGCE-SSPS) loss.

    This loss is designed for semantic segmentation with multiple annotators, 
    handling noisy annotations and uncertain labels via reliability weights.

    Args:
        annotators (int): Number of annotators (R).
        classes (int): Number of classes (K).
        ignore_value (float): Value in annotations to ignore.
        q (float): Truncation parameter for generalized cross-entropy.
        lambda_factor (float): Scaling factor for the reliability term.
        smooth (float): Small constant to prevent division by zero.
    """

    def __init__(
        self,
        annotators: int = 3,
        classes: int = 2,
        ignore_value: float = 0.6,
        q: float = 0.48029,
        lambda_factor: float = 1.0,
        smooth: float = 1e-7,
    ) -> None:
        super().__init__()
        self.K = classes
        self.R = annotators
        self.ignore_value = ignore_value
        self.q = q
        self.smooth = smooth
        self.lambda_factor = lambda_factor

    def forward(self, y_pred: Tensor, annotations: Tensor) -> Tensor:
        """
        Forward pass to compute TGCE-SS loss.

        Args:
            y_pred (Tensor): Network predictions with shape 
                (N, K + R, H, W), where first K channels are class probs
                and next R channels are reliability scores per annotator.
            annotations (Tensor): Annotator masks with shape 
                (N, K * R, H, W).

        Returns:
            Tensor: Scalar loss value.
        """
        device = y_pred[0].device
        epsilon = 1e-8

        # Split predictions: classes, reliability
        y_pred_classes = y_pred[0]
        Lambda_r = y_pred[1]

        # Reshape annotations into (N, K, R, H, W)
        N, _, H, W = y_pred_classes.shape
        annotations = annotations.reshape(N, self.K, self.R, H, W).contiguous()

        # Mask invalid values (ignore_value)
        valid_mask = torch.all(annotations != self.ignore_value, dim=1).float()
        annotations = torch.where(
            annotations != self.ignore_value,
            annotations,
            torch.zeros_like(annotations),
        )

        # Clamp predictions for numerical stability
        y_pred_classes = torch.clamp(y_pred_classes, epsilon, 1.0 - epsilon)

        # Expand predictions for annotators -> (N, K, R, H, W)
        y_pred_expanded = y_pred_classes.unsqueeze(2).repeat(1, 1, self.R, 1, 1)

        # --- Compute terms ---
        # Class term (per annotator reliability)
        term_r = torch.mean(
            annotations
            * (1 - torch.pow(y_pred_expanded, self.q))
            / (self.q + epsilon + self.smooth),
            dim=1,
        )  # (N, R, H, W)

        # Reliability regularization term
        uniform_probs = (1 / self.K + self.smooth) * torch.ones(
            (N, self.R, H, W), device=device
        )
        term_c = (1 - Lambda_r) * (1 - torch.pow(uniform_probs, self.q)) / (
            self.q + epsilon + self.smooth
        )

        # Combine terms and apply valid pixel mask
        combined_terms = Lambda_r * term_r + term_c
        masked_terms = combined_terms * valid_mask

        # Normalize by valid pixels
        valid_pixel_count = torch.maximum(
            torch.sum(valid_mask), torch.tensor(1.0, device=device)
        )
        tgce_loss = torch.sum(masked_terms) / valid_pixel_count

        return tgce_loss
