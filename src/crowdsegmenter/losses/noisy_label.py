import torch
import torch.nn as nn
from torch import Tensor


class NoisyLabelLoss(nn.Module):
    """Noisy Label Loss for crowdsourcing segmentation via learned confusion matrices.

    Models annotator reliability by routing predictions through a per-pixel
    confusion matrix before computing cross-entropy. A trace regularization
    term encourages the confusion matrices to be diagonal (reliable annotator).

    The loss receives the full annotator stack ``[B, A*C, H, W]`` and selects
    the single annotator's masks internally using the one-hot vector, following
    the layout:

        channel ``k * num_annotators + r`` → annotator ``r``, class ``k``

    Args:
        num_annotators (int): Total number of annotators (A).
        num_classes (int): Number of segmentation classes (C).
        ignored_value (float): Sentinel pixel value excluded from all calculations.
        min_trace (bool): If ``True``, adds the trace term (penalises diagonal
            confusion matrices). If ``False`` (default), subtracts it (rewards
            diagonal matrices, i.e. reliable annotators).
        alpha (float): Scaling factor for the trace regularization term.
        smooth (float): Small constant added to denominators and log arguments
            to prevent numerical instability.
    """

    def __init__(
        self,
        num_annotators: int,
        num_classes: int,
        ignored_value: float = 0.6,
        min_trace: bool = False,
        alpha: float = 0.1,
        smooth: float = 1e-8,
    ) -> None:
        super().__init__()
        self.num_annotators = num_annotators
        self.num_classes    = num_classes
        self.ignored_value  = ignored_value
        self.min_trace      = min_trace
        self.alpha          = alpha
        self.smooth         = smooth

        self.nll_loss = nn.NLLLoss(reduction='mean', ignore_index=-100)

    def _select_annotator_masks(
        self,
        annotations: Tensor,   # [B, A*C, H, W]
        anns_ids: Tensor,      # [B, A]  one-hot
    ) -> Tensor:
        """Extracts the mask channels that belong to the selected annotator.

        Given the channel layout ``k * A + r`` (annotator ``r``, class ``k``),
        for each sample the selected annotator index ``r`` is read from the
        one-hot vector and the corresponding ``C`` channels are gathered.

        Args:
            annotations (Tensor): Full annotator stack ``[B, A*C, H, W]``.
            anns_ids (Tensor): One-hot annotator selector ``[B, A]``.

        Returns:
            Tensor: Selected masks of shape ``[B, C, H, W]``.
        """
        b, _, h, w = annotations.shape
        r = anns_ids.argmax(dim=1)                # [B]  selected annotator index

        # Build channel indices for the selected annotator:
        # class k lives at channel k * num_annotators + r_i for sample i
        # indices shape: [B, C]
        k       = torch.arange(self.num_classes, device=annotations.device)
        indices = (k.unsqueeze(0) * self.num_annotators
                   + r.unsqueeze(1))              # [B, C]

        # Expand indices to gather spatial dims → [B, C, H, W]
        indices = indices.unsqueeze(-1).unsqueeze(-1).expand(b, self.num_classes, h, w)

        return annotations.gather(dim=1, index=indices)   # [B, C, H, W]

    def forward(
        self,
        pred: Tensor,           # [B, C, H, W]   after Softmax
        cms: Tensor,            # [B, C², H, W]  after Softplus
        annotations: Tensor,    # [B, A*C, H, W] full annotator stack
        anns_ids: Tensor,       # [B, A]         one-hot
    ) -> Tensor:
        """Computes the Noisy Label Loss for the selected annotator.

        Args:
            pred (Tensor): Class probability predictions ``[B, C, H, W]``
                after Softmax.
            cms (Tensor): Per-pixel confusion matrix logits ``[B, C², H, W]``
                after Softplus.
            annotations (Tensor): Full annotator mask stack ``[B, A*C, H, W]``
                using the layout ``channel = k * A + r``.
            anns_ids (Tensor): One-hot annotator selector ``[B, A]``.

        Returns:
            Tensor: Scalar loss value
                ``loss_ce ± alpha * trace_regularisation``.
        """
        b, c, h, w = pred.size()

        # 1. Select this sample's annotator masks → [B, C, H, W]
        labels = self._select_annotator_masks(annotations, anns_ids)

        # 2. Build valid-pixel mask from the float sentinel
        valid_mask = (labels != self.ignored_value).all(dim=1)  # [B, H, W]

        # 3. One-hot → integer class indices for NLLLoss
        labels_int              = labels.argmax(dim=1).long()   # [B, H, W]
        labels_int[~valid_mask] = -100                          # native NLLLoss ignore

        # 4. Flatten pred → [B*H*W, C, 1]
        pred_flat = (pred.view(b, c, h * w)
                         .permute(0, 2, 1)
                         .contiguous()
                         .view(b * h * w, c, 1))

        # 5. Reshape and row-normalise cms → [B*H*W, C, C]
        cm = (cms.view(b, c ** 2, h * w)
                 .permute(0, 2, 1)
                 .contiguous()
                 .view(b * h * w, c, c))
        cm = cm / (cm.sum(dim=1, keepdim=True) + self.smooth)

        # 6. Apply confusion matrix: corrupt clean pred → noisy pred [B, C, H, W]
        pred_noisy = torch.bmm(cm, pred_flat)
        pred_noisy = (pred_noisy.view(b, h * w, c)
                                 .permute(0, 2, 1)
                                 .contiguous()
                                 .view(b, c, h, w))

        # 7. Cross-entropy on noisy predictions
        loss_ce = self.nll_loss(torch.log(pred_noisy + self.smooth), labels_int)

        # 8. Trace regularization — rewards diagonal (reliable) confusion matrices
        regularisation = (
            self.alpha
            * torch.trace(cm.sum(dim=0).T).sum()
            / (b * h * w)
        )

        loss = loss_ce + regularisation if self.min_trace else loss_ce - regularisation
        return loss