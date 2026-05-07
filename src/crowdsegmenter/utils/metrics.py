import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Any


class MetricTracker:
    """Utility class for computing segmentation metrics with multi-annotator support.

    All evaluation is performed using a probabilistic consensus mask derived
    from all available annotators, making the protocol identical for both
    AnnotHarmony and CrowdSeg models and suitable for datasets without
    ground truth.

    The consensus mask is computed by averaging valid annotations per class
    across annotators (ignoring sentinel pixels), producing a soft reference
    in ``[0, 1]`` that captures inter-annotator uncertainty. Metrics are then
    averaged over ``probabilistic_thresholds`` to produce threshold-independent
    scores.
    """

    def __init__(self, config) -> None:
        self.num_classes            = config.num_classes
        self.num_annotators         = config.num_annotators
        self.ignored_value          = config.ignored_value
        self.threshold              = config.threshold
        self.probabilistic_thresholds = config.probabilistic_thresholds
        self.smooth                 = config.smooth

    # ------------------------------------------------------------------ #
    #  Consensus mask                                                      #
    # ------------------------------------------------------------------ #

    def compute_probability_mask(self, masks: torch.Tensor) -> torch.Tensor:
        """Builds a soft consensus mask by averaging valid annotations per class.

        Sentinel pixels (equal to ``ignored_value``) are excluded from the
        average so that missing annotations do not bias the result.

        Args:
            masks (torch.Tensor): Raw annotation masks of shape
                ``[B, num_annotators * num_classes, H, W]``.

        Returns:
            torch.Tensor: Consensus probability mask of shape
                ``[B, num_classes, H, W]`` with values in ``[0, 1]``.

        Raises:
            ValueError: If the channel dimension does not equal
                ``num_annotators * num_classes``.
        """
        expected = self.num_annotators * self.num_classes
        if masks.shape[1] != expected:
            raise ValueError(
                f"Expected masks with {expected} channels "
                f"(num_annotators={self.num_annotators} × "
                f"num_classes={self.num_classes}), "
                f"but received shape {tuple(masks.shape)}."
            )

        b = masks.shape[0]

        # Reshape to [B, num_annotators, num_classes, H, W]
        masks_reshaped = masks.view(
            b, self.num_annotators, self.num_classes, *masks.shape[2:]
        )

        # Validity mask: 1 where annotation exists, 0 at sentinel pixels
        valid_mask  = (masks_reshaped != self.ignored_value).float()
        valid_count = valid_mask.sum(dim=1)                    # [B, C, H, W]
        masks_sum   = (masks_reshaped * valid_mask).sum(dim=1) # [B, C, H, W]

        # Average over annotators; fall back to 0 where no valid annotation exists
        probability_mask = torch.where(
            valid_count > 0,
            masks_sum / valid_count,
            torch.zeros_like(masks_sum),
        )

        return probability_mask

    # ------------------------------------------------------------------ #
    #  Metrics                                                             #
    # ------------------------------------------------------------------ #

    def calculate_metrics(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        threshold: float = None,
    ) -> Dict[str, Any]:
        """Computes segmentation metrics at a single binarisation threshold.

        Both ``y_pred`` and ``y_true`` are binarised at ``threshold``.
        Sentinel pixels in ``y_true`` are excluded from all calculations.

        Args:
            y_pred (torch.Tensor): Model predictions of shape
                ``[B, num_classes, H, W]``.
            y_true (torch.Tensor): Consensus probability mask of shape
                ``[B, num_classes, H, W]``.
            threshold (float): Binarisation cut-off. Defaults to
                ``self.threshold``.

        Returns:
            Dict[str, Any]: Dictionary with keys:

                - ``"avg"`` — scalar averages over batch and classes.
                - ``"per_class"`` — per-class averages over the batch
                  (as 1-D tensors of length ``num_classes``).
        """
        threshold = threshold if threshold is not None else self.threshold

        y_pred = y_pred.contiguous().float()
        y_true = y_true.contiguous().float()

        # Exclude sentinel pixels
        mask = (y_true != self.ignored_value).float()

        # Binarise
        y_true_bin = (y_true > threshold).float()
        y_pred_bin = (y_pred > threshold).float()

        # Confusion-matrix components — [B, C]
        tp = (y_true_bin * y_pred_bin * mask).sum(dim=(2, 3))
        fp = ((1 - y_true_bin) * y_pred_bin * mask).sum(dim=(2, 3))
        fn = (y_true_bin * (1 - y_pred_bin) * mask).sum(dim=(2, 3))
        tn = ((1 - y_true_bin) * (1 - y_pred_bin) * mask).sum(dim=(2, 3))

        # Per-sample, per-class metrics — [B, C]
        dice        = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        jaccard     = (tp + self.smooth) / (tp + fp + fn + self.smooth)
        sensitivity = (tp + self.smooth) / (tp + fn + self.smooth)
        specificity = (tn + self.smooth) / (tn + fp + self.smooth)

        # Replace NaNs with 0
        _nan_to_zero = lambda t: torch.where(
            torch.isnan(t), torch.zeros_like(t), t
        )
        dice, jaccard, sensitivity, specificity = map(
            _nan_to_zero, (dice, jaccard, sensitivity, specificity)
        )

        return {
            "avg": {
                "dice":        dice.mean().item(),
                "jaccard":     jaccard.mean().item(),
                "sensitivity": sensitivity.mean().item(),
                "specificity": specificity.mean().item(),
            },
            "per_class": {
                "dice":        dice.mean(dim=0),
                "jaccard":     jaccard.mean(dim=0),
                "sensitivity": sensitivity.mean(dim=0),
                "specificity": specificity.mean(dim=0),
            },
        }

    def compute_probabilistic_metrics(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> Dict[str, Any]:
        """Averages segmentation metrics across all probabilistic thresholds.

        This is the primary evaluation method for both AnnotHarmony and
        CrowdSeg. Sweeping over thresholds captures the full range of
        possible consensus interpretations, producing a metric that does
        not depend on a single arbitrary cut-off.

        Args:
            y_pred (torch.Tensor): Model predictions of shape
                ``[B, num_classes, H, W]``.
            y_true (torch.Tensor): Consensus probability mask of shape
                ``[B, num_classes, H, W]`` from
                :meth:`compute_probability_mask`.

        Returns:
            Dict[str, Any]: Same structure as :meth:`calculate_metrics`,
                with values averaged over all thresholds.
        """
        num_thresholds = len(self.probabilistic_thresholds)
        num_classes    = y_pred.shape[1]

        avg_sums = {
            "dice": 0.0, "jaccard": 0.0,
            "sensitivity": 0.0, "specificity": 0.0,
        }
        per_class_sums = {
            key: torch.zeros(num_classes, device=y_pred.device)
            for key in avg_sums
        }

        for threshold in self.probabilistic_thresholds:
            metrics = self.calculate_metrics(y_pred, y_true, threshold)
            for key in avg_sums:
                avg_sums[key]       += metrics["avg"][key]
                per_class_sums[key] += metrics["per_class"][key]

        return {
            "avg":       {k: v / num_thresholds for k, v in avg_sums.items()},
            "per_class": {k: v / num_thresholds for k, v in per_class_sums.items()},
        }

    # ------------------------------------------------------------------ #
    #  Evaluation loop                                                     #
    # ------------------------------------------------------------------ #

    def evaluation(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: str,
    ) -> Dict[str, Any]:
        """Runs inference and computes probabilistic metrics over a full dataset.

        Expects batches of the form ``(images, masks, anns_ids)`` where
        ``masks`` has shape ``[B, num_annotators * num_classes, H, W]``.
        This layout is shared by both the AnnotHarmony and CrowdSeg
        dataloaders after unification.

        Args:
            model (nn.Module): The segmentation model to evaluate.
            loader (DataLoader): DataLoader yielding
                ``(images, masks, anns_ids)`` batches.
            device (str): Device to run inference on
                (e.g. ``'cuda'`` or ``'cpu'``).

        Returns:
            Dict[str, Any]: Probabilistic evaluation results with keys
                ``"avg"`` and ``"per_class"``.
        """
        model.eval()
        all_seg_preds: List[torch.Tensor] = []
        all_prob_masks: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in loader:
                images   = batch[0].to(device)
                masks    = batch[1].to(device)  # [B, A*C, H, W]
                anns_ids = batch[2].to(device)

                seg_pred, _   = model(images, anns_ids)

                prob_mask = self.compute_probability_mask(masks)

                all_seg_preds.append(seg_pred.cpu())
                all_prob_masks.append(prob_mask.cpu())

        seg_preds_cat  = torch.cat(all_seg_preds,  dim=0)
        prob_masks_cat = torch.cat(all_prob_masks, dim=0)

        return self.compute_probabilistic_metrics(seg_preds_cat, prob_masks_cat)

    # ------------------------------------------------------------------ #
    #  Reporting                                                           #
    # ------------------------------------------------------------------ #

    @staticmethod
    def print_summary(
        prefix: str,
        metrics: Dict[str, Any],
        class_names: List[str],
    ) -> None:
        """Prints a concise per-class metric summary for training logs.

        Args:
            prefix (str): Label for the evaluation phase
                (e.g. ``"Val"``, ``"Test"``).
            metrics (Dict[str, Any]): Output of
                :meth:`compute_probabilistic_metrics`.
            class_names (List[str]): Human-readable names for each class.
        """
        avg = metrics["avg"]
        print(
            f"\n[{prefix}] Dice: {avg['dice']:.4f} | "
            f"Jaccard: {avg['jaccard']:.4f} | "
            f"Sens.: {avg['sensitivity']:.4f} | "
            f"Spec.: {avg['specificity']:.4f}"
        )
        print(
            f"  > {'Class':<15} | {'Dice':<8} | {'Jaccard':<8} | "
            f"{'Sens.':<8} | {'Spec.':<8}"
        )
        print(f"  {'-' * 57}")

        pc = metrics["per_class"]
        for i, name in enumerate(class_names):
            print(
                f"    {name:<15} | {pc['dice'][i]:.4f}   | "
                f"{pc['jaccard'][i]:.4f}   | "
                f"{pc['sensitivity'][i]:.4f}   | "
                f"{pc['specificity'][i]:.4f}"
            )

    @staticmethod
    def print_full_report(
        prefix: str,
        metrics: Dict[str, Any],
        class_names: List[str],
    ) -> None:
        """Prints a detailed segmentation report for scientific evaluation.

        Args:
            prefix (str): Title for the report section.
            metrics (Dict[str, Any]): Output of
                :meth:`compute_probabilistic_metrics`.
            class_names (List[str]): Human-readable names for each class.
        """
        print(f"\n{' REPORT: ' + prefix + ' ':=^85}")
        header = (
            f"{'Class':<20} | {'Dice':<8} | {'Jaccard':<8} | "
            f"{'Sens.':<8} | {'Spec.':<8}"
        )
        print(header)
        print("-" * len(header))

        pc = metrics["per_class"]
        for i, name in enumerate(class_names):
            print(
                f"{name:<20} | {pc['dice'][i]:<8.4f} | "
                f"{pc['jaccard'][i]:<8.4f} | "
                f"{pc['sensitivity'][i]:<8.4f} | "
                f"{pc['specificity'][i]:<8.4f}"
            )

        print("-" * len(header))

        avg = metrics["avg"]
        print(
            f"{'Macro Avg':<20} | {avg['dice']:<8.4f} | "
            f"{avg['jaccard']:<8.4f} | "
            f"{avg['sensitivity']:<8.4f} | "
            f"{avg['specificity']:<8.4f}"
        )
        print("=" * len(header))