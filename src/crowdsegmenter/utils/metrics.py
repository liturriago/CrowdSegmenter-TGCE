import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Tuple, Any

class MetricTracker:
    """Utility class for computing segmentation metrics with multi-annotator support.

    Provides methods for probability mask computation, threshold-based metric
    calculation, and probabilistic evaluation across multiple thresholds.
    """

    def __init__(self, config):
        self.num_classes = config.num_classes
        self.num_annotators = config.num_annotators
        self.ignored_value = config.ignored_value
        self.threshold = config.threshold
        self.probabilistic_thresholds = config.probabilistic_thresholds
        self.smooth = config.smooth

    def compute_probability_mask(self, masks: torch.Tensor) -> torch.Tensor:
        """Computes a probability mask by averaging valid annotations per class.

        Ignores entries equal to ``ignored_value`` when computing the mean,
        so that missing or invalid annotations do not bias the result.

        Args:
            masks (torch.Tensor): Raw annotation masks of shape
                ``[B, num_annotators * num_classes, H, W]``.

        Returns:
            torch.Tensor: Probability mask of shape ``[B, num_classes, H, W]``
                with values in ``[0, 1]``.
        """
        batch_size = masks.shape[0]

        # Reshape to [B, num_annotators, num_classes, H, W]
        masks_reshaped = masks.view(
            batch_size, self.num_annotators, self.num_classes, *masks.shape[2:]
        )

        # Build a float validity mask (1 where annotation exists, 0 otherwise)
        valid_mask = (masks_reshaped != self.ignored_value).float()

        # Count valid annotations and sum them per class
        valid_count = valid_mask.sum(dim=1)                        # [B, num_classes, H, W]
        masks_sum = (masks_reshaped * valid_mask).sum(dim=1)       # [B, num_classes, H, W]

        # Average over valid annotations; fall back to 0 where none exist
        probability_mask = torch.where(
            valid_count > 0,
            masks_sum / valid_count,
            torch.zeros_like(masks_sum),
        )

        return probability_mask

    def calculate_metrics(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> Dict[str, Any]:
        """Computes segmentation metrics at a single decision threshold.

        Binarises both ``y_pred`` and ``y_true`` at ``threshold``, masks out
        pixels equal to ``ignored_value``, then returns Dice, Jaccard,
        Sensitivity, and Specificity — both averaged over the batch and
        broken down per class.

        Args:
            y_pred (torch.Tensor): Model predictions of shape
                ``[B, num_classes, H, W]``.
            y_true (torch.Tensor): Ground-truth masks of shape
                ``[B, num_classes, H, W]``.

        Returns:
            Dict[str, Any]: Dictionary with keys ``"avg"`` (scalar averages)
                and ``"per_class"`` (per-class averages across the batch).
        """
        y_pred = y_pred.contiguous().float()
        y_true = y_true.contiguous().float()

        # Build validity mask before binarisation
        ignore_tensor = torch.tensor(self.ignored_value, device=y_true.device)
        mask = (y_true != ignore_tensor).float()

        # Binarise predictions and ground truth
        y_true_bin = (y_true > self.threshold).float()
        y_pred_bin = (y_pred > self.threshold).float()

        # Confusion-matrix components — summed over spatial dims, shape [B, C]
        tp = (y_true_bin * y_pred_bin * mask).sum(dim=(2, 3))
        fp = ((1 - y_true_bin) * y_pred_bin * mask).sum(dim=(2, 3))
        fn = (y_true_bin * (1 - y_pred_bin) * mask).sum(dim=(2, 3))
        tn = ((1 - y_true_bin) * (1 - y_pred_bin) * mask).sum(dim=(2, 3))

        # Per-sample, per-class metrics
        dice        = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        jaccard     = (tp + self.smooth) / (tp + fp + fn + self.smooth)
        sensitivity = (tp + self.smooth) / (tp + fn + self.smooth)
        specificity = (tn + self.smooth) / (tn + fp + self.smooth)

        # Replace any NaNs with 0
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

        Iterates over ``self.probabilistic_thresholds``, calls
        :meth:`calculate_metrics` at each cut-off, then returns the mean
        result — a standard approach for threshold-independent evaluation.

        Args:
            y_pred (torch.Tensor): Model predictions of shape
                ``[B, num_classes, H, W]``.
            y_true (torch.Tensor): Ground-truth masks of shape
                ``[B, num_classes, H, W]``.

        Returns:
            Dict[str, Any]: Dictionary with keys ``"avg"`` (scalar averages)
                and ``"per_class"`` (per-class averages), each accumulated
                over all thresholds.
        """
        num_thresholds = len(self.probabilistic_thresholds)
        num_classes = y_pred.shape[1]

        # Scalar accumulators
        avg_sums = {"dice": 0.0, "jaccard": 0.0, "sensitivity": 0.0, "specificity": 0.0}

        # Per-class tensor accumulators
        per_class_sums = {
            key: torch.zeros(num_classes, device=y_pred.device)
            for key in avg_sums
        }

        for threshold in self.probabilistic_thresholds:
            metrics = self.calculate_metrics(
                y_pred, y_true, threshold, self.ignored_value, self.smooth
            )
            for key in avg_sums:
                avg_sums[key]        += metrics["avg"][key]
                per_class_sums[key]  += metrics["per_class"][key]

        return {
            "avg": {k: v / num_thresholds for k, v in avg_sums.items()},
            "per_class": {k: v / num_thresholds for k, v in per_class_sums.items()},
        }
    
    def evaluation(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: str,
    ) -> Dict[str, Any]:

        model.eval()
        all_seg_preds = []
        all_ref_masks = []

        with torch.no_grad():
            for batch in loader:
                images = batch[0].to(device)
                masks = batch[1].to(device)
                multihot = batch[2].to(device)

                seg_pred, _  = model(images, multihot)

                ref_mask = self.compute_probability_mask(masks)

                all_seg_preds.append(seg_pred.cpu())
                all_ref_masks.append(ref_mask.cpu())

        seg_preds_cat = torch.cat(all_seg_preds, dim=0)
        ref_masks_cat = torch.cat(all_ref_masks, dim=0)

        final_metrics = self.calculate_metrics(seg_preds_cat, ref_masks_cat)

        return final_metrics

    @staticmethod
    def print_summary(
        prefix: str,
        metrics: Dict[str, Any],
        class_names: List[str],
    ) -> None:
        """Prints a concise per-class metric summary for training logs.

        Args:
            prefix (str): Label for the evaluation phase (e.g., ``"Val"``).
            metrics (Dict[str, Any]): Output of :meth:`calculate_metrics` or
                :meth:`compute_probabilistic_metrics`.
            class_names (List[str]): Human-readable names for each class.
        """
        avg = metrics["avg"]
        print(f"\n[{prefix}] Dice: {avg['dice']:.4f} | "
              f"Jaccard: {avg['jaccard']:.4f} | "
              f"Sens.: {avg['sensitivity']:.4f} | "
              f"Spec.: {avg['specificity']:.4f}")
        print(f"  > {'Class':<15} | {'Dice':<8} | {'Jaccard':<8} | "
              f"{'Sens.':<8} | {'Spec.':<8}")
        print(f"  {'-' * 57}")

        pc = metrics["per_class"]
        for i, name in enumerate(class_names):
            print(f"    {name:<15} | {pc['dice'][i]:.4f}   | "
                  f"{pc['jaccard'][i]:.4f}   | "
                  f"{pc['sensitivity'][i]:.4f}   | "
                  f"{pc['specificity'][i]:.4f}")

    @staticmethod
    def print_full_report(
        prefix: str,
        metrics: Dict[str, Any],
        class_names: List[str],
    ) -> None:
        """Prints a detailed segmentation report for scientific evaluation.

        Args:
            prefix (str): Title for the report section.
            metrics (Dict[str, Any]): Output of :meth:`calculate_metrics` or
                :meth:`compute_probabilistic_metrics`.
            class_names (List[str]): Human-readable names for each class.
        """
        print(f"\n{' REPORT: ' + prefix + ' ':=^85}")
        header = (f"{'Class':<20} | {'Dice':<8} | {'Jaccard':<8} | "
                  f"{'Sens.':<8} | {'Spec.':<8}")
        print(header)
        print("-" * len(header))

        pc = metrics["per_class"]
        for i, name in enumerate(class_names):
            print(f"{name:<20} | {pc['dice'][i]:<8.4f} | "
                  f"{pc['jaccard'][i]:<8.4f} | "
                  f"{pc['sensitivity'][i]:<8.4f} | "
                  f"{pc['specificity'][i]:<8.4f}")

        print("-" * len(header))

        avg = metrics["avg"]
        print(f"{'Macro Avg':<20} | {avg['dice']:<8.4f} | "
              f"{avg['jaccard']:<8.4f} | "
              f"{avg['sensitivity']:<8.4f} | "
              f"{avg['specificity']:<8.4f}")
        print("=" * len(header))