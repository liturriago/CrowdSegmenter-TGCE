import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import List, Dict, Any, Optional


class MetricTracker:
    """Utility class for computing segmentation metrics with multi-annotator support.

    Provides two evaluation protocols:

    - **Probabilistic** (primary): consensus mask derived from all annotators,
      metrics averaged over ``probabilistic_thresholds``. Used when no ground
      truth is available — the standard protocol for both AnnotHarmony and
      CrowdSeg in this codebase.

    - **Ground-truth** (secondary): predictions compared directly against a
      binary ground-truth mask ``[B, C, H, W]`` at a single fixed threshold.
      Used when a dataset provides expert-validated labels for final
      benchmarking or cross-dataset comparison.

    Both protocols share the same ``calculate_metrics`` core and return
    identical dict structures, so reporting methods work for either.
    """

    def __init__(self, config) -> None:
        self.num_classes              = config.num_classes
        self.num_annotators           = config.num_annotators
        self.ignored_value            = config.ignored_value
        self.threshold                = config.threshold
        self.probabilistic_thresholds = config.probabilistic_thresholds
        self.smooth                   = config.smooth

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

        masks_reshaped = masks.view(
            b, self.num_annotators, self.num_classes, *masks.shape[2:]
        )

        valid_mask  = (masks_reshaped != self.ignored_value).float()
        valid_count = valid_mask.sum(dim=1)
        masks_sum   = (masks_reshaped * valid_mask).sum(dim=1)

        return torch.where(
            valid_count > 0,
            masks_sum / valid_count,
            torch.zeros_like(masks_sum),
        )

    # ------------------------------------------------------------------ #
    #  Core metric computation                                             #
    # ------------------------------------------------------------------ #

    def calculate_metrics(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
        threshold: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Computes segmentation metrics at a single binarisation threshold.

        Both ``y_pred`` and ``y_true`` are binarised at ``threshold``.
        Sentinel pixels in ``y_true`` are excluded from all calculations.

        Args:
            y_pred (torch.Tensor): Model predictions of shape
                ``[B, num_classes, H, W]``.
            y_true (torch.Tensor): Reference mask of shape
                ``[B, num_classes, H, W]``. May be a soft consensus mask
                (probabilistic protocol) or a binary ground-truth mask
                (GT protocol).
            threshold (float | None): Binarisation cut-off. Defaults to
                ``self.threshold``.

        Returns:
            Dict[str, Any]: Dictionary with keys:

                - ``"avg"`` — scalar averages over batch and classes.
                - ``"per_class"`` — per-class averages over the batch
                  (1-D tensors of length ``num_classes``).
        """
        threshold = threshold if threshold is not None else self.threshold

        y_pred = y_pred.contiguous().float()
        y_true = y_true.contiguous().float()

        mask       = (y_true != self.ignored_value).float()
        y_true_bin = (y_true > threshold).float()
        y_pred_bin = (y_pred > threshold).float()

        tp = (y_true_bin * y_pred_bin * mask).sum(dim=(2, 3))
        fp = ((1 - y_true_bin) * y_pred_bin * mask).sum(dim=(2, 3))
        fn = (y_true_bin * (1 - y_pred_bin) * mask).sum(dim=(2, 3))
        tn = ((1 - y_true_bin) * (1 - y_pred_bin) * mask).sum(dim=(2, 3))

        dice        = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        jaccard     = (tp + self.smooth) / (tp + fp + fn + self.smooth)
        sensitivity = (tp + self.smooth) / (tp + fn + self.smooth)
        specificity = (tn + self.smooth) / (tn + fp + self.smooth)

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

    # ------------------------------------------------------------------ #
    #  Probabilistic protocol (no ground truth)                           #
    # ------------------------------------------------------------------ #

    def compute_probabilistic_metrics(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> Dict[str, Any]:
        """Averages segmentation metrics across all probabilistic thresholds.

        Primary evaluation protocol for datasets without ground truth.
        Sweeping thresholds over the consensus mask captures the full range
        of possible annotation interpretations, producing a score that does
        not depend on a single arbitrary cut-off.

        Args:
            y_pred (torch.Tensor): Model predictions ``[B, num_classes, H, W]``.
            y_true (torch.Tensor): Consensus probability mask
                ``[B, num_classes, H, W]`` from
                :meth:`compute_probability_mask`.

        Returns:
            Dict[str, Any]: Same structure as :meth:`calculate_metrics`,
                values averaged over all thresholds.
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

    def evaluation(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: str,
    ) -> Dict[str, Any]:
        """Runs inference and computes probabilistic metrics over a full dataset.

        Expects batches of the form ``(images, masks, anns_ids)`` where
        ``masks`` has shape ``[B, num_annotators * num_classes, H, W]``.

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
        all_seg_preds:  List[torch.Tensor] = []
        all_prob_masks: List[torch.Tensor] = []

        with torch.no_grad():
            for batch in loader:
                images   = batch[0].to(device)
                masks    = batch[1].to(device)
                anns_ids = batch[2].to(device)

                output   = model(images, anns_ids)
                seg_pred = output[0] if isinstance(output, tuple) else output

                all_seg_preds.append(seg_pred.cpu())
                all_prob_masks.append(self.compute_probability_mask(masks).cpu())

        return self.compute_probabilistic_metrics(
            torch.cat(all_seg_preds,  dim=0),
            torch.cat(all_prob_masks, dim=0),
        )

    # ------------------------------------------------------------------ #
    #  Ground-truth protocol                                               #
    # ------------------------------------------------------------------ #

    def compute_gt_metrics(
        self,
        y_pred: torch.Tensor,
        y_true: torch.Tensor,
    ) -> Dict[str, Any]:
        """Computes metrics against a binary ground-truth mask at a fixed threshold.

        Secondary evaluation protocol used when expert-validated ground-truth
        labels are available (e.g. a held-out benchmark split). Unlike the
        probabilistic protocol, no threshold sweep is performed — the single
        ``self.threshold`` is used, reflecting the hard binary nature of the
        ground-truth reference.

        Args:
            y_pred (torch.Tensor): Model predictions ``[B, num_classes, H, W]``.
            y_true (torch.Tensor): Binary ground-truth masks
                ``[B, num_classes, H, W]`` with values in ``{0, 1}``.

        Returns:
            Dict[str, Any]: Same structure as :meth:`calculate_metrics`.
        """
        return self.calculate_metrics(y_pred, y_true, self.threshold)

    def evaluation_gt(
        self,
        model: nn.Module,
        loader: DataLoader,
        device: str,
    ) -> Dict[str, Any]:
        """Runs inference and computes ground-truth metrics over a full dataset.

        Expects batches of the form ``(images, masks, anns_ids, ground_truth)``
        where ``ground_truth`` has shape ``[B, num_classes, H, W]``.
        ``DataConfig.load_ground_truth`` must be ``True`` for the loader to
        include this item.

        Args:
            model (nn.Module): The segmentation model to evaluate.
            loader (DataLoader): DataLoader yielding
                ``(images, masks, anns_ids, ground_truth)`` batches.
            device (str): Device to run inference on.

        Returns:
            Dict[str, Any]: Ground-truth evaluation results with keys
                ``"avg"`` and ``"per_class"``.

        Raises:
            ValueError: If batches do not contain a fourth item
                (ground truth), indicating ``load_ground_truth=False``.
        """
        model.eval()
        all_seg_preds: List[torch.Tensor] = []
        all_gt_masks:  List[torch.Tensor] = []

        with torch.no_grad():
            for batch in loader:
                if len(batch) < 4:
                    raise ValueError(
                        "Batch does not contain ground truth (batch length "
                        f"{len(batch)} < 4). Set load_ground_truth=True in "
                        "DataConfig."
                    )

                images      = batch[0].to(device)
                anns_ids    = batch[2].to(device)
                ground_truth = batch[3].to(device)   # [B, C, H, W]

                output   = model(images, anns_ids)
                seg_pred = output[0] if isinstance(output, tuple) else output

                all_seg_preds.append(seg_pred.cpu())
                all_gt_masks.append(ground_truth.cpu())

        return self.compute_gt_metrics(
            torch.cat(all_seg_preds, dim=0),
            torch.cat(all_gt_masks,  dim=0),
        )

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

        Compatible with the output of both :meth:`compute_probabilistic_metrics`
        and :meth:`compute_gt_metrics`.

        Args:
            prefix (str): Label for the evaluation phase
                (e.g. ``"Val"``, ``"Test (GT)"``).
            metrics (Dict[str, Any]): Output of any compute method.
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

        Compatible with the output of both :meth:`compute_probabilistic_metrics`
        and :meth:`compute_gt_metrics`.

        Args:
            prefix (str): Title for the report section.
            metrics (Dict[str, Any]): Output of any compute method.
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