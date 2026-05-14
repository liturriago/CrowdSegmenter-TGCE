import time
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from typing import Optional, Tuple, Dict, Any, List
from tqdm import tqdm

from crowdsegmenter.utils.metrics import MetricTracker
from crowdsegmenter.utils.formatter import format_time
from crowdsegmenter.utils.train_phases import get_training_phase


class Trainer:
    """Trainer for multi-annotator segmentation models (AnnotHarmony and CrowdSeg).

    Implements a unified training and evaluation pipeline for both architectures.
    Both models share the same data contract — batches always carry the full
    annotator stack ``[B, A*C, H, W]`` — so evaluation is always performed
    against a probabilistic consensus mask, making results directly comparable
    without ground truth.

    When ``config.load_ground_truth`` is ``True``, an additional ground-truth
    evaluation pass is performed after every validation step using
    :meth:`MetricTracker.evaluation_gt`. Both metric sets are stored in
    ``history`` and printed side by side for direct comparison.

    Training uses a curriculum strategy (gradual layer unfreezing) driven by
    ``config.epochs_phases``. Mixed-precision forward/backward passes are
    applied throughout via AMP.

    Attributes:
        model (nn.Module): The segmentation model (AnnotHarmony or CrowdSeg).
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function (TGCE_SSPS or NoisyLabelLoss).
        device (torch.device): Device to run training on (cpu / cuda).
        config (Any): Training configuration. Must expose ``epochs``,
            ``threshold``, ``ignored_value``, ``smooth``,
            ``probabilistic_thresholds``, ``load_ground_truth``,
            and optionally ``epochs_phases``.
        scaler (GradScaler): AMP gradient scaler.
        tracker (MetricTracker): Metric computation and reporting utility.
        best_val_dice (float): Highest validation Dice achieved so far
            (probabilistic protocol).
        best_model_weights (Optional[Dict[str, torch.Tensor]]): State dict
            of the best checkpoint.
        history (Dict[str, List[float]]): Per-epoch training history.
            Always contains ``train_loss``, ``train_dice``, ``val_dice``.
            Contains ``val_dice_gt`` when ``load_ground_truth`` is ``True``.
        optimizer (Optional[optim.Optimizer]): Set by ``get_training_phase``.
        scheduler (Optional[LRScheduler]): Set by ``get_training_phase``.
    """

    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        device: torch.device,
        config: Any,
    ) -> None:
        """Initialises the Trainer.

        Args:
            model (nn.Module): The model to train.
            train_loader (DataLoader): Source of training batches
                ``(images, masks, anns_ids)`` or
                ``(images, masks, anns_ids, ground_truth)`` when
                ``load_ground_truth=True``.
            val_loader (DataLoader): Source of validation batches,
                same format as ``train_loader``.
            criterion (nn.Module): Loss criterion. Must accept
                ``(seg_pred, ann_pred, masks, anns_ids)``.
            device (torch.device): Device to run training on.
            config (Any): Training configuration object.
        """
        self.model        = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = criterion
        self.device       = device
        self.config       = config
        self.scaler       = GradScaler()
        self.tracker      = MetricTracker(config)

        # Whether to run the GT evaluation pass each epoch
        self._use_gt: bool = getattr(config, "load_ground_truth", False)

        # State tracking
        self.best_val_dice: float = 0.0
        self.best_model_weights: Optional[Dict[str, torch.Tensor]] = (
            copy.deepcopy(model.state_dict())
        )

        # Metrics history — gt track added conditionally
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_dice": [],
            "val_dice":   [],
        }
        if self._use_gt:
            self.history["val_dice_gt"] = []

        # Initialised inside fit() via get_training_phase
        self.optimizer: Optional[optim.Optimizer]                = None
        self.scheduler: Optional[optim.lr_scheduler.LRScheduler] = None

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def fit(self) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """Executes the full curriculum training and validation loop.

        Phase boundaries are read from ``config.epochs_phases``. A valid
        value is a list of exactly four integers. Passing an empty list
        disables the curriculum and trains the full model end-to-end from
        epoch 0. Any other value triggers a warning and falls back to
        ``[0, 5, 10, 15]``.

        When ``config.load_ground_truth`` is ``True``, a second evaluation
        pass using :meth:`evaluate_gt` is run after every probabilistic
        validation step. The GT Dice is stored in ``history["val_dice_gt"]``
        but does **not** influence checkpoint saving, which is driven
        exclusively by the probabilistic Dice. This preserves comparability
        on datasets that lack ground truth.

        Returns:
            Tuple[nn.Module, Dict[str, List[float]]]:
                The model restored to its best validation weights and the
                full training history.
        """
        epochs        = self.config.epochs
        epochs_phases = self._resolve_phases()
        min_trace_flag = 1

        if not epochs_phases:
            self.optimizer, self.scheduler = get_training_phase(
                self.model, self.config, phase=None
            )

        total_train_start = time.time()

        for epoch in range(epochs):

            if epochs_phases and epoch in epochs_phases:
                phase_idx = epochs_phases.index(epoch) + 1
                self.optimizer, self.scheduler = get_training_phase(
                    self.model, self.config, phase=phase_idx
                )

                if (self.config.model_name == "CrowdSeg") and (self.config.min_trace) and min_trace_flag:

                    self.criterion.min_trace = True
                    min_trace_flag = 1
                    print("\n [Min-Trace] Enabled for CrowdSeg model\n")

            print(f"\nEpoch {epoch + 1}/{epochs}")
            print("-" * 40)

            # 1. Training step
            train_loss, train_dice, epoch_time = self._train_epoch()
            self.history["train_loss"].append(train_loss)
            self.history["train_dice"].append(train_dice)
            print(
                f"[Train] Time: {format_time(epoch_time)} | "
                f"Loss: {train_loss:.4f} | Dice: {train_dice:.4f}"
            )

            # 2. Probabilistic validation (always)
            val_dice = self.evaluate(self.val_loader, prefix="Val (Prob.)")
            self.history["val_dice"].append(val_dice)

            # 3. Ground-truth validation (optional)
            if self._use_gt:
                val_dice_gt = self.evaluate_gt(
                    self.val_loader, prefix="Val (GT)"
                )
                self.history["val_dice_gt"].append(val_dice_gt)

            # 4. LR scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # 5. Checkpoint saving — driven by probabilistic Dice only
            if val_dice > self.best_val_dice:
                self.best_val_dice      = val_dice
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
                print(f"New best model! (Val Dice Prob.: {self.best_val_dice:.4f})")

        total_time = time.time() - total_train_start
        print(f"\n{' TRAINING COMPLETE ':=^50}")
        print(f"Total Duration      : {format_time(total_time)}")
        print(f"Best Val Dice Prob. : {self.best_val_dice:.4f}")
        if self._use_gt and self.history["val_dice_gt"]:
            print(f"Last Val Dice GT    : {self.history['val_dice_gt'][-1]:.4f}")
        print("=" * 50)

        if self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)

        return self.model, self.history

    # ------------------------------------------------------------------ #
    #  Training / evaluation internals                                     #
    # ------------------------------------------------------------------ #

    def _train_epoch(self) -> Tuple[float, float, float]:
        """Runs a single training epoch with mixed-precision forward/backward.

        Batch layout expected from the DataLoader:
            - ``batch[0]`` — images        ``[B, C_in, H, W]``
            - ``batch[1]`` — masks         ``[B, A*C,  H, W]``
            - ``batch[2]`` — anns_ids      ``[B, A]``
            - ``batch[3]`` — ground_truth  ``[B, C,    H, W]``  (optional)

        Ground truth is not used during training — it is ignored here even
        when present in the batch so that the loss always operates on the
        annotator stack.

        Returns:
            Tuple[float, float, float]:
                Average batch loss, mean Dice over the epoch, and epoch
                duration in seconds.
        """
        self.model.train()
        running_loss: float = 0.0
        dice_sum:     float = 0.0
        num_batches:  int   = 0

        epoch_start = time.time()
        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch in pbar:
            images:   torch.Tensor = batch[0].to(self.device)
            masks:    torch.Tensor = batch[1].to(self.device)
            anns_ids: torch.Tensor = batch[2].to(self.device)
            # batch[3] (ground truth) intentionally unused during training

            self.optimizer.zero_grad()

            with autocast(device_type=self.device.type):
                seg_pred, ann_pred = self.model(images, anns_ids)
                loss = self.criterion(seg_pred, ann_pred, masks, anns_ids)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * images.size(0)

            prob_mask = self.tracker.compute_probability_mask(masks)
            metrics   = self.tracker.calculate_metrics(
                y_pred=seg_pred.detach(),
                y_true=prob_mask,
                threshold=self.config.threshold,
            )
            dice_sum    += metrics["avg"]["dice"]
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_time = time.time() - epoch_start
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_dice = dice_sum / max(num_batches, 1)

        return epoch_loss, epoch_dice, epoch_time

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, prefix: str = "Val") -> float:
        """Evaluates the model using probabilistic metrics over all thresholds.

        Collects predictions and consensus masks across the full loader,
        then calls :meth:`MetricTracker.compute_probabilistic_metrics` once
        on the concatenated tensors. Ground truth in the batch (``batch[3]``)
        is ignored here — see :meth:`evaluate_gt` for that path.

        Args:
            loader (DataLoader): Dataset split to evaluate.
            prefix (str): Label for the printed summary.

        Returns:
            float: Mean Dice averaged over the full loader and all
                probabilistic thresholds.
        """
        self.model.eval()

        all_seg_preds:  List[torch.Tensor] = []
        all_prob_masks: List[torch.Tensor] = []

        pbar = tqdm(loader, desc=f"Evaluating ({prefix})", leave=False)

        for batch in pbar:
            images:   torch.Tensor = batch[0].to(self.device)
            masks:    torch.Tensor = batch[1].to(self.device)
            anns_ids: torch.Tensor = batch[2].to(self.device)

            output   = self.model(images, anns_ids)
            seg_pred = output[0] if isinstance(output, tuple) else output

            all_seg_preds.append(seg_pred.cpu())
            all_prob_masks.append(
                self.tracker.compute_probability_mask(masks).cpu()
            )

        seg_preds_cat  = torch.cat(all_seg_preds,  dim=0)
        prob_masks_cat = torch.cat(all_prob_masks, dim=0)

        metrics     = self.tracker.compute_probabilistic_metrics(
            seg_preds_cat, prob_masks_cat
        )
        class_names = [f"Class {i}" for i in range(self.config.num_classes)]
        self.tracker.print_summary(prefix, metrics, class_names)

        return metrics["avg"]["dice"]

    @torch.no_grad()
    def evaluate_gt(self, loader: DataLoader, prefix: str = "Val (GT)") -> float:
        """Evaluates the model against ground-truth masks at a fixed threshold.

        Secondary evaluation protocol. Only called when
        ``config.load_ground_truth`` is ``True``. Reads ``batch[3]`` as the
        ground-truth tensor ``[B, C, H, W]`` and delegates to
        :meth:`MetricTracker.compute_gt_metrics`.

        The resulting Dice is logged and stored in ``history["val_dice_gt"]``
        but does **not** drive checkpoint saving, which remains tied to the
        probabilistic Dice for consistency across datasets.

        Args:
            loader (DataLoader): Dataset split to evaluate. Must yield batches
                of length ≥ 4 (i.e. ``load_ground_truth=True`` in DataConfig).
            prefix (str): Label for the printed summary.

        Returns:
            float: Mean Dice against ground truth at ``config.threshold``.
        """
        self.model.eval()

        all_seg_preds: List[torch.Tensor] = []
        all_gt_masks:  List[torch.Tensor] = []

        pbar = tqdm(loader, desc=f"Evaluating ({prefix})", leave=False)

        for batch in pbar:
            if len(batch) < 4:
                raise ValueError(
                    f"evaluate_gt requires batch length ≥ 4, got {len(batch)}. "
                    "Set load_ground_truth=True in DataConfig."
                )

            images:       torch.Tensor = batch[0].to(self.device)
            anns_ids:     torch.Tensor = batch[2].to(self.device)
            ground_truth: torch.Tensor = batch[3].to(self.device)  # [B, C, H, W]

            output   = self.model(images, anns_ids)
            seg_pred = output[0] if isinstance(output, tuple) else output

            all_seg_preds.append(seg_pred.cpu())
            all_gt_masks.append(ground_truth.cpu())

        seg_preds_cat = torch.cat(all_seg_preds, dim=0)
        gt_masks_cat  = torch.cat(all_gt_masks,  dim=0)

        metrics     = self.tracker.compute_gt_metrics(seg_preds_cat, gt_masks_cat)
        class_names = [f"Class {i}" for i in range(self.config.num_classes)]
        self.tracker.print_summary(prefix, metrics, class_names)

        return metrics["avg"]["dice"]

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _resolve_phases(self) -> List[int]:
        """Validates and returns the epoch phase boundaries from config.

        Returns:
            List[int]: Resolved phase boundaries, or ``[]`` to disable
                the curriculum.
        """
        phases = getattr(self.config, "epochs_phases", None)

        if isinstance(phases, list) and len(phases) == 0:
            return []

        if (
            isinstance(phases, list)
            and len(phases) == 4
            and all(isinstance(x, (int, float)) for x in phases)
        ):
            return [int(p) for p in phases]

        print("Warning: 'epochs_phases' not valid or missing in config.")
        print("Falling back to default: [0, 5, 10, 15]")
        return [0, 5, 10, 15]