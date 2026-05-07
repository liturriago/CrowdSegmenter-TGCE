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
            ``probabilistic_thresholds``, and optionally ``epochs_phases``.
        scaler (GradScaler): AMP gradient scaler.
        tracker (MetricTracker): Metric computation and reporting utility.
        best_val_dice (float): Highest validation Dice achieved so far.
        best_model_weights (Optional[Dict[str, torch.Tensor]]): State dict
            of the best checkpoint.
        history (Dict[str, List[float]]): Per-epoch training history.
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
                ``(images, masks, anns_ids)`` where ``masks`` has shape
                ``[B, num_annotators * num_classes, H, W]``.
            val_loader (DataLoader): Source of validation batches,
                same format as ``train_loader``.
            criterion (nn.Module): Loss criterion. Must accept
                ``(seg_pred, ann_pred, masks, anns_ids)``.
            device (torch.device): Device to run training on.
            config (Any): Training configuration object.
        """
        self.model     = model.to(device)
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.criterion    = criterion
        self.device       = device
        self.config       = config
        self.scaler       = GradScaler()
        self.tracker      = MetricTracker(config)

        # State tracking
        self.best_val_dice: float = 0.0
        self.best_model_weights: Optional[Dict[str, torch.Tensor]] = (
            copy.deepcopy(model.state_dict())
        )

        # Metrics history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_dice": [],
            "val_dice":   [],
        }

        # Initialised inside fit() via get_training_phase
        self.optimizer: Optional[optim.Optimizer]                  = None
        self.scheduler: Optional[optim.lr_scheduler.LRScheduler]   = None

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

        Returns:
            Tuple[nn.Module, Dict[str, List[float]]]:
                The model restored to its best validation weights and the
                full training history.
        """
        epochs        = self.config.epochs
        epochs_phases = self._resolve_phases()

        # Phase-free mode: initialise once before the loop
        if not epochs_phases:
            self.optimizer, self.scheduler = get_training_phase(
                self.model, self.config, phase=None
            )

        total_train_start = time.time()

        for epoch in range(epochs):

            # Switch training phase at the configured epoch boundaries
            if epochs_phases and epoch in epochs_phases:
                phase_idx = epochs_phases.index(epoch) + 1
                self.optimizer, self.scheduler = get_training_phase(
                    self.model, self.config, phase=phase_idx
                )

                if self.config.model_name == "CrowdSeg":
                    self.criterion.min_trace = self.config.min_trace

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

            # 2. Validation step
            val_dice = self.evaluate(self.val_loader, prefix="Val")
            self.history["val_dice"].append(val_dice)

            # 3. LR scheduling
            if self.scheduler is not None:
                self.scheduler.step()

            # 4. Checkpoint saving
            if val_dice > self.best_val_dice:
                self.best_val_dice      = val_dice
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
                print(f"New best model! (Val Dice: {self.best_val_dice:.4f})")

        total_time = time.time() - total_train_start
        print(f"\n{' TRAINING COMPLETE ':=^50}")
        print(f"Total Duration : {format_time(total_time)}")
        print(f"Best Val Dice  : {self.best_val_dice:.4f}")
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
            - ``batch[0]`` — images       ``[B, C_in, H, W]``
            - ``batch[1]`` — masks        ``[B, A*C,  H, W]``  (all annotators)
            - ``batch[2]`` — anns_ids     ``[B, A]``           (one-hot or multi-hot)

        The full annotator stack is passed to the criterion. Each loss
        internally selects the signal it needs:

        - ``TGCE_SSPS``    — uses all annotators jointly.
        - ``NoisyLabelLoss`` — selects a single annotator via ``anns_ids``.

        Training Dice is computed against the probabilistic consensus mask
        at the fixed ``config.threshold``, keeping the training metric
        consistent with the validation protocol.

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
            images:   torch.Tensor = batch[0].to(self.device)  # [B, C_in, H, W]
            masks:    torch.Tensor = batch[1].to(self.device)  # [B, A*C,  H, W]
            anns_ids: torch.Tensor = batch[2].to(self.device)  # [B, A]

            self.optimizer.zero_grad()

            with autocast(device_type=self.device.type):
                seg_pred, ann_pred = self.model(images, anns_ids)
                # Both losses share the same positional signature:
                # criterion(seg_pred, ann_pred, masks, anns_ids)
                loss = self.criterion(seg_pred, ann_pred, masks, anns_ids)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            running_loss += loss.item() * images.size(0)

            # Training Dice: fixed threshold against probabilistic consensus
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
        on the concatenated tensors. This is both more accurate (avoids
        batch-size bias in running averages) and consistent with the paper's
        evaluation protocol.

        Args:
            loader (DataLoader): Dataset split to evaluate.
            prefix (str): Label for the printed summary
                (e.g. ``"Val"``, ``"Test"``).

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
            masks:    torch.Tensor = batch[1].to(self.device)  # [B, A*C, H, W]
            anns_ids: torch.Tensor = batch[2].to(self.device)

            output   = self.model(images, anns_ids)
            seg_pred = output[0] if isinstance(output, tuple) else output

            prob_mask = self.tracker.compute_probability_mask(masks)

            all_seg_preds.append(seg_pred.cpu())
            all_prob_masks.append(prob_mask.cpu())

        seg_preds_cat  = torch.cat(all_seg_preds,  dim=0)
        prob_masks_cat = torch.cat(all_prob_masks, dim=0)

        metrics     = self.tracker.compute_probabilistic_metrics(seg_preds_cat, prob_masks_cat)
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
            and all(isinstance(x, int) for x in phases)
        ):
            return phases

        print("Warning: 'epochs_phases' not valid or missing in config.")
        print("Falling back to default: [0, 5, 10, 15]")
        return [0, 5, 10, 15]