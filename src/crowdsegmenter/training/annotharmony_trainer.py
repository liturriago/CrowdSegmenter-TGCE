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


class AnnotHarmonyTrainer:
    """Trainer for AnnotHarmony multi-annotator segmentation model.

    Handles multi-phase curriculum training (gradual unfreezing), mixed-precision
    forward/backward passes, validation with probabilistic metrics, and best-checkpoint
    recovery.

    Attributes:
        model (nn.Module): The segmentation model AnnotHarmony.
        train_loader (DataLoader): DataLoader for training data.
        val_loader (DataLoader): DataLoader for validation data.
        criterion (nn.Module): Loss function (e.g. TGCE_SSPS).
        device (torch.device): Device to run training on (cpu / cuda).
        config (Any): Configuration object with training hyperparameters.
        scaler (GradScaler): AMP gradient scaler for mixed-precision training.
        best_val_dice (float): Highest validation Dice score achieved so far.
        best_model_weights (Optional[Dict[str, torch.Tensor]]): State dict of the best model.
        history (Dict[str, List[float]]): Per-epoch record of losses and Dice scores.
        optimizer (optim.Optimizer): Optimizer, initialised at the first training phase.
        scheduler (Optional[LRScheduler]): LR scheduler, attached in later phases.
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
        """Initialises the AnnotHarmonyTrainer.

        Args:
            model (nn.Module): The model to train.
            train_loader (DataLoader): Source of training samples.
            val_loader (DataLoader): Source of validation samples.
            criterion (nn.Module): Loss criterion.
            device (torch.device): Device to run training on.
            config (Any): Training configuration object. Must expose ``epochs``,
                ``threshold``, ``ignored_value``, ``smooth``,
                ``probabilistic_thresholds``, and optionally ``epochs_phases``.
        """
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.device = device
        self.config = config
        self.scaler = GradScaler()

        # State tracking
        self.best_val_dice: float = 0.0
        self.best_model_weights: Optional[Dict[str, torch.Tensor]] = (
            copy.deepcopy(model.state_dict())
        )

        # Metrics history
        self.history: Dict[str, List[float]] = {
            "train_loss": [],
            "train_dice": [],
            "val_dice": [],
        }

        # Optimizer / scheduler are set by get_training_phase inside fit()
        self.optimizer: Optional[optim.Optimizer] = None
        self.scheduler: Optional[optim.lr_scheduler.LRScheduler] = None
        
        # Metric tracker
        self.tracker = MetricTracker(config)

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def fit(self) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """Executes the full multi-phase training and validation process.

        Phase boundaries are read from ``config.epochs_phases`` (a list of four
        epoch indices).  If that attribute is absent or malformed the trainer
        falls back to ``[0, 5, 10, 15]``.  Passing ``epochs_phases = []``
        (empty list) disables curriculum entirely — the whole model is trained
        end-to-end from epoch 0 with a scheduler.

        Returns:
            Tuple[nn.Module, Dict[str, List[float]]]:
                The model restored to its best validation weights and the
                full training history.
        """
        epochs: int = self.config.epochs
        epochs_phases = self._resolve_phases()

        # If no curriculum is requested, initialise once in phase-free mode
        if not epochs_phases:
            self.optimizer, self.scheduler = get_training_phase(
                self.model, self.config, phase=None
            )

        total_train_start = time.time()

        for epoch in range(epochs):

            # Curriculum: switch phase when the epoch boundary is reached
            if epochs_phases and epoch in epochs_phases:
                phase_idx = epochs_phases.index(epoch) + 1
                self.optimizer, self.scheduler = get_training_phase(
                    self.model, self.config, phase=phase_idx
                )

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
                self.best_val_dice = val_dice
                self.best_model_weights = copy.deepcopy(self.model.state_dict())
                print(f"New best model! (Val Dice: {self.best_val_dice:.4f})")

        total_time = time.time() - total_train_start
        print(f"\n{' TRAINING COMPLETE ':=^50}")
        print(f"Total Duration: {format_time(total_time)}")
        print(f"Best Dice: {self.best_val_dice:.4f}")
        print("=" * 50)

        # Restore best weights before returning
        if self.best_model_weights is not None:
            self.model.load_state_dict(self.best_model_weights)

        return self.model, self.history

    # ------------------------------------------------------------------ #
    #  Training / evaluation internals                                     #
    # ------------------------------------------------------------------ #

    def _train_epoch(self) -> Tuple[float, float, float]:
        """Runs a single training epoch with mixed-precision forward/backward.

        Each batch is expected to yield ``(image, masks, onehot, gt)`` from
        the ``AnnotHarmonyDataset``.  The model is called with both the image
        and the annotator multi-hot vector; the TGCE_SSPS criterion receives the
        tuple output ``(seg_pred, ann_pred)`` together with the annotator masks.

        Returns:
            Tuple[float, float, float]:
                Average batch loss, mean Dice score over the epoch, and total
                epoch duration in seconds.
        """
        self.model.train()
        running_loss: float = 0.0

        # Accumulators for Dice across the epoch
        dice_sum: float = 0.0
        num_batches: int = 0

        epoch_start = time.time()
        pbar = tqdm(self.train_loader, desc="Training", leave=False)

        for batch in pbar:
            images: torch.Tensor = batch[0].to(self.device)
            masks: torch.Tensor = batch[1].to(self.device)
            multihot: torch.Tensor = batch[2].to(self.device)

            self.optimizer.zero_grad()

            with autocast(device_type=self.device.type):
                # Forward: returns (seg_output, annotator_output)
                seg_pred, ann_pred = self.model(images, multihot)
                loss = self.criterion(seg_pred, ann_pred, masks)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            batch_size: int = images.size(0)
            running_loss += loss.item() * batch_size

            # Compute probability mask then Dice for this batch
            prob_mask = self.tracker.compute_probability_mask(masks)
            metrics = self.tracker.calculate_metrics(y_pred=seg_pred.detach(),y_true=prob_mask,threshold=self.config.threshold)
            dice_sum += metrics["avg"]["dice"]
            num_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        epoch_time = time.time() - epoch_start
        epoch_loss: float = running_loss / len(self.train_loader.dataset)
        epoch_dice: float = dice_sum / max(num_batches, 1)

        return epoch_loss, epoch_dice, epoch_time

    @torch.no_grad()
    def evaluate(self, loader: DataLoader, prefix: str = "Val") -> float:
        """Evaluates the model using probabilistic metrics averaged over all thresholds.

        Args:
            loader (DataLoader): Dataset split to evaluate (validation or test).
            prefix (str): Label for the printed summary (e.g. ``"Val"``, ``"Test"``).

        Returns:
            float: Mean Dice score averaged over the full loader and all
                probabilistic thresholds.
        """
        self.model.eval()

        all_seg_preds: List[torch.Tensor] = []
        all_prob_masks: List[torch.Tensor] = []

        pbar = tqdm(loader, desc=f"Evaluating ({prefix})", leave=False)

        for batch in pbar:
            images: torch.Tensor = batch[0].to(self.device)
            masks: torch.Tensor = batch[1].to(self.device)
            multihot: torch.Tensor = batch[2].to(self.device)

            seg_pred, _ = self.model(images, multihot)

            prob_mask = self.tracker.compute_probability_mask(masks)

            all_seg_preds.append(seg_pred.cpu())
            all_prob_masks.append(prob_mask.cpu())

        # Concatenate all batches and compute probabilistic metrics in one pass
        seg_preds_cat = torch.cat(all_seg_preds, dim=0)
        prob_masks_cat = torch.cat(all_prob_masks, dim=0)

        # Use instance method (needs self.config thresholds)
        metrics = self.tracker.compute_probabilistic_metrics(seg_preds_cat, prob_masks_cat)

        class_names = [f"Class {i}" for i in range(self.config.num_classes)]
        self.tracker.print_summary(prefix, metrics, class_names)

        return metrics["avg"]["dice"]

    # ------------------------------------------------------------------ #
    #  Private helpers                                                     #
    # ------------------------------------------------------------------ #

    def _resolve_phases(self) -> List[int]:
        """Validates and returns the epoch phase boundaries from config.

        If ``config.epochs_phases`` is a valid list of four integers it is
        returned as-is.  An empty list disables curriculum.  Any other
        value triggers a warning and falls back to ``[0, 5, 10, 15]``.

        Returns:
            List[int]: Resolved epoch phase boundaries.
        """
        phases = getattr(self.config, "epochs_phases", None)

        # Explicit opt-out: empty list → phase-free training
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