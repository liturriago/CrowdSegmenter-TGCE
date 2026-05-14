import json
import optuna
import torch
from pathlib import Path
from typing import Dict, Any, Optional, List

from crowdsegmenter.config import ExperimentConfig
from crowdsegmenter.data.loader import CrowdSegmenterDataLoader
from crowdsegmenter.models.annot_harmony import CrowdSeg
from crowdsegmenter.losses.noisy_label import NoisyLabelLoss
from crowdsegmenter.training.trainer import Trainer
from crowdsegmenter.utils.metrics import MetricTracker
from crowdsegmenter.utils.reproducibility import set_seed


class OptunaTrainer(Trainer):
    """Subclass of Trainer that reports intermediate results to Optuna.

    Overrides :meth:`evaluate` and :meth:`evaluate_gt` so that after every
    validation step the probabilistic Dice is reported to the trial object.
    The pruner can then stop unpromising trials early.

    Ground-truth evaluation (when ``config.load_ground_truth`` is ``True``)
    is preserved and printed for logging purposes, but the value reported to
    Optuna is always the probabilistic Dice — consistent with the evaluation
    protocol used in the main training scripts.

    Args:
        trial (optuna.Trial): The current Optuna trial.
        *args: Positional arguments forwarded to :class:`Trainer`.
        **kwargs: Keyword arguments forwarded to :class:`Trainer`.
    """

    def __init__(self, trial: optuna.Trial, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trial         = trial
        self.current_epoch = 0

    def evaluate(self, loader: torch.utils.data.DataLoader, prefix: str = "Val") -> float:
        """Evaluates probabilistically and reports the result to Optuna.

        Args:
            loader (torch.utils.data.DataLoader): Dataset split to evaluate.
            prefix (str): Label for the printed summary.

        Returns:
            float: Mean probabilistic Dice over all thresholds.

        Raises:
            optuna.TrialPruned: If the pruner determines this trial should stop.
        """
        val_dice = super().evaluate(loader, prefix)

        self.trial.report(val_dice, self.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()

        self.current_epoch += 1
        return val_dice

    def evaluate_gt(
        self, loader: torch.utils.data.DataLoader, prefix: str = "Val (GT)"
    ) -> float:
        """Evaluates against ground truth and prints results without pruning.

        The GT Dice is not reported to Optuna — pruning is driven solely by
        the probabilistic Dice from :meth:`evaluate` to keep the optimisation
        objective consistent across datasets that lack ground truth.

        Args:
            loader (torch.utils.data.DataLoader): Dataset split to evaluate.
            prefix (str): Label for the printed summary.

        Returns:
            float: Mean GT Dice at ``config.threshold``.
        """
        return super().evaluate_gt(loader, prefix)


class NoisyLabelTuner:
    """Bayesian hyperparameter tuner for the ``alpha`` parameter of NoisyLabelLoss.

    Loads the dataset once and runs ``n_trials`` Optuna trials, re-initialising
    the model, loss, and :class:`OptunaTrainer` on every trial. The best ``alpha``
    is determined by the highest probabilistic validation Dice.

    When ``config.data.load_ground_truth`` is ``True``, a GT evaluation pass
    is also run each epoch (via the trainer) and printed for reference, but it
    does not influence the Optuna objective.

    Args:
        config_path (str): Path to the YAML experiment configuration file.
        n_trials (int): Number of Optuna trials. Defaults to ``50``.
        n_epochs_per_trial (int | None): Epochs per trial. Overrides
            ``config.training.n_epochs_per_trial`` when provided.
        study_name (str): Name of the Optuna study.
        output_dir (str): Directory to save ``results.json``.
        seed (int): Base random seed. Each trial uses ``seed + trial.number``
            for reproducibility while keeping trials distinct.
    """

    def __init__(
        self,
        config_path: str,
        n_trials: int = 50,
        n_epochs_per_trial: Optional[int] = None,
        study_name: str = "noisy_label_alpha_optimization",
        output_dir: str = "outputs/optuna/noisy_label",
        seed: int = 42,
    ) -> None:
        self.config_path = config_path
        self.n_trials    = n_trials
        self.study_name  = study_name
        self.output_dir  = Path(output_dir)
        self.seed        = seed

        self.config = ExperimentConfig.from_yaml(config_path)

        self.n_epochs_per_trial = (
            n_epochs_per_trial
            if n_epochs_per_trial is not None
            else self.config.training.n_epochs_per_trial
        )

        self.device = torch.device(
            self.config.training.device if torch.cuda.is_available() else "cpu"
        )

        # Dataset loaded once — not per trial
        data_manager = CrowdSegmenterDataLoader(
            self.config.data, mode="CrowdSeg"
        )
        self.train_loader, self.val_loader, _ = data_manager.get_split_loaders()

        self.study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=self.seed),
            pruner=optuna.pruners.MedianPruner(),
        )

        self.results:             List[Dict[str, Any]]              = []
        self.best_model_weights:  Optional[Dict[str, torch.Tensor]] = None
        self.overall_best_dice:   float                             = -1.0

    # ------------------------------------------------------------------ #
    #  Optuna objective                                                    #
    # ------------------------------------------------------------------ #

    def objective(self, trial: optuna.Trial) -> float:
        """Samples ``alpha``, trains for ``n_epochs_per_trial``, returns best Dice.

        Args:
            trial (optuna.Trial): The Optuna trial object.

        Returns:
            float: Best probabilistic validation Dice achieved in this trial,
                or ``0.0`` if the trial failed with an unexpected exception.

        Raises:
            optuna.TrialPruned: Re-raised transparently so Optuna can record
                the pruned state correctly.
        """
        try:
            set_seed(self.config.training.seed + trial.number)

            alpha = trial.suggest_float(
                "alpha",
                self.config.training.alpha_search_low,
                self.config.training.alpha_search_high,
            )

            print(f"\n[Trial {trial.number}] alpha = {alpha:.4f}")
            print("-" * 40)

            # Fresh model weights every trial
            model = CrowdSeg(self.config.model).to(self.device)

            criterion = NoisyLabelLoss(
                annotators=self.config.model.num_annotators,
                classes=self.config.model.num_classes,
                ignored_value=self.config.data.ignored_value,
                alpha=alpha,
                smooth=self.config.training.smooth,
            ).to(self.device)

            # Override epoch count; trim phase boundaries that fall out of range
            trial_config = self.config.training.model_copy()
            trial_config.epochs = self.n_epochs_per_trial
            if trial_config.epochs_phases is not None:
                trial_config.epochs_phases = [
                    p for p in trial_config.epochs_phases
                    if p < self.n_epochs_per_trial
                ]

            trainer = OptunaTrainer(
                trial=trial,
                model=model,
                train_loader=self.train_loader,
                val_loader=self.val_loader,
                criterion=criterion,
                device=self.device,
                config=trial_config,
            )

            _, history = trainer.fit()

            best_val_dice = max(history["val_dice"]) if history["val_dice"] else 0.0

            # Cache best weights across all trials for post-hoc reporting
            if best_val_dice > self.overall_best_dice:
                self.overall_best_dice  = best_val_dice
                self.best_model_weights = trainer.best_model_weights

            return best_val_dice

        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"[Trial {trial.number}] Failed with exception: {e}")
            return 0.0

    # ------------------------------------------------------------------ #
    #  Run                                                                 #
    # ------------------------------------------------------------------ #

    def run(self) -> Dict[str, Any]:
        """Runs the full optimisation study and generates a post-hoc report.

        After all trials complete:

        - Prints a summary table of every trial sorted by Dice descending.
        - Evaluates the best trial's model on the validation set using the
          full probabilistic protocol (and GT protocol if available) and
          prints a detailed report via :meth:`MetricTracker.print_full_report`.
        - Saves all trial results to ``output_dir/results.json``.

        Returns:
            Dict[str, Any]: Dictionary with keys:

                - ``"best_alpha"`` — the ``alpha`` value of the best trial.
                - ``"best_val_dice"`` — the corresponding Dice score.
                - ``"trials"`` — list of per-trial result dicts.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.study.optimize(self.objective, n_trials=self.n_trials)

        # ── Summary table ──────────────────────────────────────────────
        print(f"\n{' OPTIMISATION SUMMARY ':=^60}")
        print(f"{'Trial':<8} | {'alpha':<8} | {'Val Dice':<12} | {'State':<12}")
        print("-" * 46)

        sorted_trials = sorted(
            self.study.trials,
            key=lambda t: t.value if t.value is not None else 0.0,
            reverse=True,
        )

        trials_data: List[Dict[str, Any]] = []
        for t in sorted_trials:
            alpha_val  = t.params.get("alpha", "N/A")
            value  = t.value if t.value is not None else 0.0
            state  = t.state.name
            print(f"{t.number:<8} | {alpha_val:<8} | {value:<12.4f} | {state:<12}")
            trials_data.append({
                "number": t.number,
                "alpha":      t.params.get("alpha"),
                "value":  t.value,
                "state":  state,
            })

        print("=" * 60)

        best_trial    = self.study.best_trial
        best_alpha    = best_trial.params["alpha"]
        best_val_dice = best_trial.value if best_trial.value is not None else 0.0

        print(f"\n Best alpha : {best_alpha}  |  Val Dice (prob.) : {best_val_dice:.4f}")

        # ── Post-hoc evaluation of the best model ─────────────────────
        if self.best_model_weights is not None:
            print("\n Evaluating best trial model on validation set...\n")

            best_model = CrowdSeg(self.config.model).to(self.device)
            best_model.load_state_dict(self.best_model_weights)

            tracker     = MetricTracker(self.config.training)
            class_names = [f"Class {i}" for i in range(self.config.model.num_classes)]

            # Probabilistic report (always)
            prob_metrics = tracker.evaluation(best_model, self.val_loader, str(self.device))
            tracker.print_full_report("Best Trial Val — Probabilistic", prob_metrics, class_names)

            # GT report (when available)
            if self.config.data.load_ground_truth:
                gt_metrics = tracker.evaluation_gt(
                    best_model, self.val_loader, str(self.device)
                )
                tracker.print_full_report("Best Trial Val — Ground Truth", gt_metrics, class_names)

        # ── Persist results ────────────────────────────────────────────
        results_dict: Dict[str, Any] = {
            "best_alpha":        best_alpha,
            "best_val_dice": best_val_dice,
            "trials":        trials_data,
        }

        results_path = self.output_dir / "results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=4)

        print(f"\n Results saved to: {results_path}")

        return results_dict