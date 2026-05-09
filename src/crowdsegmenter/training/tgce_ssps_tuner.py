import json
import optuna
import torch
from pathlib import Path
from typing import Dict, Any, Optional

from crowdsegmenter.config import ExperimentConfig
from crowdsegmenter.data.loader import CrowdSegmenterDataLoader
from crowdsegmenter.models.annot_harmony import AnnotHarmony
from crowdsegmenter.losses.tgce_ssps import TGCE_SSPS
from crowdsegmenter.training.trainer import Trainer
from crowdsegmenter.utils.metrics import MetricTracker
from crowdsegmenter.utils.reproducibility import set_seed


class OptunaTrainer(Trainer):
    """Subclass of Trainer that reports to Optuna after each epoch.

    Overrides the `evaluate` method to report validation Dice to the Optuna
    trial and raise `TrialPruned` if the pruner determines the trial should stop.

    Args:
        trial (optuna.Trial): The current Optuna trial.
        *args: Variable length argument list for `Trainer`.
        **kwargs: Arbitrary keyword arguments for `Trainer`.
    """

    def __init__(self, trial: optuna.Trial, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.trial = trial
        self.current_epoch = 0

    def evaluate(self, loader: torch.utils.data.DataLoader, prefix: str = "Val") -> float:
        """Evaluates the model and reports intermediate values to Optuna.

        Args:
            loader (torch.utils.data.DataLoader): Dataset split to evaluate.
            prefix (str): Label for the printed summary.

        Returns:
            float: Mean validation Dice.
            
        Raises:
            optuna.TrialPruned: If the trial is pruned by the median pruner.
        """
        val_dice = super().evaluate(loader, prefix)
        
        self.trial.report(val_dice, self.current_epoch)
        if self.trial.should_prune():
            raise optuna.TrialPruned()
            
        self.current_epoch += 1
        return val_dice


class TGCESSPSTuner:
    """Optuna tuner for the TGCE_SSPS q hyperparameter.

    Loads the dataset once and runs multiple trials, re-initialising the model
    and Trainer for each trial to find the best `q` value.

    Args:
        config_path (str): Path to the YAML configuration file.
        n_trials (int): Number of Optuna trials to run.
        n_epochs_per_trial (int, optional): Number of epochs per trial. If provided,
            overrides the value in the configuration. Defaults to None.
        study_name (str): Name of the Optuna study.
        output_dir (str): Directory to save tuning results.
        seed (int): Base random seed.
    """

    def __init__(
        self,
        config_path: str,
        n_trials: int = 50,
        n_epochs_per_trial: Optional[int] = None,
        study_name: str = "tgce_ssps_q_optimization",
        output_dir: str = "outputs/optuna/tgce_ssps",
        seed: int = 42,
    ) -> None:
        self.config_path = config_path
        self.n_trials = n_trials
        self.study_name = study_name
        self.output_dir = Path(output_dir)
        self.seed = seed
        
        self.config = ExperimentConfig.from_yaml(config_path)
        
        self.n_epochs_per_trial = (
            n_epochs_per_trial if n_epochs_per_trial is not None 
            else self.config.training.n_epochs_per_trial
        )
        
        self.device = torch.device(self.config.training.device if torch.cuda.is_available() else "cpu")
        
        # Load dataset loaders once for all trials
        data_manager = CrowdSegmenterDataLoader(self.config.data, mode="Annot-Harmony")
        self.train_loader, self.val_loader, _ = data_manager.get_split_loaders()
        
        self.sampler = optuna.samplers.TPESampler(seed=self.seed)
        self.pruner = optuna.pruners.MedianPruner()
        self.study = optuna.create_study(
            study_name=self.study_name,
            direction="maximize",
            sampler=self.sampler,
            pruner=self.pruner
        )
        
        self.results: list = []
        self.best_model_weights: Optional[Dict[str, torch.Tensor]] = None
        self.overall_best_dice: float = -1.0

    def objective(self, trial: optuna.Trial) -> float:
        """Objective function for Optuna optimization.

        Samples `q`, re-initialises the model and loss, and runs a short training
        loop. Returns the best validation Dice found during the trial.

        Args:
            trial (optuna.Trial): The Optuna trial object.

        Returns:
            float: The best validation Dice achieved in this trial, or 0.0 if failed.
        """
        try:
            # Maintain reproducibility while varying per trial
            set_seed(self.config.training.seed + trial.number)
            
            q = trial.suggest_float(
                "q", 
                self.config.training.q_search_low, 
                self.config.training.q_search_high, 
                step=self.config.training.q_search_step
            )
            
            # Re-initialise model with fresh weights
            model = AnnotHarmony(self.config.model).to(self.device)
            
            # Re-initialise loss with the sampled q
            criterion = TGCE_SSPS(
                annotators=self.config.model.num_annotators,
                classes=self.config.model.num_classes,
                ignored_value=self.config.data.ignored_value,
                q=q,
                lambda_factor=self.config.training.tgce_lambda,
                smooth=self.config.training.smooth,
            ).to(self.device)
            
            # Override epochs for the trial
            trial_config = self.config.training.model_copy()
            trial_config.epochs = self.n_epochs_per_trial
            # Optionally clear phases to avoid out-of-bounds scheduling if epochs < max(phases)
            if trial_config.epochs_phases is not None:
                trial_config.epochs_phases = [p for p in trial_config.epochs_phases if p < self.n_epochs_per_trial]
            
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
            
            # Cache the best overall model weights for post-hoc analysis
            if best_val_dice > self.overall_best_dice:
                self.overall_best_dice = best_val_dice
                self.best_model_weights = trainer.best_model_weights
            
            return best_val_dice
            
        except optuna.TrialPruned:
            raise
        except Exception as e:
            print(f"Trial {trial.number} failed with exception: {e}")
            return 0.0

    def run(self) -> Dict[str, Any]:
        """Runs the optimization study and generates a post-hoc analysis.

        Returns:
            Dict[str, Any]: A dictionary containing the best `q`, the best
                validation Dice, and a list of all trial results.
        """
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.study.optimize(self.objective, n_trials=self.n_trials)
        
        print("\nOptimization Summary Table:")
        print(f"{'Trial':<10} | {'q':<10} | {'Best Val Dice':<15}")
        print("-" * 40)
        
        sorted_trials = sorted(
            self.study.trials, 
            key=lambda t: t.value if t.value is not None else 0.0, 
            reverse=True
        )
        
        trials_data = []
        for t in sorted_trials:
            q_val = t.params.get('q', 'N/A')
            val = t.value if t.value is not None else 0.0
            status = "" if t.state == optuna.trial.TrialState.COMPLETE else f" ({t.state.name})"
            print(f"{t.number:<10} | {q_val:<10} | {val:<15.4f}{status}")
            
            trials_data.append({
                "number": t.number,
                "q": t.params.get("q", None),
                "value": t.value,
                "state": t.state.name
            })
                
        best_trial = self.study.best_trial
        best_q = best_trial.params["q"]
        best_val_dice = best_trial.value if best_trial.value is not None else 0.0
        
        print(f"\nBest q found: {best_q} (Val Dice: {best_val_dice:.4f})")
        
        # Evaluate best model on validation set to print full report
        if self.best_model_weights is not None:
            print("\nEvaluating best trial model on validation set...")
            best_model = AnnotHarmony(self.config.model).to(self.device)
            best_model.load_state_dict(self.best_model_weights)
            
            tracker = MetricTracker(self.config.training)
            metrics = tracker.evaluation(best_model, self.val_loader, str(self.device))
            class_names = [f"Class {i}" for i in range(self.config.model.num_classes)]
            tracker.print_full_report("Best Trial Val", metrics, class_names)
        
        results_dict = {
            "best_q": best_q,
            "best_val_dice": best_val_dice,
            "trials": trials_data
        }
        
        with open(self.output_dir / "results.json", "w", encoding="utf-8") as f:
            json.dump(results_dict, f, indent=4)
            
        return results_dict
