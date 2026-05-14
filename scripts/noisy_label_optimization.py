"""
Command-line script for hyperparameter optimization of the NoisyLabelLoss alpha
regularization weight.

This script executes Bayesian optimization over the ``alpha`` hyperparameter
using Optuna, evaluating multiple trials with short training runs to find the
optimal trace-regularization strength for the dataset.
"""
import argparse
from pathlib import Path
import optuna

from crowdsegmenter.training.noisy_label_tuner import NoisyLabelTuner


def run_noisy_label_alpha_optimization(
    config_path: str,
    n_trials: int = 50,
    n_epochs_per_trial: int = 5,
    output_dir: str = "outputs/optuna/noisy_label_alpha",
) -> None:
    """Runs Bayesian optimisation of the NoisyLabelLoss alpha hyperparameter."""

    print(
        f"\nStarting NoisyLabelLoss Alpha Optimization "
        f"(Trials: {n_trials}, Epochs per trial: {n_epochs_per_trial})"
    )

    tuner = NoisyLabelTuner(
        config_path=config_path,
        n_trials=n_trials,
        n_epochs_per_trial=n_epochs_per_trial,
        output_dir=output_dir,
    )

    results = tuner.run()

    print(f"\nOptimization complete. Best alpha: {results['best_alpha']:.4f}")

    out_path = Path(output_dir)
    try:
        import matplotlib.pyplot as plt

        # Save optimization history plot
        fig_hist = optuna.visualization.matplotlib.plot_optimization_history(tuner.study)
        plt.tight_layout()
        plt.savefig(out_path / "optimization_history.png")
        plt.close()

        # Save parameter importance plot
        fig_imp = optuna.visualization.matplotlib.plot_param_importances(tuner.study)
        plt.tight_layout()
        plt.savefig(out_path / "param_importances.png")
        plt.close()

        print(f"Visualisations saved to {out_path}")
    except ImportError:
        print("matplotlib not installed. Skipping visualisations.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NoisyLabelLoss alpha Optimization Script"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config")
    parser.add_argument("--n_trials", type=int, default=50, help="Number of trials")
    parser.add_argument("--n_epochs_per_trial", type=int, default=5, help="Epochs per trial")
    parser.add_argument("--output_dir", type=str, default="outputs/optuna/noisy_label_alpha", help="Output directory")
    args = parser.parse_args()

    run_noisy_label_alpha_optimization(
        config_path=args.config,
        n_trials=args.n_trials,
        n_epochs_per_trial=args.n_epochs_per_trial,
        output_dir=args.output_dir,
    )