"""
Command-line script for training and evaluating AnnotHarmony on multi-annotator
segmentation data.

This script executes a single training run: data loading, model construction,
curriculum training via Trainer, final evaluation with probabilistic metrics
(and optionally ground-truth metrics), and weight serialization.
"""
import argparse
import torch
from pathlib import Path

from crowdsegmenter.config import ExperimentConfig
from crowdsegmenter.data.loader import CrowdSegmenterDataLoader
from crowdsegmenter.models.annot_harmony import AnnotHarmony
from crowdsegmenter.losses.tgce_ssps import TGCE_SSPS
from crowdsegmenter.training.trainer import Trainer
from crowdsegmenter.utils.metrics import MetricTracker
from crowdsegmenter.utils.reproducibility import set_seed


def run_annot_harmony_experiment(config_path: str) -> None:
    """Trains and evaluates AnnotHarmony on multi-annotator segmentation data.

    Stages:
        1. Configuration loading and reproducibility setup.
        2. Data loading (train / val / test splits).
        3. Model and loss construction.
        4. Curriculum training via :class:`Trainer`.
        5. Final probabilistic evaluation on the test set.
        6. Final ground-truth evaluation on the test set (when
           ``config.data.load_ground_truth`` is ``True``).
        7. Weight serialization.

    Args:
        config_path (str): Path to the YAML experiment configuration file.
    """
    # ------------------------------------------------------------------ #
    # 1. Configuration & setup                                            #
    # ------------------------------------------------------------------ #
    cfg    = ExperimentConfig.from_yaml(config_path)
    device = torch.device(
        cfg.training.device if torch.cuda.is_available() else "cpu"
    )

    if cfg.training.seed is not None:
        set_seed(cfg.training.seed)

    output_path = Path(cfg.experiment.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\n{'':=<60}")
    print(f"  Experiment : {cfg.experiment.name}")
    print(f"  Device     : {device}")
    print(f"  Output     : {output_path}")
    print(f"  GT eval    : {cfg.data.load_ground_truth}")
    print(f"{'':=<60}\n")

    # ------------------------------------------------------------------ #
    # 2. Data                                                             #
    # ------------------------------------------------------------------ #
    data_manager = CrowdSegmenterDataLoader(cfg.data, mode="annotharmony")
    train_loader, val_loader, test_loader = data_manager.get_split_loaders()

    # ------------------------------------------------------------------ #
    # 3. Model & loss                                                     #
    # ------------------------------------------------------------------ #
    model = AnnotHarmony(cfg.model).to(device)

    criterion = TGCE_SSPS(
        annotators=cfg.model.num_annotators,
        classes=cfg.model.num_classes,
        ignored_value=cfg.data.ignored_value,
        q=cfg.training.tgce_q,
        lambda_factor=cfg.training.tgce_lambda,
        smooth=cfg.training.smooth,
    ).to(device)

    # ------------------------------------------------------------------ #
    # 4. Training                                                         #
    # ------------------------------------------------------------------ #
    print(f"\n Starting AnnotHarmony training on {device}...\n")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        config=cfg.training,
    )

    trained_model, history = trainer.fit()

    # ------------------------------------------------------------------ #
    # 5. Final probabilistic evaluation on test set                       #
    # ------------------------------------------------------------------ #
    print("\n Generating final probabilistic report on test set...\n")

    class_names  = [f"Class {i}" for i in range(cfg.model.num_classes)]
    tracker      = MetricTracker(cfg.training)

    prob_metrics = tracker.evaluation(trained_model, test_loader, device)
    tracker.print_full_report("AnnotHarmony Test — Probabilistic", prob_metrics, class_names)

    # ------------------------------------------------------------------ #
    # 6. Final ground-truth evaluation on test set (optional)             #
    # ------------------------------------------------------------------ #
    if cfg.data.load_ground_truth:
        print("\n Generating final ground-truth report on test set...\n")

        gt_metrics = tracker.evaluation_gt(trained_model, test_loader, device)
        tracker.print_full_report("AnnotHarmony Test — Ground Truth", gt_metrics, class_names)
        
    # ------------------------------------------------------------------ #
    # 7. Save weights                                                     #
    # ------------------------------------------------------------------ #
    save_file = output_path / "annot_harmony_final.pth"
    torch.save(trained_model.state_dict(), save_file)
    print(f"\n Experiment complete. Weights saved to: {save_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="AnnotHarmony Training & Evaluation Script"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/training/annotharmony/oxfordiiitpet.yaml",
        help="Path to the YAML experiment configuration file.",
    )
    args = parser.parse_args()

    run_annot_harmony_experiment(args.config)