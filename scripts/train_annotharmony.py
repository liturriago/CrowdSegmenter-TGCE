"""
Command-line script for training and evaluating AnnotHarmony on multi-annotator
segmentation data.

This script executes a single training run: data loading, model construction,
curriculum training via AnnotHarmonyTrainer, final evaluation with probabilistic
metrics, and weight serialization.
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
    """
    :param config_path: Path to YAML config file.
    :return: None
    
    This function trains and evaluates the AnnotHarmony model on multi-annotator
    segmentation data. It performs the following steps:
        1. Load Configuration & Setup
        2. Data Setup
        3. Model, Loss, and Optimizer Initialization
        4. Train model
        5. Final Evaluation on Source Test Set
        6. Save the final trained weights
    """
    # 1. Load Configuration & Setup
    cfg = ExperimentConfig.from_yaml(config_path)
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    
    if cfg.training.seed is not None:
        set_seed(cfg.training.seed)
        
    output_path = Path(cfg.experiment.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 2. Data Setup
    data_manager = CrowdSegmenterDataLoader(cfg.data, mode="Annot-Harmony")
    train_loader, val_loader, test_loader = data_manager.get_split_loaders()

    # 3. Model, Loss, and Optimizer Initialization
    model = AnnotHarmony(cfg.model).to(device)

    criterion = TGCE_SSPS(
        annotators=cfg.model.num_annotators,
        classes=cfg.model.num_classes,
        ignored_value=cfg.data.ignored_value,
        q=cfg.training.tgce_q,
        lambda_factor=cfg.training.tgce_lambda,
        smooth=cfg.training.smooth,
    ).to(device)

    # 4. Train model
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        device=device,
        config=cfg.training,
    )
    
    print(f"\n Starting AnnotHarmony training on {device}...")
    trained_model, history = trainer.fit()

    # 5. Final Evaluation on Source Test Set
    print("\n Generating Final Statistical Reports on Test Set...")

    class_names = [f"Class {i}" for i in range(cfg.model.num_classes)]
    
    metric_tracker = MetricTracker(cfg.training)
    metrics = metric_tracker.evaluation(trained_model, test_loader,device)
    metric_tracker.print_full_report("AnnotHarmony Test", metrics, class_names)

    # 6. Save the final trained weights
    save_file = Path(output_path) / "model_final.pth"
    torch.save(trained_model.state_dict(), save_file)
    print(f"\n Experiment completed. Results saved in: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AnnotHarmony Training Script")
    parser.add_argument("--config", type=str, default="configs/base_experiment.yaml", help="Path to YAML config")
    args = parser.parse_args()
    
    run_annot_harmony_experiment(args.config)
