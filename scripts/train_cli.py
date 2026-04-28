import argparse
import logging
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.config import ExperimentConfig
from src.training.engine import Trainer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description="CrowdSegmenter-TGCE Training Entry Point")
    parser.add_argument(
        "--config", 
        type=str, 
        required=True, 
        help="Path to the YAML experiment configuration file"
    )
    args = parser.parse_args()

    # 1. Load Configuration
    config = ExperimentConfig.from_yaml(args.config)
    logger.info(f"Loaded configuration for experiment: {config.metadata.experiment_name}")

    # 2. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 3. Initialize Model (Mock example)
    logger.info(f"Initializing architecture: {config.model.architecture}")
    # TODO: Replace with actual model factory based on config.model
    model = nn.Conv2d(config.model.in_channels, config.model.num_classes, kernel_size=3, padding=1)

    # 4. Initialize DataLoaders (Mock example)
    logger.info(f"Setting up dataloaders from: {config.data.dataset_path}")
    # TODO: Replace with actual dataloaders based on config.data
    mock_data = torch.randn(100, config.model.in_channels, config.data.img_size, config.data.img_size)
    mock_targets = torch.randn(100, config.model.num_classes, config.data.img_size, config.data.img_size)
    dataset = TensorDataset(mock_data, mock_targets)
    train_loader = DataLoader(dataset, batch_size=config.data.batch_size, num_workers=config.data.num_workers)
    val_loader = DataLoader(dataset, batch_size=config.data.batch_size, num_workers=config.data.num_workers)

    # 5. Initialize Loss and Optimizer
    # TODO: Replace with actual loss function initialization based on config.training.loss_function
    criterion = nn.MSELoss() 
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=config.training.learning_rate, 
        weight_decay=config.training.weight_decay
    )

    # 6. Initialize and Run Trainer
    trainer = Trainer(
        config=config,
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device
    )
    
    trainer.fit()

if __name__ == "__main__":
    main()
