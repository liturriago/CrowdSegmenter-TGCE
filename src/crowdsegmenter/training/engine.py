import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
import logging

from src.config import ExperimentConfig

logger = logging.getLogger(__name__)

class Trainer:
    def __init__(
        self,
        config: ExperimentConfig,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device
    ):
        self.config = config
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion.to(device)
        self.optimizer = optimizer
        self.device = device
        
        self.scaler = GradScaler()
        self.current_epoch = 0

    def train_epoch(self) -> float:
        self.model.train()
        running_loss = 0.0
        
        for batch_idx, (inputs, targets) in enumerate(self.train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward pass with AMP
            with autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
            
            # Backward pass and optimization
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            running_loss += loss.item()
            
        return running_loss / len(self.train_loader)

    def validate_epoch(self) -> float:
        self.model.eval()
        running_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in self.val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                with autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                running_loss += loss.item()
                
        return running_loss / len(self.val_loader)

    def fit(self):
        logger.info(f"Starting training for {self.config.training.num_epochs} epochs on {self.device}")
        
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            
            train_loss = self.train_epoch()
            val_loss = self.validate_epoch()
            
            logger.info(
                f"Epoch [{epoch+1}/{self.config.training.num_epochs}] "
                f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}"
            )
            
            # Checkpointing logic can be added here
