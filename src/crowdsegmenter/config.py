import yaml
from pathlib import Path
from pydantic import BaseModel, Field

class DataConfig(BaseModel):
    dataset_path: str = Field(..., description="Path to the dataset")
    batch_size: int = Field(16, description="Training batch size", gt=0)
    num_workers: int = Field(4, description="Number of workers for DataLoader", ge=0)
    img_size: int = Field(256, description="Image resizing dimension", gt=0)

class ModelConfig(BaseModel):
    architecture: str = Field(..., description="Model architecture (e.g., UNet)")
    in_channels: int = Field(3, description="Number of input channels", gt=0)
    num_classes: int = Field(1, description="Number of output classes", gt=0)

class TrainConfig(BaseModel):
    num_epochs: int = Field(100, description="Total number of training epochs", gt=0)
    learning_rate: float = Field(1e-3, description="Learning rate for optimizer", gt=0)
    weight_decay: float = Field(1e-4, description="Weight decay for optimizer", ge=0)
    loss_function: str = Field(..., description="Loss function to use (e.g., TGCE_SSPS, Majority Voting, STAPLE)")

class ExperimentMetadata(BaseModel):
    experiment_name: str = Field(..., description="Name of the experiment")
    domain: str = Field(..., description="Domain of the dataset (e.g., Medical, Pets, Oil Wells)")
    seed: int = Field(42, description="Random seed for reproducibility")

class ExperimentConfig(BaseModel):
    
    data: DataConfig
    model: ModelConfig
    training: TrainConfig
    metadata: ExperimentMetadata

    @classmethod
    def from_yaml(cls, yaml_path: str | Path) -> "ExperimentConfig":
        """
        Loads and validates configuration from a YAML file.
        """
        path = Path(yaml_path)
        if not path.is_file():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, "r", encoding="utf-8") as f:
            raw_config = yaml.safe_load(f)
            
        return cls(**raw_config)
