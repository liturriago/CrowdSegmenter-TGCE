import yaml
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal

class DataConfig(BaseModel):
    dataset_path: str = Field(..., description="Path to the dataset")
    batch_size: int = Field(16, description="Training batch size", gt=0)
    num_workers: int = Field(4, description="Number of workers for DataLoader", ge=0)
    img_size: int = Field(256, description="Image resizing dimension", gt=0)

class ModelConfig(BaseModel):
    model_name: Literal["MV", "STAPLE", "Annot-Harmony", "CrowdSeg"] | None = Field(
        None, 
        description="Model architecture to use"
    )
    num_annotators: int | None = Field(
        None, 
        description="Number of annotators", 
        gt=0
    )
    in_channels: int | None = Field(3, description="Number of input channels", gt=0)
    out_channels: int | None = Field(1, description="Number of output channels", gt=0)
    pretrained: bool | None = Field(True, description="Whether to use pretrained weights")
    decoder_channels: List[int] | None = Field(
        default=[256, 128, 64, 64], 
        description="Decoder channel dimensions"
    )
    use_residual: bool | None = Field(True, description="Whether to use residual connections")
    seg_head_activation: Literal["sigmoid", "softmax", "tanh", "relu"] | None = Field(
        None, 
        description="Segmentation head activation"
    )
    annotator_activation: Literal["sigmoid", "softmax", "tanh", "relu"] | None = Field(
        None, 
        description="Annotator head activation"
    )

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
