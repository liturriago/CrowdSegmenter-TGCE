import yaml
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal

class DataConfig(BaseModel):
    data_dir: str = Field(..., description="Path to the root data directory")
    partitions: List[str] = Field(default=["train", "val", "test"], description="Partitions to load")
    images_folder: str = Field(default="images", description="Folder containing images")
    masks_folder: str = Field(default="masks", description="Folder containing annotator masks")
    ground_truth_folder: str = Field(default="ground_truth", description="Folder containing ground truth masks")
    
    load_annotators: bool = Field(default=True, description="Whether to load annotator masks")
    load_ground_truth: bool = Field(default=True, description="Whether to load ground truth masks")
    
    num_classes: int = Field(default=1, description="Number of classes", gt=0)
    num_annotators: int = Field(default=1, description="Number of annotators", gt=0)
    
    image_size: List[int] = Field(default=[256, 256], description="Image resizing dimensions (H, W)")
    ignored_value: int = Field(default=255, description="Pixel value to ignore")
    normalize: bool = Field(default=True, description="Whether to normalize images")
    
    mean: Optional[List[float]] = Field(default=[0.485, 0.456, 0.406], description="Mean for image normalization")
    std: Optional[List[float]] = Field(default=[0.229, 0.224, 0.225], description="Standard deviation for image normalization")
    
    batch_size: int = Field(default=16, description="Training batch size", gt=0)
    num_workers: int = Field(default=4, description="Number of workers for DataLoader", ge=0)
    pin_memory: bool = Field(default=True, description="Whether to pin memory in DataLoader")
    prefetch_factor: Optional[int] = Field(default=2, description="Prefetch factor for DataLoader", ge=1)

class ModelConfig(BaseModel):
    model_name: Literal["MV", "STAPLE", "Annot-Harmony", "CrowdSeg"] | None = Field(
        default=None, description="Model architecture to use"
    )
    num_annotators: int = Field(default=1, description="Number of annotators", gt=0)
    in_channels: int = Field(default=3, description="Number of input channels", gt=0)
    num_classes: int = Field(default=1, description="Number of classes", gt=0)
    image_size: int = Field(default=256, description="Spatial dimension of the input image (assumed square)", gt=0)
    pretrained: bool = Field(default=True, description="Whether to use pretrained weights")
    decoder_channels: List[int] = Field(
        default=[256, 128, 64, 64], description="Decoder channel dimensions"
    )
    use_residual: bool = Field(default=True, description="Whether to use residual connections")
    seg_head_activation: Literal["sigmoid", "softmax", "tanh", "relu"] | None = Field(
        default=None, description="Segmentation head activation"
    )
    annotator_activation: Literal["sigmoid", "softmax", "tanh", "relu"] | None = Field(
        default=None, description="Annotator head activation"
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
