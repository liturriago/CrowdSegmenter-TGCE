import yaml
from pathlib import Path
from pydantic import BaseModel, Field, model_validator
from typing import List, Optional, Literal

class DataConfig(BaseModel):
    """
    Configuration for data loading and preprocessing.

    Attributes:
        data_dir (str): Path to the root data directory.
        partitions (List[str]): List of partitions to load (e.g., ["train", "val", "test"]).
        images_folder (str): Folder containing images.
        masks_folder (str): Folder containing annotator masks.
        ground_truth_folder (str): Folder containing ground truth masks.
        load_annotators (bool): Whether to load annotator masks.
        load_ground_truth (bool): Whether to load ground truth masks.
        num_classes (int): Number of classes.
        num_annotators (int): Number of annotators.
        image_size (List[int]): Image resizing dimensions (H, W).
        ignored_value (int): Pixel value to ignore.
        normalize (bool): Whether to normalize images.
        transforms (bool): Whether to apply transforms.
        mean (Optional[List[float]]): Mean for image normalization.
        std (Optional[List[float]]): Standard deviation for image normalization.
        batch_size (int): Training batch size.
        num_workers (int): Number of workers for DataLoader.
        pin_memory (bool): Whether to pin memory in DataLoader.
        prefetch_factor (Optional[int]): Prefetch factor for DataLoader.
    """
    data_dir: str = Field(..., description="Path to the root data directory")
    partitions: List[str] = Field(default=["train", "val", "test"], description="Partitions to load")
    images_folder: str = Field(default="images", description="Folder containing images")
    masks_folder: str = Field(default="masks", description="Folder containing annotator masks")
    ground_truth_folder: str = Field(default="ground_truth", description="Folder containing ground truth masks")
    
    load_annotators: bool = Field(default=True, description="Whether to load annotator masks")
    load_ground_truth: bool = Field(default=False, description="Whether to load ground truth masks")
    
    num_classes: int = Field(default=1, description="Number of classes", gt=0)
    num_annotators: int = Field(default=1, description="Number of annotators", gt=0)
    
    image_size: List[int] = Field(default=[256, 256], description="Image resizing dimensions (H, W)")
    ignored_value: float = Field(default=0.6, description="Pixel value to ignore")
    normalize: bool = Field(default=True, description="Whether to normalize images")
    transforms: bool = Field(default=True, description="Whether to apply transforms")
    
    mean: Optional[List[float]] = Field(default=[0.485, 0.456, 0.406], description="Mean for image normalization")
    std: Optional[List[float]] = Field(default=[0.229, 0.224, 0.225], description="Standard deviation for image normalization")
    
    batch_size: int = Field(default=16, description="Training batch size", gt=0)
    num_workers: int = Field(default=4, description="Number of workers for DataLoader", ge=0)
    pin_memory: bool = Field(default=True, description="Whether to pin memory in DataLoader")
    prefetch_factor: Optional[int] = Field(default=2, description="Prefetch factor for DataLoader", ge=1)

class ModelConfig(BaseModel):
    """
    Configuration for model architecture.

    Attributes:
        model_name (Literal["MV", "STAPLE", "Annot-Harmony", "CrowdSeg"] | None): Model architecture to use.
        num_annotators (int): Number of annotators.
        in_channels (int): Number of input channels.
        num_classes (int): Number of classes.
        image_size (int): Spatial dimension of the input image (assumed square).
        pretrained (bool): Whether to use pretrained weights.
        decoder_channels (List[int]): Decoder channel dimensions.
        use_residual (bool): Whether to use residual connections.
        seg_head_activation (Literal["sigmoid", "softmax", "tanh", "relu"] | None): Segmentation head activation.
        annotator_activation (Literal["sigmoid", "softmax", "tanh", "relu"] | None): Annotator head activation.
    """
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
    """
    Configuration for training process.

    Attributes:
        epochs (int): Total number of training epochs.
        lr (float): Learning rate for optimizer.
        threshold (float): Threshold for binarization.
        probalistic_thresholds (List[float]): Thresholds for probabalistic metrics.
        num_classes (int): Number of classes.
        num_annotators (int): Number of annotators.
        ignored_value (int): Pixel value to ignore.
        smooth (float): Smooth value for metric calculations.
        epochs_phases (Optional[List[int]]): Epochs at which to switch training phases.
        gamma (float): Decay rate for learning rate.
        tgce_q (float): Truncation parameter for TGCE loss.
        tgce_lambda (float): Scaling factor for reliability term.
        seed (int): Random seed for reproducibility.
    """
    epochs: int = Field(100, description="Total number of training epochs", gt=0)
    lr: float = Field(1e-3, description="Learning rate for optimizer", gt=0)
    transfer_lr: float = Field(default=1e-4, description="Learning rate for transfer learning", gt=0)
    device: str = Field(default="cuda", description="Device to train on")
    
    threshold: float = Field(default=0.5, description="Threshold for binarization", ge=0, le=1)
    probabilistic_thresholds: List[float] = Field(
        default=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], 
        description="Thresholds for probabalistic metrics"
    )
    num_classes: int = Field(default=1, description="Number of classes", gt=0)
    num_annotators: int = Field(default=1, description="Number of annotators", gt=0)
    ignored_value: float = Field(default=0.6, description="Pixel value to ignore")
    smooth: float = Field(default=1e-8, description="Smooth value for metric calculations", ge=0)
    epochs_phases: Optional[List[int]] = Field(
        default=[0, 5, 10, 15],
        description="Epochs at which to switch training phases"
    )
    gamma: float = Field(default=0.94, description="Decay rate for learning rate", gt=0.0, le=1.0)
    tgce_q: float = Field(default=0.6704, description="Truncation parameter for TGCE loss", gt=0.0, le=1.0)
    tgce_lambda: float = Field(default=1.0, description="Scaling factor for reliability term", gt=0.0)
    min_trace: bool = Field(default=True, description="Whether to add (True) or subtract (False) the trace regularization")
    alpha: float = Field(default=0.1, description="Scaling factor for the trace regularization")
    
    seed: int = Field(42, description="Random seed for reproducibility")
    

class ExperimentMetadata(BaseModel):
    """
    Configuration for experiment metadata.

    Attributes:
        name (str): Name of the experiment.
        version (int): Version of the experiment.
        output_dir (Path): Directory to save experiment results.
        save_results (bool): Whether to save experiment results.
    """
    name: str = "base_experiment"
    version: int = Field(default=1, ge=0)
    output_dir: Path = Path("outputs/experiment_1")
    save_results: bool = False
    

class ExperimentConfig(BaseModel):
    
    data: DataConfig
    model: ModelConfig
    training: TrainConfig
    experiment: ExperimentMetadata

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
