import pytest
from pydantic import ValidationError
from crowdsegmenter.config import ExperimentConfig

def test_experiment_config_loading(tmp_path):
    # Create a temporary config file
    config_content = """
    metadata:
      experiment_name: "test"
      domain: "Medical"
      seed: 42
    data:
      dataset_path: "./data/medical"
      batch_size: 16
      num_workers: 4
      img_size: 256
    model:
      architecture: "UNet"
      in_channels: 1
      num_classes: 1
    training:
      num_epochs: 10
      learning_rate: 0.001
      weight_decay: 0.0001
      loss_function: "TGCE_SSPS"
    """
    config_path = tmp_path / "test_config.yaml"
    config_path.write_text(config_content)
    
    # Load and validate
    config = ExperimentConfig.from_yaml(config_path)
    
    assert config.metadata.experiment_name == "test"
    assert config.data.batch_size == 16
    assert config.model.architecture == "UNet"
    assert config.training.num_epochs == 10

def test_experiment_config_invalid():
    # Should raise validation error on invalid or missing fields if we were to pass invalid dict
    with pytest.raises(FileNotFoundError):
        ExperimentConfig.from_yaml("nonexistent_path.yaml")
