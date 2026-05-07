import pytest
from pathlib import Path
from pydantic import ValidationError
from crowdsegmenter.config import DataConfig, ModelConfig, TrainConfig, ExperimentMetadata, ExperimentConfig

def test_data_config_defaults():
    config = DataConfig(data_dir="dummy/path")
    assert config.data_dir == "dummy/path"
    assert config.partitions == ["train", "val", "test"]
    assert config.num_classes == 1
    assert config.batch_size == 16

def test_data_config_validation():
    # num_classes must be gt 0
    with pytest.raises(ValidationError):
        DataConfig(data_dir="dummy/path", num_classes=0)

def test_model_config_defaults():
    config = ModelConfig()
    assert config.num_annotators == 1
    assert config.in_channels == 3

def test_train_config_defaults():
    config = TrainConfig()
    assert config.epochs == 100
    assert config.lr == 1e-3

def test_experiment_metadata_defaults():
    metadata = ExperimentMetadata()
    assert metadata.name == "base_experiment"
    assert metadata.version == 1

def test_experiment_config_from_yaml(tmp_path):
    yaml_content = """
data:
  data_dir: "my_data"
  batch_size: 32
model:
  model_name: "MV"
  num_classes: 2
training:
  epochs: 50
experiment:
  name: "test_exp"
  version: 2
"""
    yaml_file = tmp_path / "config.yaml"
    yaml_file.write_text(yaml_content)

    config = ExperimentConfig.from_yaml(yaml_file)
    
    assert config.data.data_dir == "my_data"
    assert config.data.batch_size == 32
    assert config.model.num_classes == 2
    assert config.training.epochs == 50
    assert config.experiment.name == "test_exp"
    assert config.experiment.version == 2

def test_experiment_config_from_yaml_not_found():
    with pytest.raises(FileNotFoundError):
        ExperimentConfig.from_yaml("non_existent_file.yaml")
