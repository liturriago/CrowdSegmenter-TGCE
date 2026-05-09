import json
import pytest
import torch
import yaml
from pathlib import Path
from unittest.mock import patch

from crowdsegmenter.training.tgce_ssps_tuner import TGCESSPSTuner

@pytest.fixture
def tiny_config_path(tmp_path):
    """Creates a minimal configuration YAML for testing."""
    config = {
        "data": {
            "data_dir": str(tmp_path),
            "partitions": ["train", "val", "test"],
            "num_classes": 2,
            "num_annotators": 2,
            "image_size": [32, 32],
            "batch_size": 1,
            "ignored_value": 0.6
        },
        "model": {
            "num_classes": 2,
            "num_annotators": 2,
            "in_channels": 3,
            "image_size": 32,
            "pretrained": False,
            "decoder_channels": [16, 8, 8, 8]
        },
        "training": {
            "epochs": 1,
            "lr": 0.01,
            "transfer_lr": 0.01,
            "threshold": 0.5,
            "probabilistic_thresholds": [0.5],
            "num_classes": 2,
            "num_annotators": 2,
            "ignored_value": 0.6,
            "smooth": 1e-8,
            "epochs_phases": [],
            "gamma": 1.0,
            "tgce_q": 0.5,
            "tgce_lambda": 1.0,
            "n_epochs_per_trial": 1,
            "q_search_low": 0.1,
            "q_search_high": 0.99,
            "q_search_step": 0.01,
            "seed": 42,
            "device": "cpu"
        },
        "experiment": {
            "name": "test",
            "version": 1,
            "output_dir": str(tmp_path / "outputs"),
            "save_results": False
        }
    }
    config_file = tmp_path / "test_config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config, f)
    return str(config_file)

@pytest.fixture
def mock_loader():
    """Provides a dummy dataloader yielding tiny tensors to bypass real data IO."""
    class DummyLoader:
        def __init__(self):
            self.dataset = [1]
        def __iter__(self):
            # 1 batch, 3 channels, 32x32
            images = torch.randn(1, 3, 32, 32)
            # R=2 * K=2 = 4 channels
            masks = torch.randn(1, 4, 32, 32)
            # R=2 annotators
            anns_ids = torch.tensor([[1.0, 1.0]])
            yield (images, masks, anns_ids)
        def __len__(self):
            return 1
    return DummyLoader()

@pytest.fixture
def patch_dataloader(mock_loader):
    """Mocks the CrowdSegmenterDataLoader to return the dummy loader."""
    with patch("crowdsegmenter.training.tgce_ssps_tuner.CrowdSegmenterDataLoader") as MockLoader:
        instance = MockLoader.return_value
        instance.get_split_loaders.return_value = (mock_loader, mock_loader, mock_loader)
        yield

def test_tuner_initialization(tiny_config_path, patch_dataloader, tmp_path):
    """Verify that tuner attributes are set correctly from arguments and config."""
    output_dir = tmp_path / "optuna_out"
    tuner = TGCESSPSTuner(
        config_path=tiny_config_path,
        n_trials=5,
        n_epochs_per_trial=2,
        study_name="test_study",
        output_dir=str(output_dir)
    )
    
    assert tuner.n_trials == 5
    assert tuner.n_epochs_per_trial == 2
    assert tuner.study_name == "test_study"
    assert tuner.output_dir == output_dir
    assert tuner.train_loader is not None
    assert tuner.val_loader is not None

def test_objective_returns_float_in_valid_range(tiny_config_path, patch_dataloader, tmp_path):
    """Run 1 trial with 1 epoch and assert objective returns a float in [0, 1]."""
    tuner = TGCESSPSTuner(
        config_path=tiny_config_path,
        n_trials=1,
        n_epochs_per_trial=1,
        output_dir=str(tmp_path)
    )
    
    trial = tuner.study.ask()
    result = tuner.objective(trial)
    
    assert isinstance(result, float)
    assert 0.0 <= result <= 1.0

def test_run_returns_expected_keys(tiny_config_path, patch_dataloader, tmp_path):
    """Run 2 trials and verify the returned dictionary structure."""
    output_dir = tmp_path / "optuna_out"
    tuner = TGCESSPSTuner(
        config_path=tiny_config_path,
        n_trials=2,
        n_epochs_per_trial=1,
        output_dir=str(output_dir)
    )
    
    results = tuner.run()
    
    assert isinstance(results, dict)
    assert "best_q" in results
    assert "best_val_dice" in results
    assert "trials" in results
    assert len(results["trials"]) == 2

def test_best_q_is_within_search_bounds(tiny_config_path, patch_dataloader, tmp_path):
    """Run trials and verify the best_q respects the configured search boundaries."""
    output_dir = tmp_path / "optuna_out"
    tuner = TGCESSPSTuner(
        config_path=tiny_config_path,
        n_trials=2,
        n_epochs_per_trial=1,
        output_dir=str(output_dir)
    )
    results = tuner.run()
    
    best_q = results["best_q"]
    assert 0.1 <= best_q <= 0.99

def test_results_json_is_saved(tiny_config_path, patch_dataloader, tmp_path):
    """Verify that results.json is dumped correctly after run()."""
    output_dir = tmp_path / "optuna_out"
    tuner = TGCESSPSTuner(
        config_path=tiny_config_path,
        n_trials=1,
        n_epochs_per_trial=1,
        output_dir=str(output_dir)
    )
    tuner.run()
    
    json_path = output_dir / "results.json"
    assert json_path.exists()
    
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    assert "best_q" in data
    assert "best_val_dice" in data
    assert "trials" in data
