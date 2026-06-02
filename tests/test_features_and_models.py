import importlib.util

import numpy as np
import pandas as pd
import pytest

from digital_twin.config import PhysicsConfig, default_physics_config
from digital_twin.features import FeatureDataset, build_physics_residuals, split_by_file


def test_split_by_file_uses_file_boundaries():
    meta = pd.DataFrame(
        {
            "source_name": ["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3,
            "group_index": list(range(3)) * 4,
        }
    )
    splits = split_by_file(meta, seed=123)
    for split_indices in splits.values():
        split_files = set(meta.iloc[split_indices]["source_name"])
        for other_indices in splits.values():
            if split_indices is other_indices:
                continue
            assert split_files.isdisjoint(set(meta.iloc[other_indices]["source_name"]))


def test_split_by_file_default_seed_matches_2026():
    meta = pd.DataFrame(
        {
            "source_name": ["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3,
            "group_index": list(range(3)) * 4,
        }
    )
    default_splits = split_by_file(meta)
    explicit_splits = split_by_file(meta, seed=2026)
    for name in default_splits:
        np.testing.assert_array_equal(default_splits[name], explicit_splits[name])


def test_default_physics_config_uses_requested_values():
    config = default_physics_config()
    assert config.young_modulus_pa == 22e9
    assert config.width_m == 0.034
    assert config.thickness_m == 0.004
    assert config.y_m == (0.00145, 0.00145, 0.00145)


def test_physics_residuals_respect_missing_sensor_mask():
    x = np.zeros((1, 50, 9), dtype=float)
    x[0, :, 1] = 0.02
    x[0, :, 4] = 0.03
    dataset = FeatureDataset(
        x=x,
        y=np.array([1]),
        meta=pd.DataFrame(
            {
                "force_n": [100.0],
                "span_cm": [10.0],
                "slot_1_used": [1],
                "slot_2_used": [0],
                "slot_3_used": [0],
            }
        ),
    )
    config = PhysicsConfig(
        young_modulus_pa=1.0e9,
        width_m=0.01,
        thickness_m=0.002,
        y_m=(0.001, 0.001, 0.001),
    )
    residuals = build_physics_residuals(dataset, config)
    assert residuals.shape == (1, 50, 3)
    assert np.any(residuals[:, :, 0] != 0.0)
    assert np.all(residuals[:, :, 1] == 0.0)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch missing")
def test_pytorch_cnn_model_output_shape():
    import torch

    from digital_twin.models import build_cnn

    model = build_cnn()
    x = torch.zeros((2, 50, 9), dtype=torch.float32)
    out = model(x)
    assert tuple(out.shape) == (2,)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch missing")
def test_pytorch_physics_head_model_output_shape():
    import torch

    from digital_twin.models import build_physics_head_cnn

    model = build_physics_head_cnn()
    signal = torch.zeros((2, 50, 9), dtype=torch.float32)
    residual = torch.zeros((2, 50, 3), dtype=torch.float32)
    out = model(signal, residual)
    assert tuple(out.shape) == (2,)


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch missing")
def test_pytorch_cnn_training_smoke(tmp_path):
    from digital_twin.models import train_cnn

    rng = np.random.default_rng(2026)
    dataset = FeatureDataset(
        x=rng.normal(size=(12, 50, 9)).astype(float),
        y=np.array([0, 1, 0] * 4),
        meta=pd.DataFrame(
            {
                "source_name": ["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3,
                "group_index": list(range(3)) * 4,
            }
        ),
    )
    result = train_cnn(
        dataset,
        epochs=1,
        batch_size=4,
        seed=2026,
        model_dir=tmp_path,
        crack_class_weight=2.0,
    )
    assert result.model_path.suffix == ".pt"
    assert result.model_path.exists()
    assert result.confusion_matrix


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="PyTorch missing")
def test_pytorch_physics_head_training_smoke(tmp_path):
    from digital_twin.models import load_training_metadata, train_physics_head

    rng = np.random.default_rng(2026)
    dataset = FeatureDataset(
        x=rng.normal(size=(12, 50, 9)).astype(float),
        y=np.array([0, 1, 0] * 4),
        meta=pd.DataFrame(
            {
                "source_name": ["a"] * 3 + ["b"] * 3 + ["c"] * 3 + ["d"] * 3,
                "group_index": list(range(3)) * 4,
                "force_n": np.linspace(100.0, 200.0, 12),
                "span_cm": [15.0] * 12,
                "slot_1_used": [1] * 12,
                "slot_2_used": [1] * 12,
                "slot_3_used": [1] * 12,
            }
        ),
    )
    config = PhysicsConfig(
        young_modulus_pa=22e9,
        width_m=0.034,
        thickness_m=0.004,
        y_m=(0.00145, 0.00145, 0.00145),
    )
    result = train_physics_head(
        dataset,
        config,
        epochs=1,
        batch_size=4,
        seed=2026,
        model_dir=tmp_path,
        crack_class_weight=3.0,
    )
    assert result.model_path.suffix == ".pt"
    assert result.model_path.exists()
    assert result.confusion_matrix
    metadata = load_training_metadata(tmp_path / "physics_training_metadata.json")
    assert metadata["crack_class_weight"] == 3.0
