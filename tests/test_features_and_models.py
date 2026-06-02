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


@pytest.mark.skipif(importlib.util.find_spec("tensorflow") is None, reason="TensorFlow missing")
def test_cnn_model_output_shape():
    from digital_twin.models import build_cnn

    model = build_cnn()
    assert model.output_shape == (None, 1)
