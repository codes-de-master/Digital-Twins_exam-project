import numpy as np
import pandas as pd

from digital_twin.preprocessing import (
    BLANK_DELTA_THRESHOLD_NM,
    PHOTO_ELASTIC_COEFFICIENT,
    add_unfiltered_signals,
    apply_median_filter,
    active_raw_columns,
    build_peak_groups,
    find_wavelength_peaks,
    strain_from_delta,
)


def toy_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": ["t"] * 8,
            "time_s": [0, 0, 0, 0, 1, 1, 1, 1],
            "effective_time_s": [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, 1.75],
            "sample_index": range(8),
            "wl_1_nm": [1550.0, 1550.1, 1559.0, 1550.2, 1550.3, 1550.4, 1550.5, 1550.6],
        }
    )


def test_strain_formula():
    result = strain_from_delta(np.array([0.78]), 1550.0)
    expected = 0.78 / (1550.0 * (1.0 - PHOTO_ELASTIC_COEFFICIENT))
    np.testing.assert_allclose(result, [expected])


def test_median_filter_preserves_rows_and_adds_signals():
    frame = toy_frame()
    filtered = apply_median_filter(frame, window=3, lambda0_by_col={"wl_1_nm": 1550.0})
    assert len(filtered) == len(frame)
    assert "wl_1_filtered_nm" in filtered.columns
    assert "wl_1_delta_nm" in filtered.columns
    assert "wl_1_strain" in filtered.columns
    assert "wl_1_derivative_nm_per_s" in filtered.columns
    assert filtered["wl_1_filtered_nm"].iloc[2] < frame["wl_1_nm"].iloc[2]


def test_unfiltered_signals_use_lambda0():
    frame = toy_frame()
    out = add_unfiltered_signals(frame, {"wl_1_nm": 1550.0})
    assert out["wl_1_delta_raw_nm"].iloc[0] == 0.0
    assert out["wl_1_delta_raw_nm"].iloc[1] > 0.0


def test_peak_detection_columns():
    x = np.linspace(0, 8 * np.pi, 200)
    frame = pd.DataFrame(
        {
            "timestamp": ["t"] * len(x),
            "time_s": np.floor(x),
            "effective_time_s": np.arange(len(x)) / 10.0,
            "sample_index": np.arange(len(x)),
            "wl_1_nm": 1550 + np.sin(x),
            "wl_1_filtered_nm": 1550 + np.sin(x),
            "wl_1_delta_nm": np.sin(x),
            "wl_1_strain": np.sin(x) / 1550,
            "wl_1_derivative_nm_per_s": np.gradient(np.sin(x)),
        }
    )
    peaks = find_wavelength_peaks(frame, distance_seconds=0.2, prominence=0.1)
    assert {
        "channel",
        "peak_order",
        "effective_time_s",
        "delta_nm",
        "prominence",
    }.issubset(peaks.columns)
    assert len(peaks) > 0


def test_blank_channel_threshold_suppresses_low_delta_channel():
    frame = pd.DataFrame(
        {
            "timestamp": ["t"] * 20,
            "time_s": np.arange(20, dtype=float),
            "effective_time_s": np.arange(20, dtype=float),
            "sample_index": np.arange(20),
            "wl_1_nm": 1550.0 + np.linspace(0.0, BLANK_DELTA_THRESHOLD_NM, 20),
        }
    )
    assert active_raw_columns(frame, {"wl_1_nm": 1550.0}) == []

    filtered = apply_median_filter(frame, window=3, lambda0_by_col={"wl_1_nm": 1550.0})
    peaks = find_wavelength_peaks(filtered, distance_seconds=0.2, prominence=0.1)
    assert peaks.empty
    assert "channel" in peaks.columns


def test_peak_group_labels_any_marker_is_crack():
    peaks = pd.DataFrame(
        {
            "channel": [1] * 120,
            "peak_order": range(120),
            "row_index": range(120),
            "sample_index": range(120),
            "timestamp": ["t"] * 120,
            "time_s": np.arange(120, dtype=float),
            "effective_time_s": np.arange(120, dtype=float),
            "wavelength_nm": np.ones(120),
            "delta_nm": np.ones(120),
            "prominence": np.ones(120),
        }
    )
    metadata = pd.DataFrame(
        {
            "group_index": range(12),
            "experiment_id": ["15cm-12layers-1"] * 12,
            "pressure_bar": np.linspace(1.0, 2.0, 12),
            "layers": [12] * 12,
            "span_cm": [15] * 12,
            "force_n": np.arange(12),
            "displacement_mm": np.arange(12),
            "crack_marker": ["", "2", *[""] * 10],
            "crack_label": [0, 1, *[0] * 10],
        }
    )
    groups = build_peak_groups(peaks, metadata)[1]
    assert groups[groups["group_index"] == 0]["crack_label"].iloc[0] == 0
    assert groups[groups["group_index"] == 1]["crack_label"].iloc[0] == 1
    assert groups["is_complete_channel"].min() == 1
