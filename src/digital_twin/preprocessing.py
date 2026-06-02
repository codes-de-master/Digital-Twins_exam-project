from __future__ import annotations

import math
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import find_peaks

from .data import bragg_columns, estimate_lambda0, read_interrogator_file, safe_output_stem
from .paths import MEDIAN_FILTER_DIR, PEAK_GROUPS_DIR, PEAKS_DIR

PHOTO_ELASTIC_COEFFICIENT = 0.22
BLANK_DELTA_THRESHOLD_NM = 0.5
PEAK_COLUMNS = [
    "channel",
    "peak_order",
    "row_index",
    "sample_index",
    "timestamp",
    "time_s",
    "effective_time_s",
    "wavelength_nm",
    "delta_nm",
    "prominence",
    "distance_samples",
    "distance_seconds",
    "prominence_threshold",
]


def normalize_window(window: int) -> int:
    window = max(1, int(window))
    return window if window % 2 else window + 1


def filtered_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if re.fullmatch(r"wl_\d+_filtered_nm", col)]


def delta_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if re.fullmatch(r"wl_\d+_delta_nm", col)]


def channel_from_column(column: str) -> int:
    match = re.search(r"wl_(\d+)_", column)
    if not match:
        raise ValueError(f"Cannot parse channel from column {column!r}")
    return int(match.group(1))


def delta_amplitude(values: pd.Series | np.ndarray) -> float:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if len(array) == 0:
        return 0.0
    return float(np.nanmax(np.abs(array)))


def active_raw_columns(
    frame: pd.DataFrame,
    lambda0_by_col: dict[str, float] | None = None,
    threshold_nm: float = BLANK_DELTA_THRESHOLD_NM,
) -> list[str]:
    lambda0_by_col = lambda0_by_col or estimate_lambda0(frame)
    columns: list[str] = []
    for column in bragg_columns(frame):
        lambda0 = lambda0_by_col.get(column, float("nan"))
        if not np.isfinite(lambda0):
            continue
        if delta_amplitude(frame[column] - lambda0) > threshold_nm:
            columns.append(column)
    return columns


def active_channels_from_filtered(
    frame: pd.DataFrame,
    threshold_nm: float = BLANK_DELTA_THRESHOLD_NM,
) -> list[int]:
    channels: list[int] = []
    for column in delta_columns(frame):
        if delta_amplitude(frame[column]) > threshold_nm:
            channels.append(channel_from_column(column))
    return channels


def strain_from_delta(delta_nm: pd.Series | np.ndarray, lambda0_nm: float) -> np.ndarray:
    if not np.isfinite(lambda0_nm) or lambda0_nm == 0:
        return np.full(len(delta_nm), np.nan)
    return np.asarray(delta_nm, dtype=float) / (
        lambda0_nm * (1.0 - PHOTO_ELASTIC_COEFFICIENT)
    )


def add_unfiltered_signals(
    frame: pd.DataFrame,
    lambda0_by_col: dict[str, float] | None = None,
) -> pd.DataFrame:
    lambda0_by_col = lambda0_by_col or estimate_lambda0(frame)
    result = frame.copy()
    for column in bragg_columns(frame):
        lambda0 = lambda0_by_col.get(column, float("nan"))
        channel = channel_from_column(column)
        delta = result[column] - lambda0
        result[f"wl_{channel}_delta_raw_nm"] = delta
        result[f"wl_{channel}_strain_raw"] = strain_from_delta(delta, lambda0)
    return result


def _safe_derivative(values: np.ndarray, times: np.ndarray) -> np.ndarray:
    if len(values) < 2:
        return np.zeros(len(values), dtype=float)
    values = pd.Series(values, dtype="float64").interpolate(limit_direction="both")
    finite_times = pd.Series(times, dtype="float64").interpolate(limit_direction="both")
    if finite_times.nunique(dropna=True) < 2:
        return np.zeros(len(values), dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        derivative = np.gradient(values.to_numpy(), finite_times.to_numpy())
    return np.nan_to_num(derivative, nan=0.0, posinf=0.0, neginf=0.0)


def apply_median_filter(
    frame: pd.DataFrame,
    window: int = 5,
    lambda0_by_col: dict[str, float] | None = None,
) -> pd.DataFrame:
    window = normalize_window(window)
    lambda0_by_col = lambda0_by_col or estimate_lambda0(frame)
    keep_columns = ["timestamp", "time_s", "effective_time_s", "sample_index", *bragg_columns(frame)]
    result = frame[keep_columns].copy()
    times = result["effective_time_s"].to_numpy(dtype=float)
    extra_columns: dict[str, np.ndarray | pd.Series] = {}

    for column in bragg_columns(frame):
        channel = channel_from_column(column)
        filtered_col = f"wl_{channel}_filtered_nm"
        delta_col = f"wl_{channel}_delta_nm"
        strain_col = f"wl_{channel}_strain"
        derivative_col = f"wl_{channel}_derivative_nm_per_s"
        lambda0 = lambda0_by_col.get(column, float("nan"))

        filtered_values = (
            frame[column]
            .rolling(window=window, center=True, min_periods=1)
            .median()
        )
        delta_values = filtered_values - lambda0
        extra_columns[filtered_col] = filtered_values
        extra_columns[delta_col] = delta_values
        extra_columns[strain_col] = strain_from_delta(delta_values, lambda0)
        extra_columns[derivative_col] = _safe_derivative(
            delta_values.to_numpy(dtype=float), times
        )

    if extra_columns:
        result = pd.concat([result, pd.DataFrame(extra_columns, index=result.index)], axis=1)

    result.attrs.update(frame.attrs)
    result.attrs["median_window"] = window
    result.attrs["lambda0_by_col"] = lambda0_by_col
    return result


def filtered_output_path(source_path: Path, output_dir: Path = MEDIAN_FILTER_DIR) -> Path:
    return output_dir / f"{safe_output_stem(source_path)}_median.csv"


def save_median_filtered_file(
    source_path: Path,
    output_dir: Path = MEDIAN_FILTER_DIR,
    window: int = 5,
) -> tuple[pd.DataFrame, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = read_interrogator_file(source_path)
    filtered = apply_median_filter(raw, window=window)
    path = filtered_output_path(source_path, output_dir)
    filtered.to_csv(path, index=False)
    return filtered, path


def infer_distance_samples(times: pd.Series | np.ndarray, distance_seconds: float) -> int:
    times_array = np.asarray(times, dtype=float)
    times_array = times_array[np.isfinite(times_array)]
    if len(times_array) < 2:
        return 1
    diffs = np.diff(times_array)
    positive_diffs = diffs[diffs > 0]
    if len(positive_diffs) == 0:
        return 1
    median_dt = float(np.median(positive_diffs))
    if median_dt <= 0:
        return 1
    return max(1, int(math.ceil(distance_seconds / median_dt)))


def find_wavelength_peaks(
    filtered_frame: pd.DataFrame,
    distance_seconds: float = 0.2,
    prominence: float = 0.1,
    blank_threshold_nm: float = BLANK_DELTA_THRESHOLD_NM,
) -> pd.DataFrame:
    frame = filtered_frame.reset_index(drop=True)
    distance_samples = infer_distance_samples(frame["effective_time_s"], distance_seconds)
    rows: list[dict[str, object]] = []

    for delta_col in delta_columns(frame):
        channel = channel_from_column(delta_col)
        wavelength_col = f"wl_{channel}_filtered_nm"
        signal = frame[delta_col].to_numpy(dtype=float)
        if np.isfinite(signal).sum() < 3:
            continue
        if delta_amplitude(signal) <= blank_threshold_nm:
            continue
        clean_signal = pd.Series(signal).interpolate(limit_direction="both").to_numpy()
        peak_indices, properties = find_peaks(
            clean_signal,
            distance=distance_samples,
            prominence=prominence,
        )
        prominences = properties.get("prominences", np.full(len(peak_indices), np.nan))
        for order, row_index in enumerate(peak_indices):
            sample = frame.iloc[int(row_index)]
            rows.append(
                {
                    "channel": channel,
                    "peak_order": order,
                    "row_index": int(row_index),
                    "sample_index": int(sample["sample_index"]),
                    "timestamp": sample["timestamp"],
                    "time_s": float(sample["time_s"]),
                    "effective_time_s": float(sample["effective_time_s"]),
                    "wavelength_nm": float(sample[wavelength_col]),
                    "delta_nm": float(sample[delta_col]),
                    "prominence": float(prominences[order]),
                    "distance_samples": distance_samples,
                    "distance_seconds": float(distance_seconds),
                    "prominence_threshold": float(prominence),
                }
            )

    if not rows:
        return pd.DataFrame(columns=PEAK_COLUMNS)
    return (
        pd.DataFrame(rows, columns=PEAK_COLUMNS)
        .sort_values(["channel", "peak_order"])
        .reset_index(drop=True)
    )


def peaks_output_path(source_path: Path, output_dir: Path = PEAKS_DIR) -> Path:
    return output_dir / f"{safe_output_stem(source_path)}_peaks.csv"


def save_peaks_file(
    source_path: Path,
    filtered_frame: pd.DataFrame | None = None,
    output_dir: Path = PEAKS_DIR,
    distance_seconds: float = 0.2,
    prominence: float = 0.1,
    median_window: int = 5,
) -> tuple[pd.DataFrame, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    filtered_frame = (
        filtered_frame
        if filtered_frame is not None
        else apply_median_filter(read_interrogator_file(source_path), window=median_window)
    )
    peaks = find_wavelength_peaks(
        filtered_frame,
        distance_seconds=distance_seconds,
        prominence=prominence,
    )
    path = peaks_output_path(source_path, output_dir)
    peaks.to_csv(path, index=False)
    return peaks, path


def group_output_path(
    source_path: Path,
    channel: int,
    output_dir: Path = PEAK_GROUPS_DIR,
) -> Path:
    return output_dir / f"{safe_output_stem(source_path)}_wl{channel}_groups.csv"


def build_peak_groups(
    peaks: pd.DataFrame,
    metadata: pd.DataFrame,
    peaks_per_group: int = 10,
    expected_groups: int = 12,
) -> dict[int, pd.DataFrame]:
    if metadata is None or metadata.empty:
        return {}
    if peaks.empty or "channel" not in peaks.columns:
        return {}
    groups_by_channel: dict[int, pd.DataFrame] = {}
    for channel_value in sorted(peaks["channel"].dropna().astype(int).unique()):
        channel = int(channel_value)
        channel_peaks = peaks[peaks["channel"] == channel].sort_values("peak_order")
        total_peaks = len(channel_peaks)
        usable = channel_peaks.head(peaks_per_group * expected_groups)
        rows: list[dict[str, object]] = []
        for group_index in range(expected_groups):
            start = group_index * peaks_per_group
            stop = start + peaks_per_group
            group_peaks = usable.iloc[start:stop]
            if group_peaks.empty:
                continue
            if group_index >= len(metadata):
                continue
            meta = metadata.iloc[group_index]
            for peak_in_group, (_, peak) in enumerate(group_peaks.iterrows()):
                rows.append(
                    {
                        "experiment_id": meta["experiment_id"],
                        "channel": channel,
                        "group_index": group_index,
                        "peak_in_group": peak_in_group,
                        "group_peak_count": len(group_peaks),
                        "is_complete_group": int(len(group_peaks) == peaks_per_group),
                        "is_complete_channel": int(
                            total_peaks >= peaks_per_group * expected_groups
                        ),
                        "channel_peak_count": total_peaks,
                        "extra_peaks_ignored": max(
                            0, total_peaks - peaks_per_group * expected_groups
                        ),
                        "pressure_bar": meta["pressure_bar"],
                        "force_n": meta["force_n"],
                        "displacement_mm": meta["displacement_mm"],
                        "layers": meta["layers"],
                        "span_cm": meta["span_cm"],
                        "crack_marker": meta["crack_marker"],
                        "crack_label": int(meta["crack_label"]),
                        "row_index": int(peak["row_index"]),
                        "sample_index": int(peak["sample_index"]),
                        "timestamp": peak["timestamp"],
                        "time_s": float(peak["time_s"]),
                        "effective_time_s": float(peak["effective_time_s"]),
                        "wavelength_nm": float(peak["wavelength_nm"]),
                        "delta_nm": float(peak["delta_nm"]),
                        "prominence": float(peak["prominence"]),
                    }
                )
        groups_by_channel[channel] = pd.DataFrame(rows)
    return groups_by_channel


def save_peak_groups_file(
    source_path: Path,
    peaks: pd.DataFrame,
    metadata: pd.DataFrame,
    output_dir: Path = PEAK_GROUPS_DIR,
) -> dict[int, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    groups_by_channel = build_peak_groups(peaks, metadata)
    paths: dict[int, Path] = {}
    for channel, group_frame in groups_by_channel.items():
        path = group_output_path(source_path, channel, output_dir)
        group_frame.to_csv(path, index=False)
        paths[channel] = path
    return paths
