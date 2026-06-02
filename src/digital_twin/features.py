from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .config import PhysicsConfig
from .data import (
    bragg_columns,
    load_crack_workbook,
    metadata_for_file,
    parse_experiment_id,
)
from .paths import INTERROGATOR_DIR, MEDIAN_FILTER_DIR, MODEL_DIR, PEAK_GROUPS_DIR
from .preprocessing import (
    active_channels_from_filtered,
    filtered_output_path,
    filtered_columns,
    group_output_path,
)


TIMESTEPS = 50
FEATURES = 9
SENSOR_SLOTS = 3


@dataclass
class FeatureDataset:
    x: np.ndarray
    y: np.ndarray
    meta: pd.DataFrame

    @property
    def is_empty(self) -> bool:
        return len(self.y) == 0


def _resample_to_grid(
    times: np.ndarray,
    values: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    mask = np.isfinite(times) & np.isfinite(values)
    if mask.sum() == 0:
        return np.zeros(len(grid), dtype=float)
    clean_times = times[mask]
    clean_values = values[mask]
    order = np.argsort(clean_times)
    clean_times = clean_times[order]
    clean_values = clean_values[order]
    unique_times, unique_indices = np.unique(clean_times, return_index=True)
    clean_values = clean_values[unique_indices]
    if len(unique_times) == 1:
        return np.full(len(grid), clean_values[0], dtype=float)
    return np.interp(grid, unique_times, clean_values)


def _group_frame_for_channel(source_path: Path, channel: int) -> pd.DataFrame | None:
    path = group_output_path(source_path, channel, PEAK_GROUPS_DIR)
    if not path.exists():
        return None
    frame = pd.read_csv(path)
    if frame.empty:
        return None
    return frame


def build_feature_dataset(
    files: list[Path] | None = None,
    workbook: dict[str, pd.DataFrame] | None = None,
    filtered_dir: Path = MEDIAN_FILTER_DIR,
    require_complete_groups: bool = True,
) -> FeatureDataset:
    files = files or sorted(INTERROGATOR_DIR.glob("*-interrogator.txt"))
    workbook = workbook or load_crack_workbook()
    samples: list[np.ndarray] = []
    labels: list[int] = []
    meta_rows: list[dict[str, object]] = []

    for source_path in files:
        experiment_id = parse_experiment_id(source_path)
        metadata = metadata_for_file(source_path, workbook)
        if metadata is None or metadata.empty:
            continue

        filtered_path = filtered_output_path(source_path, filtered_dir)
        if not filtered_path.exists():
            continue
        filtered = pd.read_csv(filtered_path)
        channels = active_channels_from_filtered(filtered)
        if not channels and not any(col.endswith("_delta_nm") for col in filtered.columns):
            channels = sorted(
                int(col.split("_")[1])
                for col in filtered_columns(filtered)
                if col.endswith("_filtered_nm")
            )
        if not channels:
            continue

        channel_groups = {
            channel: _group_frame_for_channel(source_path, channel) for channel in channels
        }
        channel_groups = {
            channel: frame for channel, frame in channel_groups.items() if frame is not None
        }
        if not channel_groups:
            continue

        for group_index in range(min(12, len(metadata))):
            complete_channels: list[int] = []
            group_windows: list[tuple[float, float]] = []
            for channel, groups in channel_groups.items():
                group_rows = groups[groups["group_index"] == group_index]
                if group_rows.empty:
                    continue
                is_complete = (
                    int(group_rows["is_complete_channel"].iloc[0]) == 1
                    and int(group_rows["is_complete_group"].min()) == 1
                    and len(group_rows) == 10
                )
                if require_complete_groups and not is_complete:
                    continue
                complete_channels.append(channel)
                group_windows.append(
                    (
                        float(group_rows["effective_time_s"].min()),
                        float(group_rows["effective_time_s"].max()),
                    )
                )

            if not complete_channels:
                continue
            start = min(item[0] for item in group_windows)
            stop = max(item[1] for item in group_windows)
            if not np.isfinite(start) or not np.isfinite(stop) or stop <= start:
                continue
            grid = np.linspace(start, stop, TIMESTEPS)
            window = filtered[
                (filtered["effective_time_s"] >= start)
                & (filtered["effective_time_s"] <= stop)
            ]
            if len(window) < 2:
                continue

            tensor = np.zeros((TIMESTEPS, FEATURES), dtype=float)
            slot_used = [0, 0, 0]
            for slot in range(SENSOR_SLOTS):
                channel = slot + 1
                if channel not in complete_channels:
                    continue
                delta_col = f"wl_{channel}_delta_nm"
                strain_col = f"wl_{channel}_strain"
                derivative_col = f"wl_{channel}_derivative_nm_per_s"
                if not {delta_col, strain_col, derivative_col}.issubset(window.columns):
                    continue
                times = window["effective_time_s"].to_numpy(dtype=float)
                base = slot * 3
                tensor[:, base] = _resample_to_grid(
                    times, window[delta_col].to_numpy(dtype=float), grid
                )
                tensor[:, base + 1] = _resample_to_grid(
                    times, window[strain_col].to_numpy(dtype=float), grid
                )
                tensor[:, base + 2] = _resample_to_grid(
                    times, window[derivative_col].to_numpy(dtype=float), grid
                )
                slot_used[slot] = 1

            if not any(slot_used):
                continue
            label = int(metadata.loc[metadata["group_index"] == group_index, "crack_label"].iloc[0])
            meta = metadata.loc[metadata["group_index"] == group_index].iloc[0]
            samples.append(tensor)
            labels.append(label)
            meta_rows.append(
                {
                    "source_name": source_path.name,
                    "experiment_id": experiment_id.normalized_stem,
                    "group_index": group_index,
                    "pressure_bar": float(meta["pressure_bar"]),
                    "force_n": float(meta["force_n"]) if pd.notna(meta["force_n"]) else np.nan,
                    "span_cm": float(meta["span_cm"]),
                    "layers": float(meta["layers"]),
                    "displacement_mm": (
                        float(meta["displacement_mm"])
                        if pd.notna(meta["displacement_mm"])
                        else np.nan
                    ),
                    "crack_marker": meta["crack_marker"],
                    "crack_label": label,
                    "slot_1_used": slot_used[0],
                    "slot_2_used": slot_used[1],
                    "slot_3_used": slot_used[2],
                    "window_start_s": start,
                    "window_stop_s": stop,
                }
            )

    if not samples:
        return FeatureDataset(
            x=np.zeros((0, TIMESTEPS, FEATURES), dtype=float),
            y=np.zeros((0,), dtype=int),
            meta=pd.DataFrame(),
        )
    return FeatureDataset(
        x=np.stack(samples),
        y=np.asarray(labels, dtype=int),
        meta=pd.DataFrame(meta_rows),
    )


def split_by_file(
    meta: pd.DataFrame,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    seed: int = 2026,
) -> dict[str, np.ndarray]:
    if meta.empty:
        return {"train": np.array([], dtype=int), "val": np.array([], dtype=int), "test": np.array([], dtype=int)}

    file_ids = np.asarray(sorted(meta["source_name"].unique()))
    rng = np.random.default_rng(seed)
    rng.shuffle(file_ids)
    n_files = len(file_ids)

    if n_files < 3:
        train_files = set(file_ids)
        val_files: set[str] = set()
        test_files: set[str] = set()
    else:
        n_test = max(1, int(round(n_files * test_fraction)))
        n_val = max(1, int(round(n_files * val_fraction)))
        if n_test + n_val >= n_files:
            n_test = 1
            n_val = 1
        test_files = set(file_ids[:n_test])
        val_files = set(file_ids[n_test : n_test + n_val])
        train_files = set(file_ids[n_test + n_val :])

    source_names = meta["source_name"].to_numpy()
    return {
        "train": np.flatnonzero(np.isin(source_names, list(train_files))),
        "val": np.flatnonzero(np.isin(source_names, list(val_files))),
        "test": np.flatnonzero(np.isin(source_names, list(test_files))),
    }


def fit_standardizer(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if len(x) == 0:
        return np.zeros((FEATURES,), dtype=float), np.ones((FEATURES,), dtype=float)
    mean = np.nanmean(x, axis=(0, 1))
    std = np.nanstd(x, axis=(0, 1))
    std = np.where(std > 1e-12, std, 1.0)
    return mean.astype(float), std.astype(float)


def apply_standardizer(
    x: np.ndarray,
    mean: np.ndarray,
    std: np.ndarray,
) -> np.ndarray:
    return np.nan_to_num((x - mean.reshape(1, 1, -1)) / std.reshape(1, 1, -1))


def save_standardizer(
    mean: np.ndarray,
    std: np.ndarray,
    path: Path = MODEL_DIR / "feature_standardizer.json",
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump({"mean": mean.tolist(), "std": std.tolist()}, handle, indent=2)


def load_standardizer(
    path: Path = MODEL_DIR / "feature_standardizer.json",
) -> tuple[np.ndarray, np.ndarray] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return np.asarray(payload["mean"], dtype=float), np.asarray(payload["std"], dtype=float)


def physics_strain(
    force_n: float,
    span_m: float,
    y_m: float,
    config: PhysicsConfig,
) -> float:
    denominator = config.young_modulus_pa * config.width_m * (config.thickness_m**3)
    if denominator <= 0 or not np.isfinite(force_n):
        return np.nan
    return 3.0 * force_n * span_m * y_m / denominator


def build_physics_residuals(dataset: FeatureDataset, config: PhysicsConfig) -> np.ndarray:
    residuals = np.zeros((len(dataset.y), TIMESTEPS, SENSOR_SLOTS), dtype=float)
    if dataset.is_empty or not config.is_complete():
        return residuals
    for row_index, meta in dataset.meta.reset_index(drop=True).iterrows():
        force = float(meta["force_n"])
        span_m = float(meta["span_cm"]) / 100.0
        for slot in range(SENSOR_SLOTS):
            if int(meta.get(f"slot_{slot + 1}_used", 0)) != 1:
                continue
            eps_physics = physics_strain(force, span_m, config.y_m[slot], config)
            if not np.isfinite(eps_physics):
                continue
            strain_feature = dataset.x[row_index, :, slot * 3 + 1]
            residuals[row_index, :, slot] = strain_feature - eps_physics
    return residuals
