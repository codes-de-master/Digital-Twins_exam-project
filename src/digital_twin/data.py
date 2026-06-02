from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .paths import HANDHELD_XLSX, INTERROGATOR_DIR


EXPERIMENT_RE = re.compile(
    r"(?P<span>\d+)\s*cm\s*[- ]+\s*"
    r"(?P<layers>\d+)\s*[- ]*layers\s*[- ]+"
    r"(?P<cycle>\d+)"
    r"(?:\s*[- ]+\s*(?P<small>s))?",
    re.IGNORECASE,
)


@dataclass(frozen=True, order=True)
class ExperimentId:
    span_cm: int
    layers: int
    cycle: int
    small: bool = False

    @property
    def normalized_stem(self) -> str:
        suffix = "-s" if self.small else ""
        return f"{self.span_cm}cm-{self.layers}layers-{self.cycle}{suffix}"

    @property
    def span_m(self) -> float:
        return self.span_cm / 100.0


def parse_experiment_id(value: str | Path) -> ExperimentId:
    text = Path(value).name if isinstance(value, Path) else str(value)
    text = text.strip()
    text = re.sub(r"\.txt$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"-?interrogator$", "", text, flags=re.IGNORECASE)
    match = EXPERIMENT_RE.search(text)
    if not match:
        raise ValueError(f"Could not parse experiment id from {value!r}")
    groups = match.groupdict()
    return ExperimentId(
        span_cm=int(groups["span"]),
        layers=int(groups["layers"]),
        cycle=int(groups["cycle"]),
        small=bool(groups.get("small")),
    )


def safe_output_stem(path: str | Path) -> str:
    stem = Path(path).stem
    stem = re.sub(r"-?interrogator$", "", stem, flags=re.IGNORECASE)
    stem = stem.strip().replace(" ", "_")
    return re.sub(r"[^A-Za-z0-9._-]+", "_", stem)


def list_interrogator_files(directory: Path = INTERROGATOR_DIR) -> list[Path]:
    return sorted(directory.glob("*-interrogator.txt"), key=lambda p: p.name.lower())


def bragg_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in frame.columns if re.fullmatch(r"wl_\d+_nm", col)]


def infer_effective_time(time_s: Iterable[float]) -> np.ndarray:
    """Create a sub-second effective time for repeated integer-second rows."""
    series = pd.Series(list(time_s), dtype="float64")
    occurrence = series.groupby(series, sort=False).cumcount()
    counts = series.groupby(series, sort=False).transform("count")
    offsets = occurrence / counts.replace(0, np.nan)
    return (series + offsets.fillna(0.0)).to_numpy(dtype=float)


def read_interrogator_file(path: Path) -> pd.DataFrame:
    rows: list[tuple[str, float, list[float]]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        next(handle, None)
        for line in handle:
            parts = re.split(r"\s+", line.strip())
            if len(parts) < 3:
                continue
            try:
                time_s = float(parts[1])
                wavelengths = [float(value) for value in parts[2:]]
            except ValueError:
                continue
            if wavelengths:
                rows.append((parts[0], time_s, wavelengths))

    if not rows:
        raise ValueError(f"No wavelength rows found in {path}")

    channel_count = max(len(row[2]) for row in rows)
    payload: dict[str, list[float] | list[str] | np.ndarray] = {
        "timestamp": [row[0] for row in rows],
        "time_s": [row[1] for row in rows],
        "sample_index": np.arange(len(rows), dtype=int),
    }
    for channel in range(channel_count):
        payload[f"wl_{channel + 1}_nm"] = [
            row[2][channel] if channel < len(row[2]) else np.nan for row in rows
        ]

    frame = pd.DataFrame(payload)
    frame["effective_time_s"] = infer_effective_time(frame["time_s"])
    frame = frame[
        ["timestamp", "time_s", "effective_time_s", "sample_index", *bragg_columns(frame)]
    ]
    experiment_id = parse_experiment_id(path)
    frame.attrs["source_name"] = path.name
    frame.attrs["source_path"] = str(path)
    frame.attrs["output_stem"] = safe_output_stem(path)
    frame.attrs["normalized_stem"] = experiment_id.normalized_stem
    frame.attrs["span_cm"] = experiment_id.span_cm
    frame.attrs["layers"] = experiment_id.layers
    frame.attrs["cycle"] = experiment_id.cycle
    frame.attrs["small_sample"] = experiment_id.small
    return frame


def estimate_lambda0(
    frame: pd.DataFrame,
    columns: list[str] | None = None,
    n_samples: int = 50,
) -> dict[str, float]:
    values: dict[str, float] = {}
    for column in columns or bragg_columns(frame):
        valid = frame[column].dropna().head(n_samples)
        values[column] = float(valid.median()) if not valid.empty else float("nan")
    return values


def _clean_marker(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    return text


def load_crack_workbook(path: Path = HANDHELD_XLSX) -> dict[str, pd.DataFrame]:
    raw_sheets = pd.read_excel(path, sheet_name=None, engine="openpyxl")
    metadata: dict[str, pd.DataFrame] = {}
    for sheet_name, sheet in raw_sheets.items():
        normalized_name = sheet_name.strip()
        try:
            experiment_id = parse_experiment_id(normalized_name)
        except ValueError:
            continue

        first_columns = sheet.iloc[:, :6].copy()
        first_columns.columns = [
            "pressure_bar",
            "layers",
            "span_cm",
            "force_n",
            "displacement_mm",
            "crack_marker",
        ][: len(first_columns.columns)]
        while len(first_columns.columns) < 6:
            first_columns[f"missing_{len(first_columns.columns)}"] = np.nan
        first_columns = first_columns.iloc[:, :6]
        first_columns.columns = [
            "pressure_bar",
            "layers",
            "span_cm",
            "force_n",
            "displacement_mm",
            "crack_marker",
        ]

        first_columns["pressure_bar"] = pd.to_numeric(
            first_columns["pressure_bar"], errors="coerce"
        )
        first_columns = first_columns[first_columns["pressure_bar"].notna()].head(12)
        if first_columns.empty:
            continue

        first_columns["layers"] = pd.to_numeric(
            first_columns["layers"], errors="coerce"
        ).ffill()
        first_columns["span_cm"] = pd.to_numeric(
            first_columns["span_cm"], errors="coerce"
        ).ffill()
        first_columns["force_n"] = pd.to_numeric(
            first_columns["force_n"], errors="coerce"
        )
        first_columns["displacement_mm"] = pd.to_numeric(
            first_columns["displacement_mm"], errors="coerce"
        )
        first_columns["layers"] = first_columns["layers"].fillna(experiment_id.layers)
        first_columns["span_cm"] = first_columns["span_cm"].fillna(experiment_id.span_cm)
        first_columns["crack_marker"] = first_columns["crack_marker"].map(_clean_marker)
        first_columns["crack_label"] = (first_columns["crack_marker"] != "").astype(int)
        first_columns.insert(0, "group_index", np.arange(len(first_columns), dtype=int))
        first_columns.insert(1, "experiment_id", experiment_id.normalized_stem)
        metadata[experiment_id.normalized_stem] = first_columns.reset_index(drop=True)
    return metadata


def metadata_for_file(
    file_path: Path,
    workbook: dict[str, pd.DataFrame] | None = None,
) -> pd.DataFrame | None:
    workbook = workbook if workbook is not None else load_crack_workbook()
    experiment_id = parse_experiment_id(file_path)
    return workbook.get(experiment_id.normalized_stem)


def workbook_coverage(
    files: Iterable[Path],
    workbook: dict[str, pd.DataFrame],
) -> tuple[list[str], list[str]]:
    file_ids = {parse_experiment_id(path).normalized_stem for path in files}
    sheet_ids = set(workbook)
    return sorted(file_ids - sheet_ids), sorted(sheet_ids - file_ids)
