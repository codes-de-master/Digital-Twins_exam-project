from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from digital_twin.config import PhysicsConfig, load_physics_config, save_physics_config
from digital_twin.data import (
    bragg_columns,
    estimate_lambda0,
    list_interrogator_files,
    load_crack_workbook,
    metadata_for_file,
    read_interrogator_file,
    workbook_coverage,
)
from digital_twin.features import build_feature_dataset, build_physics_residuals, split_by_file
from digital_twin.models import (
    CNN_MODEL_PATH,
    PHYSICS_MODEL_PATH,
    train_cnn,
    train_physics_head,
)
from digital_twin.paths import (
    HANDHELD_XLSX,
    INTERROGATOR_DIR,
    MEDIAN_FILTER_DIR,
    MODEL_DIR,
    PEAK_GROUPS_DIR,
    PEAKS_DIR,
    ensure_computed_dirs,
)
from digital_twin.preprocessing import (
    BLANK_DELTA_THRESHOLD_NM,
    add_unfiltered_signals,
    apply_median_filter,
    active_channels_from_filtered,
    active_raw_columns,
    build_peak_groups,
    delta_amplitude,
    find_wavelength_peaks,
    filtered_output_path,
    group_output_path,
    peaks_output_path,
    save_median_filtered_file,
    save_peak_groups_file,
    save_peaks_file,
)


st.set_page_config(page_title="FBG Bending Digital Twin", layout="wide")
ensure_computed_dirs()


@st.cache_data(show_spinner=False)
def cached_files() -> list[str]:
    return [str(path) for path in list_interrogator_files(INTERROGATOR_DIR)]


@st.cache_data(show_spinner=False)
def cached_workbook() -> dict[str, pd.DataFrame]:
    return load_crack_workbook(HANDHELD_XLSX)


@st.cache_data(show_spinner=False)
def cached_raw(path: str) -> pd.DataFrame:
    return read_interrogator_file(Path(path))


def plot_lines(
    frame: pd.DataFrame,
    y_columns: list[str],
    title: str,
    x_column: str = "effective_time_s",
    x_axis_title: str = "Effective time [s]",
    y_axis_title: str = "Value",
) -> go.Figure:
    fig = go.Figure()
    for column in y_columns:
        if column not in frame.columns:
            continue
        fig.add_trace(
            go.Scattergl(
                x=frame[x_column],
                y=frame[column],
                mode="lines",
                name=column,
            )
        )
    fig.update_layout(
        title=title,
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        height=520,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_correlation_pair(
    frame: pd.DataFrame,
    left_col: str,
    right_col: str,
) -> go.Figure:
    pair = frame[[left_col, right_col]].dropna()
    pearson = float(pair[left_col].corr(pair[right_col])) if len(pair) >= 2 else np.nan
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=pair[left_col],
            y=pair[right_col],
            mode="markers",
            marker=dict(size=4, opacity=0.45),
            name="Samples",
        )
    )
    if len(pair) >= 2 and pair[left_col].nunique() > 1:
        slope, intercept = np.polyfit(pair[left_col], pair[right_col], 1)
        xs = np.linspace(pair[left_col].min(), pair[left_col].max(), 100)
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=slope * xs + intercept,
                mode="lines",
                line=dict(color="red", width=2),
                name=f"Regression, r={pearson:.3f}",
            )
        )
    fig.update_layout(
        title=f"{left_col} vs {right_col} (Pearson r={pearson:.3f})",
        xaxis_title=f"{left_col} [nm]",
        yaxis_title=f"{right_col} [nm]",
        height=500,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_peak_overlay(
    filtered: pd.DataFrame,
    peaks: pd.DataFrame,
    channel: int,
) -> go.Figure:
    delta_col = f"wl_{channel}_delta_nm"
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=filtered["effective_time_s"],
            y=filtered[delta_col],
            mode="lines",
            name=delta_col,
        )
    )
    if not peaks.empty:
        channel_peaks = peaks[peaks["channel"] == channel]
        fig.add_trace(
            go.Scattergl(
                x=channel_peaks["effective_time_s"],
                y=channel_peaks["delta_nm"],
                mode="markers",
                marker=dict(color="green", size=7),
                name="Detected peaks",
            )
        )
    fig.update_layout(
        title=f"Detected peaks for WL {channel}",
        xaxis_title="Effective time [s]",
        yaxis_title="Filtered delta-lambda [nm]",
        height=520,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def plot_convolution_outputs(
    filtered: pd.DataFrame,
    channel: int,
    kernels: dict[str, np.ndarray],
) -> go.Figure:
    delta_col = f"wl_{channel}_delta_nm"
    signal = (
        filtered[delta_col]
        .astype(float)
        .interpolate(limit_direction="both")
        .fillna(0.0)
        .to_numpy()
    )
    fig = go.Figure()
    fig.add_trace(
        go.Scattergl(
            x=filtered["effective_time_s"],
            y=signal,
            mode="lines",
            name="filtered delta-lambda",
        )
    )
    for name, kernel in kernels.items():
        convolved = np.convolve(signal, kernel, mode="same")
        fig.add_trace(
            go.Scattergl(
                x=filtered["effective_time_s"],
                y=convolved,
                mode="lines",
                name=name,
            )
        )
    fig.update_layout(
        title=f"WL {channel}: filtered signal convolved by reference filters",
        xaxis_title="Effective time [s]",
        yaxis_title="Delta-lambda / convolution output [nm]",
        height=560,
        margin=dict(l=40, r=20, t=50, b=40),
    )
    return fig


def confusion_matrix_frame(confusion: dict[str, int]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "actual": "no crack",
                "predicted no crack": confusion.get("true_negative", 0),
                "predicted crack": confusion.get("false_positive", 0),
            },
            {
                "actual": "crack",
                "predicted no crack": confusion.get("false_negative", 0),
                "predicted crack": confusion.get("true_positive", 0),
            },
        ]
    )


def weight_label(weight: float) -> str:
    return f"{float(weight):.4g}".replace("-", "m").replace(".", "p")


def weighted_cnn_model_dir(weight: float, force_subdir: bool = False) -> Path:
    if np.isclose(weight, 1.0) and not force_subdir:
        return MODEL_DIR
    return MODEL_DIR / "cnn_crack_weight_models" / f"crack_weight_{weight_label(weight)}"


def preprocess_one(
    source_path: Path,
    workbook: dict[str, pd.DataFrame],
    median_window: int,
    distance_seconds: float,
    prominence: float,
) -> dict[str, object]:
    filtered, filtered_path = save_median_filtered_file(
        source_path,
        output_dir=MEDIAN_FILTER_DIR,
        window=median_window,
    )
    peaks, peaks_path = save_peaks_file(
        source_path,
        filtered_frame=filtered,
        output_dir=PEAKS_DIR,
        distance_seconds=distance_seconds,
        prominence=prominence,
        median_window=median_window,
    )
    metadata = metadata_for_file(source_path, workbook)
    group_paths: dict[int, Path] = {}
    if metadata is not None:
        group_paths = save_peak_groups_file(
            source_path,
            peaks,
            metadata,
            output_dir=PEAK_GROUPS_DIR,
        )
    return {
        "filtered_path": filtered_path,
        "peaks_path": peaks_path,
        "group_paths": group_paths,
        "peak_count": len(peaks),
        "matched_metadata": metadata is not None,
    }


def feature_dataset_summary(dataset, seed: int = 2026) -> pd.DataFrame:
    if dataset.is_empty:
        return pd.DataFrame()
    splits = split_by_file(dataset.meta, seed=seed)
    return pd.DataFrame(
        {
            "metric": [
                "samples",
                "files",
                "no_crack",
                "crack",
                "train",
                "validation",
                "test",
            ],
            "value": [
                len(dataset.y),
                dataset.meta["source_name"].nunique(),
                int((dataset.y == 0).sum()),
                int((dataset.y == 1).sum()),
                len(splits["train"]),
                len(splits["val"]),
                len(splits["test"]),
            ],
        }
    )


files = [Path(path) for path in cached_files()]
workbook = cached_workbook()

st.title("FBG Bending Digital Twin")

with st.sidebar:
    st.header("Dataset")
    selected_file = st.selectbox(
        "Interrogator file",
        files,
        format_func=lambda path: path.name,
    )
    median_window = st.number_input(
        "Median window [samples]",
        min_value=1,
        max_value=101,
        value=5,
        step=2,
    )
    distance_seconds = st.number_input(
        "Peak distance [s]",
        min_value=0.01,
        max_value=5.0,
        value=0.2,
        step=0.05,
    )
    peak_prominence = st.number_input(
        "Peak prominence [nm]",
        min_value=0.0,
        max_value=10.0,
        value=0.1,
        step=0.05,
    )
    st.caption(f"Interrogator files: {len(files)}")
    st.caption(f"Workbook sheets: {len(workbook)}")

raw = cached_raw(str(selected_file))
lambda0 = estimate_lambda0(raw)
raw_signals = add_unfiltered_signals(raw, lambda0)
filtered = apply_median_filter(raw, window=int(median_window), lambda0_by_col=lambda0)
peaks = find_wavelength_peaks(
    filtered,
    distance_seconds=float(distance_seconds),
    prominence=float(peak_prominence),
)
metadata = metadata_for_file(selected_file, workbook)
all_channels = [int(col.split("_")[1]) for col in bragg_columns(raw)]
active_raw_cols = active_raw_columns(raw, lambda0)
active_filtered_channels = active_channels_from_filtered(filtered)
channels = active_filtered_channels or [int(col.split("_")[1]) for col in active_raw_cols]
channel_diagnostics = pd.DataFrame(
    [
        {
            "channel": int(col.split("_")[1]),
            "lambda0_nm": lambda0[col],
            "max_abs_delta_lambda_nm": delta_amplitude(raw[col] - lambda0[col]),
            "active": col in active_raw_cols,
        }
        for col in bragg_columns(raw)
    ]
)

tabs = st.tabs(
    [
        "1 Raw",
        "2 Correlation",
        "3 Strain",
        "4 Median Filter",
        "5 Peak Detection",
        "6 Peak Groups",
        "7 CNN Filters",
        "8 CNN Training",
        "9 Physics Head",
    ]
)

with tabs[0]:
    st.subheader("Unfiltered wavelength vs time")
    st.write(
        f"{selected_file.name}: {len(raw):,} samples, "
        f"{len(channels)} active Bragg peak channel(s), {len(all_channels)} raw channel column(s)."
    )
    st.caption(
        f"Blank-channel rule: max absolute delta-lambda <= "
        f"{BLANK_DELTA_THRESHOLD_NM:g} nm is treated as no data."
    )
    st.dataframe(channel_diagnostics, width="stretch")
    if not active_raw_cols:
        st.warning("No active wavelength channels exceed the blank-channel threshold.")
    else:
        st.plotly_chart(
            plot_lines(
                raw,
                active_raw_cols,
                "Raw Bragg wavelengths",
                "effective_time_s",
                y_axis_title="Wavelength [nm]",
            ),
            width="stretch",
        )

with tabs[1]:
    st.subheader("Delta-lambda channel correlation")
    delta_cols = [f"wl_{channel}_delta_raw_nm" for channel in channels]
    if len(delta_cols) < 2:
        st.info("This file has fewer than two detected Bragg peak channels.")
    else:
        corr = raw_signals[delta_cols].corr(method="pearson")
        st.dataframe(corr, width="stretch")
        for index, left_col in enumerate(delta_cols):
            for right_col in delta_cols[index + 1 :]:
                st.plotly_chart(
                    plot_correlation_pair(raw_signals, left_col, right_col),
                    width="stretch",
                )

with tabs[2]:
    st.subheader("Unfiltered FBG strain")
    strain_cols = [f"wl_{channel}_strain_raw" for channel in channels]
    if not strain_cols:
        st.info("No active channel is available for strain plotting.")
    else:
        st.plotly_chart(
            plot_lines(
                raw_signals,
                strain_cols,
                "Unfiltered strain",
                "effective_time_s",
                y_axis_title="Strain [-]",
            ),
            width="stretch",
        )

with tabs[3]:
    st.subheader("Centered rolling median spike filtering")
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Save filtered CSV for selected file"):
            _, path = save_median_filtered_file(
                selected_file,
                output_dir=MEDIAN_FILTER_DIR,
                window=int(median_window),
            )
            st.success(f"Saved {path}")
    with col_b:
        if st.button("Save filtered CSVs for all files"):
            progress = st.progress(0)
            for index, path in enumerate(files):
                save_median_filtered_file(path, MEDIAN_FILTER_DIR, int(median_window))
                progress.progress((index + 1) / len(files))
            st.success(f"Saved filtered files in {MEDIAN_FILTER_DIR}")

    if not channels:
        st.info("No active channel is available for raw/filtered comparison.")
    else:
        channel = st.selectbox("Channel to compare", channels, format_func=lambda v: f"WL {v}")
        compare = pd.DataFrame(
            {
                "effective_time_s": raw["effective_time_s"],
                f"wl_{channel}_raw_nm": raw[f"wl_{channel}_nm"],
                f"wl_{channel}_filtered_nm": filtered[f"wl_{channel}_filtered_nm"],
            }
        )
        st.plotly_chart(
            plot_lines(
                compare,
                [f"wl_{channel}_raw_nm", f"wl_{channel}_filtered_nm"],
                f"Raw vs filtered WL {channel}",
                y_axis_title="Wavelength [nm]",
            ),
            width="stretch",
        )

with tabs[4]:
    st.subheader("Peak detection")
    st.write(
        f"Current settings detect {len(peaks):,} peaks across {len(channels)} channel(s)."
    )
    if not peaks.empty:
        st.dataframe(
            peaks.groupby("channel")
            .agg(peaks=("peak_order", "count"), distance_samples=("distance_samples", "first"))
            .reset_index(),
            width="stretch",
        )
    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Save peaks CSV for selected file"):
            _, path = save_peaks_file(
                selected_file,
                filtered_frame=filtered,
                output_dir=PEAKS_DIR,
                distance_seconds=float(distance_seconds),
                prominence=float(peak_prominence),
            )
            st.success(f"Saved {path}")
    with col_b:
        if st.button("Save peaks CSVs for all files"):
            progress = st.progress(0)
            for index, path in enumerate(files):
                filtered_frame, _ = save_median_filtered_file(
                    path,
                    MEDIAN_FILTER_DIR,
                    int(median_window),
                )
                save_peaks_file(
                    path,
                    filtered_frame=filtered_frame,
                    output_dir=PEAKS_DIR,
                    distance_seconds=float(distance_seconds),
                    prominence=float(peak_prominence),
                )
                progress.progress((index + 1) / len(files))
            st.success(f"Saved peak files in {PEAKS_DIR}")
    if not channels:
        st.info("No active channel is available for peak overlay.")
    else:
        overlay_channel = st.selectbox(
            "Peak overlay channel",
            channels,
            format_func=lambda v: f"WL {v}",
            key="peak_overlay_channel",
        )
        st.plotly_chart(
            plot_peak_overlay(filtered, peaks, overlay_channel),
            width="stretch",
        )

with tabs[5]:
    st.subheader("Pressure groups and crack labels")
    files_without_sheet, sheets_without_file = workbook_coverage(files, workbook)
    if metadata is None:
        st.warning("This interrogator file has no matching Excel sheet for labels.")
    else:
        st.dataframe(metadata, width="stretch")
        groups_by_channel = build_peak_groups(peaks, metadata)
        if groups_by_channel:
            diagnostics = []
            for channel, group_frame in groups_by_channel.items():
                diagnostics.append(
                    {
                        "channel": channel,
                        "rows": len(group_frame),
                        "channel_peak_count": (
                            int(group_frame["channel_peak_count"].iloc[0])
                            if not group_frame.empty
                            else 0
                        ),
                        "complete_channel": (
                            int(group_frame["is_complete_channel"].iloc[0])
                            if not group_frame.empty
                            else 0
                        ),
                        "extra_peaks_ignored": (
                            int(group_frame["extra_peaks_ignored"].iloc[0])
                            if not group_frame.empty
                            else 0
                        ),
                    }
                )
            st.dataframe(pd.DataFrame(diagnostics), width="stretch")
    with st.expander("Workbook coverage diagnostics"):
        st.write("Files without matching sheet:", files_without_sheet)
        st.write("Sheets without matching file:", sheets_without_file)

    col_a, col_b = st.columns([1, 1])
    with col_a:
        if st.button("Save groups for selected file"):
            if metadata is None:
                st.error("No matching workbook metadata for selected file.")
            else:
                peak_path = peaks_output_path(selected_file, PEAKS_DIR)
                filtered_path = filtered_output_path(selected_file, MEDIAN_FILTER_DIR)
                filtered.to_csv(filtered_path, index=False)
                peaks.to_csv(peak_path, index=False)
                paths = save_peak_groups_file(
                    selected_file,
                    peaks,
                    metadata,
                    PEAK_GROUPS_DIR,
                )
                st.success(f"Saved {len(paths)} group file(s) in {PEAK_GROUPS_DIR}")
    with col_b:
        if st.button("Generate filter, peaks, and groups for all files"):
            progress = st.progress(0)
            rows = []
            for index, path in enumerate(files):
                result = preprocess_one(
                    path,
                    workbook,
                    int(median_window),
                    float(distance_seconds),
                    float(peak_prominence),
                )
                rows.append(
                    {
                        "file": path.name,
                        "matched_metadata": result["matched_metadata"],
                        "peak_count": result["peak_count"],
                        "group_files": len(result["group_paths"]),
                    }
                )
                progress.progress((index + 1) / len(files))
            st.success("Computed preprocessing artifacts.")
            st.dataframe(pd.DataFrame(rows), width="stretch")

with tabs[6]:
    st.subheader("Reference convolution filters on the signal")
    kernels = {
        "central_difference": np.array([-0.5, 0.0, 0.5]),
        "long_range_trend": np.ones(9) / 9.0,
        "laplacian": np.array([1.0, -2.0, 1.0]),
    }
    if not channels:
        st.info("No active channel is available for convolution plotting.")
    else:
        conv_channel = st.selectbox(
            "Signal channel",
            channels,
            format_func=lambda v: f"WL {v}",
            key="conv_channel",
        )
        st.plotly_chart(
            plot_convolution_outputs(filtered, conv_channel, kernels),
            width="stretch",
        )
        with st.expander("Kernel weights"):
            fig = go.Figure()
            for name, kernel in kernels.items():
                x = np.arange(len(kernel)) - (len(kernel) - 1) / 2
                fig.add_trace(
                    go.Scatter(x=x, y=kernel, mode="lines+markers", name=name)
                )
            fig.update_layout(
                xaxis_title="Kernel position [samples]",
                yaxis_title="Kernel weight [-]",
                height=420,
                margin=dict(l=40, r=20, t=20, b=40),
            )
            st.plotly_chart(fig, width="stretch")

with tabs[7]:
    st.subheader("CNN crack classifier")
    st.caption(f"Checkpoint: {CNN_MODEL_PATH}")
    if CNN_MODEL_PATH.exists():
        st.success("A saved CNN checkpoint exists.")
    dataset = None
    if st.button("Build training dataset from computed artifacts"):
        dataset = build_feature_dataset(files, workbook)
        st.session_state["feature_dataset"] = dataset
    dataset = st.session_state.get("feature_dataset", dataset)
    if dataset is not None:
        if dataset.is_empty:
            st.warning(
                "No samples were built. Generate median, peak, and group files first."
            )
        else:
            epochs = st.number_input("Epochs", min_value=1, max_value=500, value=25)
            batch_size = st.number_input("Batch size", min_value=1, max_value=256, value=32)
            model_seed = st.number_input(
                "Seed",
                min_value=0,
                max_value=2**31 - 1,
                value=2026,
                step=1,
                help="Controls the file-level train/validation/test split and TensorFlow initialization.",
            )
            crack_weight = st.number_input(
                "Crack class weight",
                min_value=0.1,
                max_value=1000.0,
                value=1.0,
                step=0.5,
                help=(
                    "Weight for class 1 during training. Increase it to penalize "
                    "missed cracks more strongly and try to reduce false negatives."
                ),
            )
            target_model_dir = weighted_cnn_model_dir(float(crack_weight))
            st.dataframe(feature_dataset_summary(dataset, seed=int(model_seed)), width="stretch")
            st.write(f"Tensor shape: `{dataset.x.shape}`")
            st.caption(f"Weighted checkpoint directory: {target_model_dir}")
            if st.button("Train CNN"):
                try:
                    result = train_cnn(
                        dataset,
                        epochs=int(epochs),
                        batch_size=int(batch_size),
                        seed=int(model_seed),
                        model_dir=target_model_dir,
                        crack_class_weight=float(crack_weight),
                    )
                    st.success(f"Saved model to {result.model_path}")
                    st.write(result.metrics)
                    if result.confusion_matrix:
                        st.dataframe(
                            confusion_matrix_frame(result.confusion_matrix),
                            width="stretch",
                        )
                    st.line_chart(pd.DataFrame(result.history))
                except Exception as exc:
                    st.error(str(exc))
            with st.expander("Crack-weight sweep"):
                sweep_min = st.number_input(
                    "Sweep minimum crack weight",
                    min_value=0.1,
                    max_value=1000.0,
                    value=1.0,
                    step=0.5,
                )
                sweep_max = st.number_input(
                    "Sweep maximum crack weight",
                    min_value=0.1,
                    max_value=1000.0,
                    value=8.0,
                    step=0.5,
                )
                sweep_count = st.number_input(
                    "Number of weights",
                    min_value=2,
                    max_value=50,
                    value=5,
                    step=1,
                )
                if st.button("Train CNN weight sweep"):
                    if float(sweep_max) < float(sweep_min):
                        st.error("Sweep maximum must be greater than or equal to minimum.")
                    else:
                        sweep_rows = []
                        weights = np.linspace(
                            float(sweep_min),
                            float(sweep_max),
                            int(sweep_count),
                        )
                        progress = st.progress(0)
                        for index, weight in enumerate(weights):
                            weight_dir = weighted_cnn_model_dir(
                                float(weight),
                                force_subdir=True,
                            )
                            try:
                                result = train_cnn(
                                    dataset,
                                    epochs=int(epochs),
                                    batch_size=int(batch_size),
                                    seed=int(model_seed),
                                    model_dir=weight_dir,
                                    crack_class_weight=float(weight),
                                )
                                confusion = result.confusion_matrix
                                actual_cracks = confusion.get("false_negative", 0) + confusion.get(
                                    "true_positive", 0
                                )
                                false_negative_rate = (
                                    confusion.get("false_negative", 0) / actual_cracks
                                    if actual_cracks
                                    else np.nan
                                )
                                sweep_rows.append(
                                    {
                                        "crack_class_weight": float(weight),
                                        "model_dir": str(weight_dir),
                                        "false_negative": confusion.get(
                                            "false_negative", 0
                                        ),
                                        "false_negative_rate": false_negative_rate,
                                        "false_positive": confusion.get(
                                            "false_positive", 0
                                        ),
                                        "true_positive": confusion.get("true_positive", 0),
                                        "true_negative": confusion.get("true_negative", 0),
                                        **result.metrics,
                                    }
                                )
                            except Exception as exc:
                                sweep_rows.append(
                                    {
                                        "crack_class_weight": float(weight),
                                        "model_dir": str(weight_dir),
                                        "error": str(exc),
                                    }
                                )
                            progress.progress((index + 1) / len(weights))
                        sweep_frame = pd.DataFrame(sweep_rows)
                        st.session_state["cnn_weight_sweep"] = sweep_frame
                sweep_frame = st.session_state.get("cnn_weight_sweep")
                if sweep_frame is not None and not sweep_frame.empty:
                    st.dataframe(sweep_frame, width="stretch")
                    if "false_negative" in sweep_frame.columns:
                        plot_frame = sweep_frame.dropna(
                            subset=["crack_class_weight", "false_negative"]
                        )
                        if not plot_frame.empty:
                            fig = go.Figure()
                            fig.add_trace(
                                go.Scatter(
                                    x=plot_frame["crack_class_weight"],
                                    y=plot_frame["false_negative"],
                                    mode="lines+markers",
                                    name="False negatives",
                                )
                            )
                            if "false_negative_rate" in plot_frame.columns:
                                fig.add_trace(
                                    go.Scatter(
                                        x=plot_frame["crack_class_weight"],
                                        y=plot_frame["false_negative_rate"],
                                        mode="lines+markers",
                                        name="False negative rate",
                                        yaxis="y2",
                                    )
                                )
                            fig.update_layout(
                                title="False negatives vs crack class weight",
                                xaxis_title="Crack class weight [-]",
                                yaxis_title="False negatives [count]",
                                yaxis2=dict(
                                    title="False negative rate [-]",
                                    overlaying="y",
                                    side="right",
                                    range=[0, 1],
                                ),
                                height=500,
                                margin=dict(l=40, r=60, t=50, b=40),
                            )
                            st.plotly_chart(fig, width="stretch")

with tabs[8]:
    st.subheader("Physics Head + CNN")
    config = load_physics_config()
    with st.form("physics_config"):
        young_modulus = st.number_input(
            "Young modulus E [Pa]",
            min_value=0.0,
            value=float(config.young_modulus_pa),
            format="%.6g",
        )
        width = st.number_input(
            "Beam width b [m]",
            min_value=0.0,
            value=float(config.width_m),
            format="%.6g",
        )
        thickness = st.number_input(
            "Beam thickness h [m]",
            min_value=0.0,
            value=float(config.thickness_m),
            format="%.6g",
        )
        shared_y = st.checkbox("Use the same y for all sensor slots", value=True)
        if shared_y:
            y_value = st.number_input(
                "FBG distance from neutral axis y [m]",
                min_value=0.0,
                value=float(config.y_m[0]),
                format="%.6g",
            )
            y_values = (y_value, y_value, y_value)
        else:
            cols = st.columns(3)
            y_values = tuple(
                cols[index].number_input(
                    f"Sensor {index + 1} y [m]",
                    min_value=0.0,
                    value=float(config.y_m[index]),
                    format="%.6g",
                )
                for index in range(3)
            )
        submitted = st.form_submit_button("Save physics constants")
        if submitted:
            config = PhysicsConfig(
                young_modulus_pa=float(young_modulus),
                width_m=float(width),
                thickness_m=float(thickness),
                y_m=tuple(float(v) for v in y_values),
            )
            save_physics_config(config)
            st.success("Saved physics constants.")

    if PHYSICS_MODEL_PATH.exists():
        st.success("A saved physics-head checkpoint exists.")
    if not config.is_complete():
        st.warning("Complete and save all physics constants before training.")
    else:
        dataset = st.session_state.get("feature_dataset")
        if dataset is None and st.button("Build training dataset for physics head"):
            dataset = build_feature_dataset(files, workbook)
            st.session_state["feature_dataset"] = dataset
        dataset = st.session_state.get("feature_dataset", dataset)
        if dataset is not None and not dataset.is_empty:
            residuals = build_physics_residuals(dataset, config)
            st.write(f"Residual tensor shape: `{residuals.shape}`")
            epochs = st.number_input(
                "Physics-head epochs",
                min_value=1,
                max_value=500,
                value=25,
            )
            batch_size = st.number_input(
                "Physics-head batch size",
                min_value=1,
                max_value=256,
                value=32,
            )
            if st.button("Train Physics Head + CNN"):
                try:
                    result = train_physics_head(
                        dataset,
                        config,
                        epochs=int(epochs),
                        batch_size=int(batch_size),
                        model_dir=MODEL_DIR,
                    )
                    st.success(f"Saved model to {result.model_path}")
                    st.write(result.metrics)
                    if result.confusion_matrix:
                        st.dataframe(
                            confusion_matrix_frame(result.confusion_matrix),
                            width="stretch",
                        )
                    st.line_chart(pd.DataFrame(result.history))
                except Exception as exc:
                    st.error(str(exc))
