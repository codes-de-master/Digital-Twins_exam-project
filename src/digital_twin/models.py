from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .config import PhysicsConfig
from .features import (
    FeatureDataset,
    TIMESTEPS,
    apply_standardizer,
    build_physics_residuals,
    fit_standardizer,
    save_standardizer,
    split_by_file,
)
from .paths import MODEL_DIR


CNN_MODEL_PATH = MODEL_DIR / "cnn_crack_classifier.keras"
PHYSICS_MODEL_PATH = MODEL_DIR / "physics_head_cnn.keras"
CNN_SCALER_PATH = MODEL_DIR / "cnn_feature_standardizer.json"
PHYSICS_FEATURE_SCALER_PATH = MODEL_DIR / "physics_feature_standardizer.json"
PHYSICS_RESIDUAL_SCALER_PATH = MODEL_DIR / "physics_residual_standardizer.json"


@dataclass
class TrainingResult:
    model_path: Path
    history: dict[str, list[float]]
    metrics: dict[str, float]
    split_counts: dict[str, int]
    confusion_matrix: dict[str, int]


def _tensorflow():
    try:
        import tensorflow as tf
    except ImportError as exc:
        raise RuntimeError(
            "TensorFlow is required for model training. Install requirements.txt first."
        ) from exc
    return tf


def build_cnn(input_shape: tuple[int, int] = (TIMESTEPS, 9), dropout: float = 0.30):
    tf = _tensorflow()
    model = tf.keras.Sequential(
        [
            tf.keras.layers.Input(shape=input_shape),
            tf.keras.layers.Conv1D(32, kernel_size=3, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(64, kernel_size=3, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.MaxPooling1D(pool_size=2),
            tf.keras.layers.Conv1D(128, kernel_size=3, padding="same"),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.ReLU(),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_physics_head_cnn(
    signal_shape: tuple[int, int] = (TIMESTEPS, 9),
    residual_shape: tuple[int, int] = (TIMESTEPS, 3),
    dropout: float = 0.30,
):
    tf = _tensorflow()
    signal_input = tf.keras.layers.Input(shape=signal_shape, name="fbg_signal")
    residual_input = tf.keras.layers.Input(shape=residual_shape, name="physics_residual")

    x = tf.keras.layers.Conv1D(32, kernel_size=3, padding="same")(signal_input)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(64, kernel_size=3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    x = tf.keras.layers.MaxPooling1D(pool_size=2)(x)
    x = tf.keras.layers.Conv1D(128, kernel_size=3, padding="same")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.ReLU()(x)
    cnn_features = tf.keras.layers.GlobalAveragePooling1D()(x)
    cnn_features = tf.keras.layers.Dropout(dropout)(cnn_features)

    residual_features = tf.keras.layers.GlobalAveragePooling1D()(residual_input)
    merged = tf.keras.layers.Concatenate()([cnn_features, residual_features])
    output = tf.keras.layers.Dense(1, activation="sigmoid")(merged)
    model = tf.keras.Model(
        inputs=[signal_input, residual_input],
        outputs=output,
        name="physics_head_cnn",
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def _save_training_metadata(
    path: Path,
    payload: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def _split_counts(splits: dict[str, np.ndarray]) -> dict[str, int]:
    return {name: int(len(indices)) for name, indices in splits.items()}


def _class_counts(y: np.ndarray) -> dict[str, int]:
    return {"no_crack": int((y == 0).sum()), "crack": int((y == 1).sum())}


def _confusion_matrix(y_true: np.ndarray, probabilities: np.ndarray) -> dict[str, int]:
    y_pred = (probabilities.reshape(-1) >= 0.5).astype(int)
    y_true = y_true.astype(int).reshape(-1)
    return {
        "true_negative": int(((y_true == 0) & (y_pred == 0)).sum()),
        "false_positive": int(((y_true == 0) & (y_pred == 1)).sum()),
        "false_negative": int(((y_true == 1) & (y_pred == 0)).sum()),
        "true_positive": int(((y_true == 1) & (y_pred == 1)).sum()),
    }


def _history_dict(history: Any) -> dict[str, list[float]]:
    return {key: [float(v) for v in values] for key, values in history.history.items()}


def train_cnn(
    dataset: FeatureDataset,
    epochs: int = 25,
    batch_size: int = 32,
    seed: int = 2026,
    model_dir: Path = MODEL_DIR,
    crack_class_weight: float = 1.0,
) -> TrainingResult:
    tf = _tensorflow()
    if dataset.is_empty:
        raise ValueError("No training samples were built from computed peak groups.")
    crack_class_weight = max(0.0, float(crack_class_weight))

    splits = split_by_file(dataset.meta, seed=seed)
    if len(splits["train"]) == 0:
        raise ValueError("The train split is empty.")

    tf.keras.utils.set_random_seed(seed)
    mean, std = fit_standardizer(dataset.x[splits["train"]])
    x_scaled = apply_standardizer(dataset.x, mean, std)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / CNN_MODEL_PATH.name
    save_standardizer(mean, std, model_dir / CNN_SCALER_PATH.name)

    model = build_cnn(input_shape=dataset.x.shape[1:])
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            save_best_only=True,
            monitor="val_loss" if len(splits["val"]) else "loss",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if len(splits["val"]) else "loss",
            patience=5,
            restore_best_weights=True,
        ),
    ]
    validation_data = None
    if len(splits["val"]):
        validation_data = (x_scaled[splits["val"]], dataset.y[splits["val"]])

    history = model.fit(
        x_scaled[splits["train"]],
        dataset.y[splits["train"]],
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        class_weight={0: 1.0, 1: crack_class_weight},
        verbose=0,
    )
    model.save(model_path)

    metrics: dict[str, float] = {}
    confusion_matrix: dict[str, int] = {}
    if len(splits["test"]):
        loss, accuracy = model.evaluate(
            x_scaled[splits["test"]],
            dataset.y[splits["test"]],
            verbose=0,
        )
        metrics = {"test_loss": float(loss), "test_accuracy": float(accuracy)}
        probabilities = model.predict(x_scaled[splits["test"]], verbose=0)
        confusion_matrix = _confusion_matrix(dataset.y[splits["test"]], probabilities)

    metadata_path = model_dir / "cnn_training_metadata.json"
    _save_training_metadata(
        metadata_path,
        {
            "model_path": str(model_path),
            "input_shape": list(dataset.x.shape[1:]),
            "samples": int(len(dataset.y)),
            "split_counts": _split_counts(splits),
            "class_counts": _class_counts(dataset.y),
            "seed": seed,
            "crack_class_weight": crack_class_weight,
            "confusion_matrix": confusion_matrix,
        },
    )
    return TrainingResult(
        model_path=model_path,
        history=_history_dict(history),
        metrics=metrics,
        split_counts=_split_counts(splits),
        confusion_matrix=confusion_matrix,
    )


def train_physics_head(
    dataset: FeatureDataset,
    config: PhysicsConfig,
    epochs: int = 25,
    batch_size: int = 32,
    seed: int = 2026,
    model_dir: Path = MODEL_DIR,
) -> TrainingResult:
    tf = _tensorflow()
    if not config.is_complete():
        raise ValueError("Physics constants must be complete before training.")
    if dataset.is_empty:
        raise ValueError("No training samples were built from computed peak groups.")

    residuals = build_physics_residuals(dataset, config)
    splits = split_by_file(dataset.meta, seed=seed)
    if len(splits["train"]) == 0:
        raise ValueError("The train split is empty.")

    feature_mean, feature_std = fit_standardizer(dataset.x[splits["train"]])
    residual_mean, residual_std = fit_standardizer(residuals[splits["train"]])
    x_scaled = apply_standardizer(dataset.x, feature_mean, feature_std)
    r_scaled = apply_standardizer(residuals, residual_mean, residual_std)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / PHYSICS_MODEL_PATH.name
    save_standardizer(feature_mean, feature_std, model_dir / PHYSICS_FEATURE_SCALER_PATH.name)
    save_standardizer(residual_mean, residual_std, model_dir / PHYSICS_RESIDUAL_SCALER_PATH.name)

    model = build_physics_head_cnn(
        signal_shape=dataset.x.shape[1:],
        residual_shape=residuals.shape[1:],
    )
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_path,
            save_best_only=True,
            monitor="val_loss" if len(splits["val"]) else "loss",
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss" if len(splits["val"]) else "loss",
            patience=5,
            restore_best_weights=True,
        ),
    ]
    validation_data = None
    if len(splits["val"]):
        validation_data = (
            [x_scaled[splits["val"]], r_scaled[splits["val"]]],
            dataset.y[splits["val"]],
        )

    history = model.fit(
        [x_scaled[splits["train"]], r_scaled[splits["train"]]],
        dataset.y[splits["train"]],
        validation_data=validation_data,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=0,
    )
    model.save(model_path)

    metrics: dict[str, float] = {}
    confusion_matrix: dict[str, int] = {}
    if len(splits["test"]):
        loss, accuracy = model.evaluate(
            [x_scaled[splits["test"]], r_scaled[splits["test"]]],
            dataset.y[splits["test"]],
            verbose=0,
        )
        metrics = {"test_loss": float(loss), "test_accuracy": float(accuracy)}
        probabilities = model.predict(
            [x_scaled[splits["test"]], r_scaled[splits["test"]]],
            verbose=0,
        )
        confusion_matrix = _confusion_matrix(dataset.y[splits["test"]], probabilities)

    metadata_path = model_dir / "physics_training_metadata.json"
    _save_training_metadata(
        metadata_path,
        {
            "model_path": str(model_path),
            "input_shape": list(dataset.x.shape[1:]),
            "residual_shape": list(residuals.shape[1:]),
            "samples": int(len(dataset.y)),
            "split_counts": _split_counts(splits),
            "class_counts": _class_counts(dataset.y),
            "seed": seed,
            "physics_config": {
                "young_modulus_pa": config.young_modulus_pa,
                "width_m": config.width_m,
                "thickness_m": config.thickness_m,
                "y_m": list(config.y_m),
            },
            "confusion_matrix": confusion_matrix,
        },
    )
    return TrainingResult(
        model_path=model_path,
        history=_history_dict(history),
        metrics=metrics,
        split_counts=_split_counts(splits),
        confusion_matrix=confusion_matrix,
    )


def load_keras_model(path: Path):
    tf = _tensorflow()
    if not path.exists():
        return None
    return tf.keras.models.load_model(path)
