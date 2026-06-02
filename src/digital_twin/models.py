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


CNN_MODEL_PATH = MODEL_DIR / "cnn_crack_classifier.pt"
PHYSICS_MODEL_PATH = MODEL_DIR / "physics_head_cnn.pt"
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


def _torch():
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError(
            "PyTorch is required for CNN training. Install requirements.txt first."
        ) from exc
    return torch


def build_cnn(input_shape: tuple[int, int] = (TIMESTEPS, 9), dropout: float = 0.30):
    _torch()
    from torch import nn

    class PyTorchCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            _, feature_count = input_shape
            self.network = nn.Sequential(
                nn.Conv1d(feature_count, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(128, 1),
            )

        def forward(self, x):
            x = x.transpose(1, 2)
            return self.network(x).squeeze(-1)

    return PyTorchCNN()


def build_physics_head_cnn(
    signal_shape: tuple[int, int] = (TIMESTEPS, 9),
    residual_shape: tuple[int, int] = (TIMESTEPS, 3),
    dropout: float = 0.30,
):
    torch = _torch()
    from torch import nn

    class PyTorchPhysicsHeadCNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            _, signal_feature_count = signal_shape
            _, residual_feature_count = residual_shape
            self.signal_network = nn.Sequential(
                nn.Conv1d(signal_feature_count, 32, kernel_size=3, padding=1),
                nn.BatchNorm1d(32),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(32, 64, kernel_size=3, padding=1),
                nn.BatchNorm1d(64),
                nn.ReLU(),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(64, 128, kernel_size=3, padding=1),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Dropout(dropout),
            )
            self.classifier = nn.Linear(128 + residual_feature_count, 1)

        def forward(self, signal, residual):
            signal_features = self.signal_network(signal.transpose(1, 2))
            residual_features = residual.mean(dim=1)
            merged = torch.cat(
                [signal_features, residual_features],
                dim=1,
            )
            return self.classifier(merged).squeeze(-1)

    return PyTorchPhysicsHeadCNN()


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


def _torch_dataset(torch, x: np.ndarray, y: np.ndarray):
    return torch.utils.data.TensorDataset(
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )


def _torch_loader(
    torch,
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    generator = torch.Generator()
    generator.manual_seed(seed)
    return torch.utils.data.DataLoader(
        _torch_dataset(torch, x, y),
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _torch_physics_loader(
    torch,
    x: np.ndarray,
    residuals: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(residuals, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32),
    )
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
    )


def _torch_epoch(
    torch,
    model,
    loader,
    criterion,
    optimizer=None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for x_batch, y_batch in loader:
            if training:
                optimizer.zero_grad()
            logits = model(x_batch)
            loss = criterion(logits, y_batch)
            if training:
                loss.backward()
                optimizer.step()
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()
            total_loss += float(loss.item()) * len(y_batch)
            total_correct += int((predictions == y_batch).sum().item())
            total_count += int(len(y_batch))
    if total_count == 0:
        return float("nan"), float("nan")
    return total_loss / total_count, total_correct / total_count


def _torch_physics_epoch(
    torch,
    model,
    loader,
    criterion,
    optimizer=None,
) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_correct = 0
    total_count = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for signal_batch, residual_batch, y_batch in loader:
            if training:
                optimizer.zero_grad()
            logits = model(signal_batch, residual_batch)
            loss = criterion(logits, y_batch)
            if training:
                loss.backward()
                optimizer.step()
            probabilities = torch.sigmoid(logits)
            predictions = (probabilities >= 0.5).float()
            total_loss += float(loss.item()) * len(y_batch)
            total_correct += int((predictions == y_batch).sum().item())
            total_count += int(len(y_batch))
    if total_count == 0:
        return float("nan"), float("nan")
    return total_loss / total_count, total_correct / total_count


def _torch_probabilities(torch, model, x: np.ndarray, batch_size: int) -> np.ndarray:
    if len(x) == 0:
        return np.zeros((0,), dtype=float)
    loader = torch.utils.data.DataLoader(
        torch.as_tensor(x, dtype=torch.float32),
        batch_size=batch_size,
        shuffle=False,
    )
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for x_batch in loader:
            logits = model(x_batch)
            batches.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(batches) if batches else np.zeros((0,), dtype=float)


def _torch_physics_probabilities(
    torch,
    model,
    x: np.ndarray,
    residuals: np.ndarray,
    batch_size: int,
) -> np.ndarray:
    if len(x) == 0:
        return np.zeros((0,), dtype=float)
    dataset = torch.utils.data.TensorDataset(
        torch.as_tensor(x, dtype=torch.float32),
        torch.as_tensor(residuals, dtype=torch.float32),
    )
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model.eval()
    batches: list[np.ndarray] = []
    with torch.no_grad():
        for signal_batch, residual_batch in loader:
            logits = model(signal_batch, residual_batch)
            batches.append(torch.sigmoid(logits).detach().cpu().numpy())
    return np.concatenate(batches) if batches else np.zeros((0,), dtype=float)


def _save_torch_checkpoint(
    torch,
    model_path: Path,
    model,
    input_shape: tuple[int, int],
    metadata: dict[str, Any],
) -> None:
    torch.save(
        {
            "state_dict": model.state_dict(),
            "input_shape": input_shape,
            "metadata": metadata,
        },
        model_path,
    )


def load_training_metadata(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_torch_cnn(path: Path = CNN_MODEL_PATH):
    torch = _torch()
    if not path.exists():
        return None
    checkpoint = torch.load(path, map_location="cpu")
    input_shape = tuple(checkpoint.get("input_shape", (TIMESTEPS, 9)))
    model = build_cnn(input_shape=input_shape)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def predict_cnn_probabilities(model, x: np.ndarray, batch_size: int = 256) -> np.ndarray:
    torch = _torch()
    return _torch_probabilities(torch, model, x, batch_size=batch_size)


def train_cnn(
    dataset: FeatureDataset,
    epochs: int = 25,
    batch_size: int = 32,
    seed: int = 2026,
    model_dir: Path = MODEL_DIR,
    crack_class_weight: float = 1.0,
) -> TrainingResult:
    torch = _torch()
    if dataset.is_empty:
        raise ValueError("No training samples were built from computed peak groups.")
    crack_class_weight = max(0.0, float(crack_class_weight))

    splits = split_by_file(dataset.meta, seed=seed)
    if len(splits["train"]) == 0:
        raise ValueError("The train split is empty.")

    np.random.seed(seed)
    torch.manual_seed(seed)
    mean, std = fit_standardizer(dataset.x[splits["train"]])
    x_scaled = apply_standardizer(dataset.x, mean, std)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / CNN_MODEL_PATH.name
    save_standardizer(mean, std, model_dir / CNN_SCALER_PATH.name)

    model = build_cnn(input_shape=dataset.x.shape[1:])
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(crack_class_weight), dtype=torch.float32)
    )
    train_loader = _torch_loader(
        torch,
        x_scaled[splits["train"]],
        dataset.y[splits["train"]],
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_loader = None
    if len(splits["val"]):
        val_loader = _torch_loader(
            torch,
            x_scaled[splits["val"]],
            dataset.y[splits["val"]],
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        )

    history: dict[str, list[float]] = {"loss": [], "accuracy": []}
    if val_loader is not None:
        history["val_loss"] = []
        history["val_accuracy"] = []
    best_score = float("inf")
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    stale_epochs = 0
    patience = 5
    for _ in range(int(epochs)):
        train_loss, train_accuracy = _torch_epoch(
            torch,
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
        )
        history["loss"].append(float(train_loss))
        history["accuracy"].append(float(train_accuracy))
        monitor_score = train_loss
        if val_loader is not None:
            val_loss, val_accuracy = _torch_epoch(torch, model, val_loader, criterion)
            history["val_loss"].append(float(val_loss))
            history["val_accuracy"].append(float(val_accuracy))
            monitor_score = val_loss
        if monitor_score < best_score:
            best_score = float(monitor_score)
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    model.load_state_dict(best_state)

    metrics: dict[str, float] = {}
    confusion_matrix: dict[str, int] = {}
    if len(splits["test"]):
        test_loader = _torch_loader(
            torch,
            x_scaled[splits["test"]],
            dataset.y[splits["test"]],
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        )
        loss, accuracy = _torch_epoch(torch, model, test_loader, criterion)
        metrics = {"test_loss": float(loss), "test_accuracy": float(accuracy)}
        probabilities = _torch_probabilities(
            torch,
            model,
            x_scaled[splits["test"]],
            batch_size=batch_size,
        )
        confusion_matrix = _confusion_matrix(dataset.y[splits["test"]], probabilities)

    metadata_path = model_dir / "cnn_training_metadata.json"
    metadata = {
        "framework": "pytorch",
        "model_path": str(model_path),
        "input_shape": list(dataset.x.shape[1:]),
        "samples": int(len(dataset.y)),
        "split_counts": _split_counts(splits),
        "class_counts": _class_counts(dataset.y),
        "seed": seed,
        "crack_class_weight": crack_class_weight,
        "confusion_matrix": confusion_matrix,
    }
    _save_torch_checkpoint(torch, model_path, model, dataset.x.shape[1:], metadata)
    _save_training_metadata(metadata_path, metadata)
    return TrainingResult(
        model_path=model_path,
        history=history,
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
    crack_class_weight: float = 1.0,
) -> TrainingResult:
    torch = _torch()
    if not config.is_complete():
        raise ValueError("Physics constants must be complete before training.")
    if dataset.is_empty:
        raise ValueError("No training samples were built from computed peak groups.")
    crack_class_weight = max(0.0, float(crack_class_weight))

    residuals = build_physics_residuals(dataset, config)
    splits = split_by_file(dataset.meta, seed=seed)
    if len(splits["train"]) == 0:
        raise ValueError("The train split is empty.")

    np.random.seed(seed)
    torch.manual_seed(seed)
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
    optimizer = torch.optim.Adam(model.parameters())
    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(float(crack_class_weight), dtype=torch.float32)
    )
    train_loader = _torch_physics_loader(
        torch,
        x_scaled[splits["train"]],
        r_scaled[splits["train"]],
        dataset.y[splits["train"]],
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
    )
    val_loader = None
    if len(splits["val"]):
        val_loader = _torch_physics_loader(
            torch,
            x_scaled[splits["val"]],
            r_scaled[splits["val"]],
            dataset.y[splits["val"]],
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        )

    history: dict[str, list[float]] = {"loss": [], "accuracy": []}
    if val_loader is not None:
        history["val_loss"] = []
        history["val_accuracy"] = []
    best_score = float("inf")
    best_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    stale_epochs = 0
    patience = 5
    for _ in range(int(epochs)):
        train_loss, train_accuracy = _torch_physics_epoch(
            torch,
            model,
            train_loader,
            criterion,
            optimizer=optimizer,
        )
        history["loss"].append(float(train_loss))
        history["accuracy"].append(float(train_accuracy))
        monitor_score = train_loss
        if val_loader is not None:
            val_loss, val_accuracy = _torch_physics_epoch(
                torch,
                model,
                val_loader,
                criterion,
            )
            history["val_loss"].append(float(val_loss))
            history["val_accuracy"].append(float(val_accuracy))
            monitor_score = val_loss
        if monitor_score < best_score:
            best_score = float(monitor_score)
            best_state = {
                key: value.detach().clone() for key, value in model.state_dict().items()
            }
            stale_epochs = 0
        else:
            stale_epochs += 1
        if stale_epochs >= patience:
            break
    model.load_state_dict(best_state)

    metrics: dict[str, float] = {}
    confusion_matrix: dict[str, int] = {}
    if len(splits["test"]):
        test_loader = _torch_physics_loader(
            torch,
            x_scaled[splits["test"]],
            r_scaled[splits["test"]],
            dataset.y[splits["test"]],
            batch_size=batch_size,
            shuffle=False,
            seed=seed,
        )
        loss, accuracy = _torch_physics_epoch(torch, model, test_loader, criterion)
        metrics = {"test_loss": float(loss), "test_accuracy": float(accuracy)}
        probabilities = _torch_physics_probabilities(
            torch,
            model,
            x_scaled[splits["test"]],
            r_scaled[splits["test"]],
            batch_size=batch_size,
        )
        confusion_matrix = _confusion_matrix(dataset.y[splits["test"]], probabilities)

    metadata_path = model_dir / "physics_training_metadata.json"
    metadata = {
        "framework": "pytorch",
        "model_path": str(model_path),
        "input_shape": list(dataset.x.shape[1:]),
        "residual_shape": list(residuals.shape[1:]),
        "samples": int(len(dataset.y)),
        "split_counts": _split_counts(splits),
        "class_counts": _class_counts(dataset.y),
        "seed": seed,
        "crack_class_weight": crack_class_weight,
        "physics_config": {
            "young_modulus_pa": config.young_modulus_pa,
            "width_m": config.width_m,
            "thickness_m": config.thickness_m,
            "y_m": list(config.y_m),
        },
        "confusion_matrix": confusion_matrix,
    }
    _save_torch_checkpoint(
        torch,
        model_path,
        model,
        dataset.x.shape[1:],
        metadata,
    )
    _save_training_metadata(metadata_path, metadata)
    return TrainingResult(
        model_path=model_path,
        history=history,
        metrics=metrics,
        split_counts=_split_counts(splits),
        confusion_matrix=confusion_matrix,
    )
