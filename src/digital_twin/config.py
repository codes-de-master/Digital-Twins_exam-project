from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from .paths import PHYSICS_CONFIG_PATH


@dataclass(frozen=True)
class PhysicsConfig:
    young_modulus_pa: float
    width_m: float
    thickness_m: float
    y_m: tuple[float, float, float]

    def is_complete(self) -> bool:
        values = [self.young_modulus_pa, self.width_m, self.thickness_m, *self.y_m]
        return all(value > 0 for value in values)


def default_physics_config() -> PhysicsConfig:
    return PhysicsConfig(
        young_modulus_pa=22e9,
        width_m=0.034,
        thickness_m=0.004,
        y_m=(0.00145, 0.00145, 0.00145),
    )


def load_physics_config(path: Path = PHYSICS_CONFIG_PATH) -> PhysicsConfig:
    if not path.exists():
        return default_physics_config()
    with path.open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    y_raw = payload.get("y_m", (0.0, 0.0, 0.0))
    y_values = tuple(float(v) for v in list(y_raw)[:3])
    if len(y_values) < 3:
        y_values = (*y_values, *(0.0 for _ in range(3 - len(y_values))))
    return PhysicsConfig(
        young_modulus_pa=float(payload.get("young_modulus_pa", 0.0)),
        width_m=float(payload.get("width_m", 0.0)),
        thickness_m=float(payload.get("thickness_m", 0.0)),
        y_m=y_values,
    )


def save_physics_config(config: PhysicsConfig, path: Path = PHYSICS_CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(config)
    payload["y_m"] = list(config.y_m)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
