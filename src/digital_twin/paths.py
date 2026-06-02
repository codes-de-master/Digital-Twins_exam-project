from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATASET_DIR = ROOT_DIR / "dataset"
INTERROGATOR_DIR = DATASET_DIR / "interrogator-data"
HANDHELD_XLSX = DATASET_DIR / "hand-held-data" / "data.xlsx"

COMPUTED_DIR = DATASET_DIR / "_computed"
MEDIAN_FILTER_DIR = COMPUTED_DIR / "04-median-filter"
PEAKS_DIR = COMPUTED_DIR / "05-find-peaks"
PEAK_GROUPS_DIR = COMPUTED_DIR / "06-peaks-groups"
MODEL_DIR = COMPUTED_DIR / "08-model-checkpoints"
PHYSICS_CONFIG_PATH = COMPUTED_DIR / "physics_config.json"


def ensure_computed_dirs() -> None:
    for path in (MEDIAN_FILTER_DIR, PEAKS_DIR, PEAK_GROUPS_DIR, MODEL_DIR):
        path.mkdir(parents=True, exist_ok=True)
