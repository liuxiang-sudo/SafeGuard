"""Shared resource paths, independent of the current working directory."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DATASETS_DIR = DATA_DIR / "datasets"
PROCESSED_DIR = DATA_DIR / "processed"
TRAINING_SAMPLES_DIR = DATA_DIR / "training_samples"
TWIBOT_DIR = DATA_DIR / "raw" / "TwiBot-20"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
EXAMPLES_DIR = PROJECT_ROOT / "examples" / "data"
WEB_DIR = Path(__file__).resolve().parent / "web"


def ensure_runtime_dirs():
    """Create writable runtime directories when an entry point needs them."""
    for directory in (DATASETS_DIR, PROCESSED_DIR, TRAINING_SAMPLES_DIR, MODELS_DIR, REPORTS_DIR):
        directory.mkdir(parents=True, exist_ok=True)
