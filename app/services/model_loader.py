from pathlib import Path
from typing import Any

import joblib

from app.config import get_settings


def get_model_path() -> Path:
    settings = get_settings()
    return Path(settings.model_path)


def is_model_ready() -> bool:
    return get_model_path().exists()


def load_model() -> Any:
    model_path = get_model_path()

    if not model_path.exists():
        return None

    return joblib.load(model_path)
