from __future__ import annotations

import os
from pathlib import Path

import yaml

from .schemas import Config

DEFAULT_CONFIG_PATH = Path(__file__).with_name("config.yaml")


def load_config(path: str | None = None) -> Config:
    config_path = Path(path or os.environ.get("SCT_CONFIG", str(DEFAULT_CONFIG_PATH)))
    with open(config_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config.model_validate(data)
