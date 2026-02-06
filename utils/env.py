import json
import os
from typing import Optional, Dict


def load_secrets(secrets_path: str = "secrets.json") -> Dict[str, str]:
    with open(secrets_path, "r") as f:
        secrets = json.load(f)
    return secrets


def setup_kaggle_env(secrets: Dict[str, str]) -> None:
    if "KAGGLE_USERNAME" in secrets:
        os.environ["KAGGLE_USERNAME"] = secrets["KAGGLE_USERNAME"]
    if "KAGGLE_KEY" in secrets:
        os.environ["KAGGLE_KEY"] = secrets["KAGGLE_KEY"]


def setup_wandb_env(secrets: Dict[str, str]) -> None:
    if "wandb_API_KEY" in secrets:
        os.environ["WANDB_API_KEY"] = secrets["wandb_API_KEY"]


def load_and_setup_env(
    secrets_path: str = "secrets.json",
    set_kaggle: bool = True,
    set_wandb: bool = True,
) -> Dict[str, str]:
    secrets = load_secrets(secrets_path)
    if set_kaggle:
        setup_kaggle_env(secrets)
    if set_wandb:
        setup_wandb_env(secrets)
    return secrets
