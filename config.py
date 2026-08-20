"""OpenBroadcast — Configuration."""

import os
import json
from pathlib import Path

APP_NAME = "OpenBroadcast"
APP_DIR = Path(os.path.expanduser("~")) / ".openbroadcast"
CONFIG_FILE = APP_DIR / "config.json"


def ensure_app_dir():
    APP_DIR.mkdir(parents=True, exist_ok=True)


def load_config():
    ensure_app_dir()
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return get_default_config()


def save_config(config):
    ensure_app_dir()
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_default_config():
    return {
        "camera_index": 0,
        "processing_resolution": [640, 480],
        "correction_strength": 0.85,
        "amplification": 1.0,
        "first_run": True,
    }
