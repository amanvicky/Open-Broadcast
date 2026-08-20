"""
OpenBroadcast — Application Configuration

Default settings that get overridden by system detection at runtime.
Supports hardware from 8GB RAM / 8GB disk up to 30GB+ RAM / 40GB+ disk.
"""
import os
import json
from pathlib import Path

# Application info
APP_NAME = "OpenBroadcast"
APP_VERSION = "1.0.0"
APP_AUTHOR = "OpenBroadcast Team"

# Paths
APP_DIR = Path(os.path.expanduser("~")) / ".openbroadcast"
CONFIG_FILE = APP_DIR / "config.json"
LOG_FILE = APP_DIR / "openbroadcast.log"
WEIGHTS_DIR = Path(__file__).parent / "models" / "weights"

# Default camera settings
DEFAULT_CAMERA_INDEX = 0
DEFAULT_RESOLUTION = (640, 480)
DEFAULT_TARGET_FPS = 20

# Default processing settings
DEFAULT_MODE = "geometric_only"
DEFAULT_CORRECTION_STRENGTH = 0.85
DEFAULT_NEURAL_FRAME_SKIP = 3
DEFAULT_INFERENCE_THREADS = 2
DEFAULT_COLOR_CORRECTION = True
DEFAULT_FEATHER_RADIUS = 15

# ─── Performance tiers ─────────────────────────────────────────────
# Covers 8GB RAM / 8GB disk → 30GB+ RAM / 40GB+ disk
PERFORMANCE_TIERS = {
    "ULTRA_LOW": {
        "mode": "geometric_only",
        "model_file": None,
        "processing_resolution": (480, 360),
        "display_resolution": (640, 480),
        "neural_frame_skip": 0,
        "inference_threads": 1,
        "max_fps": 15,
        "color_correction": False,
        "feather_radius": 10,
        "correction_strength_default": 0.7,
        "enable_virtual_camera": False,
        "description": "Geometric-only mode for older hardware",
    },
    "LOW": {
        "mode": "geometric_with_smoothing",
        "model_file": None,
        "processing_resolution": (640, 480),
        "display_resolution": (640, 480),
        "neural_frame_skip": 0,
        "inference_threads": 1,
        "max_fps": 20,
        "color_correction": True,
        "feather_radius": 12,
        "correction_strength_default": 0.75,
        "enable_virtual_camera": True,
        "description": "Geometric correction with smoothing",
    },
    "MEDIUM": {
        "mode": "hybrid_balanced",
        "model_file": "gaze_net_quantized.onnx",
        "processing_resolution": (1280, 720),
        "display_resolution": (1280, 720),
        "neural_frame_skip": 3,
        "inference_threads": 2,
        "max_fps": 25,
        "color_correction": True,
        "feather_radius": 15,
        "correction_strength_default": 0.85,
        "enable_virtual_camera": True,
        "description": "Hybrid mode — geometric + neural correction",
    },
    "HIGH": {
        "mode": "hybrid_quality",
        "model_file": "gaze_net.onnx",
        "processing_resolution": (1920, 1080),
        "display_resolution": (1920, 1080),
        "neural_frame_skip": 2,
        "inference_threads": 4,
        "max_fps": 30,
        "color_correction": True,
        "feather_radius": 18,
        "correction_strength_default": 0.9,
        "enable_virtual_camera": True,
        "description": "High quality hybrid correction",
    },
    "ULTRA_HIGH": {
        "mode": "hybrid_quality",
        "model_file": "gaze_net.onnx",
        "processing_resolution": (1920, 1080),
        "display_resolution": (1920, 1080),
        "neural_frame_skip": 1,
        "inference_threads": 6,
        "max_fps": 30,
        "color_correction": True,
        "feather_radius": 20,
        "correction_strength_default": 0.95,
        "enable_virtual_camera": True,
        "description": "Maximum quality — dedicated GPU recommended",
    },
}


def ensure_app_dir():
    """Create app directory if it doesn't exist."""
    APP_DIR.mkdir(parents=True, exist_ok=True)
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)


def load_config():
    """Load saved configuration or return defaults."""
    ensure_app_dir()
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    return get_default_config()


def save_config(config):
    """Save configuration to disk."""
    ensure_app_dir()
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def get_default_config():
    """Return default configuration."""
    return {
        "camera_index": DEFAULT_CAMERA_INDEX,
        "resolution": list(DEFAULT_RESOLUTION),
        "target_fps": DEFAULT_TARGET_FPS,
        "mode": DEFAULT_MODE,
        "correction_strength": DEFAULT_CORRECTION_STRENGTH,
        "neural_frame_skip": DEFAULT_NEURAL_FRAME_SKIP,
        "inference_threads": DEFAULT_INFERENCE_THREADS,
        "color_correction": DEFAULT_COLOR_CORRECTION,
        "feather_radius": DEFAULT_FEATHER_RADIUS,
        "enable_virtual_camera": False,
        "enable_landmark_overlay": False,
        "performance_tier": "MEDIUM",
        "first_run": True,
    }
