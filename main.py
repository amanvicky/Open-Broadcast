"""
OpenBroadcast — Eye Gaze Correction for Low-End PCs

Main entry point. Detects hardware, configures automatically,
and launches the application.

Auto-installs dependencies on first run if missing.
"""

import sys
import os
import subprocess
import importlib

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


# ─── Auto-install dependencies if missing ──────────────────────────────
def ensure_dependencies():
    """
    Check if required packages are installed.
    If not, automatically install them via pip.
    """
    required = {
        "cv2": "opencv-python",
        "mediapipe": "mediapipe",
        "numpy": "numpy",
        "PyQt6": "PyQt6",
        "psutil": "psutil",
        "cpuinfo": "py-cpuinfo",
    }

    optional = {
        "pyvirtualcam": "pyvirtualcam",
        "onnxruntime": "onnxruntime",
        "wmi": "WMI",
        "screeninfo": "screeninfo",
    }

    missing_required = []
    missing_optional = []

    # Check required packages
    for module, package in required.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing_required.append(package)

    # Check optional packages
    for module, package in optional.items():
        try:
            importlib.import_module(module)
        except ImportError:
            missing_optional.append(package)

    # Install missing required packages
    if missing_required:
        print(f"[OpenBroadcast] Installing missing packages: {', '.join(missing_required)}")
        print("[OpenBroadcast] This may take a few minutes on first run...\n")

        try:
            # Find pip
            pip_cmd = [sys.executable, "-m", "pip", "install"]

            # Install all missing required packages at once
            result = subprocess.run(
                pip_cmd + missing_required + ["--quiet"],
                capture_output=True,
                text=True,
                timeout=300,
            )

            if result.returncode == 0:
                print("[OpenBroadcast] Required packages installed successfully!\n")
            else:
                print(f"[OpenBroadcast] Some packages failed to install:\n{result.stderr}\n")
                # Try one by one
                for pkg in missing_required:
                    try:
                        subprocess.run(
                            pip_cmd + [pkg, "--quiet"],
                            capture_output=True,
                            timeout=120,
                        )
                        print(f"  [OK] {pkg}")
                    except Exception as e:
                        print(f"  [FAIL] {pkg}: {e}")

        except subprocess.TimeoutExpired:
            print("[OpenBroadcast] Installation timed out. Please run: pip install -r requirements.txt")
        except Exception as e:
            print(f"[OpenBroadcast] Auto-install failed: {e}")
            print("[OpenBroadcast] Please run manually: pip install -r requirements.txt")

    # Try to install optional packages silently
    if missing_optional:
        try:
            pip_cmd = [sys.executable, "-m", "pip", "install"]
            subprocess.run(
                pip_cmd + missing_optional + ["--quiet"],
                capture_output=True,
                timeout=120,
            )
        except Exception:
            pass  # Optional, don't worry if they fail

    return len(missing_required) == 0


# Run dependency check before importing other modules
print("[OpenBroadcast] Checking dependencies...")
ensure_dependencies()


from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from config import (
    APP_NAME, APP_VERSION, load_config, save_config,
    get_default_config, ensure_app_dir,
)
from core.system_detector import detect_system, format_system_report
from ui.styles import apply_theme
from ui.main_window import MainWindow
from ui.first_run_wizard import FirstRunWizard


def main():
    """Application entry point."""
    # High DPI support
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)

    # Apply dark theme
    apply_theme(app)

    # Set default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Ensure app directory exists
    ensure_app_dir()

    # Load or create configuration
    config = load_config()

    # Detect system hardware
    print(f"[{APP_NAME}] Detecting system hardware...")
    try:
        system_info = detect_system()
    except Exception as e:
        print(f"[{APP_NAME}] System detection failed: {e}")
        QMessageBox.warning(
            None,
            "System Detection",
            f"Could not detect system hardware:\n{e}\n\n"
            "Using default settings."
        )
        system_info = {
            "cpu": {"brand": "Unknown", "physical_cores": 4, "logical_cores": 8,
                    "has_avx2": False, "has_avx512": False},
            "ram": {"total_gb": 8, "available_gb": 4, "speed_mhz": None},
            "gpu": {"has_dedicated": False, "has_integrated": True,
                    "name": "Unknown", "vram_mb": 0, "vendor": "Unknown"},
            "cameras": [],
            "os": {"name": "Windows", "version": "Unknown"},
            "disk_free_gb": 10,
            "tier": "MEDIUM",
            "config": get_default_config(),
        }

    # Print system report
    print(format_system_report(system_info))

    # Check if first run
    is_first_run = config.get("first_run", True)

    if is_first_run:
        # Show first-run wizard
        wizard = FirstRunWizard(system_info)
        result = wizard.exec()

        if result == FirstRunWizard.DialogCode.Accepted:
            # Apply auto-detected config + hardware info
            config.update(system_info["config"])
            config["cpu"] = system_info["cpu"]
            config["ram"] = system_info["ram"]
            config["gpu"] = system_info["gpu"]
            config["os"] = system_info["os"]
            config["first_run"] = False
            save_config(config)
            print(f"[{APP_NAME}] Configuration saved.")
        else:
            # User closed wizard — use defaults
            config = get_default_config()
            config.update(system_info["config"])
            config["cpu"] = system_info["cpu"]
            config["ram"] = system_info["ram"]
            config["gpu"] = system_info["gpu"]
            config["os"] = system_info["os"]
            config["first_run"] = False
    else:
        # Not first run — just merge detected hardware info
        config.update(system_info["config"])
        config["cpu"] = system_info["cpu"]
        config["ram"] = system_info["ram"]
        config["gpu"] = system_info["gpu"]
        config["os"] = system_info["os"]

    # Check for camera
    if not system_info["cameras"]:
        QMessageBox.warning(
            None,
            "No Camera Found",
            "No camera was detected.\n\n"
            "Please connect a webcam and restart the application."
        )

    # Auto-detect trained neural model and upgrade mode if found
    model_path = os.path.join(os.path.dirname(__file__), "models", "weights")
    quantized = os.path.join(model_path, "gaze_net_quantized.onnx")
    full_model = os.path.join(model_path, "gaze_net.onnx")

    if os.path.exists(quantized):
        config["model_file"] = "gaze_net_quantized.onnx"
        if config.get("mode") == "geometric_only":
            config["mode"] = "hybrid_balanced"
            print(f"[{APP_NAME}] Trained model found — upgrading to hybrid_balanced mode")
    elif os.path.exists(full_model):
        config["model_file"] = "gaze_net.onnx"
        if config.get("mode") == "geometric_only":
            config["mode"] = "hybrid_balanced"
            print(f"[{APP_NAME}] Trained model found — upgrading to hybrid_balanced mode")
    else:
        config["model_file"] = None
        if config.get("mode", "").startswith("hybrid"):
            config["mode"] = "geometric_with_smoothing"
            print(f"[{APP_NAME}] No trained model found — using geometric mode")

    # Create and show main window
    window = MainWindow(config)
    window.show()

    # Print startup info
    print(f"[{APP_NAME}] Application started.")
    print(f"[{APP_NAME}] Tier: {config.get('tier', 'UNKNOWN')}")
    print(f"[{APP_NAME}] Mode: {config.get('mode', 'unknown')}")
    print(f"[{APP_NAME}] Resolution: {config.get('processing_resolution', 'unknown')}")
    print(f"[{APP_NAME}] Target FPS: {config.get('max_fps', 'unknown')}")

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
