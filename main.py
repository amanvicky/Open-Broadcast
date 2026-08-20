"""OpenBroadcast — Eye Gaze Correction for Low-End PCs."""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtGui import QFont

from config import load_config, save_config, get_default_config, ensure_app_dir
from ui.styles import apply_theme
from ui.main_window import MainWindow


def main():
    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    app = QApplication(sys.argv)
    app.setApplicationName("OpenBroadcast")
    apply_theme(app)
    app.setFont(QFont("Segoe UI", 10))

    ensure_app_dir()
    config = load_config()

    # On first run, apply defaults
    if config.get("first_run", True):
        config = get_default_config()
        config["first_run"] = False
        save_config(config)

    window = MainWindow(config)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
