"""
OpenBroadcast — Dark Theme Stylesheet
Professional dark UI matching broadcast software aesthetics.
"""

DARK_THEME = """
QMainWindow {
    background-color: #1a1a2e;
}

QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
}

QLabel {
    color: #e0e0e0;
    background: transparent;
}

QLabel#title {
    font-size: 18px;
    font-weight: bold;
    color: #00d4ff;
}

QLabel#subtitle {
    font-size: 11px;
    color: #888;
}

QLabel#fps_label {
    font-size: 14px;
    font-weight: bold;
    color: #00ff88;
    background-color: rgba(0, 0, 0, 0.3);
    padding: 4px 8px;
    border-radius: 4px;
}

QLabel#tier_label {
    font-size: 12px;
    font-weight: bold;
    padding: 4px 12px;
    border-radius: 4px;
    background-color: #2a5298;
    color: white;
}

QPushButton {
    background-color: #2a5298;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 8px 20px;
    font-size: 13px;
    font-weight: bold;
}

QPushButton:hover {
    background-color: #3a62b8;
}

QPushButton:pressed {
    background-color: #1a3278;
}

QPushButton#toggle_btn {
    background-color: #cc3333;
    min-width: 120px;
    min-height: 36px;
    font-size: 14px;
}

QPushButton#toggle_btn:checked {
    background-color: #00aa55;
}

QPushButton#calibrate_btn {
    background-color: #ff8800;
}

QPushButton#calibrate_btn:hover {
    background-color: #ffaa33;
}

QPushButton#virtual_cam_btn {
    background-color: #6633cc;
}

QPushButton#virtual_cam_btn:checked {
    background-color: #9933ff;
}

QSlider::groove:horizontal {
    border: none;
    height: 6px;
    background: #333355;
    border-radius: 3px;
}

QSlider::handle:horizontal {
    background: #00d4ff;
    width: 16px;
    height: 16px;
    margin: -5px 0;
    border-radius: 8px;
}

QSlider::handle:horizontal:hover {
    background: #00eeff;
}

QSlider::sub-page:horizontal {
    background: #00d4ff;
    border-radius: 3px;
}

QComboBox {
    background-color: #2a2a4a;
    color: #e0e0e0;
    border: 1px solid #444466;
    border-radius: 4px;
    padding: 6px 12px;
    min-width: 120px;
}

QComboBox:hover {
    border-color: #00d4ff;
}

QComboBox::drop-down {
    border: none;
    width: 24px;
}

QComboBox QAbstractItemView {
    background-color: #2a2a4a;
    color: #e0e0e0;
    selection-background-color: #2a5298;
    border: 1px solid #444466;
}

QGroupBox {
    border: 1px solid #333355;
    border-radius: 8px;
    margin-top: 12px;
    padding-top: 16px;
    font-weight: bold;
    color: #00d4ff;
}

QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
}

QCheckBox {
    spacing: 8px;
    color: #e0e0e0;
}

QCheckBox::indicator {
    width: 18px;
    height: 18px;
    border: 2px solid #444466;
    border-radius: 4px;
    background-color: #2a2a4a;
}

QCheckBox::indicator:checked {
    background-color: #00d4ff;
    border-color: #00d4ff;
}

QProgressBar {
    border: none;
    border-radius: 4px;
    background-color: #333355;
    text-align: center;
    color: white;
    height: 8px;
}

QProgressBar::chunk {
    background-color: #00d4ff;
    border-radius: 4px;
}

QScrollArea {
    border: none;
    background: transparent;
}

QScrollBar:vertical {
    background-color: #1a1a2e;
    width: 8px;
    border-radius: 4px;
}

QScrollBar::handle:vertical {
    background-color: #444466;
    border-radius: 4px;
    min-height: 30px;
}

QScrollBar::handle:vertical:hover {
    background-color: #666688;
}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
    height: 0px;
}

QStatusBar {
    background-color: #111122;
    color: #888;
    font-size: 12px;
}

QToolTip {
    background-color: #2a2a4a;
    color: #e0e0e0;
    border: 1px solid #444466;
    border-radius: 4px;
    padding: 6px;
}
"""


def apply_theme(app):
    """Apply the dark theme to the application."""
    app.setStyleSheet(DARK_THEME)
