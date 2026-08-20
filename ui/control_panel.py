"""
OpenBroadcast — Control Panel Widget

Settings sidebar with all user-facing controls.
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QCheckBox, QGroupBox, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal


class ControlPanel(QWidget):
    """Sidebar with all application controls."""

    # Signals
    correction_toggled = pyqtSignal(bool)
    strength_changed = pyqtSignal(float)
    amplification_changed = pyqtSignal(float)
    landmark_overlay_toggled = pyqtSignal(bool)
    compare_mode_toggled = pyqtSignal(bool)
    demo_mode_toggled = pyqtSignal(bool)
    calibrate_requested = pyqtSignal()
    virtual_cam_toggled = pyqtSignal(bool)

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setFixedWidth(280)
        self._setup_ui()

    def _setup_ui(self):
        """Build the control panel UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        # Title
        title = QLabel("OpenBroadcast")
        title.setObjectName("title")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)

        subtitle = QLabel("Eye Gaze Correction")
        subtitle.setObjectName("subtitle")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)

        # Performance tier badge
        tier = self.config.get("tier", "MEDIUM")
        self.tier_label = QLabel(f"Tier: {tier}")
        self.tier_label.setObjectName("tier_label")
        self.tier_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.tier_label)

        layout.addSpacing(8)

        # Scroll area for controls
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 0, 0)
        scroll_layout.setSpacing(10)

        # --- Correction Group ---
        correction_group = QGroupBox("Eye Correction")
        correction_layout = QVBoxLayout(correction_group)

        # Toggle
        self.toggle_btn = QPushButton("ENABLED")
        self.toggle_btn.setObjectName("toggle_btn")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(True)
        self.toggle_btn.clicked.connect(self._on_toggle_clicked)
        correction_layout.addWidget(self.toggle_btn)

        # Strength slider
        strength_label = QLabel("Correction Strength")
        correction_layout.addWidget(strength_label)

        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(int(self.config.get("correction_strength", 0.85) * 100))
        self.strength_slider.valueChanged.connect(self._on_strength_changed)
        correction_layout.addWidget(self.strength_slider)

        self.strength_value = QLabel(f"{self.strength_slider.value()}%")
        self.strength_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        correction_layout.addWidget(self.strength_value)

        # Amplification slider
        amp_label = QLabel("Correction Amplification")
        amp_label.setToolTip("Multiplier for correction shift. Higher = more dramatic effect.")
        correction_layout.addWidget(amp_label)

        self.amp_slider = QSlider(Qt.Orientation.Horizontal)
        self.amp_slider.setRange(100, 500)  # 1.0x to 5.0x
        self.amp_slider.setValue(400)  # Default 4.0x
        self.amp_slider.valueChanged.connect(self._on_amp_changed)
        correction_layout.addWidget(self.amp_slider)

        self.amp_value = QLabel("4.0x")
        self.amp_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        correction_layout.addWidget(self.amp_value)

        scroll_layout.addWidget(correction_group)

        # --- Display Group ---
        display_group = QGroupBox("Display")
        display_layout = QVBoxLayout(display_group)

        self.compare_cb = QCheckBox("Compare: Original vs Corrected")
        self.compare_cb.setToolTip("Show split-screen: left=original, right=corrected")
        self.compare_cb.toggled.connect(self.compare_mode_toggled.emit)
        display_layout.addWidget(self.compare_cb)

        self.landmark_cb = QCheckBox("Show Face Landmarks")
        self.landmark_cb.toggled.connect(self.landmark_overlay_toggled.emit)
        display_layout.addWidget(self.landmark_cb)

        self.demo_cb = QCheckBox("Demo: Simulate 20° Gaze Offset")
        self.demo_cb.setToolTip("Shows what correction looks like at 20 degrees")
        self.demo_cb.toggled.connect(self.demo_mode_toggled.emit)
        display_layout.addWidget(self.demo_cb)

        scroll_layout.addWidget(display_group)

        # --- Virtual Camera Group ---
        vcam_group = QGroupBox("Virtual Camera")
        vcam_layout = QVBoxLayout(vcam_group)

        self.vcam_btn = QPushButton("Start Virtual Camera")
        self.vcam_btn.setObjectName("virtual_cam_btn")
        self.vcam_btn.setCheckable(True)
        self.vcam_btn.clicked.connect(self._on_vcam_clicked)
        vcam_layout.addWidget(self.vcam_btn)

        vcam_hint = QLabel("Outputs to OBS / Zoom / Teams")
        vcam_hint.setObjectName("subtitle")
        vcam_layout.addWidget(vcam_hint)

        scroll_layout.addWidget(vcam_group)

        # --- Calibration Group ---
        cal_group = QGroupBox("Calibration")
        cal_layout = QVBoxLayout(cal_group)

        self.calibrate_btn = QPushButton("Calibrate Gaze")
        self.calibrate_btn.setObjectName("calibrate_btn")
        self.calibrate_btn.clicked.connect(self.calibrate_requested.emit)
        cal_layout.addWidget(self.calibrate_btn)

        cal_hint = QLabel("Look at camera center for 2 seconds")
        cal_hint.setObjectName("subtitle")
        cal_layout.addWidget(cal_hint)

        scroll_layout.addWidget(cal_group)

        # --- System Info Group ---
        info_group = QGroupBox("System Info")
        info_layout = QVBoxLayout(info_group)

        cpu = self.config.get("cpu", {})
        ram = self.config.get("ram", {})
        self.info_label = QLabel(
            f"CPU: {cpu.get('brand', 'N/A')}\n"
            f"Cores: {cpu.get('physical_cores', '?')} / {cpu.get('logical_cores', '?')}\n"
            f"RAM: {ram.get('total_gb', '?')} GB"
        )
        self.info_label.setObjectName("subtitle")
        self.info_label.setWordWrap(True)
        info_layout.addWidget(self.info_label)

        scroll_layout.addWidget(info_group)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

    def _on_toggle_clicked(self, checked):
        """Handle correction toggle."""
        self.toggle_btn.setText("ENABLED" if checked else "DISABLED")
        self.correction_toggled.emit(checked)

    def _on_strength_changed(self, value):
        """Handle strength slider change."""
        self.strength_value.setText(f"{value}%")
        self.strength_changed.emit(value / 100.0)

    def _on_amp_changed(self, value):
        """Handle amplification slider change."""
        amp = value / 100.0
        self.amp_value.setText(f"{amp:.1f}x")
        self.amplification_changed.emit(amp)

    def _on_vcam_clicked(self, checked):
        """Handle virtual camera toggle."""
        self.vcam_btn.setText("Stop Virtual Camera" if checked else "Start Virtual Camera")
        self.virtual_cam_toggled.emit(checked)

    def update_fps(self, fps):
        """Update FPS display if needed."""
        pass

    def set_calibrating(self, active):
        """Update UI during calibration."""
        self.calibrate_btn.setEnabled(not active)
        if active:
            self.calibrate_btn.setText("Looking at camera...")
        else:
            self.calibrate_btn.setText("Calibrate Gaze")
