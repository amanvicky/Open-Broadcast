"""
OpenBroadcast — First Run Wizard

Shown on first launch to display detected hardware and auto-configured settings.
"""

from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QGroupBox, QFrame, QScrollArea, QWidget
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont


class FirstRunWizard(QDialog):
    """
    First-run dialog showing system specs and recommended settings.
    User can accept defaults or customize before proceeding.
    """

    def __init__(self, system_info, parent=None):
        super().__init__(parent)
        self.system_info = system_info
        self.accepted_config = None
        self.setWindowTitle("OpenBroadcast — First Launch Setup")
        self.setMinimumWidth(600)
        self.setMinimumHeight(500)
        self._setup_ui()

    def _setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Title
        title = QLabel("Welcome to OpenBroadcast!")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #00d4ff;")
        layout.addWidget(title)

        subtitle = QLabel("We've detected your hardware and configured optimal settings.")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle.setStyleSheet("color: #aaa;")
        layout.addWidget(subtitle)

        # Scroll area for system info
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # System specs
        self._add_system_section(scroll_layout)

        # Recommended settings
        self._add_config_section(scroll_layout)

        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

        # Buttons
        btn_layout = QHBoxLayout()

        self.accept_btn = QPushButton("Accept & Start")
        self.accept_btn.setMinimumHeight(40)
        self.accept_btn.setStyleSheet(
            "background-color: #00aa55; font-size: 14px; font-weight: bold;"
        )
        self.accept_btn.clicked.connect(self._on_accept)
        btn_layout.addWidget(self.accept_btn)

        layout.addLayout(btn_layout)

    def _add_system_section(self, layout):
        """Add system hardware section."""
        group = QGroupBox("Detected Hardware")
        group_layout = QVBoxLayout(group)

        cpu = self.system_info["cpu"]
        ram = self.system_info["ram"]
        gpu = self.system_info["gpu"]
        os_info = self.system_info["os"]

        specs = [
            f"CPU: {cpu['brand']}",
            f"Cores: {cpu['physical_cores']} physical / {cpu['logical_cores']} logical threads",
            f"AVX2: {'Yes' if cpu['has_avx2'] else 'No'} | AVX-512: {'Yes' if cpu['has_avx512'] else 'No'}",
            f"RAM: {ram['total_gb']} GB total, {ram['available_gb']} GB available",
            f"GPU: {gpu['name']}{' (Dedicated, ' + str(gpu['vram_mb']) + ' MB)' if gpu['has_dedicated'] else ' (Integrated)' if gpu['has_integrated'] else ' (None)'}",
            f"OS: {os_info['name']} ({os_info['version']})",
        ]

        # Cameras
        for cam in self.system_info["cameras"]:
            w, h = cam["resolution"]
            specs.append(f"Camera {cam['index']}: {w}x{h} @ {cam['max_fps']:.0f} FPS")

        if not self.system_info["cameras"]:
            specs.append("Camera: None detected!")

        for spec in specs:
            label = QLabel(spec)
            label.setStyleSheet("font-size: 13px; padding: 2px;")
            group_layout.addWidget(label)

        layout.addWidget(group)

    def _add_config_section(self, layout):
        """Add recommended configuration section."""
        config = self.system_info["config"]
        tier = config.get("tier", "MEDIUM")
        ram = self.system_info["ram"]

        group = QGroupBox(f"Recommended Settings (Tier: {tier})")
        group_layout = QVBoxLayout(group)

        # Tier color coding
        tier_colors = {
            "ULTRA_LOW": "#ff4444",
            "LOW": "#ff8844",
            "MEDIUM": "#ffcc44",
            "HIGH": "#44cc44",
        }
        tier_color = tier_colors.get(tier, "#888888")

        tier_label = QLabel(f"Performance Tier: {tier}")
        tier_label.setStyleSheet(f"font-size: 16px; font-weight: bold; color: {tier_color};")
        group_layout.addWidget(tier_label)

        settings = [
            f"Mode: {config['description']}",
            f"Processing Resolution: {config['processing_resolution'][0]}x{config['processing_resolution'][1]}",
            f"Target FPS: {config['max_fps']}",
            f"Correction Strength: {int(config['correction_strength_default'] * 100)}%",
            f"Neural Model: {'Enabled' if config['mode'].startswith('hybrid') else 'Disabled (geometric only)'}",
            f"Color Correction: {'On' if config['color_correction'] else 'Off'}",
            f"Virtual Camera: {'Supported' if config.get('enable_virtual_camera') else 'Disabled (low-end mode)'}",
        ]

        # 8GB RAM specific notes
        if 7.5 <= ram['total_gb'] < 9:
            settings.append("")
            settings.append("📋 8GB RAM Optimization:")
            settings.append("  • Using geometric + smoothing mode")
            settings.append("  • 720p processing for best balance")
            settings.append("  • Neural model runs every 3rd frame")
            settings.append("  • Memory monitoring active")

        for setting in settings:
            label = QLabel(setting)
            label.setStyleSheet("font-size: 13px; padding: 2px;")
            group_layout.addWidget(label)

        layout.addWidget(group)

    def _on_accept(self):
        """Accept the configuration and close."""
        self.accepted_config = self.system_info["config"]
        self.accept()

    def get_config(self):
        """Return the accepted configuration."""
        return self.accepted_config
