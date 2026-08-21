"""OpenBroadcast — Control Panel."""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QPushButton,
    QSlider, QCheckBox, QGroupBox, QScrollArea, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal


class ControlPanel(QWidget):
    correction_toggled = pyqtSignal(bool)
    strength_changed = pyqtSignal(float)
    amplification_changed = pyqtSignal(float)
    compare_mode_toggled = pyqtSignal(bool)
    landmarks_toggled = pyqtSignal(bool)
    calibrate_clicked = pyqtSignal()
    virtual_cam_toggled = pyqtSignal(bool)
    mode_changed = pyqtSignal(int)
    wizard_clicked = pyqtSignal()
    train_clicked = pyqtSignal()
    record_toggled = pyqtSignal(bool)
    camera_changed = pyqtSignal(int)
    camera_restart = pyqtSignal()

    def __init__(self, config, parent=None):
        super().__init__(parent)
        self.config = config
        self.setFixedWidth(280)
        self._setup_ui()

    def _setup_ui(self):
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

        layout.addSpacing(8)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        content = QWidget()
        cl = QVBoxLayout(content)
        cl.setContentsMargins(0, 0, 0, 0)
        cl.setSpacing(10)

        # Correction group (always open)
        group = QGroupBox("Eye Correction")
        group.setCheckable(False)
        gl = QVBoxLayout(group)

        self.toggle_btn = QPushButton("ENABLED")
        self.toggle_btn.setObjectName("toggle_btn")
        self.toggle_btn.setCheckable(True)
        self.toggle_btn.setChecked(True)
        self.toggle_btn.clicked.connect(self._on_toggle)
        gl.addWidget(self.toggle_btn)

        gl.addWidget(QLabel("Correction Strength"))
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(int(self.config.get("correction_strength", 0.85) * 100))
        self.strength_slider.valueChanged.connect(self._on_strength)
        gl.addWidget(self.strength_slider)

        self.strength_value = QLabel(f"{self.strength_slider.value()}%")
        self.strength_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gl.addWidget(self.strength_value)

        amp_label = QLabel("Correction Amplification")
        amp_label.setToolTip("Multiplier for correction shift")
        gl.addWidget(amp_label)

        self.amp_slider = QSlider(Qt.Orientation.Horizontal)
        self.amp_slider.setRange(100, 500)
        amp_default = int(self.config.get("amplification", 4.0) * 100)
        # Ensure minimum 4.0x on startup for visible correction
        self.amp_slider.setValue(max(400, min(500, amp_default)))
        self.amp_slider.valueChanged.connect(self._on_amp)
        gl.addWidget(self.amp_slider)

        self.amp_value = QLabel(f"{self.amp_slider.value() / 100:.1f}x")
        self.amp_value.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gl.addWidget(self.amp_value)

        cl.addWidget(group)

        # Camera group
        cam_group = QGroupBox("Camera")
        cam_group.setCheckable(False)
        cam_gl = QVBoxLayout(cam_group)

        cam_gl.addWidget(QLabel("Camera"))
        self.camera_combo = QComboBox()
        self.camera_combo.currentIndexChanged.connect(self._on_camera_changed)
        cam_gl.addWidget(self.camera_combo)

        self.camera_restart_btn = QPushButton("Restart Camera")
        self.camera_restart_btn.setToolTip("Try again if camera is stuck")
        self.camera_restart_btn.clicked.connect(self.camera_restart.emit)
        cam_gl.addWidget(self.camera_restart_btn)

        self.camera_status = QLabel("")
        self.camera_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cam_gl.addWidget(self.camera_status)

        cl.addWidget(cam_group)

        # Display group (collapsible)
        dg = QGroupBox("Display")
        dg.setCheckable(True)
        dg.setChecked(False)
        dg.toggled.connect(lambda checked: self._toggle_group(dg, checked))
        dl = QVBoxLayout(dg)

        self.compare_cb = QCheckBox("Compare: Original vs Corrected")
        self.compare_cb.toggled.connect(self.compare_mode_toggled.emit)
        dl.addWidget(self.compare_cb)

        self.landmarks_cb = QCheckBox("Show Face Landmarks")
        self.landmarks_cb.toggled.connect(self.landmarks_toggled.emit)
        dl.addWidget(self.landmarks_cb)

        self.demo_cb = QCheckBox("Demo Mode (Simulate 20° Offset)")
        self.demo_cb.setToolTip("Prove correction works by simulating a 20-degree gaze offset")
        dl.addWidget(self.demo_cb)

        self.batch_btn = QPushButton("Process Video File (Ctrl+B)")
        self.batch_btn.setToolTip("Apply correction to an existing video file")
        dl.addWidget(self.batch_btn)

        cl.addWidget(dg)

        # Performance group (collapsible)
        pg = QGroupBox("Performance")
        pg.setCheckable(True)
        pg.setChecked(False)
        pg.toggled.connect(lambda checked: self._toggle_group(pg, checked))
        pl = QVBoxLayout(pg)

        pl.addWidget(QLabel("Mode"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Geometric (Fastest)", "Neural (Best)"])
        self.mode_combo.setCurrentIndex(0)
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        pl.addWidget(self.mode_combo)

        cl.addWidget(pg)

        # Calibration group (collapsible)
        cg = QGroupBox("Calibration")
        cg.setCheckable(True)
        cg.setChecked(False)
        cg.toggled.connect(lambda checked: self._toggle_group(cg, checked))
        cgl = QVBoxLayout(cg)

        self.calibrate_btn = QPushButton("Calibrate Gaze")
        self.calibrate_btn.setObjectName("calibrate_btn")
        self.calibrate_btn.setToolTip("Look at camera for 2 seconds, then click")
        self.calibrate_btn.clicked.connect(self.calibrate_clicked.emit)
        cgl.addWidget(self.calibrate_btn)

        self.calibrate_status = QLabel("")
        self.calibrate_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cgl.addWidget(self.calibrate_status)

        self.wizard_btn = QPushButton("Interactive Calibration")
        self.wizard_btn.setToolTip("Follow the moving dot to calibrate your gaze")
        self.wizard_btn.clicked.connect(self.wizard_clicked.emit)
        cgl.addWidget(self.wizard_btn)

        cl.addWidget(cg)

        # Virtual Camera group (collapsible)
        vg = QGroupBox("Output")
        vg.setCheckable(True)
        vg.setChecked(False)
        vg.toggled.connect(lambda checked: self._toggle_group(vg, checked))
        vl = QVBoxLayout(vg)

        self.virtcam_btn = QPushButton("Virtual Camera: OFF")
        self.virtcam_btn.setObjectName("virtual_cam_btn")
        self.virtcam_btn.setCheckable(True)
        self.virtcam_btn.setToolTip("Output corrected video to OBS/Zoom/Teams")
        self.virtcam_btn.clicked.connect(self._on_virtcam)
        vl.addWidget(self.virtcam_btn)

        self.virtcam_status = QLabel("")
        self.virtcam_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        vl.addWidget(self.virtcam_status)

        cl.addWidget(vg)

        # Recording group (collapsible)
        rg = QGroupBox("Recording")
        rg.setCheckable(True)
        rg.setChecked(False)
        rg.toggled.connect(lambda checked: self._toggle_group(rg, checked))
        rgl = QVBoxLayout(rg)

        self.record_btn = QPushButton("Record Corrected Video")
        self.record_btn.setObjectName("record_btn")
        self.record_btn.setCheckable(True)
        self.record_btn.setToolTip("Save corrected video to file (press R)")
        self.record_btn.clicked.connect(self._on_record)
        rgl.addWidget(self.record_btn)

        self.record_status = QLabel("")
        self.record_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        rgl.addWidget(self.record_status)

        cl.addWidget(rg)

        # Training group (collapsible)
        tg = QGroupBox("Neural Training")
        tg.setCheckable(True)
        tg.setChecked(False)
        tg.toggled.connect(lambda checked: self._toggle_group(tg, checked))
        tgl = QVBoxLayout(tg)

        self.train_btn = QPushButton("Train Model")
        self.train_btn.setToolTip("Collect data + train neural model (~2 min)")
        self.train_btn.clicked.connect(self.train_clicked.emit)
        tgl.addWidget(self.train_btn)

        self.train_status = QLabel("")
        self.train_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        tgl.addWidget(self.train_status)

        cl.addWidget(tg)

        # About button
        self.about_btn = QPushButton("About OpenBroadcast")
        self.about_btn.setToolTip("Version info and credits")
        cl.addWidget(self.about_btn)

        cl.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

    def _on_toggle(self, checked):
        self.toggle_btn.setText("ENABLED" if checked else "DISABLED")
        self.correction_toggled.emit(checked)

    def _on_strength(self, value):
        self.strength_value.setText(f"{value}%")
        self.strength_changed.emit(value / 100.0)

    def _on_amp(self, value):
        amp = value / 100.0
        self.amp_value.setText(f"{amp:.1f}x")
        self.amplification_changed.emit(amp)

    def _on_virtcam(self, checked):
        self.virtcam_btn.setText("Virtual Camera: ON" if checked else "Virtual Camera: OFF")
        self.virtual_cam_toggled.emit(checked)

    def set_calibrate_status(self, text):
        self.calibrate_status.setText(text)

    def set_virtcam_status(self, text):
        self.virtcam_status.setText(text)

    def set_train_status(self, text):
        self.train_status.setText(text)

    def _on_record(self, checked):
        self.record_btn.setText("Stop Recording" if checked else "Record Corrected Video")
        self.record_toggled.emit(checked)

    def set_record_status(self, text):
        self.record_status.setText(text)

    def set_mode_available(self, index, available):
        """Enable/disable a mode option."""
        self.mode_combo.setItemData(index, available, Qt.ItemDataRole.UserRole - 1)

    def _on_camera_changed(self, index):
        if index >= 0:
            self.camera_changed.emit(index)

    def populate_cameras(self, cameras):
        """Fill camera dropdown from enumerated list."""
        self.camera_combo.clear()
        for cam in cameras:
            w, h = cam["default_resolution"]
            self.camera_combo.addItem(f"Camera {cam['index']} ({w}x{h})", cam["index"])

    def set_camera_status(self, text):
        self.camera_status.setText(text)

    def _on_mode_changed(self, index):
        self.mode_changed.emit(index)

    def _toggle_group(self, group, checked):
        """Show/hide all children in a collapsible group."""
        for i in range(group.layout().count()):
            child = group.layout().itemAt(i)
            if child and child.widget():
                child.widget().setVisible(checked)
