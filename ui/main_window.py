"""OpenBroadcast — Main Window. Camera → Face → Gaze → Correction → Display."""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout,
    QMessageBox, QSplitter
)
from PyQt6.QtCore import Qt

from core.camera import CameraThread
from core.face_detector import FaceDetector
from core.gaze_estimator import GeometricGazeEstimator
from core.eye_corrector import EyeCorrector
from utils.performance import FPSCounter
from ui.preview_widget import PreviewWidget
from ui.control_panel import ControlPanel


class MainWindow(QMainWindow):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setWindowTitle("OpenBroadcast — Eye Gaze Correction")
        self.setMinimumSize(900, 600)

        # Core pipeline
        self.face_detector = FaceDetector(detection_interval=3)
        self.gaze_estimator = GeometricGazeEstimator(
            ema_alpha=0.6,
            look_at_camera_threshold=0.15,
        )
        self.eye_corrector = EyeCorrector(
            strength=config.get("correction_strength", 0.85),
        )
        self.fps_counter = FPSCounter()

        # State
        self.correction_enabled = True

        self._setup_ui()
        self._start_camera()

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        self.preview = PreviewWidget()
        splitter.addWidget(self.preview)

        self.control_panel = ControlPanel(self.config)
        splitter.addWidget(self.control_panel)

        splitter.setSizes([700, 280])
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 0)

        main_layout.addWidget(splitter)

        # Signals
        self.control_panel.correction_toggled.connect(self._on_correction_toggled)
        self.control_panel.strength_changed.connect(self._on_strength_changed)
        self.control_panel.amplification_changed.connect(self._on_amplification_changed)
        self.control_panel.compare_mode_toggled.connect(self._on_compare_toggled)

        # Sync initial values
        self.eye_corrector.strength = self.control_panel.strength_slider.value() / 100.0
        self.eye_corrector.amplification = self.control_panel.amp_slider.value() / 100.0

        self.preview.correction_enabled = self.correction_enabled

    def _start_camera(self):
        cam_index = self.config.get("camera_index", 0)
        resolution = self.config.get("processing_resolution", [640, 480])
        w, h = resolution[0], resolution[1]

        self.camera_thread = CameraThread(
            camera_index=cam_index,
            target_width=w,
            target_height=h,
        )
        self.camera_thread.frame_ready.connect(self._on_frame_ready)
        self.camera_thread.error_occurred.connect(self._on_camera_error)
        self.camera_thread.start()

    def _on_frame_ready(self, frame, timestamp):
        self.fps_counter.update()

        landmarks = self.face_detector.detect(frame)

        if landmarks is None:
            self.preview.update_frame(frame)
            self.preview.set_gaze_info(None)
            self.statusBar().showMessage("No face detected")
            self.preview.set_fps(self.fps_counter.fps)
            return

        # Eye data → gaze → correction
        eye_data = self.face_detector.get_eye_data(landmarks, frame.shape)
        gaze = self.gaze_estimator.estimate(eye_data)

        if self.correction_enabled and not eye_data["is_blinking"]:
            display_frame = self.eye_corrector.correct_frame(frame, eye_data)
        else:
            display_frame = frame

        # Display
        self.preview.update_frame(frame, display_frame)

        self.preview.set_fps(self.fps_counter.fps)
        self.preview.set_gaze_info({
            "yaw": gaze.yaw,
            "pitch": gaze.pitch,
            "is_looking_at_camera": gaze.is_looking_at_camera,
        })

        # Status bar
        mode = "ON" if self.correction_enabled else "OFF"
        self.statusBar().showMessage(
            f"Correction:{mode} | "
            f"Y={gaze.yaw:+.1f}° P={gaze.pitch:+.1f}° | "
            f"FPS:{self.fps_counter.fps:.1f}"
        )

    def _on_camera_error(self, message):
        QMessageBox.critical(self, "Camera Error", message)

    def _on_correction_toggled(self, enabled):
        self.correction_enabled = enabled
        self.preview.correction_enabled = enabled

    def _on_strength_changed(self, strength):
        self.eye_corrector.strength = strength
        self.config["correction_strength"] = strength

    def _on_amplification_changed(self, amp):
        self.eye_corrector.amplification = amp
        self.config["amplification"] = amp

    def _on_compare_toggled(self, enabled):
        self.preview.set_compare_mode(enabled)

    def closeEvent(self, event):
        if hasattr(self, "camera_thread") and self.camera_thread:
            self.camera_thread.stop()
        self.face_detector.cleanup()
        event.accept()
