"""
OpenBroadcast — Main Window

Orchestrates camera, face detection, gaze estimation, eye correction, and display.
Includes demo mode that simulates large gaze offsets to prove correction works.
"""

import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout, QVBoxLayout,
    QMessageBox, QSplitter
)
from PyQt6.QtCore import Qt, QTimer

from core.camera import CameraThread
from core.face_detector import FaceDetector
from core.gaze_estimator import GeometricGazeEstimator
from core.eye_corrector import EyeCorrector
from utils.performance import FPSCounter, AdaptivePerformanceController
from ui.preview_widget import PreviewWidget
from ui.control_panel import ControlPanel


class MainWindow(QMainWindow):
    """Main application window."""

    def __init__(self, config):
        super().__init__()
        self.config = config
        self.setWindowTitle("OpenBroadcast — Eye Gaze Correction")
        self.setMinimumSize(900, 600)

        # Core components
        self.face_detector = FaceDetector(detection_interval=3)
        self.gaze_estimator = GeometricGazeEstimator(
            ema_alpha=0.6,
            look_at_camera_threshold=0.15,
        )
        self.eye_corrector = EyeCorrector(
            strength=config.get("correction_strength", 0.85),
        )
        self.fps_counter = FPSCounter()
        self.perf_controller = AdaptivePerformanceController(
            target_fps=config.get('max_fps', 20),
            total_ram_gb=config.get('ram', {}).get('total_gb', 8),
        )

        # State
        self.correction_enabled = True
        self.show_landmarks = config.get("enable_landmark_overlay", False)
        self.compare_mode = False
        self.demo_mode = False  # Synthetic 20° offset demo
        self.camera_running = False
        self.current_frame = None
        self.original_frame = None
        self.calibrating = False
        self.last_landmarks = None
        self._skip_frame = False

        # Build UI
        self._setup_ui()
        self._start_camera()

    def _setup_ui(self):
        """Build the main window layout."""
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

        # Connect signals
        self.control_panel.correction_toggled.connect(self._on_correction_toggled)
        self.control_panel.strength_changed.connect(self._on_strength_changed)
        self.control_panel.amplification_changed.connect(self._on_amplification_changed)
        self.control_panel.landmark_overlay_toggled.connect(self._on_landmark_toggled)
        self.control_panel.compare_mode_toggled.connect(self._on_compare_toggled)
        self.control_panel.demo_mode_toggled.connect(self._on_demo_toggled)
        self.control_panel.calibrate_requested.connect(self._on_calibrate)
        self.control_panel.virtual_cam_toggled.connect(self._on_virtual_cam_toggled)

        # Sync initial slider values to corrector
        self.eye_corrector.strength = self.control_panel.strength_slider.value() / 100.0
        self.eye_corrector.amplification = self.control_panel.amp_slider.value() / 100.0

        tier = self.config.get('tier', 'UNKNOWN')
        self.statusBar().showMessage(f"Tier: {tier} — Initializing camera...")

        self.preview.correction_enabled = self.correction_enabled
        self.preview.show_landmarks = self.show_landmarks

    def _start_camera(self):
        """Start the camera capture thread."""
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
        self.camera_thread.fps_updated.connect(self._on_camera_fps)
        self.camera_thread.start()
        self.camera_running = True

    def _apply_demo_offset(self, eye_data):
        """
        Apply synthetic 20° gaze offset to eye data.
        Returns a NEW dict — never mutates the original.
        """
        demo_offset = 0.33  # ~20° gaze offset
        import copy
        data = copy.deepcopy(eye_data)

        for side in ["left_eye", "right_eye"]:
            eye = data[side]
            eye["offset_x"] = eye["offset_x"] + demo_offset
            eye["iris"] = eye["center"].copy()
            eye["iris"][0] += eye["offset_x"] * eye["width"]

        return data

    def _on_frame_ready(self, frame, timestamp):
        """Handle new camera frame — run the full pipeline."""
        self.current_frame = frame
        self.fps_counter.update()
        self.perf_controller.update()
        self.original_frame = frame.copy()

        # Face detection
        if self.perf_controller.current_mode == "minimal":
            self._skip_frame = not self._skip_frame
            if self._skip_frame and self.last_landmarks is not None:
                landmarks = self.last_landmarks
            else:
                landmarks = self.face_detector.detect(frame)
        else:
            landmarks = self.face_detector.detect(frame)

        if landmarks is None:
            self.preview.update_frame(frame)
            self.preview.set_gaze_info(None)
            self.preview.set_eye_data(None)
            self.statusBar().showMessage("No face detected — looking for face...")
            self.preview.set_fps(self.fps_counter.fps)
            return

        self.last_landmarks = landmarks

        # Extract eye data
        eye_data = self.face_detector.get_eye_data(landmarks, frame.shape)

        # Gaze estimation (on REAL eye data, before demo offset)
        if self.calibrating:
            self.gaze_estimator.add_calibration_sample(eye_data)

        gaze = self.gaze_estimator.estimate(eye_data)

        # Apply demo offset if enabled (AFTER gaze estimation, BEFORE correction)
        if self.demo_mode:
            eye_data = self._apply_demo_offset(eye_data)

        # Eye correction
        should_correct = (
            self.correction_enabled
            and not eye_data["is_blinking"]
            and not (self.perf_controller.current_mode == "minimal" and self._skip_frame)
        )

        # Calculate correction displacement
        left = eye_data["left_eye"]
        right = eye_data["right_eye"]
        avg_offset_x = (left["offset_x"] + right["offset_x"]) / 2
        avg_offset_y = (left["offset_y"] + right["offset_y"]) / 2
        avg_eye_width = (left["width"] + right["width"]) / 2

        if should_correct:
            display_frame = self.eye_corrector.correct_frame(frame, eye_data)
            corr_px_x = abs(avg_offset_x * avg_eye_width * self.eye_corrector.strength)
            corr_px_y = abs(avg_offset_y * avg_eye_width * self.eye_corrector.strength * 0.7)
            corr_total = (corr_px_x**2 + corr_px_y**2) ** 0.5
        else:
            display_frame = frame.copy()
            corr_total = 0

        # Draw overlays
        if self.show_landmarks and self.perf_controller.current_mode != "minimal":
            display_frame = self.face_detector.draw_landmarks(display_frame, landmarks)

        # Update display
        if self.compare_mode:
            self.preview.update_frame(self.original_frame, display_frame)
        else:
            self.preview.update_frame(frame, display_frame)

        self.preview.set_fps(self.fps_counter.fps)
        self.preview.set_eye_data(eye_data)
        self.preview.set_gaze_info({
            "yaw": gaze.yaw,
            "pitch": gaze.pitch,
            "is_looking_at_camera": gaze.is_looking_at_camera,
            "confidence": gaze.confidence,
            "demo_mode": self.demo_mode,
        })

        # Status bar
        mode = "Corrected" if self.correction_enabled else "Original"
        perf_mode = self.perf_controller.current_mode
        demo_str = " [DEMO 20°]" if self.demo_mode else ""

        amp = self.eye_corrector.amplification
        amp_str = f"Amp:{amp:.1f}x" if amp != 1.0 else ""

        if should_correct and corr_total > 0.5:
            corr_str = f"Shift:{corr_total:.1f}px"
        elif should_correct:
            corr_str = "Shift:<1px"
        else:
            corr_str = ""

        self.statusBar().showMessage(
            f"{mode}{demo_str} | Perf:{perf_mode} | "
            f"Y={gaze.yaw:+.1f}° P={gaze.pitch:+.1f}° | "
            f"FPS:{self.fps_counter.fps:.1f} | "
            f"Frame:{self.fps_counter.frame_time_ms:.1f}ms | "
            f"{corr_str} {amp_str}"
        )

    def _on_camera_error(self, message):
        self.statusBar().showMessage(f"Camera Error: {message}")
        QMessageBox.critical(self, "Camera Error", message)

    def _on_camera_fps(self, _fps):
        pass

    def _on_correction_toggled(self, enabled):
        self.correction_enabled = enabled
        self.preview.correction_enabled = enabled

    def _on_strength_changed(self, strength):
        self.eye_corrector.strength = strength
        self.config["correction_strength"] = strength

    def _on_amplification_changed(self, amp):
        self.eye_corrector.amplification = amp

    def _on_landmark_toggled(self, enabled):
        self.show_landmarks = enabled
        self.preview.show_landmarks = enabled

    def _on_compare_toggled(self, enabled):
        self.compare_mode = enabled
        self.preview.set_compare_mode(enabled)

    def _on_demo_toggled(self, enabled):
        self.demo_mode = enabled

    def _on_calibrate(self):
        self.calibrating = True
        self.gaze_estimator.start_calibration()
        self.control_panel.set_calibrating(True)
        self.statusBar().showMessage("Calibrating — look straight at camera...")
        QTimer.singleShot(2000, self._finish_calibration)

    def _finish_calibration(self):
        success = self.gaze_estimator.finish_calibration()
        self.calibrating = False
        self.control_panel.set_calibrating(False)
        if success:
            self.statusBar().showMessage(
                f"Calibration complete! "
                f"Offset: Y={self.gaze_estimator.calibration_offset_yaw:+.3f} "
                f"P={self.gaze_estimator.calibration_offset_pitch:+.3f}"
            )
        else:
            self.statusBar().showMessage("Calibration failed — not enough data. Try again.")

    def _on_virtual_cam_toggled(self, enabled):
        if enabled:
            self.statusBar().showMessage("Virtual camera: Starting...")
            QMessageBox.information(
                self, "Virtual Camera",
                "Virtual camera output requires OBS Studio.\n\n"
                "Please install OBS Studio and its virtual camera plugin.\n"
                "Feature will be available in the next update."
            )
            self.control_panel.vcam_btn.setChecked(False)
        else:
            self.statusBar().showMessage("Virtual camera: Stopped")

    def closeEvent(self, event):
        if hasattr(self, "camera_thread") and self.camera_thread:
            self.camera_thread.stop()
        self.face_detector.cleanup()
        event.accept()
