"""OpenBroadcast — Main Window. Camera → Face → Gaze → Correction → Display."""

import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QHBoxLayout,
    QMessageBox, QSplitter
)
from PyQt6.QtCore import Qt

import os
from core.camera import CameraThread
from core.face_detector import FaceDetector
from core.gaze_estimator import GeometricGazeEstimator
from core.eye_corrector import EyeCorrector
from utils.performance import FPSCounter
from ui.preview_widget import PreviewWidget
from ui.control_panel import ControlPanel

# Try to import neural corrector
try:
    from core.neural_corrector import NeuralCorrector
    HAS_NEURAL = True
except ImportError:
    HAS_NEURAL = False


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
        self.show_landmarks = False
        self.last_landmarks = None
        self._calibrating = False
        self._virtcam_active = False
        self._virtcam_writer = None
        self._neural_corrector = None
        self._use_neural = False
        self._wizard_active = False
        self._wizard_step = 0
        self._wizard_positions = [
            (0.3, 0.5),   # Left
            (0.7, 0.5),   # Right
            (0.5, 0.3),   # Up
            (0.5, 0.7),   # Down
            (0.5, 0.5),   # Center
            (0.3, 0.3),   # Left-Up
            (0.7, 0.7),   # Right-Down
            (0.5, 0.5),   # Center again
        ]
        self._wizard_gaze_samples = []

        # Try to load neural model
        model_path = os.path.join(os.path.dirname(__file__), '..', 'models', 'gaze_correction.pth')
        if HAS_NEURAL and os.path.exists(model_path):
            try:
                self._neural_corrector = NeuralCorrector(model_path)
                self._use_neural = True
                print('[MainWindow] Neural correction model loaded')
            except Exception as e:
                print(f'[MainWindow] Could not load neural model: {e}')

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
        self.control_panel.landmarks_toggled.connect(self._on_landmarks_toggled)
        self.control_panel.calibrate_clicked.connect(self._on_calibrate_clicked)
        self.control_panel.wizard_clicked.connect(self._on_wizard_clicked)
        self.control_panel.virtual_cam_toggled.connect(self._on_virtual_cam_toggled)
        self.control_panel.mode_changed.connect(self._on_mode_changed)

        # Disable neural option if model not loaded
        if not self._use_neural:
            self.control_panel.mode_combo.setItemEnabled(1, False)

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
            self.last_landmarks = None
            self.preview.update_frame(frame)
            self.preview.set_gaze_info(None)
            self.preview.set_eye_data(None)
            self.statusBar().showMessage("No face detected")
            self.preview.set_fps(self.fps_counter.fps)
            return

        self.last_landmarks = landmarks

        # Eye data → gaze → correction
        eye_data = self.face_detector.get_eye_data(landmarks, frame.shape)
        gaze = self.gaze_estimator.estimate(eye_data)

        if self.correction_enabled and not eye_data["is_blinking"]:
            if self._use_neural and self._neural_corrector:
                display_frame = self._neural_corrector.correct_frame(frame, eye_data)
            else:
                display_frame = self.eye_corrector.correct_frame(frame, eye_data)
        else:
            display_frame = frame

        # Draw landmarks on display if enabled
        if self.show_landmarks:
            display_frame = self.face_detector.draw_landmarks(display_frame, landmarks)

        # Calibration mode: collect samples
        if self._calibrating:
            self.gaze_estimator.add_calibration_sample(eye_data)

        # Virtual camera output
        if self._virtcam_active and self._virtcam_writer is not None:
            try:
                self._virtcam_writer.write(display_frame)
            except Exception:
                pass

        # Display
        self.preview.update_frame(frame, display_frame)
        self.preview.set_eye_data(eye_data)

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

    def _on_landmarks_toggled(self, enabled):
        self.show_landmarks = enabled

    def _on_calibrate_clicked(self):
        if self._calibrating:
            # Finish calibration
            success = self.gaze_estimator.finish_calibration()
            self._calibrating = False
            self.control_panel.set_calibrate_status(
                "Calibrated!" if success else "Not enough samples"
            )
            self.control_panel.calibrate_btn.setText("Calibrate Gaze")
        else:
            # Start calibration
            self.gaze_estimator.start_calibration()
            self._calibrating = True
            self.control_panel.set_calibrate_status("Looking at camera...")
            self.control_panel.calibrate_btn.setText("Stop Calibration")

    def _on_wizard_clicked(self):
        if self._wizard_active:
            # Stop wizard
            self._wizard_active = False
            self.preview.set_wizard_dot(None)
            self.control_panel.wizard_btn.setText("Interactive Calibration")
            self.control_panel.set_calibrate_status(
                f"Collected {len(self._wizard_gaze_samples)} samples"
            )
            # Apply calibration if we have enough samples
            if len(self._wizard_gaze_samples) >= 10:
                samples = np.array(self._wizard_gaze_samples)
                self.gaze_estimator.calibration_offset_yaw = float(np.mean(samples[:, 0]))
                self.gaze_estimator.calibration_offset_pitch = float(np.mean(samples[:, 1]))
                self.control_panel.set_calibrate_status("Calibrated!")
            self._wizard_gaze_samples = []
        else:
            # Start wizard
            self._wizard_active = True
            self._wizard_step = 0
            self._wizard_gaze_samples = []
            self.control_panel.wizard_btn.setText("Stop Wizard")
            self.control_panel.set_calibrate_status("Follow the dot...")
            self._update_wizard_dot()

    def _update_wizard_dot(self):
        if not self._wizard_active:
            return
        if self._wizard_step >= len(self._wizard_positions):
            # Wizard complete
            self._wizard_active = False
            self.preview.set_wizard_dot(None)
            self.control_panel.wizard_btn.setText("Interactive Calibration")
            if len(self._wizard_gaze_samples) >= 10:
                samples = np.array(self._wizard_gaze_samples)
                self.gaze_estimator.calibration_offset_yaw = float(np.mean(samples[:, 0]))
                self.gaze_estimator.calibration_offset_pitch = float(np.mean(samples[:, 1]))
                self.control_panel.set_calibrate_status("Calibrated!")
            self._wizard_gaze_samples = []
            return

        # Show dot at next position
        x, y = self._wizard_positions[self._wizard_step]
        self.preview.set_wizard_dot((x, y))
        self.control_panel.set_calibrate_status(
            f"Step {self._wizard_step + 1}/{len(self._wizard_positions)}: Look at the dot"
        )

        # After 2 seconds, collect samples and move to next position
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(2000, self._wizard_collect_and_advance)

    def _wizard_collect_and_advance(self):
        if not self._wizard_active:
            return
        # Collect current gaze offset as calibration sample
        if self.last_landmarks is not None:
            eye_data = self.face_detector.get_eye_data(self.last_landmarks, (480, 640))
            left = eye_data["left_eye"]
            right = eye_data["right_eye"]
            avg_x = (left["offset_x"] + right["offset_x"]) / 2
            avg_y = (left["offset_y"] + right["offset_y"]) / 2
            self._wizard_gaze_samples.append((avg_x, avg_y))

        self._wizard_step += 1
        self._update_wizard_dot()

    def _on_mode_changed(self, index):
        self._use_neural = (index == 1 and self._neural_corrector is not None)

    def _on_virtual_cam_toggled(self, active):
        if active:
            try:
                import pyvirtualcam
                self._virtcam_writer = pyvirtualcam.Camera(
                    width=640, height=480, fps=30
                )
                self._virtcam_active = True
                self.control_panel.set_virtcam_status(
                    f"Outputting to: {self._virtcam_writer.device}"
                )
            except ImportError:
                self.control_panel.set_virtcam_status("pip install pyvirtualcam")
                self.control_panel.virtcam_btn.setChecked(False)
            except Exception as e:
                self.control_panel.set_virtcam_status(str(e)[:50])
                self.control_panel.virtcam_btn.setChecked(False)
        else:
            self._virtcam_active = False
            if self._virtcam_writer:
                try:
                    self._virtcam_writer.close()
                except Exception:
                    pass
                self._virtcam_writer = None
            self.control_panel.set_virtcam_status("")

    def closeEvent(self, event):
        if hasattr(self, "camera_thread") and self.camera_thread:
            self.camera_thread.stop()
        self.face_detector.cleanup()
        event.accept()
