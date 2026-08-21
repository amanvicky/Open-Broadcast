"""
OpenBroadcast — Gaze Estimation Module

Geometric gaze estimation — runs in <1ms, no neural network needed.

How it works:
1. MediaPipe gives us iris position (landmarks 468/473)
2. MediaPipe gives us eye corner positions (landmarks 33/133, 362/263)
3. We measure how far the iris is from the eye center
4. That offset = gaze direction

This is surprisingly accurate for our use case because:
- We don't need EXACT gaze angle (we just need "how far from center")
- We're measuring physical iris position, which directly correlates with gaze
- For correction, we just need to know "how far to shift the iris back"

Accuracy: ~10-15° error (enough for correction)
Speed: <0.5ms (just math, no model inference)
"""

import numpy as np
from dataclasses import dataclass
from collections import deque
import time


@dataclass
class GazeDirection:
    """Represents estimated gaze direction."""
    yaw: float          # Horizontal angle in degrees (+ = looking right)
    pitch: float        # Vertical angle in degrees (+ = looking down)
    confidence: float   # 0-1
    raw_offset_x: float # Raw normalized iris offset (-0.5 to 0.5)
    raw_offset_y: float
    is_looking_at_camera: bool  # True if looking roughly at camera


class GeometricGazeEstimator:
    """
    Estimates gaze direction from iris position relative to eye corners.

    No neural network needed. Pure geometry.
    Fast enough to run on every frame even on 2-core CPUs.
    """

    def __init__(self, ema_alpha=0.6, look_at_camera_threshold=0.08):
        """
        Args:
            ema_alpha: EMA smoothing factor (0.6 = responsive, 0.3 = smooth)
            look_at_camera_threshold: Offset below which we consider
                                      the user is "looking at camera"
        """
        self.ema_alpha = ema_alpha
        self.look_at_camera_threshold = look_at_camera_threshold

        # Smoothed values
        self.smoothed_yaw = 0.0
        self.smoothed_pitch = 0.0

        # History for noise detection
        self.history = deque(maxlen=15)

        # Calibration offset (set during first-run calibration)
        self.calibration_offset_yaw = 0.0
        self.calibration_offset_pitch = 0.0

        # Calibration collection
        self._calibration_samples = []
        self._calibrating = False

    def estimate(self, eye_data):
        """
        Estimate gaze from eye data extracted by FaceDetector.

        Args:
            eye_data: dict from FaceDetector.get_eye_data()

        Returns:
            GazeDirection object
        """
        left = eye_data["left_eye"]
        right = eye_data["right_eye"]

        # Skip if eyes are too small (face far away or partial)
        if left["width"] < 10 or right["width"] < 10:
            return GazeDirection(
                yaw=0.0, pitch=0.0, confidence=0.0,
                raw_offset_x=0.0, raw_offset_y=0.0,
                is_looking_at_camera=True,
            )

        # Skip during blink
        if eye_data["is_blinking"]:
            return GazeDirection(
                yaw=self.smoothed_yaw,
                pitch=self.smoothed_pitch,
                confidence=0.3,
                raw_offset_x=0.0, raw_offset_y=0.0,
                is_looking_at_camera=abs(self.smoothed_yaw) < 5,
            )

        # Average both eyes for robustness
        avg_offset_x = (left["offset_x"] + right["offset_x"]) / 2
        avg_offset_y = (left["offset_y"] + right["offset_y"]) / 2

        # Apply calibration offset
        avg_offset_x -= self.calibration_offset_yaw
        avg_offset_y -= self.calibration_offset_pitch

        # Convert normalized offset to degrees
        # Empirical mapping: offset of 0.5 ≈ 15° gaze angle
        raw_yaw = avg_offset_x * 30.0
        raw_pitch = avg_offset_y * 20.0

        # EMA smoothing
        self.smoothed_yaw = (
            self.ema_alpha * self.smoothed_yaw
            + (1 - self.ema_alpha) * raw_yaw
        )
        self.smoothed_pitch = (
            self.ema_alpha * self.smoothed_pitch
            + (1 - self.ema_alpha) * raw_pitch
        )

        # Noise detection: if gaze jumps >20° in one frame, revert
        self.history.append((self.smoothed_yaw, self.smoothed_pitch))
        if len(self.history) >= 2:
            prev_yaw, prev_pitch = self.history[-2]
            delta = abs(self.smoothed_yaw - prev_yaw) + abs(self.smoothed_pitch - prev_pitch)
            if delta > 20:
                self.smoothed_yaw = prev_yaw * 0.8 + self.smoothed_yaw * 0.2
                self.smoothed_pitch = prev_pitch * 0.8 + self.smoothed_pitch * 0.2

        # Determine if looking at camera
        is_at_camera = (
            abs(avg_offset_x) < self.look_at_camera_threshold
            and abs(avg_offset_y) < self.look_at_camera_threshold
        )

        # Confidence based on offset magnitude (closer to center = more confident)
        distance = np.sqrt(avg_offset_x**2 + avg_offset_y**2)
        confidence = max(0.3, 1.0 - distance * 2)

        return GazeDirection(
            yaw=self.smoothed_yaw,
            pitch=self.smoothed_pitch,
            confidence=confidence,
            raw_offset_x=avg_offset_x,
            raw_offset_y=avg_offset_y,
            is_looking_at_camera=is_at_camera,
        )

    def start_calibration(self):
        """Begin collecting calibration samples."""
        self._calibrating = True
        self._calibration_samples = []

    def add_calibration_sample(self, eye_data):
        """Add a sample during calibration (user looking at camera)."""
        if not self._calibrating:
            return
        left = eye_data["left_eye"]
        right = eye_data["right_eye"]
        avg_x = (left["offset_x"] + right["offset_x"]) / 2
        avg_y = (left["offset_y"] + right["offset_y"]) / 2
        self._calibration_samples.append((avg_x, avg_y))

    def finish_calibration(self):
        """
        Finish calibration and compute offset.
        Call this after user has looked at camera for ~2 seconds.
        """
        self._calibrating = False
        if len(self._calibration_samples) < 5:
            return False  # Not enough samples

        samples = np.array(self._calibration_samples)
        self.calibration_offset_yaw = np.mean(samples[:, 0])
        self.calibration_offset_pitch = float(np.mean(samples[:, 1]))
        # Reset smoothed state so calibration takes effect immediately
        self.smoothed_yaw = 0.0
        self.smoothed_pitch = 0.0
        self.history.clear()
        return True

    @property
    def is_calibrating(self):
        return self._calibrating

    def reset(self):
        """Reset all state."""
        self.smoothed_yaw = 0.0
        self.smoothed_pitch = 0.0
        self.history.clear()
