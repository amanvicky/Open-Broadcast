"""
OpenBroadcast — Eye Preprocessor

Extracts and normalizes eye regions from MediaPipe landmarks
for input to the GazeNet-Lite neural model.

Pipeline:
1. Get eye landmarks from MediaPipe (12 points per eye + 5 iris)
2. Calculate eye bounding box with padding
3. Compute affine transform to align + center + resize
4. Apply transform to get normalized eye crop (36×60 per eye)
5. Concatenate both eyes → (36×120) for model input
"""

import cv2
import numpy as np
from dataclasses import dataclass


@dataclass
class EyeROI:
    """Normalized eye region of interest."""
    image: np.ndarray          # 36×60 grayscale crop
    iris_center: tuple         # (x, y) in crop coordinates
    eye_corners: tuple         # ((outer_x, outer_y), (inner_x, inner_y))
    gaze_offset: tuple         # (offset_x, offset_y) — normalized
    head_rotation: float       # estimated head yaw from eye geometry


class EyePreprocessor:
    """
    Extracts normalized eye crops from MediaPipe face landmarks.
    """

    # MediaPipe landmark indices
    LEFT_EYE = {
        "outer": 33,
        "inner": 133,
        "top": 159,
        "bottom": 145,
    }
    RIGHT_EYE = {
        "outer": 362,
        "inner": 263,
        "top": 386,
        "bottom": 374,
    }
    LEFT_IRIS = [468, 469, 470, 471, 472]
    RIGHT_IRIS = [473, 474, 475, 476, 477]

    TARGET_WIDTH = 60
    TARGET_HEIGHT = 36

    def extract_both_eyes(self, frame, landmarks):
        """
        Extract and concatenate both eye crops for model input.

        Args:
            frame: BGR image
            landmarks: MediaPipe face mesh landmarks

        Returns:
            tensor: (1, 1, 36, 120) normalized float32
            left_eye: EyeROI
            right_eye: EyeROI
        """
        left = self._extract_eye(frame, landmarks, "left")
        right = self._extract_eye(frame, landmarks, "right")

        if left is None or right is None:
            return None, left, right

        # Concatenate: (36, 60) + (36, 60) → (36, 120)
        combined = np.concatenate([left.image, right.image], axis=1)

        # Normalize to [0, 1]
        combined = combined.astype(np.float32) / 255.0

        # Add batch + channel dims: (1, 1, 36, 120)
        tensor = combined[np.newaxis, np.newaxis, ...]

        return tensor, left, right

    def _extract_eye(self, frame, landmarks, side):
        """Extract a single normalized eye crop."""
        config = self.LEFT_EYE if side == "left" else self.RIGHT_EYE
        iris_indices = self.LEFT_IRIS if side == "left" else self.RIGHT_IRIS

        h, w = frame.shape[:2]

        def pt(idx):
            lm = landmarks.landmark[idx]
            return np.array([lm.x * w, lm.y * h])

        outer = pt(config["outer"])
        inner = pt(config["inner"])
        iris_center = pt(iris_indices[0])

        eye_width = np.linalg.norm(inner - outer)
        eye_center = (outer + inner) / 2
        eye_angle = np.arctan2(inner[1] - outer[1], inner[0] - outer[0])

        if eye_width < 5:
            return None

        # Build affine transform: align eye horizontally, center it
        src_pts = np.float32([
            outer,
            inner,
            eye_center + np.array([0, -eye_width * 0.4]),
        ])

        half_w = self.TARGET_WIDTH / 2
        half_h = self.TARGET_HEIGHT / 2

        dst_pts = np.float32([
            [half_w - eye_width / 2, half_h],
            [half_w + eye_width / 2, half_h],
            [half_w, half_h - eye_width * 0.4],
        ])

        M = cv2.getAffineTransform(src_pts, dst_pts)
        eye_crop = cv2.warpAffine(
            frame, M, (self.TARGET_WIDTH, self.TARGET_HEIGHT),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        # Convert to grayscale
        if len(eye_crop.shape) == 3:
            eye_crop = cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)

        # Transform iris center to crop coordinates
        iris_in_crop = M @ np.array([iris_center[0], iris_center[1], 1.0])

        # Calculate gaze offset
        gaze_offset_x = (iris_in_crop[0] - half_w) / half_w
        gaze_offset_y = (iris_in_crop[1] - half_h) / half_h

        return EyeROI(
            image=eye_crop,
            iris_center=(iris_in_crop[0], iris_in_crop[1]),
            eye_corners=(outer.tolist(), inner.tolist()),
            gaze_offset=(gaze_offset_x, gaze_offset_y),
            head_rotation=eye_angle,
        )
