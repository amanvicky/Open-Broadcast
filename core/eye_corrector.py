"""Eye gaze correction via affine translation with temporal smoothing."""

import cv2
import numpy as np

# Vertical offset below eye center for natural "attentive" look (fraction of eye width)
ATTENTION_OFFSET = 0.21


class EyeCorrector:
    """Shift iris toward eye center using warpAffine + EMA smoothing."""

    def __init__(self, strength=0.85, max_shift=14.0, smoothing=0.6,
                 amplification=1.0, **_kwargs):
        self._strength = float(np.clip(strength, 0.0, 1.0))
        self._max_shift = float(max(0.0, max_shift))
        self._smoothing = float(np.clip(smoothing, 0.0, 0.98))
        self.amplification = amplification
        self._shift_ema = {}

    @property
    def strength(self):
        return self._strength

    @strength.setter
    def strength(self, value):
        self._strength = float(np.clip(value, 0.0, 1.0))

    def correct_frame(self, frame, eye_data):
        """Apply gaze correction to an entire frame."""
        if self._strength <= 0.0:
            return frame.copy()
        output = frame
        for side in ("left", "right"):
            output = self._recentre_eye(output, eye_data, side)
        return output

    def _recentre_eye(self, frame, eye_data, side):
        """Shift iris toward eye center using affine translation."""
        eye = eye_data[f"{side}_eye"]
        h, w = frame.shape[:2]
        pupil = np.array(eye["iris"], dtype=np.float32)
        socket = np.array(eye["center"], dtype=np.float32)
        eye_width = eye["width"]

        if eye_width < 10 or eye_data["is_blinking"]:
            return frame

        # Anchor slightly below eye center for natural "attentive" look
        anchor = socket.copy()
        anchor[1] += ATTENTION_OFFSET * eye_width

        target = (anchor - pupil) * self._strength * self.amplification

        # Minimum visibility: ensure correction is always perceptible
        MIN_SHIFT_PX = 6.0
        target_norm = float(np.linalg.norm(target))
        if 0.01 < target_norm < MIN_SHIFT_PX:
            target = target * (MIN_SHIFT_PX / target_norm)

        # EMA smoothing
        prev = self._shift_ema.get(side)
        if prev is None:
            smoothed = target.astype(np.float32)
        else:
            smoothed = (self._smoothing * prev + (1.0 - self._smoothing) * target).astype(np.float32)
        self._shift_ema[side] = smoothed

        shift = smoothed
        norm = float(np.linalg.norm(shift))
        if norm < 0.5:
            return frame
        if norm > self._max_shift:
            shift = shift * (self._max_shift / norm)

        # ROI around eye
        roi = self._eye_roi((h, w), eye)
        if roi is None:
            return frame
        x0, y0, x1, y1 = roi
        patch = frame[y0:y1, x0:x1]
        if patch.size == 0:
            return frame

        # Affine translation
        M = np.array([[1, 0, shift[0]], [0, 1, shift[1]]], dtype=np.float32)
        warped = cv2.warpAffine(patch, M, (patch.shape[1], patch.shape[0]),
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

        # Elliptical blend mask
        blend = self._eye_blend_mask(patch.shape[:2], eye, (x0, y0))
        result = frame.copy()
        result[y0:y1, x0:x1] = np.clip(
            patch * (1.0 - blend) + warped * blend, 0, 255
        ).astype(np.uint8)
        return result

    @staticmethod
    def _eye_roi(shape, eye, pad=1.3):
        h, w = shape
        cx, cy = float(eye["center"][0]), float(eye["center"][1])
        ew = eye["width"]
        hw = (ew * pad) / 2.0 + 4.0
        hh = (ew * 0.6 * pad) / 2.0 + 4.0
        x0, y0 = int(np.clip(cx - hw, 0, w - 1)), int(np.clip(cy - hh, 0, h - 1))
        x1, y1 = int(np.clip(cx + hw, 0, w)), int(np.clip(cy + hh, 0, h))
        return (x0, y0, x1, y1) if (x1 - x0 >= 3 and y1 - y0 >= 3) else None

    def _eye_blend_mask(self, patch_shape, eye, origin):
        ph, pw = patch_shape
        mask = np.zeros((ph, pw), dtype=np.float32)
        cx = float(eye["center"][0]) - origin[0]
        cy = float(eye["center"][1]) - origin[1]
        ax, ay = max(3.0, eye["width"] * 0.55), max(3.0, eye["width"] * 0.4)
        cv2.ellipse(mask, (int(cx), int(cy)), (int(ax), int(ay)), 0, 0, 360, 1.0, -1)
        return cv2.GaussianBlur(mask, (0, 0), sigmaX=max(1.5, min(ax, ay) * 0.25))[:, :, None]
