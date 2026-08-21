"""Eye gaze correction via iris transplant.

Extracts the iris and pastes it at the eye center using Poisson blending,
leaving the surrounding skin untouched. No sclera fill — the feathered
paste naturally covers the old position.
"""

import cv2
import numpy as np

# Slightly below eye center for natural "attentive" look (fraction of eye width)
ATTENTION_OFFSET = 0.21
# Iris radius as fraction of eye width
IRIS_RADIUS_FRAC = 0.28
# Feather radius as fraction of iris radius
FEATHER_FRAC = 0.35


class EyeCorrector:
    """Transplant iris to eye center with temporal smoothing."""

    def __init__(self, strength=0.85, max_shift=40.0, smoothing=0.6,
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
        """Apply iris transplant to both eyes."""
        if self._strength <= 0.0:
            return frame.copy()
        output = frame.copy()
        for side in ("left", "right"):
            output = self._transplant_iris(output, eye_data, side)
        return output

    def _transplant_iris(self, frame, eye_data, side):
        """Extract iris, calculate shift to center, paste at new position."""
        eye = eye_data[f"{side}_eye"]
        h, w = frame.shape[:2]

        pupil = np.array(eye["iris"], dtype=np.float32)
        socket = np.array(eye["center"], dtype=np.float32)
        eye_width = float(eye["width"])

        if eye_width < 10 or eye_data["is_blinking"]:
            return frame

        # Anchor slightly below center for natural attentive look
        anchor = socket.copy()
        anchor[1] += ATTENTION_OFFSET * eye_width

        # Raw shift from current iris position to target
        raw_shift = (anchor - pupil) * self._strength * self.amplification

        # Minimum visibility: force shift to be at least 3px
        raw_norm = float(np.linalg.norm(raw_shift))
        if 0.01 < raw_norm < 3.0:
            raw_shift = raw_shift * (3.0 / raw_norm)

        # EMA smoothing
        prev = self._shift_ema.get(side)
        if prev is None:
            smoothed = raw_shift.astype(np.float32)
        else:
            smoothed = (self._smoothing * prev +
                        (1.0 - self._smoothing) * raw_shift).astype(np.float32)
        self._shift_ema[side] = smoothed

        shift = smoothed
        norm = float(np.linalg.norm(shift))
        if norm < 0.3:
            return frame
        if norm > self._max_shift:
            shift = shift * (self._max_shift / norm)

        # Iris transplant
        iris_r = max(5, int(eye_width * IRIS_RADIUS_FRAC))
        ix, iy = int(round(pupil[0])), int(round(pupil[1]))
        new_ix = int(round(pupil[0] + shift[0]))
        new_iy = int(round(pupil[1] + shift[1]))

        # Clamp to frame
        ix = np.clip(ix, iris_r, w - iris_r - 1)
        iy = np.clip(iy, iris_r, h - iris_r - 1)
        new_ix = np.clip(new_ix, iris_r, w - iris_r - 1)
        new_iy = np.clip(new_iy, iris_r, h - iris_r - 1)

        if abs(new_ix - ix) < 1 and abs(new_iy - iy) < 1:
            return frame

        # Extract iris from original frame
        iris_patch, iris_mask = self._extract_iris(frame, ix, iy, iris_r)
        if iris_patch is None:
            return frame

        # Paste at new position with feathered blend
        result = frame.copy()
        self._paste_iris(result, iris_patch, iris_mask, new_ix, new_iy, iris_r)
        return result

    def _extract_iris(self, frame, cx, cy, radius):
        """Extract circular iris region and its feathered mask."""
        h, w = frame.shape[:2]
        x0 = max(0, cx - radius)
        y0 = max(0, cy - radius)
        x1 = min(w, cx + radius + 1)
        y1 = min(h, cy + radius + 1)

        patch = frame[y0:y1, x0:x1]
        if patch.size == 0:
            return None, None

        # Circular mask with feathered edges
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        local_cx = cx - x0
        local_cy = cy - y0
        cv2.circle(mask, (local_cx, local_cy), radius, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0),
                                sigmaX=max(1.0, radius * FEATHER_FRAC))

        return patch, mask

    def _paste_iris(self, frame, patch, mask, new_cx, new_cy, radius):
        """Paste extracted iris at new position with feathered blending."""
        h, w = frame.shape[:2]
        ph, pw = patch.shape[:2]

        # Destination region
        nx0 = max(0, new_cx - radius)
        ny0 = max(0, new_cy - radius)
        nx1 = min(w, new_cx + radius + 1)
        ny1 = min(h, new_cy + radius + 1)

        # Source region in the extracted patch
        sx0 = nx0 - new_cx + radius
        sy0 = ny0 - new_cy + radius
        sx1 = sx0 + (nx1 - nx0)
        sy1 = sy0 + (ny1 - ny0)

        # Clamp to patch bounds
        sx0 = max(0, min(sx0, pw))
        sy0 = max(0, min(sy0, ph))
        sx1 = max(0, min(sx1, pw))
        sy1 = max(0, min(sy1, ph))

        if sx1 <= sx0 or sy1 <= sy0:
            return

        ow = sx1 - sx0
        oh = sy1 - sy0
        if ow <= 0 or oh <= 0:
            return

        # Corresponding region in output
        ox0 = sx0 + new_cx - radius
        oy0 = sy0 + new_cy - radius

        src = patch[sy0:sy1, sx0:sx1].astype(np.float32)
        m = mask[sy0:sy1, sx0:sx1][:, :, None]
        dst = frame[oy0:oy0 + oh, ox0:ox0 + ow].astype(np.float32)

        blended = dst * (1.0 - m) + src * m
        frame[oy0:oy0 + oh, ox0:ox0 + ow] = np.clip(blended, 0, 255).astype(np.uint8)
