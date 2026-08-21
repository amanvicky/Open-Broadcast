"""Eye gaze correction via iris transplant.

Unlike warpAffine (which shifts the entire eye region making the correction
invisible), this extracts ONLY the iris and pastes it at the eye center,
leaving the surrounding skin/eyelids untouched. The result is a visible
iris repositioning even at small gaze offsets.
"""

import cv2
import numpy as np

# Fraction of eye width for the attention anchor (slightly below center)
ATTENTION_OFFSET = 0.21
# Iris radius as fraction of eye width
IRIS_RADIUS_FRAC = 0.28
# Blend feather radius as fraction of iris radius
FEATHER_FRAC = 0.35


class EyeCorrector:
    """Transplant iris to eye center with temporal smoothing."""

    def __init__(self, strength=0.85, max_shift=20.0, smoothing=0.6,
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

        # --- Iris transplant ---
        iris_r = max(5, int(eye_width * IRIS_RADIUS_FRAC))
        ix, iy = int(round(pupil[0])), int(round(pupil[1]))
        new_ix = int(round(pupil[0] + shift[0]))
        new_iy = int(round(pupil[1] + shift[1]))

        # Clamp to frame
        ix = np.clip(ix, iris_r, w - iris_r - 1)
        iy = np.clip(iy, iris_r, h - iris_r - 1)
        new_ix = np.clip(new_ix, iris_r, w - iris_r - 1)
        new_iy = np.clip(new_iy, iris_r, h - iris_r - 1)

        # If barely moved, skip
        if abs(new_ix - ix) < 1 and abs(new_iy - iy) < 1:
            return frame

        result = frame.copy()

        # 1. Fill old iris position with sclera (average white around iris)
        self._fill_with_sclera(result, ix, iy, iris_r, frame)

        # 2. Extract iris pixels from original frame (before fill)
        iris_patch = self._extract_iris(frame, ix, iy, iris_r)

        # 3. Paste iris at new position with feathered blend
        if iris_patch is not None:
            self._paste_iris(result, iris_patch, new_ix, new_iy, iris_r, frame)

        return result

    def _extract_iris(self, frame, cx, cy, radius):
        """Extract circular iris region from frame."""
        h, w = frame.shape[:2]
        x0 = max(0, cx - radius)
        y0 = max(0, cy - radius)
        x1 = min(w, cx + radius + 1)
        y1 = min(h, cy + radius + 1)

        patch = frame[y0:y1, x0:x1]
        if patch.size == 0:
            return None

        # Create circular mask
        mask = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        local_cx = cx - x0
        local_cy = cy - y0
        cv2.circle(mask, (local_cx, local_cy), radius, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0),
                                sigmaX=max(1.0, radius * FEATHER_FRAC))

        return {"patch": patch, "mask": mask, "origin": (x0, y0),
                "center": (cx, cy), "radius": radius}

    def _paste_iris(self, frame, iris_data, new_cx, new_cy, radius, orig_frame):
        """Paste extracted iris at new position with feathered blending."""
        h, w = frame.shape[:2]
        patch = iris_data["patch"]
        mask = iris_data["mask"]
        old_ox, old_oy = iris_data["origin"]
        ph, pw = patch.shape[:2]

        # New paste region
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

        # Corresponding region in output
        ox0 = nx0 + (sx0 - (nx0 - new_cx + radius))
        oy0 = ny0 + (sy0 - (ny0 - new_cy + radius))
        ow = sx1 - sx0
        oh = sy1 - sy0

        if ow <= 0 or oh <= 0:
            return

        # Extract the source pixels and mask
        src = patch[sy0:sy1, sx0:sx1].astype(np.float32)
        m = mask[sy0:sy1, sx0:sx1][:, :, None]

        # Destination
        dst = frame[oy0:oy0 + oh, ox0:ox0 + ow].astype(np.float32)

        # Blend
        blended = dst * (1.0 - m) + src * m
        frame[oy0:oy0 + oh, ox0:ox0 + ow] = np.clip(blended, 0, 255).astype(np.uint8)

    def _fill_with_sclera(self, frame, cx, cy, radius, orig_frame):
        """Fill the old iris position with surrounding sclera color."""
        h, w = frame.shape[:2]
        # Sample sclera from just outside the iris (left and right of iris)
        sample_positions = [
            (cx - int(radius * 1.3), cy),  # left of iris
            (cx + int(radius * 1.3), cy),  # right of iris
            (cx, cy - int(radius * 1.1)),  # above iris
        ]

        colors = []
        for sx, sy in sample_positions:
            sx = np.clip(sx, 0, w - 1)
            sy = np.clip(sy, 0, h - 1)
            colors.append(orig_frame[sy, sx].astype(np.float32))

        if not colors:
            return

        # Average sclera color
        avg_color = np.mean(colors, axis=0).astype(np.uint8)

        # Fill circle with feathered blend
        x0 = max(0, cx - radius)
        y0 = max(0, cy - radius)
        x1 = min(w, cx + radius + 1)
        y1 = min(h, cy + radius + 1)

        fill_mask = np.zeros((y1 - y0, x1 - x0), dtype=np.float32)
        local_cx = cx - x0
        local_cy = cy - y0
        cv2.circle(fill_mask, (local_cx, local_cy), radius, 1.0, -1)
        fill_mask = cv2.GaussianBlur(fill_mask, (0, 0),
                                     sigmaX=max(1.0, radius * 0.5))
        fill_mask = fill_mask[:, :, None]

        fill_region = frame[y0:y1, x0:x1].astype(np.float32)
        sclera = np.full_like(fill_region, avg_color)
        blended = fill_region * (1.0 - fill_mask) + sclera * fill_mask
        frame[y0:y1, x0:x1] = np.clip(blended, 0, 255).astype(np.uint8)
