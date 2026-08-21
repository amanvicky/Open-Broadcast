"""Neural gaze correction using trained GazeCorrectionNet.

Replaces the geometric iris transplant with a learned model that generates
"looking at camera" eye patches. Produces more natural results at small
gaze offsets where geometric methods fail.

Usage:
    corrector = NeuralCorrector("models/gaze_correction.pth")
    corrected_frame = corrector.correct_frame(frame, eye_data)
"""

import os
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from core.gaze_model import GazeCorrectionNet


class NeuralCorrector:
    """Neural gaze correction using trained U-Net model."""

    PATCH_SIZE = 64

    def __init__(self, model_path, device=None):
        """
        Args:
            model_path: Path to trained model checkpoint
            device: torch device (auto-detected if None)
        """
        if not HAS_TORCH:
            raise ImportError("PyTorch required: pip install torch torchvision")

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load model
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = GazeCorrectionNet(in_channels=3, offset_dim=2)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.model.eval()

        print(f"[NeuralCorrector] Loaded model from {model_path}")
        print(f"  Device: {self.device}, Val loss: {checkpoint.get('val_loss', 'N/A')}")

    def correct_frame(self, frame, eye_data):
        """Apply neural correction to both eyes in a frame."""
        output = frame.copy()

        for side in ("left", "right"):
            output = self._correct_eye(output, eye_data, side)

        return output

    def _correct_eye(self, frame, eye_data, side):
        """Correct a single eye using the neural model."""
        import cv2

        eye = eye_data[f"{side}_eye"]
        pupil = np.array(eye["iris"], dtype=np.float32)
        socket = np.array(eye["center"], dtype=np.float32)
        eye_width = float(eye["width"])

        if eye_width < 15 or eye_data["is_blinking"]:
            return frame

        # Calculate normalized offset
        offset_x = (pupil[0] - socket[0]) / (eye_width + 1e-6)
        offset_y = (pupil[1] - socket[1]) / (eye_width + 1e-6)
        offset = np.array([offset_x, offset_y], dtype=np.float32)

        # If already near center, skip
        if np.linalg.norm(offset) < 0.05:
            return frame

        # Extract eye patch centered on eye center
        h, w = frame.shape[:2]
        cx, cy = int(socket[0]), int(socket[1])
        half = self.PATCH_SIZE // 2

        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        x1 = min(w, cx + half)
        y1 = min(h, cy + half)

        if (x1 - x0) < self.PATCH_SIZE or (y1 - y0) < self.PATCH_SIZE:
            return frame

        input_patch = frame[y0:y1, x0:x1].copy()

        # Prepare tensors
        # (1, C, H, W) normalized to [0, 1]
        input_tensor = torch.from_numpy(input_patch).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        input_tensor = input_tensor.to(self.device)

        offset_tensor = torch.from_numpy(offset).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            output_tensor = self.model(input_tensor, offset_tensor)

        # Convert back to numpy
        output_patch = output_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
        output_patch = (output_patch * 255.0).clip(0, 255).astype(np.uint8)

        # Blend with original using feathered mask
        mask = np.zeros((self.PATCH_SIZE, self.PATCH_SIZE), dtype=np.float32)
        cv2.circle(mask, (half, half), half - 2, 1.0, -1)
        mask = cv2.GaussianBlur(mask, (0, 0), sigmaX=8.0)
        mask = mask[:, :, None]

        result = frame.copy()
        blend = (output_patch.astype(np.float32) * mask +
                 frame[y0:y1, x0:x1].astype(np.float32) * (1.0 - mask))
        result[y0:y1, x0:x1] = np.clip(blend, 0, 255).astype(np.uint8)

        return result

    def correct_frame_batch(self, frames, eye_data_list):
        """Batch correction for multiple frames (GPU-optimized)."""
        if not HAS_TORCH or len(frames) == 0:
            return frames

        # For batch processing, process each frame individually
        # (batch mode would require padding to same size)
        return [self.correct_frame(f, e) for f, e in zip(frames, eye_data_list)]
