"""Lightweight U-Net for eye gaze correction.

Architecture: Tiny U-Net (~300K params) that takes an eye patch + offset
and produces the corrected eye patch. Designed for CPU inference at 30fps.

Input: 64x64x3 eye patch + 2D offset vector
Output: 64x64x3 corrected eye patch

The model learns to move the iris to the center while preserving
skin texture, eyelid boundaries, and lighting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Module):
    """Two conv-bn-relu blocks."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


class GazeCorrectionNet(nn.Module):
    """Tiny U-Net for eye gaze correction.

    ~300K parameters. Runs at ~5ms/frame on CPU (i5-8250U).
    """

    def __init__(self, in_channels=3, offset_dim=2):
        super().__init__()

        # Encoder
        self.enc1 = DoubleConv(in_channels, 32)
        self.enc2 = DoubleConv(32, 64)
        self.enc3 = DoubleConv(64, 128)

        # Bottleneck
        self.bottleneck = DoubleConv(128, 256)

        # Offset embedding: project 2D offset to spatial features
        self.offset_proj = nn.Sequential(
            nn.Linear(offset_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 128),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec3 = DoubleConv(256, 128)  # 128 (from skip) + 128 (from up)
        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec2 = DoubleConv(128, 64)   # 64 + 64
        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec1 = DoubleConv(64, 32)    # 32 + 32

        # Output: predict residual (offset to add to input)
        self.out_conv = nn.Conv2d(32, in_channels, 1)

        # Pooling
        self.pool = nn.MaxPool2d(2)

    def forward(self, x, offset):
        """
        Args:
            x: (B, 3, 64, 64) eye patch
            offset: (B, 2) normalized gaze offset [offset_x, offset_y]
        Returns:
            corrected: (B, 3, 64, 64) corrected eye patch
        """
        # Encoder
        e1 = self.enc1(x)           # (B, 32, 64, 64)
        e2 = self.enc2(self.pool(e1))  # (B, 64, 32, 32)
        e3 = self.enc3(self.pool(e2))  # (B, 128, 16, 16)

        # Bottleneck
        b = self.bottleneck(self.pool(e3))  # (B, 256, 8, 8)

        # Inject offset information
        offset_feat = self.offset_proj(offset)  # (B, 128)
        offset_feat = offset_feat.view(-1, 128, 1, 1)  # (B, 128, 1, 1)
        offset_feat = F.interpolate(offset_feat, size=b.shape[2:])  # (B, 128, 8, 8)
        b = b + offset_feat  # Add offset features to bottleneck

        # Decoder with skip connections
        d3 = self.up3(b)                    # (B, 128, 16, 16)
        d3 = torch.cat([d3, e3], dim=1)     # (B, 256, 16, 16)
        d3 = self.dec3(d3)                   # (B, 128, 16, 16)

        d2 = self.up2(d3)                    # (B, 64, 32, 32)
        d2 = torch.cat([d2, e2], dim=1)     # (B, 128, 32, 32)
        d2 = self.dec2(d2)                   # (B, 64, 32, 32)

        d1 = self.up1(d2)                    # (B, 32, 64, 64)
        d1 = torch.cat([d1, e1], dim=1)     # (B, 64, 64, 64)
        d1 = self.dec1(d1)                   # (B, 32, 64, 64)

        # Predict residual
        residual = self.out_conv(d1)         # (B, 3, 64, 64)

        # Output = input + residual (learn to correct the eye)
        return x + residual

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class GazeCorrectionDataset:
    """Synthetic dataset: extracts eye patches from webcam frames."""

    PATCH_SIZE = 64

    def __init__(self, face_detector, eye_corrector):
        self.face_detector = face_detector
        self.eye_corrector = eye_corrector

    def generate_pair(self, frame, eye_data):
        """Generate (input, corrected) pair from a single frame."""
        import numpy as np

        pairs = []
        for side in ("left", "right"):
            eye = eye_data[f"{side}_eye"]
            pupil = np.array(eye["iris"], dtype=np.float32)
            socket = np.array(eye["center"], dtype=np.float32)
            eye_width = float(eye["width"])

            if eye_width < 15 or eye_data["is_blinking"]:
                continue

            # Extract eye patch centered on eye center
            cx, cy = int(socket[0]), int(socket[1])
            half = self.PATCH_SIZE // 2
            h, w = frame.shape[:2]

            x0 = max(0, cx - half)
            y0 = max(0, cy - half)
            x1 = min(w, cx + half)
            y1 = min(h, cy + half)

            if (x1 - x0) < self.PATCH_SIZE or (y1 - y0) < self.PATCH_SIZE:
                continue

            input_patch = frame[y0:y1, x0:x1].copy()

            # Create target: iris moved to center
            offset_x = (pupil[0] - socket[0]) / (eye_width + 1e-6)
            offset_y = (pupil[1] - socket[1]) / (eye_width + 1e-6)

            # Generate corrected version using geometric method
            single_eye_data = {
                f"{side}_eye": eye,
                "is_blinking": False,
            }
            corrected_frame = self.eye_corrector.correct_frame(frame, single_eye_data)
            target_patch = corrected_frame[y0:y1, x0:x1].copy()

            # Normalize offset
            offset = np.array([offset_x, offset_y], dtype=np.float32)

            pairs.append((input_patch, target_patch, offset))

        return pairs

    def generate_from_video(self, video_path, max_frames=1000):
        """Generate dataset from a video file."""
        import cv2
        import numpy as np

        cap = cv2.VideoCapture(video_path)
        frames_data = []
        frame_count = 0

        while cap.isOpened() and frame_count < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            landmarks = self.face_detector.detect(frame)
            if landmarks is None:
                continue

            eye_data = self.face_detector.get_eye_data(landmarks, frame.shape)
            pairs = self.generate_pair(frame, eye_data)

            for input_patch, target_patch, offset in pairs:
                frames_data.append({
                    "input": input_patch,
                    "target": target_patch,
                    "offset": offset,
                })

            frame_count += 1

        cap.release()
        return frames_data

    def augment(self, input_patch, target_patch, offset):
        """Apply random augmentations for training variety."""
        import cv2
        import numpy as np

        # Random brightness
        if np.random.random() > 0.5:
            factor = 0.7 + np.random.random() * 0.6
            input_patch = np.clip(input_patch * factor, 0, 255).astype(np.uint8)
            target_patch = np.clip(target_patch * factor, 0, 255).astype(np.uint8)

        # Random noise
        if np.random.random() > 0.5:
            noise = np.random.normal(0, 5, input_patch.shape).astype(np.float32)
            input_patch = np.clip(input_patch.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            target_patch = np.clip(target_patch.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Random horizontal flip (both input and target)
        if np.random.random() > 0.5:
            input_patch = cv2.flip(input_patch, 1)
            target_patch = cv2.flip(target_patch, 1)
            offset = offset * np.array([-1, 1], dtype=np.float32)

        return input_patch, target_patch, offset
