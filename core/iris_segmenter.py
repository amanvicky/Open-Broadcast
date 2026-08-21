"""Tiny iris segmentation model for precise iris boundary detection.

Replaces the fixed-radius circular extraction with a learned segmentation
that produces exact iris masks. This improves transplant quality because:
- Irises are not perfectly circular
- Size varies per person (18-35% of eye width)
- Partial occlusion from eyelids is handled correctly

Architecture: Lightweight encoder-decoder (~50K params)
Input: 64x64x3 eye patch
Output: 64x64x1 iris mask (binary)

Uses MediaPipe landmarks as pseudo-labels during training.
"""

import numpy as np

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class IrisSegmentationNet(nn.Module):
    """Lightweight iris segmentation network (~50K params).
    
    Uses depthwise separable convolutions for efficiency on CPU.
    """

    def __init__(self, in_channels=3):
        super().__init__()
        
        # Encoder
        self.enc1 = self._depthwise_sep_conv(in_channels, 16)
        self.enc2 = self._depthwise_sep_conv(16, 32)
        self.enc3 = self._depthwise_sep_conv(32, 64)
        
        # Bottleneck
        self.bottleneck = self._depthwise_sep_conv(64, 64)
        
        # Decoder with skip connections (regular convs for variable input channels)
        self.up3 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(32 + 64, 32, 3, padding=1, bias=False),  # 32 up + 64 skip = 96
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
        )
        
        self.up2 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(16 + 32, 16, 3, padding=1, bias=False),  # 16 up + 32 skip = 48
            nn.BatchNorm2d(16), nn.ReLU(inplace=True),
        )
        
        self.up1 = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(8 + 16, 8, 3, padding=1, bias=False),   # 8 up + 16 skip = 24
            nn.BatchNorm2d(8), nn.ReLU(inplace=True),
        )
        
        # Output: single channel mask
        self.out_conv = nn.Conv2d(8, 1, 1)
        
        self.pool = nn.MaxPool2d(2)

    def _depthwise_sep_conv(self, in_ch, out_ch):
        """Depthwise separable convolution: depthwise + pointwise."""
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        """
        Args:
            x: (B, 3, 64, 64) eye patch normalized to [0, 1]
        Returns:
            mask: (B, 1, 64, 64) iris mask (logits)
        """
        # Encoder
        e1 = self.enc1(x)           # (B, 16, 64, 64)
        e2 = self.enc2(self.pool(e1))  # (B, 32, 32, 32)
        e3 = self.enc3(self.pool(e2))  # (B, 64, 16, 16)
        
        # Bottleneck
        b = self.bottleneck(self.pool(e3))  # (B, 64, 8, 8)
        
        # Decoder with skip connections
        d3 = self.up3(b)                    # (B, 32, 16, 16)
        d3 = torch.cat([d3, e3], dim=1)     # (B, 64, 16, 16)
        d3 = self.dec3(d3)                   # (B, 32, 16, 16)
        
        d2 = self.up2(d3)                    # (B, 16, 32, 32)
        d2 = torch.cat([d2, e2], dim=1)     # (B, 32, 32, 32)
        d2 = self.dec2(d2)                   # (B, 16, 32, 32)
        
        d1 = self.up1(d2)                    # (B, 8, 64, 64)
        d1 = torch.cat([d1, e1], dim=1)     # (B, 16, 64, 64)
        d1 = self.dec1(d1)                   # (B, 8, 64, 64)
        
        return self.out_conv(d1)             # (B, 1, 64, 64)

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class IrisSegmenter:
    """Inference wrapper for iris segmentation."""
    
    PATCH_SIZE = 64
    
    def __init__(self, model_path=None, device=None):
        if not HAS_TORCH:
            raise ImportError("PyTorch required")
        
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = IrisSegmentationNet(in_channels=3)
        
        if model_path and __import__("os").path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"[IrisSegmenter] Loaded model from {model_path}")
        
        self.model.to(self.device)
        self.model.eval()
    
    def segment(self, frame, eye_data):
        """Segment iris in both eyes.
        
        Args:
            frame: BGR image (numpy array)
            eye_data: dict from FaceDetector.get_eye_data()
            
        Returns:
            dict with 'left_mask' and 'right_mask' (64x64 float32 arrays)
        """
        import cv2
        masks = {}
        
        for side in ("left", "right"):
            eye = eye_data[f"{side}_eye"]
            pupil = np.array(eye["iris"], dtype=np.float32)
            socket = np.array(eye["center"], dtype=np.float32)
            eye_width = float(eye["width"])
            
            if eye_width < 15 or eye_data["is_blinking"]:
                masks[f"{side}_mask"] = None
                continue
            
            # Extract eye patch centered on eye center
            h, w = frame.shape[:2]
            cx, cy = int(socket[0]), int(socket[1])
            half = self.PATCH_SIZE // 2
            
            x0 = max(0, cx - half)
            y0 = max(0, cy - half)
            x1 = min(w, cx + half)
            y1 = min(h, cy + half)
            
            if (x1 - x0) < self.PATCH_SIZE or (y1 - y0) < self.PATCH_SIZE:
                masks[f"{side}_mask"] = None
                continue
            
            patch = frame[y0:y1, x0:x1].copy()
            
            # Prepare tensor
            tensor = torch.from_numpy(patch).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            tensor = tensor.to(self.device)
            
            # Inference
            with torch.no_grad():
                mask_logits = self.model(tensor)
            
            # Convert to numpy
            mask = torch.sigmoid(mask_logits).squeeze().cpu().numpy()
            mask = (mask > 0.5).astype(np.float32)
            
            # Smooth mask edges
            mask = cv2.GaussianBlur(mask, (5, 5), 1.0)
            mask = (mask > 0.3).astype(np.float32)
            
            masks[f"{side}_mask"] = mask
        
        return masks
    
    def segment_and_extract(self, frame, eye_data):
        """Segment iris and extract patches with exact masks.
        
        Returns:
            dict with 'left' and 'right' containing:
                'patch': 64x64x3 uint8
                'mask': 64x64 float32 (exact iris boundary)
                'center': (cx, cy) in frame coordinates
                'iris_center': (ix, iy) in frame coordinates
        """
        import cv2
        
        masks = self.segment(frame, eye_data)
        results = {}
        
        for side in ("left", "right"):
            eye = eye_data[f"{side}_eye"]
            pupil = np.array(eye["iris"], dtype=np.float32)
            socket = np.array(eye["center"], dtype=np.float32)
            eye_width = float(eye["width"])
            
            if eye_width < 15 or masks.get(f"{side}_mask") is None:
                results[side] = None
                continue
            
            h, w = frame.shape[:2]
            cx, cy = int(socket[0]), int(socket[1])
            half = self.PATCH_SIZE // 2
            
            x0 = max(0, cx - half)
            y0 = max(0, cy - half)
            x1 = min(w, cx + half)
            y1 = min(h, cy + half)
            
            if (x1 - x0) < self.PATCH_SIZE or (y1 - y0) < self.PATCH_SIZE:
                results[side] = None
                continue
            
            patch = frame[y0:y1, x0:x1].copy()
            mask = masks[f"{side}_mask"]
            
            results[side] = {
                "patch": patch,
                "mask": mask,
                "center": (cx, cy),
                "iris_center": (int(pupil[0]), int(pupil[1])),
                "eye_width": eye_width,
                "offset": (x0, y0),  # top-left corner in frame coords
            }
        
        return results
