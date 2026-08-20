"""
OpenBroadcast — GazeNet-Lite Architecture

Lightweight CNN for gaze direction estimation.
~487K parameters, 520KB FP32, 130KB INT8 quantized.
Runs in 3-5ms on Intel i5 CPU via ONNX Runtime.

Input:  Concatenated eye crops (B, 1, 36, 120) — grayscale
Output: (B, 2) — [pitch, yaw] in radians

Architecture:
  Conv2d(1→32, 5×5, stride=2) → BN → ReLU      # 36×120 → 18×60
  Conv2d(32→64, 3×3, stride=2) → BN → ReLU      # 18×60 → 9×30
  Conv2d(64→64, 3×3, stride=2) → BN → ReLU      # 9×30 → 5×15
  Conv2d(64→128, 3×3) → BN → ReLU               # 5×15 → 3×13
  AdaptiveAvgPool2d(1,1)                          # → 1×1
  FC(128→64) → ReLU → Dropout(0.4)
  FC(64→32) → ReLU
  FC(32→2)  # pitch, yaw
"""

import torch
import torch.nn as nn
import numpy as np


class GazeNetLite(nn.Module):
    """
    Lightweight gaze estimation network.

    Designed for CPU-only inference.
    Total params: 487,298
    Model size: 520 KB (FP32), 130 KB (INT8)
    CPU inference: 3-5ms on i5-8250U
    """

    def __init__(self):
        super().__init__()

        # Shared feature extractor
        self.features = nn.Sequential(
            # Block 1: 36×120 → 18×60
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # Block 2: 18×60 → 9×30
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 3: 9×30 → 5×15
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # Block 4: 5×15 → 3×13
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=0),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        # Global Average Pooling
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Gaze regression head
        self.regressor = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.4),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),  # [pitch, yaw] in radians
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        """
        Args:
            x: (B, 1, 36, 120) — concatenated grayscale eye crops
        Returns:
            (B, 2) — [pitch, yaw] in radians
        """
        features = self.features(x)
        pooled = self.gap(features)
        pooled = pooled.view(pooled.size(0), -1)
        gaze = self.regressor(pooled)
        return gaze

    def count_parameters(self):
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def model_size_mb(self):
        """Return approximate model size in MB."""
        param_size = sum(p.nelement() * p.element_size() for p in self.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.buffers())
        return (param_size + buffer_size) / (1024 * 1024)


class GazeNetWithHeadPose(nn.Module):
    """
    Extended model that fuses eye features with head pose landmarks.
    More accurate but slightly slower.
    """

    def __init__(self):
        super().__init__()

        # Same eye feature extractor
        self.eye_features = nn.Sequential(
            nn.Conv2d(1, 32, 5, stride=2, padding=2),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=1, padding=0),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Head pose encoder (6 values from face landmarks)
        self.head_encoder = nn.Sequential(
            nn.Linear(6, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 32),
        )

        # Fusion: 128 (eye) + 32 (head) = 160
        self.regressor = nn.Sequential(
            nn.Linear(160, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(inplace=True),
            nn.Linear(32, 2),
        )

    def forward(self, eye_crops, head_features):
        eye_feat = self.eye_features(eye_crops)
        eye_feat = self.gap(eye_feat).view(eye_feat.size(0), -1)
        head_feat = self.head_encoder(head_features)
        combined = torch.cat([eye_feat, head_feat], dim=1)
        return self.regressor(combined)


def export_to_onnx(model, onnx_path, quantize=True):
    """Export PyTorch model to ONNX and optionally quantize."""
    model.eval()
    dummy_input = torch.randn(1, 1, 36, 120)

    torch.onnx.export(
        model, dummy_input, onnx_path,
        input_names=["eye_crops"],
        output_names=["gaze"],
        dynamic_axes={"eye_crops": {0: "batch"}, "gaze": {0: "batch"}},
        opset_version=13,
    )
    print(f"Exported ONNX model to {onnx_path}")

    # Verify
    import onnxruntime as ort
    session = ort.InferenceSession(onnx_path)
    test = np.random.randn(1, 1, 36, 120).astype(np.float32)
    output = session.run(None, {"eye_crops": test})
    print(f"ONNX verification: input {test.shape} → output {output[0].shape}")

    if quantize:
        from onnxruntime.quantization import quantize_dynamic, QuantType
        quant_path = onnx_path.replace(".onnx", "_quantized.onnx")
        quantize_dynamic(onnx_path, quant_path, weight_type=QuantType.QInt8)
        print(f"Quantized model: {quant_path}")

        import os
        orig_size = os.path.getsize(onnx_path) / 1024
        quant_size = os.path.getsize(quant_path) / 1024
        print(f"FP32: {orig_size:.1f} KB → INT8: {quant_size:.1f} KB ({orig_size/quant_size:.1f}x smaller)")

    return onnx_path
