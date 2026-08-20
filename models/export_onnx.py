"""
OpenBroadcast — ONNX Export & Quantization

Exports trained PyTorch model to ONNX and quantizes to INT8
for fast CPU inference via ONNX Runtime.

Usage:
    python -m models.export_onnx --weights models/weights/gaze_net_best.pth
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch

from models.gaze_net import GazeNetLite


def export_to_onnx(weights_path, output_dir=None, quantize=True):
    """
    Export PyTorch model to ONNX and optionally quantize to INT8.

    Steps:
    1. Load PyTorch model weights
    2. Export to ONNX format (opset 13)
    3. Verify ONNX model runs correctly
    4. Quantize to INT8 (4x smaller, 2-3x faster on CPU)
    5. Verify quantized model accuracy
    """
    weights_path = Path(weights_path)
    if output_dir is None:
        output_dir = weights_path.parent
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {weights_path}...")
    model = GazeNetLite()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()

    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,}")
    print(f"PyTorch model size: {model.model_size_mb():.2f} MB")

    # Dummy input
    dummy = torch.randn(1, 1, 36, 120)

    # Export to ONNX
    onnx_path = output_dir / "gaze_net.onnx"
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["eye_crops"],
        output_names=["gaze"],
        dynamic_axes={"eye_crops": {0: "batch"}, "gaze": {0: "batch"}},
        opset_version=13,
    )
    print(f"\nExported ONNX model: {onnx_path}")

    # Verify ONNX model
    print("\nVerifying ONNX model...")
    import onnxruntime as ort
    session = ort.InferenceSession(str(onnx_path))
    test_input = np.random.randn(1, 1, 36, 120).astype(np.float32)
    output = session.run(None, {"eye_crops": test_input})
    print(f"  Input: {test_input.shape} → Output: {output[0].shape}")
    print(f"  Sample output: pitch={np.degrees(output[0][0][0]):.2f}° yaw={np.degrees(output[0][0][1]):.2f}°")

    onnx_size = os.path.getsize(str(onnx_path)) / 1024
    print(f"  ONNX model size: {onnx_size:.1f} KB")

    if quantize:
        print("\nQuantizing to INT8...")
        quant_path = output_dir / "gaze_net_quantized.onnx"

        try:
            from onnxruntime.quantization import quantize_dynamic, QuantType
            quantize_dynamic(
                str(onnx_path), str(quant_path),
                weight_type=QuantType.QInt8,
            )

            # Verify quantized model
            session_q = ort.InferenceSession(str(quant_path))
            output_q = session_q.run(None, {"eye_crops": test_input})

            diff = np.abs(output[0] - output_q[0])
            print(f"  Max output difference: {diff.max():.6f}")

            quant_size = os.path.getsize(str(quant_path)) / 1024
            print(f"  Quantized model size: {quant_size:.1f} KB")
            print(f"  Compression ratio: {onnx_size/quant_size:.1f}x")

            print(f"\n  Quantized model: {quant_path}")

        except ImportError:
            print("  onnxruntime.quantization not available. Skipping quantization.")
            print("  Install with: pip install onnxruntime")

    print("\nExport complete!")
    print(f"  FP32 model: {onnx_path}")
    if quantize:
        print(f"  INT8 model: {quant_path}")

    return str(onnx_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export GazeNet-Lite to ONNX")
    parser.add_argument("--weights", type=str, required=True,
                       help="Path to PyTorch .pth weights file")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory (default: same as weights)")
    parser.add_argument("--no_quantize", action="store_true",
                       help="Skip INT8 quantization")
    args = parser.parse_args()

    export_to_onnx(args.weights, args.output_dir, quantize=not args.no_quantize)
