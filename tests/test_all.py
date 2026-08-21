"""Headless test: exercise the full pipeline — data generation, model, training."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import traceback


def test_data_generator():
    """Generate synthetic pairs from a random image."""
    import cv2
    from core.face_detector import FaceDetector
    from core.eye_corrector import EyeCorrector
    from core.gaze_model import GazeCorrectionDataset

    detector = FaceDetector(detection_interval=1)
    corrector = EyeCorrector(strength=0.85, amplification=3.0)
    dataset = GazeCorrectionDataset(detector, corrector)

    # Create a synthetic face-like image (won't detect, but tests the pipeline)
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    eye_data = {
        "left_eye": {
            "outer": np.array([280.0, 240.0]),
            "inner": np.array([340.0, 240.0]),
            "top": np.array([310.0, 230.0]),
            "bottom": np.array([310.0, 250.0]),
            "center": np.array([310.0, 240.0]),
            "iris": np.array([320.0, 240.0]),
            "width": 60.0,
            "height": 20.0,
            "offset_x": 0.17,
            "offset_y": 0.0,
        },
        "right_eye": {
            "outer": np.array([380.0, 240.0]),
            "inner": np.array([440.0, 240.0]),
            "top": np.array([410.0, 230.0]),
            "bottom": np.array([410.0, 250.0]),
            "center": np.array([410.0, 240.0]),
            "iris": np.array([420.0, 240.0]),
            "width": 60.0,
            "height": 20.0,
            "offset_x": 0.17,
            "offset_y": 0.0,
        },
        "is_blinking": False,
        "eye_ratio": 0.33,
    }

    pairs = dataset.generate_pair(frame, eye_data)
    assert len(pairs) == 2, f"Expected 2 pairs, got {len(pairs)}"

    for inp, tgt, offset in pairs:
        assert inp.shape == (64, 64, 3), f"Input shape: {inp.shape}"
        assert tgt.shape == (64, 64, 3), f"Target shape: {tgt.shape}"
        assert offset.shape == (2,), f"Offset shape: {offset.shape}"

    # Test augmentation
    aug_inp, aug_tgt, aug_off = dataset.augment(pairs[0][0], pairs[0][1], pairs[0][2])
    assert aug_inp.shape == (64, 64, 3)
    assert aug_tgt.shape == (64, 64, 3)

    detector.cleanup()
    print("[PASS] test_data_generator")


def test_model_forward_backward():
    """Model forward + backward pass."""
    import torch
    from core.gaze_model import GazeCorrectionNet

    model = GazeCorrectionNet(in_channels=3, offset_dim=2)
    params = model.count_parameters()
    assert params > 10000, f"Too few params: {params}"
    assert params < 5000000, f"Too many params: {params}"

    # Forward pass
    x = torch.randn(4, 3, 64, 64)
    offset = torch.randn(4, 2)
    out = model(x, offset)

    assert out.shape == (4, 3, 64, 64), f"Output shape: {out.shape}"

    # Backward pass
    loss = torch.nn.functional.l1_loss(out, torch.randn(4, 3, 64, 64))
    loss.backward()

    # Check gradients exist
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"

    print(f"[PASS] test_model_forward_backward ({params:,} params)")


def test_training_loop():
    """Mini training loop — 3 epochs on synthetic data."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from core.gaze_model import GazeCorrectionNet

    # Create tiny dataset
    n = 32
    inputs = torch.randn(n, 3, 64, 64)
    targets = torch.randn(n, 3, 64, 64)
    offsets = torch.randn(n, 2)
    dataset = TensorDataset(inputs, targets, offsets)
    loader = DataLoader(dataset, batch_size=8)

    model = GazeCorrectionNet(in_channels=3, offset_dim=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.L1Loss()

    losses = []
    model.train()
    for epoch in range(3):
        epoch_loss = 0
        for batch_in, batch_tgt, batch_off in loader:
            pred = model(batch_in, batch_off)
            loss = loss_fn(pred, batch_tgt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        losses.append(epoch_loss / len(loader))

    # Loss should decrease or stay stable
    assert losses[-1] <= losses[0] * 1.5, f"Loss diverged: {losses}"
    print(f"[PASS] test_training_loop (losses: {[f'{l:.4f}' for l in losses]})")


def test_model_save_load():
    """Save and load model checkpoint."""
    import torch
    from core.gaze_model import GazeCorrectionNet

    model = GazeCorrectionNet(in_channels=3, offset_dim=2)
    model.eval()
    x = torch.randn(1, 3, 64, 64)
    offset = torch.randn(1, 2)

    with torch.no_grad():
        out1 = model(x, offset)

    # Save
    os.makedirs("models", exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_shape": (3, 64, 64),
        "offset_dim": 2,
    }, "models/test_checkpoint.pth")

    # Load into new model
    model2 = GazeCorrectionNet(in_channels=3, offset_dim=2)
    model2.eval()
    checkpoint = torch.load("models/test_checkpoint.pth", weights_only=False)
    model2.load_state_dict(checkpoint["model_state_dict"])

    with torch.no_grad():
        out2 = model2(x, offset)

    diff = (out1 - out2).abs().max().item()
    assert diff < 1e-5, f"Output mismatch after load: {diff}"

    os.remove("models/test_checkpoint.pth")
    print("[PASS] test_model_save_load")


def test_large_scale_generator():
    """Test the large-scale pair generator augmentation."""
    from data.large_scale_generator import LargeScaleGenerator

    gen = LargeScaleGenerator()

    # Create synthetic patches
    input_patch = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    target_patch = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
    offset = np.array([0.15, -0.05], dtype=np.float32)

    # Generate 10 augmented versions
    pairs = []
    for _ in range(10):
        aug_inp, tgt, off = gen._augment_pair(input_patch.copy(), target_patch.copy(), offset.copy())
        pairs.append((aug_inp, tgt, off))

    # Verify all pairs are valid
    for inp, tgt, off in pairs:
        assert inp.shape == (64, 64, 3), f"Shape: {inp.shape}"
        assert inp.dtype == np.uint8, f"dtype: {inp.dtype}"
        assert off.shape == (2,), f"Offset shape: {off.shape}"

    # Verify augmentations produce variety (not all identical)
    inputs = np.stack([p[0] for p in pairs])
    std = inputs.std(axis=0).mean()
    assert std > 0, "Augmentations produce no variety"

    gen.cleanup()
    print(f"[PASS] test_large_scale_generator (std={std:.2f})")


def test_eye_corrector_with_iris_overlay():
    """Test the eye corrector produces valid output for overlay display."""
    import cv2
    from core.eye_corrector import EyeCorrector

    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    eye_data = {
        "left_eye": {
            "outer": np.array([280.0, 240.0]),
            "inner": np.array([340.0, 240.0]),
            "center": np.array([310.0, 240.0]),
            "iris": np.array([325.0, 240.0]),
            "width": 60.0,
            "height": 20.0,
            "offset_x": 0.25,
            "offset_y": 0.0,
        },
        "right_eye": {
            "outer": np.array([380.0, 240.0]),
            "inner": np.array([440.0, 240.0]),
            "center": np.array([410.0, 240.0]),
            "iris": np.array([425.0, 240.0]),
            "width": 60.0,
            "height": 20.0,
            "offset_x": 0.25,
            "offset_y": 0.0,
        },
        "is_blinking": False,
        "eye_ratio": 0.33,
    }

    corrector = EyeCorrector(strength=0.85, amplification=4.0)
    result = corrector.correct_frame(frame, eye_data)

    assert result.shape == frame.shape, f"Shape mismatch: {result.shape}"
    assert result is not frame, "Returned same object"

    diff = np.abs(result.astype(int) - frame.astype(int)).sum()
    assert diff > 0, "No correction applied"

    # Verify iris data is accessible for overlay
    for side in ("left", "right"):
        iris = eye_data[f"{side}_eye"]["iris"]
        center = eye_data[f"{side}_eye"]["center"]
        assert iris.shape == (2,)
        assert center.shape == (2,)

    print(f"[PASS] test_eye_corrector_with_iris_overlay (diff={diff})")


if __name__ == "__main__":
    tests = [
        test_data_generator,
        test_model_forward_backward,
        test_training_loop,
        test_model_save_load,
        test_large_scale_generator,
        test_eye_corrector_with_iris_overlay,
    ]

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            traceback.print_exc()
            failed += 1

    print(f"\n{'='*60}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed:
        sys.exit(1)
