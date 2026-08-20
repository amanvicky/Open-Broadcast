"""
OpenBroadcast — Core Component Tests

Tests for gaze estimation, eye correction, and system detection.
"""

import sys
import os
import numpy as np

# Add parent dir to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_gaze_estimator():
    """Test geometric gaze estimator."""
    from core.gaze_estimator import GeometricGazeEstimator

    estimator = GeometricGazeEstimator()

    # Create mock eye data
    eye_data = {
        "left_eye": {
            "outer": np.array([100, 200]),
            "inner": np.array([160, 200]),
            "top": np.array([130, 190]),
            "bottom": np.array([130, 210]),
            "center": np.array([130, 200]),
            "iris": np.array([130, 200]),  # Looking straight
            "width": 60.0,
            "height": 20.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
        },
        "right_eye": {
            "outer": np.array([220, 200]),
            "inner": np.array([280, 200]),
            "top": np.array([250, 190]),
            "bottom": np.array([250, 210]),
            "center": np.array([250, 200]),
            "iris": np.array([250, 200]),  # Looking straight
            "width": 60.0,
            "height": 20.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
        },
        "is_blinking": False,
        "eye_ratio": 0.33,
    }

    # Looking straight
    gaze = estimator.estimate(eye_data)
    assert abs(gaze.yaw) < 2.0, f"Expected yaw ~0, got {gaze.yaw}"
    assert gaze.is_looking_at_camera, "Should be looking at camera"

    # Looking right (iris offset positive)
    eye_data["left_eye"]["offset_x"] = 0.3
    eye_data["right_eye"]["offset_x"] = 0.3
    eye_data["left_eye"]["iris"] = np.array([148, 200])
    eye_data["right_eye"]["iris"] = np.array([268, 200])

    gaze = estimator.estimate(eye_data)
    assert gaze.yaw > 5.0, f"Expected positive yaw, got {gaze.yaw}"
    assert not gaze.is_looking_at_camera, "Should NOT be looking at camera"

    print("  ✓ Gaze estimator tests passed")


def test_eye_corrector():
    """Test eye correction engine."""
    from core.eye_corrector import EyeCorrector

    corrector = EyeCorrector(strength=0.85)

    # Create a test frame
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)

    # Create mock eye data with looking-away gaze
    eye_data = {
        "left_eye": {
            "outer": np.array([200.0, 240.0]),
            "inner": np.array([260.0, 240.0]),
            "top": np.array([230.0, 230.0]),
            "bottom": np.array([230.0, 250.0]),
            "center": np.array([230.0, 240.0]),
            "iris": np.array([248.0, 240.0]),  # Offset right
            "width": 60.0,
            "height": 20.0,
            "offset_x": 0.3,
            "offset_y": 0.0,
        },
        "right_eye": {
            "outer": np.array([380.0, 240.0]),
            "inner": np.array([440.0, 240.0]),
            "top": np.array([410.0, 230.0]),
            "bottom": np.array([410.0, 250.0]),
            "center": np.array([410.0, 240.0]),
            "iris": np.array([428.0, 240.0]),  # Offset right
            "width": 60.0,
            "height": 20.0,
            "offset_x": 0.3,
            "offset_y": 0.0,
        },
        "is_blinking": False,
        "eye_ratio": 0.33,
    }

    # Apply correction
    corrected = corrector.correct_frame(frame, eye_data)

    # Check that output is valid
    assert corrected.shape == frame.shape, f"Shape mismatch: {corrected.shape}"
    assert corrected.dtype == np.uint8, f"Wrong dtype: {corrected.dtype}"
    assert not np.array_equal(corrected, frame), "Correction should change the frame"

    # Test with blinking (should not crash)
    eye_data["is_blinking"] = True
    corrected_blink = corrector.correct_frame(frame, eye_data)
    assert corrected_blink.shape == frame.shape

    print("  ✓ Eye corrector tests passed")


def test_performance():
    """Test FPS counter and performance controller."""
    from utils.performance import FPSCounter, AdaptivePerformanceController

    # Test FPS counter
    counter = FPSCounter(window_size=5)
    for _ in range(10):
        counter.update()
    assert counter.fps > 0, "FPS should be positive"

    # Test adaptive controller with different RAM sizes
    for ram in [8, 16, 32]:
        controller = AdaptivePerformanceController(target_fps=20, total_ram_gb=ram)
        for _ in range(50):
            controller.update()
        assert controller.current_mode in ["full", "reduced", "minimal"]
        assert controller.min_free_gb > 0

    print("  ✓ Performance tests passed")


def test_gaze_net_model():
    """Test GazeNet-Lite model architecture."""
    import torch
    from models.gaze_net import GazeNetLite

    model = GazeNetLite()

    # Test forward pass
    dummy = torch.randn(1, 1, 36, 120)
    output = model(dummy)

    assert output.shape == (1, 2), f"Expected (1, 2), got {output.shape}"

    # Test batch processing
    batch = torch.randn(8, 1, 36, 120)
    output = model(batch)
    assert output.shape == (8, 2), f"Expected (8, 2), got {output.shape}"

    # Check parameter count
    params = model.count_parameters()
    assert params < 1_000_000, f"Too many params: {params}"
    print(f"  Model parameters: {params:,}")

    size_mb = model.model_size_mb()
    print(f"  Model size: {size_mb:.2f} MB")

    print("  ✓ GazeNet model tests passed")


def test_system_detector():
    """Test system detection."""
    from core.system_detector import detect_system, format_system_report

    info = detect_system()

    assert "cpu" in info
    assert "ram" in info
    assert "gpu" in info
    assert "tier" in info
    assert "config" in info

    assert info["cpu"]["physical_cores"] > 0
    assert info["ram"]["total_gb"] > 0
    assert info["tier"] in ["ULTRA_LOW", "LOW", "MEDIUM", "HIGH", "ULTRA_HIGH"]

    report = format_system_report(info)
    assert len(report) > 100, "Report should be substantial"

    print(f"  Detected tier: {info['tier']}")
    print(f"  CPU: {info['cpu']['brand']}")
    print(f"  RAM: {info['ram']['total_gb']} GB")
    print("  ✓ System detector tests passed")


if __name__ == "__main__":
    print("Running OpenBroadcast tests...\n")

    tests = [
        test_gaze_estimator,
        test_eye_corrector,
        test_performance,
        test_gaze_net_model,
        test_system_detector,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ {test.__name__} FAILED: {e}")
            failed += 1

    print(f"\n{'='*50}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)} tests")
    print(f"{'='*50}")

    if failed > 0:
        sys.exit(1)
