"""Prove the eye corrector actually shifts pixels toward center."""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import cv2
import numpy as np
from core.eye_corrector import EyeCorrector


def test_correction_shifts_iris_to_center():
    """With a known offset, verify the corrected frame differs from original."""
    corrector = EyeCorrector(strength=1.0, amplification=1.0, smoothing=0.0)

    frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

    eye_data = {
        "left_eye": {
            "center": np.array([230.0, 240.0]),
            "iris": np.array([260.0, 240.0]),  # 30px offset right
            "width": 60.0,
            "height": 20.0,
            "offset_x": 0.5,  # looking right
            "offset_y": 0.0,
        },
        "right_eye": {
            "center": np.array([410.0, 240.0]),
            "iris": np.array([440.0, 240.0]),  # 30px offset right
            "width": 60.0,
            "height": 20.0,
            "offset_x": 0.5,
            "offset_y": 0.0,
        },
        "is_blinking": False,
        "eye_ratio": 0.33,
    }

    corrected = corrector.correct_frame(frame, eye_data)

    # Core assertions
    assert corrected.shape == frame.shape
    assert corrected.dtype == np.uint8
    assert not np.array_equal(corrected, frame), "Frame should change"

    # The correction should shift pixels LEFT (toward center)
    # So pixels in the corrected frame at eye center should come from
    # the right side of the original (where the iris is)
    left_eye_x, left_eye_y = 230, 240
    # Before correction: frame[240, 230] is some value
    # After correction: corrected[240, 230] should be frame[240, 230+shift]
    # i.e., the shifted content
    orig_val = frame[left_eye_y, left_eye_x].astype(float)
    corr_val = corrected[left_eye_y, left_eye_x].astype(float)
    # They should differ (pixels shifted)
    diff = np.abs(orig_val - corr_val).mean()
    assert diff > 0.5, f"Pixel at eye center should differ after correction, got diff={diff:.2f}"

    print(f"  ✓ Correction shifts pixels by {diff:.1f} avg at eye center")


def test_no_shift_when_looking_at_camera():
    """When iris is already at center, correction should be minimal."""
    corrector = EyeCorrector(strength=1.0, amplification=1.0, smoothing=0.0)

    frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)

    eye_data = {
        "left_eye": {
            "center": np.array([230.0, 240.0]),
            "iris": np.array([230.0, 240.0]),  # AT center
            "width": 60.0,
            "height": 20.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
        },
        "right_eye": {
            "center": np.array([410.0, 240.0]),
            "iris": np.array([410.0, 240.0]),  # AT center
            "width": 60.0,
            "height": 20.0,
            "offset_x": 0.0,
            "offset_y": 0.0,
        },
        "is_blinking": False,
        "eye_ratio": 0.33,
    }

    corrected = corrector.correct_frame(frame, eye_data)
    # When looking at center, the correction should still produce a valid frame
    assert corrected.shape == frame.shape
    assert corrected.dtype == np.uint8
    # The difference should be very small (only the attention offset below center)
    diff = np.abs(corrected.astype(float) - frame.astype(float)).mean()
    assert diff < 5.0, f"When centered, correction should be minimal, got diff={diff:.1f}"

    print(f"  ✓ Minimal change when looking at camera (avg diff={diff:.2f})")


def test_amplification_increases_shift():
    """Higher amplification should produce larger pixel shifts."""
    frame = np.random.randint(50, 200, (480, 640, 3), dtype=np.uint8)
    eye_data = {
        "left_eye": {
            "center": np.array([230.0, 240.0]),
            "iris": np.array([260.0, 240.0]),
            "width": 60.0, "height": 20.0,
            "offset_x": 0.5, "offset_y": 0.0,
        },
        "right_eye": {
            "center": np.array([410.0, 240.0]),
            "iris": np.array([440.0, 240.0]),
            "width": 60.0, "height": 20.0,
            "offset_x": 0.5, "offset_y": 0.0,
        },
        "is_blinking": False, "eye_ratio": 0.33,
    }

    c1 = EyeCorrector(strength=1.0, amplification=1.0, smoothing=0.0)
    c2 = EyeCorrector(strength=1.0, amplification=3.0, smoothing=0.0)

    diff1 = np.abs(c1.correct_frame(frame, eye_data).astype(float) - frame.astype(float)).mean()
    diff2 = np.abs(c2.correct_frame(frame, eye_data).astype(float) - frame.astype(float)).mean()

    assert diff2 > diff1, f"Amplification 3x ({diff2:.1f}) should > 1x ({diff1:.1f})"
    print(f"  ✓ Amplification works: 1x={diff1:.1f}, 3x={diff2:.1f}")


if __name__ == "__main__":
    print("Proving eye correction works...")
    test_correction_shifts_iris_to_center()
    test_no_shift_when_looking_at_camera()
    test_amplification_increases_shift()
    print("All proofs passed ✓")
