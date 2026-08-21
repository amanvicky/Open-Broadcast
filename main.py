"""OpenBroadcast — Eye Gaze Correction for Low-End PCs."""

import sys
import os
import argparse
import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def test_image_mode(image_path, output_path, strength=0.85, amplification=4.0):
    """Run correction pipeline on a static image (no Qt needed)."""
    from core.face_detector import FaceDetector
    from core.eye_corrector import EyeCorrector

    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Error: Cannot read image: {image_path}")
        sys.exit(1)

    print(f"Loaded: {image_path} ({frame.shape[1]}x{frame.shape[0]})")

    detector = FaceDetector(detection_interval=1)
    corrector = EyeCorrector(strength=strength, amplification=amplification)

    landmarks = detector.detect(frame)
    if landmarks is None:
        print("No face detected.")
        sys.exit(1)

    eye_data = detector.get_eye_data(landmarks, frame.shape)
    corrected = corrector.correct_frame(frame, eye_data)

    # Side-by-side comparison
    h, w = frame.shape[:2]
    comparison = np.zeros((h, w * 2 + 4, 3), dtype=np.uint8)
    comparison[:, :w] = frame
    comparison[:, w + 4:] = corrected
    comparison[:, w:w + 4] = (255, 255, 255)

    cv2.imwrite(output_path, comparison)
    print(f"Saved: {output_path}")
    print(f"Left=Original, Right=Corrected (strength={strength}, amp={amplification}x)")

    detector.cleanup()


def main():
    parser = argparse.ArgumentParser(description="OpenBroadcast — Eye Gaze Correction")
    parser.add_argument("--test-image", type=str, help="Run on a static image instead of webcam")
    parser.add_argument("--output", type=str, default="result.jpg", help="Output path for --test-image")
    parser.add_argument("--strength", type=float, default=0.85, help="Correction strength (0-1)")
    parser.add_argument("--amplification", type=float, default=4.0, help="Correction amplification")
    args = parser.parse_args()

    if args.test_image:
        test_image_mode(args.test_image, args.output, args.strength, args.amplification)
        return

    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtGui import QFont
    from config import load_config, save_config, get_default_config, ensure_app_dir
    from ui.styles import apply_theme
    from ui.main_window import MainWindow

    os.environ.setdefault("QT_AUTO_SCREEN_SCALE_FACTOR", "1")

    app = QApplication(sys.argv)
    app.setApplicationName("OpenBroadcast")
    apply_theme(app)
    app.setFont(QFont("Segoe UI", 10))

    ensure_app_dir()
    config = load_config()

    if config.get("first_run", True):
        config = get_default_config()
        config["first_run"] = False
        save_config(config)

    window = MainWindow(config)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
