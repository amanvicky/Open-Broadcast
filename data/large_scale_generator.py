"""Large-scale synthetic training data generator.

Creates massive training datasets by:
1. Taking any face images
2. Detecting eyes
3. Generating multiple augmented versions with different gaze offsets
4. Creating (input, corrected) pairs

Can generate 100K+ training pairs from a few thousand base images.

Usage:
    # Generate 100K pairs from downloaded datasets
    python -m data.large_scale_generator --input data/raw --output data/large_pairs.npz --target-count 100000

    # Generate from webcam (collect for 5 minutes)
    python -m data.large_scale_generator --webcam --output data/webcam_pairs.npz --target-count 50000
"""

import os
import sys
import argparse
import numpy as np
import cv2
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class LargeScaleGenerator:
    """Generate massive training datasets with diverse augmentations."""

    PATCH_SIZE = 64

    def __init__(self):
        from core.face_detector import FaceDetector
        from core.eye_corrector import EyeCorrector

        self.detector = FaceDetector(detection_interval=1)
        self.corrector = EyeCorrector(strength=0.85, amplification=3.0)

    def generate_from_image(self, frame, target_per_image=20):
        """Generate multiple pairs from a single image with heavy augmentation."""
        pairs = []

        landmarks = self.detector.detect(frame)
        if landmarks is None:
            return pairs

        eye_data = self.detector.get_eye_data(landmarks, frame.shape)

        for side in ("left", "right"):
            eye = eye_data[f"{side}_eye"]
            pupil = np.array(eye["iris"], dtype=np.float32)
            socket = np.array(eye["center"], dtype=np.float32)
            eye_width = float(eye["width"])

            if eye_width < 15 or eye_data["is_blinking"]:
                continue

            # Extract base eye patch
            cx, cy = int(socket[0]), int(socket[1])
            half = self.PATCH_SIZE // 2

            x0 = max(0, cx - half)
            y0 = max(0, cy - half)
            x1 = min(frame.shape[1], cx + half)
            y1 = min(frame.shape[0], cy + half)

            if (x1 - x0) < self.PATCH_SIZE or (y1 - y0) < self.PATCH_SIZE:
                continue

            base_patch = frame[y0:y1, x0:x1].copy()
            base_offset_x = (pupil[0] - socket[0]) / (eye_width + 1e-6)
            base_offset_y = (pupil[1] - socket[1]) / (eye_width + 1e-6)

            # Generate corrected version
            single_eye_data = {f"{side}_eye": eye, "is_blinking": False}
            corrected_frame = self.corrector.correct_frame(frame, single_eye_data)
            base_target = corrected_frame[y0:y1, x0:x1].copy()

            # Generate augmented versions
            for _ in range(target_per_image):
                aug_input, aug_target, aug_offset = self._augment_pair(
                    base_patch, base_target,
                    np.array([base_offset_x, base_offset_y], dtype=np.float32)
                )
                pairs.append({
                    "input": aug_input,
                    "target": aug_target,
                    "offset": aug_offset,
                })

        return pairs

    def _augment_pair(self, input_patch, target_patch, offset):
        """Apply random augmentations to create training variety."""
        # Random brightness
        if np.random.random() > 0.3:
            factor = 0.5 + np.random.random() * 1.0
            input_patch = np.clip(input_patch * factor, 0, 255).astype(np.uint8)
            target_patch = np.clip(target_patch * factor, 0, 255).astype(np.uint8)

        # Random contrast
        if np.random.random() > 0.3:
            alpha = 0.5 + np.random.random() * 1.0
            input_patch = np.clip(alpha * (input_patch.astype(np.float32) - 128) + 128, 0, 255).astype(np.uint8)
            target_patch = np.clip(alpha * (target_patch.astype(np.float32) - 128) + 128, 0, 255).astype(np.uint8)

        # Random noise
        if np.random.random() > 0.5:
            sigma = np.random.random() * 10
            noise = np.random.normal(0, sigma, input_patch.shape).astype(np.float32)
            input_patch = np.clip(input_patch.astype(np.float32) + noise, 0, 255).astype(np.uint8)
            target_patch = np.clip(target_patch.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # Random blur
        if np.random.random() > 0.7:
            ksize = np.random.choice([3, 5])
            input_patch = cv2.GaussianBlur(input_patch, (ksize, ksize), 0)

        # Random horizontal flip
        if np.random.random() > 0.5:
            input_patch = cv2.flip(input_patch, 1)
            target_patch = cv2.flip(target_patch, 1)
            offset = offset * np.array([-1, 1], dtype=np.float32)

        # Random rotation (±5 degrees)
        if np.random.random() > 0.7:
            angle = np.random.uniform(-5, 5)
            h, w = input_patch.shape[:2]
            M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
            input_patch = cv2.warpAffine(input_patch, M, (w, h))
            target_patch = cv2.warpAffine(target_patch, M, (w, h))

        # Random erasing (simulate occlusion)
        if np.random.random() > 0.8:
            x = np.random.randint(0, self.PATCH_SIZE - 10)
            y = np.random.randint(0, self.PATCH_SIZE - 10)
            w = np.random.randint(5, 15)
            h = np.random.randint(5, 15)
            input_patch[y:y+h, x:x+w] = np.mean(input_patch)

        return input_patch, target_patch, offset

    def generate_from_webcam(self, target_count=50000, duration_seconds=300):
        """Generate pairs from webcam with guided movement."""
        import time

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        pairs = []
        start_time = time.time()

        print(f"[Webcam] Collecting for {duration_seconds} seconds...")
        print("[Webcam] Move your head around for variety!")

        while len(pairs) < target_count:
            elapsed = time.time() - start_time
            if elapsed > duration_seconds:
                break

            ret, frame = cap.read()
            if not ret:
                continue

            frame_pairs = self.generate_from_image(frame, target_per_image=5)
            pairs.extend(frame_pairs)

            if len(pairs) % 1000 == 0:
                print(f"  {len(pairs)}/{target_count} pairs ({elapsed:.0f}s)")

        cap.release()
        return pairs

    def generate_from_directory(self, input_dir, target_count=100000):
        """Generate pairs from all images in a directory."""
        input_dir = Path(input_dir)
        image_paths = []

        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_paths.extend(input_dir.rglob(ext))

        print(f"[Generate] Found {len(image_paths)} images in {input_dir}")

        if len(image_paths) == 0:
            return []

        # Calculate pairs per image
        pairs_per_image = max(1, target_count // len(image_paths))
        pairs_per_image = min(pairs_per_image, 50)  # Cap at 50 per image

        pairs = []
        processed = 0

        for img_path in image_paths:
            frame = cv2.imread(str(img_path))
            if frame is None:
                continue

            # Resize to consistent size
            h, w = frame.shape[:2]
            if w != 640 or h != 480:
                frame = cv2.resize(frame, (640, 480))

            frame_pairs = self.generate_from_image(frame, target_per_image=pairs_per_image)
            pairs.extend(frame_pairs)

            processed += 1
            if processed % 500 == 0:
                print(f"  Processed {processed}/{len(image_paths)} images, {len(pairs)} pairs")

            if len(pairs) >= target_count:
                break

        return pairs

    def save_pairs(self, pairs, output_path):
        """Save generated pairs to npz file."""
        if len(pairs) == 0:
            print("[Save] No pairs to save")
            return

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        inputs = np.stack([p["input"] for p in pairs])
        targets = np.stack([p["target"] for p in pairs])
        offsets = np.stack([p["offset"] for p in pairs])

        np.savez(str(output_path), inputs=inputs, targets=targets, offsets=offsets)

        size_mb = output_path.stat().st_size / (1024 * 1024)
        print(f"\n[Save] Saved {len(pairs)} pairs to {output_path}")
        print(f"  Input shape: {inputs.shape}")
        print(f"  Target shape: {targets.shape}")
        print(f"  File size: {size_mb:.1f} MB")

    def cleanup(self):
        self.detector.cleanup()


def main():
    parser = argparse.ArgumentParser(description="Large-scale training data generator")
    parser.add_argument("--input", type=str, help="Input directory with images")
    parser.add_argument("--output", type=str, default="data/large_pairs.npz",
                        help="Output file path")
    parser.add_argument("--webcam", action="store_true", help="Generate from webcam")
    parser.add_argument("--target-count", type=int, default=100000,
                        help="Target number of pairs to generate")
    parser.add_argument("--duration", type=int, default=300,
                        help="Webcam collection duration (seconds)")
    args = parser.parse_args()

    generator = LargeScaleGenerator()

    try:
        if args.webcam:
            pairs = generator.generate_from_webcam(
                target_count=args.target_count,
                duration_seconds=args.duration
            )
        elif args.input:
            pairs = generator.generate_from_directory(
                args.input,
                target_count=args.target_count
            )
        else:
            parser.print_help()
            return

        generator.save_pairs(pairs, args.output)

    finally:
        generator.cleanup()

    print("\n[Done] Training data ready!")
    print(f"Next: python train.py --train --data {args.output}")


if __name__ == "__main__":
    main()
