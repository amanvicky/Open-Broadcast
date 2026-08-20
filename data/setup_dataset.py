"""
OpenBroadcast — Dataset Setup Script

Downloads and preprocesses gaze estimation datasets.

Usage:
    python -m data.setup_dataset --output_dir data/processed
    python -m data.setup_dataset --skip_gaze360  # Skip large download
"""

import argparse
import os
import sys
import json
import zipfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np


def download_file(url, dest, description=""):
    """Download a file with progress display."""
    print(f"Downloading {description}...")
    print(f"  URL: {url}")
    print(f"  Destination: {dest}")

    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    try:
        urllib.request.urlretrieve(url, str(dest))
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  Downloaded: {size_mb:.1f} MB")
        return True
    except Exception as e:
        print(f"  Download failed: {e}")
        return False


class MPIIGazePreprocessor:
    """
    Preprocess MPIIFaceGaze dataset.

    Source: https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/
    Requires free registration for download.
    """

    def __init__(self, raw_dir, output_dir):
        self.raw_dir = Path(raw_dir)
        self.output_dir = Path(output_dir)
        (self.output_dir / "eyes").mkdir(parents=True, exist_ok=True)

    def check_prerequisites(self):
        """Check if dataset files exist."""
        data_dir = self.raw_dir / "data"
        annot_dir = self.raw_dir / "annotation"

        if not data_dir.exists() or not annot_dir.exists():
            print("\n" + "=" * 60)
            print("MPIIFaceGaze Dataset Not Found")
            print("=" * 60)
            print(f"Expected location: {self.raw_dir}")
            print("\nTo download:")
            print("1. Visit: https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/research/gaze-based-human-computer-interaction/the-mpiigaze-dataset-15000-images-under-real-world-lighting-conditions/")
            print("2. Register for free")
            print("3. Download the dataset zip file")
            print(f"4. Extract to: {self.raw_dir}")
            print(f"   (should contain 'data/' and 'annotation/' folders)")
            return False
        return True

    def process_all(self):
        """Process all subjects."""
        if not self.check_prerequisites():
            return 0

        subjects = sorted(os.listdir(self.raw_dir / "data"))
        all_samples = []

        for subject in subjects:
            annotation_file = self.raw_dir / "annotation" / f"{subject}.txt"
            if not annotation_file.exists():
                continue

            annotations = self._load_annotations(annotation_file)
            count = 0

            for idx, annot in enumerate(annotations):
                img_path = self.raw_dir / "data" / subject / annot["image_path"]
                if not img_path.exists():
                    continue

                face_img = cv2.imread(str(img_path))
                if face_img is None:
                    continue

                eyes = self._extract_eyes(face_img)
                if eyes is None:
                    continue

                sample_id = f"mpii_{subject}_{idx:05d}"
                np.save(self.output_dir / "eyes" / f"{sample_id}.npy", eyes)

                all_samples.append({
                    "id": sample_id,
                    "gaze_pitch": annot["gaze_y"],
                    "gaze_yaw": annot["gaze_x"],
                    "subject": subject,
                    "source": "mpiigaze",
                })
                count += 1

            print(f"  Subject {subject}: {count} samples")

        return all_samples

    def _load_annotations(self, filepath):
        annotations = []
        with open(filepath) as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 3:
                    annotations.append({
                        "image_path": parts[0],
                        "gaze_x": float(parts[1]),
                        "gaze_y": float(parts[2]),
                    })
        return annotations

    def _extract_eyes(self, face_img):
        """Extract eye crops using dlib or simple face detection."""
        try:
            import dlib

            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
            detector = dlib.get_frontal_face_detector()
            faces = detector(gray, 1)

            if not faces:
                return None

            predictor_path = Path("models/shape_predictor_68_face_landmarks.dat")
            if not predictor_path.exists():
                # Fallback: use simple Haar cascade
                return self._extract_eyes_simple(face_img)

            predictor = dlib.shape_predictor(str(predictor_path))
            landmarks = predictor(gray, faces[0])

            left_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(36, 42)])
            right_eye = np.array([(landmarks.part(i).x, landmarks.part(i).y) for i in range(42, 48)])

            left_crop = self._normalize_eye(face_img, left_eye)
            right_crop = self._normalize_eye(face_img, right_eye)

            if left_crop is None or right_crop is None:
                return None

            return np.concatenate([left_crop, right_crop], axis=1)

        except ImportError:
            return self._extract_eyes_simple(face_img)

    def _extract_eyes_simple(self, face_img):
        """Simple eye extraction without dlib (less accurate but always works)."""
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)

        if len(faces) == 0:
            return None

        x, y, w, h = faces[0]
        face = face_img[y:y+h, x:x+w]

        # Estimate eye positions (rough approximation)
        eye_h = h // 5
        eye_w = w // 3
        left_eye_y = int(y + h * 0.35)
        right_eye_y = int(y + h * 0.35)
        left_eye_x = int(x + w * 0.15)
        right_eye_x = int(x + w * 0.55)

        left_eye_pts = np.array([
            [left_eye_x, left_eye_y],
            [left_eye_x + eye_w//2, left_eye_y - eye_h//4],
            [left_eye_x + eye_w, left_eye_y],
            [left_eye_x + eye_w//2, left_eye_y + eye_h//4],
        ])
        right_eye_pts = np.array([
            [right_eye_x, right_eye_y],
            [right_eye_x + eye_w//2, right_eye_y - eye_h//4],
            [right_eye_x + eye_w, right_eye_y],
            [right_eye_x + eye_w//2, right_eye_y + eye_h//4],
        ])

        left_crop = self._normalize_eye(face_img, left_eye_pts)
        right_crop = self._normalize_eye(face_img, right_eye_pts)

        if left_crop is None or right_crop is None:
            return None

        return np.concatenate([left_crop, right_crop], axis=1)

    def _normalize_eye(self, img, eye_points, target_w=60, target_h=36):
        """Normalize eye crop: align, center, resize."""
        outer = eye_points[0]
        inner = eye_points[2] if len(eye_points) > 2 else eye_points[-1]

        eye_center = (outer + inner) / 2
        eye_width = np.linalg.norm(inner - outer)
        eye_angle = np.arctan2(inner[1] - outer[1], inner[0] - outer[0])

        M = cv2.getRotationMatrix2D(tuple(eye_center.astype(int)), np.degrees(eye_angle), 1.0)
        scale = target_w / (eye_width * 1.8)
        M[0, 0] *= scale
        M[1, 1] *= scale
        M[0, 2] += target_w / 2 - eye_center[0] * scale
        M[1, 2] += target_h / 2 - eye_center[1] * scale

        eye_crop = cv2.warpAffine(
            img, M, (target_w, target_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )

        return cv2.cvtColor(eye_crop, cv2.COLOR_BGR2GRAY)


def create_splits(samples, output_dir, seed=42):
    """Create train/val/test splits using leave-one-subject-out."""
    subjects = list(set(s["subject"] for s in samples))
    rng = np.random.RandomState(seed)
    rng.shuffle(subjects)

    n = len(subjects)
    n_train = max(1, int(n * 0.75))
    n_val = max(1, int(n * 0.1))
    n_test = max(1, n - n_train - n_val)

    train_subjects = set(subjects[:n_train])
    val_subjects = set(subjects[n_train:n_train+n_val])
    test_subjects = set(subjects[n_train+n_val:])

    splits = {
        "train": [s for s in samples if s["subject"] in train_subjects],
        "val": [s for s in samples if s["subject"] in val_subjects],
        "test": [s for s in samples if s["subject"] in test_subjects],
    }

    for split_name, split_samples in splits.items():
        path = output_dir / f"{split_name}.json"
        with open(path, "w") as f:
            json.dump(split_samples, f, indent=2)
        print(f"  {split_name}: {len(split_samples)} samples "
              f"({len(set(s['subject'] for s in split_samples))} subjects)")

    return splits


def main():
    parser = argparse.ArgumentParser(description="Setup gaze estimation dataset")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument("--mpiigaze_dir", type=str, default="data/raw/mpiigaze")
    parser.add_argument("--skip_mpiigaze", action="store_true")
    parser.add_argument("--skip_gaze360", action="store_true", default=True)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "eyes").mkdir(exist_ok=True)

    all_samples = []

    # Process MPIIGaze
    if not args.skip_mpiigaze:
        print("\n" + "=" * 60)
        print("Processing MPIIFaceGaze")
        print("=" * 60)
        mpii = MPIIGazePreprocessor(args.mpiigaze_dir, output)
        samples = mpii.process_all()
        if samples:
            all_samples.extend(samples)
            print(f"  Total MPIIGaze samples: {len(samples)}")

    if not all_samples:
        print("\nNo dataset samples found.")
        print("Please download MPIIFaceGaze and run again.")
        print("See: https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/")
        return

    # Create splits
    print("\nCreating train/val/test splits...")
    create_splits(all_samples, output)

    print(f"\nDataset setup complete! {len(all_samples)} total samples.")
    print(f"Next step: python -m models.train --data_dir {args.output_dir}")


if __name__ == "__main__":
    main()
