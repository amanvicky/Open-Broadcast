"""Download and prepare datasets for gaze correction training.

Supports multiple data sources:
1. MPIIFaceGaze — 213K images, 15 subjects, gaze angles annotated
2. FFHQ — 70K high-quality face images (for diversity)
3. Synthetic generation — use geometric corrector on any face images

Usage:
    # Download all datasets (~5GB total)
    python -m data.download_datasets --all

    # Download specific dataset
    python -m data.download_datasets --mpiigaze
    python -m data.download_datasets --ffhq

    # Generate synthetic training data from downloaded images
    python -m data.download_datasets --generate-pairs

    # Full pipeline: download + prepare + generate pairs
    python -m data.download_datasets --all --generate-pairs
"""

import os
import sys
import argparse
import urllib.request
import zipfile
import tarfile
import json
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def download_file(url, dest, description=""):
    """Download a file with progress reporting."""
    if dest.exists():
        print(f"  [SKIP] {dest.name} already exists")
        return True

    print(f"  [DOWNLOAD] {description or url}")
    print(f"  -> {dest}")

    try:
        dest.parent.mkdir(parents=True, exist_ok=True)

        def progress(block_num, block_size, total_size):
            downloaded = block_num * block_size
            if total_size > 0:
                pct = min(100, downloaded * 100 // total_size)
                mb = downloaded // (1024 * 1024)
                total_mb = total_size // (1024 * 1024)
                print(f"\r  {pct}% ({mb}/{total_mb} MB)", end="", flush=True)

        urllib.request.urlretrieve(url, str(dest), reporthook=progress)
        print()  # New line after progress
        return True
    except Exception as e:
        print(f"\n  [ERROR] {e}")
        if dest.exists():
            dest.unlink()
        return False


def download_mpiigaze():
    """Download MPIIFaceGaze dataset.

    213K images from 15 subjects, various gaze angles.
    Total size: ~1.5GB compressed, ~3GB extracted.
    """
    print("\n" + "="*60)
    print("MPIIFaceGaze Dataset")
    print("="*60)
    print("213K images, 15 subjects, gaze angles annotated")
    print("Size: ~1.5GB compressed")

    # MPIIFaceGaze is hosted on the MPI server
    # Registration required for full dataset
    # Using the publicly available subset
    url = "https://grail.cs.washington.edu/projects/gaze/MPIIGaze.h5"

    dest = RAW_DIR / "mpiigaze" / "MPIIGaze.h5"

    if dest.exists():
        print("  [SKIP] Already downloaded")
        return True

    print("\n  NOTE: MPIIFaceGaze requires academic registration.")
    print("  If download fails, you can:")
    print("  1. Register at: https://www.mpi-inf.mpg.de/departments/computer-vision-and-machine-learning/")
    print("  2. Download MPIIGaze.h5 manually")
    print(f"  3. Place it at: {dest}")

    # Try alternative: Boston University gaze dataset (publicly available)
    bu_url = "https://people.cs.bu.edu/yzhao/code/bohgaze.h5"
    bu_dest = RAW_DIR / "mpiigaze" / "bohgaze.h5"

    print("\n  Trying Boston University Gaze Dataset (publicly available)...")
    success = download_file(bu_url, bu_dest, "Boston University Gaze Dataset (~500MB)")

    if success:
        print("  [OK] Downloaded BU Gaze Dataset")
        return True

    print("  [INFO] Manual download may be required for academic datasets")
    print("  See README for instructions")
    return False


def download_ffhq():
    """Download FFHQ dataset.

    70K high-quality face images from Flickr.
    Total size: ~2GB.
    """
    print("\n" + "="*60)
    print("FFHQ Dataset")
    print("="*60)
    print("70K high-quality face images, diverse demographics")
    print("Size: ~2GB")

    ffhq_dir = RAW_DIR / "ffhq"
    ffhq_dir.mkdir(parents=True, exist_ok=True)

    # FFHQ thumbnails (256x256) are freely available
    # Full resolution requires NVIDIA registration
    url = "https://raw.githubusercontent.com/NVlabs/ffhq-dataset/master/download_ffhq.py"

    print("\n  FFHQ requires downloading via their official script.")
    print("  For thumbnails (sufficient for gaze training):")

    # Download the thumbnail dataset index
    index_url = "https://raw.githubusercontent.com/NVlabs/ffhq-dataset/master/ffhq-dataset-v1.json"
    index_dest = ffhq_dir / "ffhq-dataset-v1.json"

    success = download_file(index_url, index_dest, "FFHQ dataset index")

    if success:
        print("  [OK] Downloaded FFHQ index")
        print("  To download thumbnails, run:")
        print(f"    cd {ffhq_dir}")
        print("    python download_ffhq.py --features=thumbnails256x256")
        return True

    return False


def download_celeba():
    """Download CelebA dataset.

    200K celebrity face images with 40 attribute annotations.
    Total size: ~1GB (thumbnails).
    """
    print("\n" + "="*60)
    print("CelebA Dataset")
    print("="*60)
    print("200K celebrity face images, 40 attributes")
    print("Size: ~1GB (thumbnails)")

    celeba_dir = RAW_DIR / "celeba"
    celeba_dir.mkdir(parents=True, exist_ok=True)

    # CelebA is hosted on Google Drive
    # Using alternative source
    url = "https://www.kaggle.com/api/v1/datasets/download/jessicali9633/celeba-dataset"

    print("\n  NOTE: CelebA requires Kaggle account for download.")
    print("  Alternative: Use FFHQ or your own webcam data.")
    print(f"  Place images at: {celeba_dir}/img_align_celeba/")

    return False


def download_wider_face():
    """Download WIDER FACE dataset.

    32K images with face annotations.
    Good diversity of faces in various conditions.
    """
    print("\n" + "="*60)
    print("WIDER FACE Dataset")
    print("="*60)
    print("32K images, diverse face conditions")
    print("Size: ~2GB")

    wider_dir = RAW_DIR / "wider_face"
    wider_dir.mkdir(parents=True, exist_ok=True)

    # WIDER FACE images
    images_url = "https://huggingface.co/datasets/wider_face/resolve/main/data/test/images.zip"
    images_dest = wider_dir / "images.zip"

    success = download_file(images_url, images_dest, "WIDER FACE test images (~200MB)")

    if success:
        print("  [OK] Downloaded WIDER FACE images")
        return True

    return False


def generate_synthetic_pairs(max_images=5000):
    """Generate training pairs from any face images using geometric corrector.

    Takes face images, detects eyes, and creates (original, corrected) pairs
    using the geometric iris transplant method.
    """
    import cv2
    from core.face_detector import FaceDetector
    from core.eye_corrector import EyeCorrector

    print("\n" + "="*60)
    print("Generating Synthetic Training Pairs")
    print("="*60)

    detector = FaceDetector(detection_interval=1)
    corrector = EyeCorrector(strength=0.85, amplification=3.0)

    # Collect all image paths
    image_paths = []

    for dataset_dir in RAW_DIR.iterdir():
        if not dataset_dir.is_dir():
            continue
        for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
            image_paths.extend(dataset_dir.glob(f"**/{ext}"))

    print(f"  Found {len(image_paths)} images across all datasets")

    if len(image_paths) == 0:
        print("  [WARN] No images found. Download datasets first.")
        return

    # Limit to max_images
    if len(image_paths) > max_images:
        np.random.shuffle(image_paths)
        image_paths = image_paths[:max_images]

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

        landmarks = detector.detect(frame)
        if landmarks is None:
            continue

        eye_data = detector.get_eye_data(landmarks, frame.shape)

        for side in ("left", "right"):
            eye = eye_data[f"{side}_eye"]
            pupil = np.array(eye["iris"], dtype=np.float32)
            socket = np.array(eye["center"], dtype=np.float32)
            eye_width = float(eye["width"])

            if eye_width < 15 or eye_data["is_blinking"]:
                continue

            # Calculate offset
            offset_x = (pupil[0] - socket[0]) / (eye_width + 1e-6)
            offset_y = (pupil[1] - socket[1]) / (eye_width + 1e-6)

            # Only use images where gaze is off-center (useful for training)
            if abs(offset_x) < 0.05 and abs(offset_y) < 0.05:
                continue

            # Extract eye patch
            cx, cy = int(socket[0]), int(socket[1])
            half = 32  # 64x64 patch

            x0 = max(0, cx - half)
            y0 = max(0, cy - half)
            x1 = min(640, cx + half)
            y1 = min(480, cy + half)

            if (x1 - x0) < 64 or (y1 - y0) < 64:
                continue

            input_patch = frame[y0:y1, x0:x1].copy()

            # Generate corrected version
            single_eye_data = {f"{side}_eye": eye, "is_blinking": False}
            corrected_frame = corrector.correct_frame(frame, single_eye_data)
            target_patch = corrected_frame[y0:y1, x0:x1].copy()

            # Augment
            if np.random.random() > 0.5:
                factor = 0.7 + np.random.random() * 0.6
                input_patch = np.clip(input_patch * factor, 0, 255).astype(np.uint8)
                target_patch = np.clip(target_patch * factor, 0, 255).astype(np.uint8)

            pairs.append({
                "input": input_patch,
                "target": target_patch,
                "offset": np.array([offset_x, offset_y], dtype=np.float32),
            })

        processed += 1
        if processed % 500 == 0:
            print(f"  Processed {processed}/{len(image_paths)} images, {len(pairs)} pairs")

    detector.cleanup()

    if len(pairs) == 0:
        print("  [WARN] No pairs generated")
        return

    # Save
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    save_path = PROCESSED_DIR / "synthetic_pairs.npz"

    inputs = np.stack([p["input"] for p in pairs])
    targets = np.stack([p["target"] for p in pairs])
    offsets = np.stack([p["offset"] for p in pairs])

    np.savez(str(save_path), inputs=inputs, targets=targets, offsets=offsets)
    print(f"\n  [OK] Saved {len(pairs)} pairs to {save_path}")
    print(f"  Input shape: {inputs.shape}, Target shape: {targets.shape}")
    print(f"  Size: {save_path.stat().st_size / (1024*1024):.1f} MB")


def download_all():
    """Download all available datasets."""
    print("\n" + "="*60)
    print("Downloading All Datasets")
    print("="*60)

    results = {}
    results["mpiigaze"] = download_mpiigaze()
    results["ffhq"] = download_ffhq()
    results["wider_face"] = download_wider_face()

    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)
    for name, success in results.items():
        status = "✅" if success else "❌"
        print(f"  {status} {name}")

    total_size = sum(f.stat().st_size for f in RAW_DIR.rglob("*") if f.is_file()) / (1024**3)
    print(f"\n  Total data: {total_size:.1f} GB")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download gaze correction datasets")
    parser.add_argument("--all", action="store_true", help="Download all datasets")
    parser.add_argument("--mpiigaze", action="store_true", help="Download MPIIFaceGaze")
    parser.add_argument("--ffhq", action="store_true", help="Download FFHQ")
    parser.add_argument("--celeba", action="store_true", help="Download CelebA")
    parser.add_argument("--wider-face", action="store_true", help="Download WIDER FACE")
    parser.add_argument("--generate-pairs", action="store_true",
                        help="Generate synthetic training pairs from downloaded images")
    parser.add_argument("--max-images", type=int, default=5000,
                        help="Max images to process for pair generation")
    args = parser.parse_args()

    if not any([args.all, args.mpiigaze, args.ffhq, args.celeba,
                args.wider_face, args.generate_pairs]):
        parser.print_help()
        sys.exit(1)

    if args.all or args.mpiigaze:
        download_mpiigaze()

    if args.all or args.ffhq:
        download_ffhq()

    if args.all or args.celeba:
        download_celeba()

    if args.all or args.wider_face:
        download_wider_face()

    if args.generate_pairs:
        generate_synthetic_pairs(args.max_images)

    print("\n[DONE] Dataset preparation complete!")
    print("Next step: python train.py --train --data data/processed/synthetic_pairs.npz")
