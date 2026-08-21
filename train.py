"""Training script for GazeCorrectionNet.

Usage:
    # Generate training data from webcam (run for ~2 minutes)
    python train.py --collect --output data/training_pairs.npy

    # Or generate from a video file
    python train.py --collect --video path/to/video.mp4 --output data/training_pairs.npy

    # Train the model
    python train.py --train --data data/training_pairs.npy --epochs 50

    # Or collect + train in one step
    python train.py --collect --train --epochs 50

The model saves to models/gaze_correction.pth (~1.2MB).
"""

import sys
import os
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def collect_data(args):
    """Generate training pairs from webcam or video."""
    import cv2
    from core.face_detector import FaceDetector
    from core.eye_corrector import EyeCorrector
    from core.gaze_model import GazeCorrectionDataset

    print("[Collect] Initializing detector and corrector...")
    detector = FaceDetector(detection_interval=1)
    corrector = EyeCorrector(strength=0.85, amplification=3.0)
    dataset = GazeCorrectionDataset(detector, corrector)

    if args.video:
        print(f"[Collect] Processing video: {args.video}")
        pairs = dataset.generate_from_video(args.video, max_frames=args.max_frames)
    else:
        # Guided collection: prompt user to look in specific directions
        directions = [
            ("Look straight at camera", 3),
            ("Look LEFT (15-20 degrees)", 3),
            ("Look RIGHT (15-20 degrees)", 3),
            ("Look UP (10-15 degrees)", 3),
            ("Look DOWN (10-15 degrees)", 3),
            ("Look LEFT and UP", 2),
            ("Look RIGHT and DOWN", 2),
            ("Move your head around naturally", 5),
        ]

        if args.guided:
            print("[Collect] Guided collection mode")
            print("[Collect] Follow the prompts. Keep your face in frame.")
        else:
            print("[Collect] Free capture mode (use --guided for prompts)")

        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        pairs = []
        frame_count = 0

        if args.guided:
            import time
            for direction, duration in directions:
                print(f"\n  >>> {direction} (for {duration} seconds)")
                print("      3... 2... 1... GO!")
                time.sleep(1)

                end_time = time.time() + duration
                dir_pairs = 0
                while time.time() < end_time:
                    ret, frame = cap.read()
                    if not ret:
                        continue

                    landmarks = detector.detect(frame)
                    if landmarks is None:
                        continue

                    eye_data = detector.get_eye_data(landmarks, frame.shape)
                    frame_pairs = dataset.generate_pair(frame, eye_data)

                    for input_patch, target_patch, offset in frame_pairs:
                        aug_input, aug_target, aug_offset = dataset.augment(
                            input_patch, target_patch, offset
                        )
                        pairs.append({
                            "input": aug_input,
                            "target": aug_target,
                            "offset": aug_offset,
                        })
                        dir_pairs += 1

                print(f"      Collected {dir_pairs} pairs")
        else:
            while frame_count < args.max_frames:
                ret, frame = cap.read()
                if not ret:
                    continue

                landmarks = detector.detect(frame)
                if landmarks is None:
                    continue

                eye_data = detector.get_eye_data(landmarks, frame.shape)
                frame_pairs = dataset.generate_pair(frame, eye_data)

                for input_patch, target_patch, offset in frame_pairs:
                    aug_input, aug_target, aug_offset = dataset.augment(
                        input_patch, target_patch, offset
                    )
                    pairs.append({
                        "input": aug_input,
                        "target": aug_target,
                        "offset": aug_offset,
                    })

                frame_count += 1
                if frame_count % 100 == 0:
                    print(f"  Processed {frame_count} frames, {len(pairs)} pairs collected")

        cap.release()

    detector.cleanup()

    if len(pairs) == 0:
        print("[Collect] No pairs collected. Check camera/video.")
        return

    # Save as numpy archive
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    inputs = np.stack([p["input"] for p in pairs])
    targets = np.stack([p["target"] for p in pairs])
    offsets = np.stack([p["offset"] for p in pairs])

    np.savez(args.output, inputs=inputs, targets=targets, offsets=offsets)
    print(f"[Collect] Saved {len(pairs)} pairs to {args.output}")
    print(f"  Input shape: {inputs.shape}, Target shape: {targets.shape}")


def train_model(args):
    """Train GazeCorrectionNet on collected data."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from core.gaze_model import GazeCorrectionNet

    # Load data
    print(f"[Train] Loading data from {args.data}...")
    data = np.load(args.output if hasattr(args, 'output') and args.output else args.data)
    inputs = data["inputs"]
    targets = data["targets"]
    offsets = data["offsets"]

    print(f"[Train] Loaded {len(inputs)} pairs")
    print(f"  Inputs: {inputs.shape}, Targets: {targets.shape}")

    # Normalize inputs to [0, 1]
    inputs = inputs.astype(np.float32) / 255.0
    targets = targets.astype(np.float32) / 255.0

    # Transpose to CHW: (N, H, W, C) -> (N, C, H, W)
    inputs = np.transpose(inputs, (0, 3, 1, 2))
    targets = np.transpose(targets, (0, 3, 1, 2))

    # Split train/val (80/20)
    n = len(inputs)
    split = int(n * 0.8)
    indices = np.random.permutation(n)
    train_idx = indices[:split]
    val_idx = indices[split:]

    train_dataset = TensorDataset(
        torch.tensor(inputs[train_idx]),
        torch.tensor(targets[train_idx]),
        torch.tensor(offsets[train_idx]),
    )
    val_dataset = TensorDataset(
        torch.tensor(inputs[val_idx]),
        torch.tensor(targets[val_idx]),
        torch.tensor(offsets[val_idx]),
    )

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Train] Using device: {device}")

    model = GazeCorrectionNet(in_channels=3, offset_dim=2).to(device)
    print(f"[Train] Model parameters: {model.count_parameters():,}")

    # Loss: L1 + perceptual (SSIM-like via gradient matching)
    l1_loss = nn.L1Loss()

    def combined_loss(pred, target):
        # L1 pixel loss
        l1 = l1_loss(pred, target)

        # Gradient loss (preserves edges)
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
        grad_loss = l1_loss(pred_dx, target_dx) + l1_loss(pred_dy, target_dy)

        return l1 + 0.5 * grad_loss

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    # Training loop
    best_val_loss = float("inf")
    os.makedirs("models", exist_ok=True)

    print(f"[Train] Starting training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        # Train
        model.train()
        train_loss = 0.0
        for batch_input, batch_target, batch_offset in train_loader:
            batch_input = batch_input.to(device)
            batch_target = batch_target.to(device)
            batch_offset = batch_offset.to(device)

            pred = model(batch_input, batch_offset)
            loss = combined_loss(pred, batch_target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(batch_input)

        train_loss /= len(train_dataset)

        # Validate
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_input, batch_target, batch_offset in val_loader:
                batch_input = batch_input.to(device)
                batch_target = batch_target.to(device)
                batch_offset = batch_offset.to(device)

                pred = model(batch_input, batch_offset)
                loss = combined_loss(pred, batch_target)
                val_loss += loss.item() * len(batch_input)

        val_loss /= len(val_dataset)
        scheduler.step(val_loss)

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_path = os.path.join("models", "gaze_correction.pth")
            torch.save({
                "model_state_dict": model.state_dict(),
                "input_shape": (3, 64, 64),
                "offset_dim": 2,
                "val_loss": val_loss,
            }, save_path)

        if (epoch + 1) % 5 == 0 or epoch == 0:
            lr = optimizer.param_groups[0]["lr"]
            print(f"  Epoch {epoch+1}/{args.epochs} | "
                  f"Train: {train_loss:.6f} | Val: {val_loss:.6f} | "
                  f"LR: {lr:.2e} | Best: {best_val_loss:.6f}")

    print(f"\n[Train] Done! Best val loss: {best_val_loss:.6f}")
    print(f"[Train] Model saved to models/gaze_correction.pth")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GazeCorrectionNet")
    parser.add_argument("--collect", action="store_true", help="Collect training data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--guided", action="store_true", help="Guided collection with direction prompts")
    parser.add_argument("--video", type=str, help="Video file for data collection")
    parser.add_argument("--output", type=str, default="data/training_pairs.npz",
                        help="Output path for collected data")
    parser.add_argument("--data", type=str, default="data/training_pairs.npz",
                        help="Path to training data for --train")
    parser.add_argument("--max-frames", type=int, default=1000,
                        help="Max frames to collect")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    args = parser.parse_args()

    if not args.collect and not args.train:
        parser.print_help()
        sys.exit(1)

    if args.collect:
        collect_data(args)

    if args.train:
        train_model(args)
