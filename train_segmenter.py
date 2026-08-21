"""Train iris segmentation model using webcam data.

Uses MediaPipe iris landmarks as pseudo-labels to train a tiny
segmentation model that detects exact iris boundaries.

Usage:
    python train_segmenter.py --collect     # Collect training data (2 min)
    python train_segmenter.py --train       # Train the model (~5 min)
    python train_segmenter.py --all         # Collect + train
"""

import argparse
import os
import sys
import time
import numpy as np
import cv2

sys.path.insert(0, os.path.dirname(__file__))


def collect_data(duration_seconds=120, output_path="data/iris_segments.npz"):
    """Collect iris segmentation training data from webcam.
    
    Uses MediaPipe iris landmarks to create pseudo-labels:
    - Iris center + radius → circular mask
    - Iris landmarks 469-472/474-477 → refined mask
    """
    from core.face_detector import FaceDetector
    
    print("=== Iris Segmentation Data Collection ===")
    print(f"Collecting for {duration_seconds} seconds...")
    print("Look around at different angles and distances.")
    print("Press 'q' to stop early.\n")
    
    detector = FaceDetector(detection_interval=1)
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    
    if not cap.isOpened():
        print("ERROR: Cannot open camera")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    patches = []
    masks = []
    start_time = time.time()
    frame_count = 0
    
    while time.time() - start_time < duration_seconds:
        ret, frame = cap.read()
        if not ret:
            continue
        
        landmarks = detector.detect(frame)
        if landmarks is None:
            continue
        
        eye_data = detector.get_eye_data(landmarks, frame.shape)
        h, w = frame.shape[:2]
        
        for side in ("left", "right"):
            eye = eye_data[f"{side}_eye"]
            socket = np.array(eye["center"], dtype=np.float32)
            pupil = np.array(eye["iris"], dtype=np.float32)
            eye_width = float(eye["width"])
            
            if eye_width < 20 or eye_data["is_blinking"]:
                continue
            
            # Extract patch centered on eye
            cx, cy = int(socket[0]), int(socket[1])
            half = 32
            x0 = max(0, cx - half)
            y0 = max(0, cy - half)
            x1 = min(w, cx + half)
            y1 = min(h, cy + half)
            
            if (x1 - x0) < 64 or (y1 - y0) < 64:
                continue
            
            patch = frame[y0:y1, x0:x1].copy()
            
            # Create pseudo-label mask from iris landmarks
            mask = np.zeros((64, 64), dtype=np.float32)
            
            # Iris center in patch coordinates
            local_cx = int(pupil[0]) - x0
            local_cy = int(pupil[1]) - y0
            
            # Iris radius from landmarks (average of iris boundary points)
            if side == "left":
                iris_indices = [469, 470, 471, 472]
            else:
                iris_indices = [474, 475, 476, 477]
            
            iris_points = []
            for idx in iris_indices:
                lm = landmarks[idx]
                px = lm.x * w - x0
                py = lm.y * h - y0
                iris_points.append((px, py))
            
            if iris_points:
                # Compute radius from center to boundary points
                distances = [np.sqrt((px - local_cx)**2 + (py - local_cy)**2) 
                            for px, py in iris_points]
                iris_r = int(np.mean(distances))
                
                # Draw filled circle as mask
                cv2.circle(mask, (local_cx, local_cy), max(3, iris_r), 1.0, -1)
                
                # Also use the actual boundary points for better shape
                pts = np.array(iris_points, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 1.0)
                
                # Smooth edges
                mask = cv2.GaussianBlur(mask, (5, 5), 1.5)
                mask = (mask > 0.3).astype(np.float32)
            
            patches.append(patch)
            masks.append(mask)
            frame_count += 1
        
        # Show preview
        elapsed = time.time() - start_time
        remaining = duration_seconds - elapsed
        
        info_frame = frame.copy()
        cv2.putText(info_frame, f"Collected: {frame_count} pairs", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(info_frame, f"Time: {remaining:.0f}s remaining", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Collecting iris data...", info_frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    detector.cleanup()
    
    if len(patches) < 100:
        print(f"WARNING: Only {len(patches)} samples collected. Need 100+.")
        return
    
    # Save
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path,
                        patches=np.array(patches),
                        masks=np.array(masks))
    print(f"\nSaved {len(patches)} samples to {output_path}")
    print(f"Patches shape: {np.array(patches).shape}")
    print(f"Masks shape: {np.array(masks).shape}")


def train_model(data_path="data/iris_segments.npz", 
                output_path="models/iris_segmenter.pth",
                epochs=50, lr=1e-3):
    """Train iris segmentation model on collected data."""
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    from core.iris_segmenter import IrisSegmentationNet
    
    print("=== Iris Segmentation Training ===")
    
    # Load data
    if not os.path.exists(data_path):
        print(f"ERROR: No data found at {data_path}")
        print("Run: python train_segmenter.py --collect")
        return
    
    data = np.load(data_path)
    patches = data["patches"].astype(np.float32) / 255.0
    masks = data["masks"].astype(np.float32)
    
    print(f"Loaded {len(patches)} samples")
    print(f"Patches: {patches.shape}, Masks: {masks.shape}")
    
    # Split train/val
    n = len(patches)
    n_train = int(0.8 * n)
    
    # Convert to tensors: (N, C, H, W)
    patches_t = torch.from_numpy(patches).permute(0, 3, 1, 2)
    masks_t = torch.from_numpy(masks).unsqueeze(1)  # (N, 1, H, W)
    
    train_dataset = TensorDataset(patches_t[:n_train], masks_t[:n_train])
    val_dataset = TensorDataset(patches_t[n_train:], masks_t[n_train:])
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32)
    
    # Model
    model = IrisSegmentationNet(in_channels=3)
    print(f"Model parameters: {model.count_parameters():,}")
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    # Loss: BCE + Dice
    def bce_dice_loss(pred, target):
        bce = nn.functional.binary_cross_entropy_with_logits(pred, target)
        pred_mask = torch.sigmoid(pred)
        intersection = (pred_mask * target).sum()
        union = pred_mask.sum() + target.sum()
        dice = 1 - (2 * intersection + 1) / (union + 1)
        return bce + dice
    
    # Training loop
    best_val_loss = float("inf")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    for epoch in range(epochs):
        # Train
        model.train()
        train_loss = 0
        for batch_patches, batch_masks in train_loader:
            pred = model(batch_patches)
            loss = bce_dice_loss(pred, batch_masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        train_loss /= len(train_loader)
        
        # Validate
        model.eval()
        val_loss = 0
        correct_pixels = 0
        total_pixels = 0
        
        with torch.no_grad():
            for batch_patches, batch_masks in val_loader:
                pred = model(batch_patches)
                loss = bce_dice_loss(pred, batch_masks)
                val_loss += loss.item()
                
                # Accuracy
                pred_mask = (torch.sigmoid(pred) > 0.5).float()
                correct_pixels += (pred_mask == batch_masks).sum().item()
                total_pixels += batch_masks.numel()
        
        val_loss /= len(val_loader)
        accuracy = correct_pixels / total_pixels * 100
        
        scheduler.step()
        
        # Print progress
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d}/{epochs} | "
                  f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                  f"Acc: {accuracy:.1f}%")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "model_state_dict": model.state_dict(),
                "val_loss": val_loss,
                "accuracy": accuracy,
                "input_shape": (3, 64, 64),
            }, output_path)
    
    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Model saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Train iris segmentation model")
    parser.add_argument("--collect", action="store_true", help="Collect training data")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--all", action="store_true", help="Collect + train")
    parser.add_argument("--duration", type=int, default=120, help="Collection duration (seconds)")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs")
    args = parser.parse_args()
    
    if args.all or (args.collect and args.train):
        collect_data(duration_seconds=args.duration)
        train_model(epochs=args.epochs)
    elif args.collect:
        collect_data(duration_seconds=args.duration)
    elif args.train:
        train_model(epochs=args.epochs)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
