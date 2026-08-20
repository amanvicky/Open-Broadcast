"""
OpenBroadcast — Training Pipeline

Trains GazeNet-Lite on MPIIGaze + Gaze360 datasets.
Uses angular loss (great-circle distance) for proper direction learning.

Usage:
    python -m models.train --data_dir data/processed --epochs 100
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import json
from pathlib import Path

from models.gaze_net import GazeNetLite
from data.dataset import GazeDataset
from data.augmentations import GazeAugmentation


class AngularLoss(nn.Module):
    """
    Great-circle distance between predicted and true gaze vectors.

    Why not MSE? Gaze directions live on a unit sphere.
    Angular loss measures the actual angle between vectors,
    which is the correct metric for direction prediction.
    """

    def forward(self, pred, target):
        pred_vec = self._to_unit_vector(pred)
        target_vec = self._to_unit_vector(target)

        cos_sim = torch.sum(pred_vec * target_vec, dim=1)
        cos_sim = torch.clamp(cos_sim, -0.9999, 0.9999)
        return torch.acos(cos_sim).mean()

    @staticmethod
    def _to_unit_vector(angles):
        pitch, yaw = angles[:, 0], angles[:, 1]
        x = torch.cos(pitch) * torch.sin(yaw)
        y = torch.sin(pitch)
        z = torch.cos(pitch) * torch.cos(yaw)
        return torch.stack([x, y, z], dim=1)


def evaluate(model, loader, device="cpu"):
    """Evaluate model on validation/test set."""
    model.eval()
    criterion = AngularLoss()

    total_loss = 0
    total_samples = 0
    pitch_errors = []
    yaw_errors = []

    with torch.no_grad():
        for eyes, gaze in loader:
            eyes = eyes.to(device)
            gaze = gaze.to(device)

            pred = model(eyes)
            loss = criterion(pred, gaze)

            total_loss += loss.item() * eyes.size(0)
            total_samples += eyes.size(0)

            pitch_errors.extend(torch.abs(pred[:, 0] - gaze[:, 0]).cpu().numpy())
            yaw_errors.extend(torch.abs(pred[:, 1] - gaze[:, 1]).cpu().numpy())

    avg_loss = total_loss / max(total_samples, 1)
    avg_pitch_err = np.degrees(np.mean(pitch_errors)) if pitch_errors else 0
    avg_yaw_err = np.degrees(np.mean(yaw_errors)) if yaw_errors else 0

    return {
        "angular_error": np.degrees(avg_loss),
        "pitch_error": avg_pitch_err,
        "yaw_error": avg_yaw_err,
    }


def train(config):
    """Main training loop."""

    device = "cpu"

    # Load datasets
    train_aug = GazeAugmentation(p=0.5)
    train_dataset = GazeDataset(config["data_dir"], "train", transform=train_aug)
    val_dataset = GazeDataset(config["data_dir"], "val", transform=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=min(2, config.get("workers", 2)),
        pin_memory=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
    )

    # Initialize model
    model = GazeNetLite().to(device)
    param_count = model.count_parameters()
    print(f"Model parameters: {param_count:,}")
    print(f"Model size: {model.model_size_mb():.2f} MB")

    # Optimizer + scheduler
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config["lr"],
        weight_decay=config["weight_decay"],
    )
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2
    )
    criterion = AngularLoss()

    # Training
    best_val_error = float("inf")
    history = []
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(config["epochs"]):
        model.train()
        epoch_loss = 0
        epoch_samples = 0
        t_start = time.time()

        for eyes, gaze in train_loader:
            eyes = eyes.to(device)
            gaze = gaze.to(device)

            pred = model(eyes)
            loss = criterion(pred, gaze)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item() * eyes.size(0)
            epoch_samples += eyes.size(0)

        scheduler.step()

        # Validate
        val_metrics = evaluate(model, val_loader, device)
        epoch_time = time.time() - t_start

        print(
            f"Epoch {epoch+1}/{config['epochs']} "
            f"({epoch_time:.1f}s) | "
            f"Train: {epoch_loss/max(epoch_samples,1):.4f} | "
            f"Val Angular: {val_metrics['angular_error']:.2f}° | "
            f"Pitch: {val_metrics['pitch_error']:.2f}° | "
            f"Yaw: {val_metrics['yaw_error']:.2f}°"
        )

        history.append({
            "epoch": epoch + 1,
            "train_loss": epoch_loss / max(epoch_samples, 1),
            "val_angular_error": val_metrics["angular_error"],
            "val_pitch_error": val_metrics["pitch_error"],
            "val_yaw_error": val_metrics["yaw_error"],
        })

        # Save best model
        if val_metrics["angular_error"] < best_val_error:
            best_val_error = val_metrics["angular_error"]
            torch.save(model.state_dict(), output_dir / "gaze_net_best.pth")
            print(f"  → New best! ({best_val_error:.2f}°)")

    # Save training history
    with open(output_dir / "training_history.json", "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nTraining complete. Best angular error: {best_val_error:.2f}°")
    return model


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train GazeNet-Lite")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="models/weights")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()

    config = {
        "data_dir": args.data_dir,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "workers": args.workers,
    }

    model = train(config)
