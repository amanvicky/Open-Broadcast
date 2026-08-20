"""
OpenBroadcast — Gaze Dataset Loader

Loads preprocessed eye crops and gaze labels from:
- MPIIFaceGaze (213K images, 15 subjects)
- Gaze360 (172K images, 238 subjects)

Handles train/val/test splits and augmentation.
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class GazeDataset(Dataset):
    """
    PyTorch Dataset for gaze estimation.

    Loads preprocessed eye crops (36×120 grayscale) and gaze labels.
    Returns:
        eye_crops: (1, 36, 120) float32 tensor, normalized [0, 1]
        gaze: (2,) float32 tensor, [pitch, yaw] in radians
    """

    def __init__(self, data_dir, split="train", transform=None):
        """
        Args:
            data_dir: path to preprocessed data directory
            split: 'train', 'val', or 'test'
            transform: augmentation callable (None for val/test)
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform

        split_file = self.data_dir / f"{split}.json"
        if not split_file.exists():
            raise FileNotFoundError(
                f"Split file not found: {split_file}\n"
                f"Run 'python -m data.setup_dataset' first."
            )

        with open(split_file) as f:
            self.samples = json.load(f)

        print(f"[Dataset] Loaded {split}: {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        # Load eye crop
        eyes_path = self.data_dir / "eyes" / f"{sample['id']}.npy"
        eyes = np.load(str(eyes_path))

        # Get gaze angles
        pitch = sample["gaze_pitch"]
        yaw = sample["gaze_yaw"]

        # Apply augmentation
        if self.transform:
            eyes, pitch, yaw = self.transform(eyes, pitch, yaw)

        # Normalize to [0, 1]
        eyes = eyes.astype(np.float32) / 255.0

        # Convert to tensors
        eyes_tensor = torch.from_numpy(eyes).unsqueeze(0)  # (1, 36, 120)
        gaze_tensor = torch.tensor([pitch, yaw], dtype=torch.float32)

        return eyes_tensor, gaze_tensor

    def get_subject_distribution(self):
        """Show sample count per subject."""
        dist = {}
        for s in self.samples:
            subj = s["subject"]
            dist[subj] = dist.get(subj, 0) + 1
        return dist


class CombinedDataset(Dataset):
    """
    Combine multiple gaze datasets for training.
    Handles different label formats and normalizes everything.
    """

    def __init__(self, datasets, transform=None):
        self.datasets = datasets
        self.transform = transform

        self.index_map = []
        for ds_idx, ds in enumerate(datasets):
            for sample_idx in range(len(ds)):
                self.index_map.append((ds_idx, sample_idx))

        print(f"[Combined] {len(self)} samples from {len(datasets)} sources")

    def __len__(self):
        return len(self.index_map)

    def __getitem__(self, idx):
        ds_idx, sample_idx = self.index_map[idx]
        return self.datasets[ds_idx][sample_idx]
