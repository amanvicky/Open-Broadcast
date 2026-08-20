"""
OpenBroadcast — Data Augmentation Pipeline

Augmentations designed to improve gaze model generalization:
1. Horizontal flip — doubles data, teaches left/right invariance
2. Brightness — simulates different lighting conditions
3. Contrast — simulates different camera qualities
4. Gaussian noise — simulates low-quality webcams
5. Rotation — simulates head tilt
6. Random crop jitter — simulates imperfect face detection
7. Motion blur — simulates fast movement
"""

import cv2
import numpy as np
import random


class GazeAugmentation:
    """
    Data augmentation for gaze estimation training.

    Each augmentation is applied independently with probability p.
    Horizontal flip also mirrors the gaze direction.
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, eyes, gaze_pitch, gaze_yaw):
        """
        Args:
            eyes: (36, 120) grayscale eye crop
            gaze_pitch: float (radians)
            gaze_yaw: float (radians)

        Returns:
            augmented_eyes, augmented_pitch, augmented_yaw
        """
        # 1. Horizontal flip
        if random.random() < self.p:
            eyes = np.flip(eyes, axis=1).copy()
            gaze_yaw = -gaze_yaw

        # 2. Brightness adjustment
        if random.random() < self.p:
            brightness = random.uniform(-30, 30)
            eyes = np.clip(eyes.astype(np.float32) + brightness, 0, 255).astype(np.uint8)

        # 3. Contrast adjustment
        if random.random() < self.p:
            contrast = random.uniform(0.7, 1.3)
            mean = eyes.mean()
            eyes = np.clip((eyes - mean) * contrast + mean, 0, 255).astype(np.uint8)

        # 4. Gaussian noise
        if random.random() < self.p:
            noise = np.random.normal(0, 8, eyes.shape)
            eyes = np.clip(eyes.astype(np.float32) + noise, 0, 255).astype(np.uint8)

        # 5. Rotation (small head tilt)
        if random.random() < self.p:
            angle = random.uniform(-10, 10)
            h, w = eyes.shape
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            eyes = cv2.warpAffine(eyes, M, (w, h), borderMode=cv2.BORDER_REFLECT)

        # 6. Random crop jitter
        if random.random() < self.p:
            dx = random.randint(-5, 5)
            dy = random.randint(-3, 3)
            eyes = np.roll(np.roll(eyes, dx, axis=1), dy, axis=0)

        # 7. Motion blur (lower probability)
        if random.random() < self.p * 0.3:
            kernel_size = random.choice([3, 5])
            kernel = np.zeros((kernel_size, kernel_size))
            kernel[kernel_size // 2, :] = 1.0 / kernel_size
            eyes = cv2.filter2D(eyes, -1, kernel)

        return eyes, gaze_pitch, gaze_yaw


class SyntheticGazeGenerator:
    """
    Generate synthetic training data by applying gaze offsets to existing images.

    Given an image at known gaze angle, shift the iris region to create
    synthetic images at different gaze angles. Useful for:
    - Extreme gaze angles (±60° to ±90°) rare in real data
    - Filling gaps in training distribution
    - Data augmentation for underrepresented subjects
    """

    def generate_offset_samples(self, eye_image, gaze_pitch, gaze_yaw,
                                 num_offsets=5, max_offset_degrees=20):
        """
        Generate synthetic samples by shifting iris position.

        Args:
            eye_image: (36, 60) grayscale eye crop
            gaze_pitch: original pitch (radians)
            gaze_yaw: original yaw (radians)
            num_offsets: number of synthetic samples to generate
            max_offset_degrees: maximum angular offset

        Returns:
            list of (synthetic_image, new_pitch, new_yaw)
        """
        h, w = eye_image.shape[:2]
        center_x, center_y = w // 2, h // 2

        offsets = np.linspace(-max_offset_degrees, max_offset_degrees, num_offsets)
        # Convert degrees to pixels (empirical: 1° ≈ 1.5px for 60px eye)
        pixel_per_degree = 1.5

        results = []
        for deg in offsets:
            dx = int(deg * pixel_per_degree)

            map_x = np.zeros((h, w), dtype=np.float32)
            map_y = np.zeros((h, w), dtype=np.float32)

            for row in range(h):
                for col in range(w):
                    dist = np.sqrt((col - center_x)**2 + (row - center_y)**2)
                    influence = np.exp(-dist**2 / (2 * 12**2))
                    map_x[row, col] = col - dx * influence
                    map_y[row, col] = row

            synthetic = cv2.remap(
                eye_image, map_x, map_y,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_REFLECT,
            )

            new_yaw = gaze_yaw + np.radians(deg)
            results.append((synthetic, gaze_pitch, new_yaw))

        return results
