"""
OpenBroadcast — Image Utilities

Blending, color transfer, and compositing functions for eye correction.
"""

import cv2
import numpy as np


def create_feathered_mask(height, width, radius, feather_radius=15):
    """
    Create elliptical mask with soft feathered edges.

    Args:
        height: Mask height
        width: Mask width
        radius: Ellipse radius
        feather_radius: Feathering radius in pixels

    Returns:
        Float32 mask (height, width) with values 0-1
    """
    mask = np.zeros((height, width), dtype=np.float32)
    center = (width // 2, height // 2)

    cv2.ellipse(mask, center, (radius, int(radius * 0.7)),
                0, 0, 360, 1.0, -1)

    ksize = max(3, (feather_radius // 3) * 2 + 1)
    mask = cv2.GaussianBlur(mask, (ksize, ksize), feather_radius / 3)

    return mask


def color_transfer_lab(source, target):
    """
    Match color statistics of source to target using LAB color space.

    This ensures corrected regions don't look brighter/darker
    than surrounding skin.

    Args:
        source: BGR image to adjust
        target: BGR reference image

    Returns:
        Color-adjusted BGR image
    """
    try:
        source_lab = cv2.cvtColor(source, cv2.COLOR_BGR2LAB).astype(np.float32)
        target_lab = cv2.cvtColor(target, cv2.COLOR_BGR2LAB).astype(np.float32)

        for ch in range(3):
            s_mean = source_lab[:, :, ch].mean()
            s_std = source_lab[:, :, ch].std() + 1e-6
            t_mean = target_lab[:, :, ch].mean()
            t_std = target_lab[:, :, ch].std() + 1e-6

            source_lab[:, :, ch] = (
                (source_lab[:, :, ch] - s_mean) * (t_std / s_std) + t_mean
            )

        source_lab = np.clip(source_lab, 0, 255).astype(np.uint8)
        return cv2.cvtColor(source_lab, cv2.COLOR_LAB2BGR)
    except Exception:
        return source


def alpha_blend(background, foreground, mask):
    """
    Alpha blend foreground onto background using mask.

    Args:
        background: BGR image
        foreground: BGR image (same size as background region)
        mask: Float32 mask (0-1), same height as foreground

    Returns:
        Blended image
    """
    if len(mask.shape) == 2:
        mask_3ch = np.stack([mask] * 3, axis=-1)
    else:
        mask_3ch = mask

    blended = (foreground * mask_3ch + background * (1 - mask_3ch))
    return np.clip(blended, 0, 255).astype(np.uint8)


def poisson_blend(source, target, center, mask=None):
    """
    Poisson seamless cloning for invisible seams.

    Args:
        source: Source image
        target: Target image (to blend into)
        center: Center point for cloning
        mask: Optional binary mask

    Returns:
        Blended image
    """
    try:
        if mask is None:
            h, w = source.shape[:2]
            mask = np.ones((h, w), dtype=np.uint8) * 255

        result = cv2.seamlessClone(
            source, target, mask, center,
            cv2.NORMAL_CLONE
        )
        return result
    except Exception:
        return target


def adjust_brightness_contrast(image, brightness=0, contrast=1.0):
    """Adjust brightness and contrast of an image."""
    mean = image.mean()
    adjusted = (image.astype(np.float32) - mean) * contrast + mean + brightness
    return np.clip(adjusted, 0, 255).astype(np.uint8)
