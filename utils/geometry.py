"""
OpenBroadcast — Eye Geometry Utilities

Mathematical functions for eye geometry calculations.
"""

import numpy as np


def calculate_iris_offset(iris_pos, eye_outer, eye_inner):
    """
    Calculate normalized iris offset from eye center.

    Args:
        iris_pos: (x, y) iris center position
        eye_outer: (x, y) outer corner of eye
        eye_inner: (x, y) inner corner of eye

    Returns:
        (offset_x, offset_y) normalized to [-0.5, 0.5]
    """
    eye_center = np.array([(eye_outer[0] + eye_inner[0]) / 2,
                           (eye_outer[1] + eye_inner[1]) / 2])
    eye_width = np.linalg.norm(np.array(eye_inner) - np.array(eye_outer))

    if eye_width < 1e-6:
        return 0.0, 0.0

    offset_x = (iris_pos[0] - eye_center[0]) / eye_width
    offset_y = (iris_pos[1] - eye_center[1]) / eye_width

    return offset_x, offset_y


def offset_to_degrees(offset_x, offset_y, yaw_scale=30.0, pitch_scale=20.0):
    """
    Convert normalized offset to gaze angles in degrees.

    Empirical mapping: offset of 0.5 ≈ 15° gaze angle
    """
    yaw = offset_x * yaw_scale * 2  # *2 because offset is -0.5 to 0.5
    pitch = offset_y * pitch_scale * 2
    return yaw, pitch


def degrees_to_offset(yaw, pitch, yaw_scale=30.0, pitch_scale=20.0):
    """Convert gaze angles in degrees to normalized offset."""
    offset_x = yaw / (yaw_scale * 2)
    offset_y = pitch / (pitch_scale * 2)
    return offset_x, offset_y


def calculate_displacement_correction(offset_x, eye_width, strength=0.85, scale=0.8):
    """
    Calculate pixel displacement for eye correction.

    Args:
        offset_x: Normalized iris offset (-0.5 to 0.5)
        eye_width: Width of eye in pixels
        strength: Correction strength (0-1)
        scale: Correction scale factor (< 1 to avoid over-correction)

    Returns:
        Pixel displacement (negative = shift left)
    """
    return -offset_x * eye_width * strength * scale


def create_influence_map(width, height, center_x, center_y, sigma):
    """
    Create Gaussian influence map centered on the eye.

    Used for distance-weighted displacement during warping.
    Pixels closer to center get more displacement.
    """
    y_coords, x_coords = np.mgrid[0:height, 0:width].astype(np.float32)
    dx = x_coords - center_x
    dy = y_coords - center_y
    return np.exp(-(dx**2 + dy**2) / (2 * sigma**2))


def is_eye_blinking(eye_top, eye_bottom, eye_width, threshold=0.15):
    """
    Detect if eye is blinking based on eye opening ratio.

    Args:
        eye_top: (x, y) top of eye
        eye_bottom: (x, y) bottom of eye
        eye_width: Width of eye
        threshold: Ratio below which eye is considered closed

    Returns:
        True if blinking
    """
    eye_height = np.linalg.norm(np.array(eye_bottom) - np.array(eye_top))
    ratio = eye_height / (eye_width + 1e-6)
    return ratio < threshold
