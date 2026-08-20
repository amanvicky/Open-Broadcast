"""
OpenBroadcast — Gaze Calibration Utility

Calibrates the gaze estimator to the individual user.
Records where the iris sits when the user looks straight at camera,
then subtracts that baseline from all future measurements.
"""

import numpy as np
import time
from collections import deque


class GazeCalibrator:
    """
    Collects iris position samples while user looks at camera,
    then computes calibration offset.

    Usage:
        calibrator = GazeCalibrator(duration=2.0)
        # In frame processing loop:
        if calibrator.is_calibrating:
            calibrator.add_sample(left_iris_offset, right_iris_offset)
        # After duration:
        offset = calibrator.finish()
    """

    def __init__(self, duration=2.0, min_samples=10):
        """
        Args:
            duration: Calibration duration in seconds
            min_samples: Minimum samples needed for valid calibration
        """
        self.duration = duration
        self.min_samples = min_samples
        self.samples = []
        self.start_time = None
        self._calibrating = False

    def start(self):
        """Start calibration."""
        self._calibrating = True
        self.start_time = time.perf_counter()
        self.samples = []
        return True

    def add_sample(self, left_offset_x, left_offset_y,
                   right_offset_x, right_offset_y):
        """Add an iris offset sample during calibration."""
        if not self._calibrating:
            return

        avg_x = (left_offset_x + right_offset_x) / 2
        avg_y = (left_offset_y + right_offset_y) / 2
        self.samples.append((avg_x, avg_y))

    @property
    def is_calibrating(self):
        """Whether calibration is in progress."""
        if not self._calibrating:
            return False
        elapsed = time.perf_counter() - self.start_time
        return elapsed < self.duration

    @property
    def progress(self):
        """Calibration progress 0-1."""
        if not self._calibrating:
            return 0.0
        elapsed = time.perf_counter() - self.start_time
        return min(1.0, elapsed / self.duration)

    def finish(self):
        """
        Finish calibration and compute offset.

        Returns:
            (offset_x, offset_y) or None if not enough samples
        """
        self._calibrating = False

        if len(self.samples) < self.min_samples:
            return None

        samples = np.array(self.samples)
        offset_x = float(np.mean(samples[:, 0]))
        offset_y = float(np.mean(samples[:, 1]))

        return (offset_x, offset_y)
