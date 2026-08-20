"""OpenBroadcast — FPS Counter."""

import time
from collections import deque


class FPSCounter:
    def __init__(self, window_size=30):
        self.frame_times = deque(maxlen=window_size)
        self.last_time = None
        self.fps = 0.0

    def update(self):
        now = time.perf_counter()
        if self.last_time is not None:
            self.frame_times.append(now - self.last_time)
        self.last_time = now
        if len(self.frame_times) > 2:
            avg = sum(self.frame_times) / len(self.frame_times)
            self.fps = 1.0 / avg if avg > 0 else 0

    @property
    def frame_time_ms(self):
        if not self.frame_times:
            return 0
        return (sum(self.frame_times) / len(self.frame_times)) * 1000
