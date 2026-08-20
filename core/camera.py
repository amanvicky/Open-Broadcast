"""
OpenBroadcast — Camera Capture Module

Background thread that captures frames from webcam.
Optimized for low-end PCs:
- DirectShow backend on Windows (fastest)
- Minimal buffer (low latency)
- Downscale in background thread
- Never blocks UI thread
"""

import cv2
import numpy as np
import time
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QMutexLocker


class CameraThread(QThread):
    """
    Background thread that captures frames from webcam.

    Emits:
        frame_ready(np.ndarray, float): Captured frame and timestamp
        error_occurred(str): Error message if camera fails
        fps_updated(float): Current capture FPS
    """

    frame_ready = pyqtSignal(np.ndarray, float)
    error_occurred = pyqtSignal(str)
    fps_updated = pyqtSignal(float)

    def __init__(self, camera_index=0, target_width=640, target_height=480):
        super().__init__()
        self.camera_index = camera_index
        self.target_width = target_width
        self.target_height = target_height
        self.running = False
        self.mutex = QMutex()
        self.cap = None

        # FPS tracking
        self.frame_times = []
        self.last_fps_time = time.perf_counter()

    def run(self):
        """Main capture loop."""
        # Try DirectShow first (fastest on Windows), fallback to default
        try:
            self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            if not self.cap.isOpened():
                self.cap = cv2.VideoCapture(self.camera_index)
        except Exception:
            self.cap = cv2.VideoCapture(self.camera_index)

        if not self.cap or not self.cap.isOpened():
            self.error_occurred.emit(
                "Cannot open camera.\n\n"
                "Possible causes:\n"
                "• Camera is not connected\n"
                "• Camera is being used by another application\n"
                "• Camera drivers are not installed\n\n"
                "Please close other apps using the camera and try again."
            )
            return

        # Optimize for low latency
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.target_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.target_height)

        # Verify actual resolution
        actual_w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(f"[Camera] Opened: {actual_w}x{actual_h}")

        self.running = True

        while self.running:
            ret, frame = self.cap.read()
            if not ret or frame is None:
                # Brief sleep to avoid busy-waiting on error
                time.sleep(0.01)
                continue

            # Downscale if camera gave higher resolution than needed
            h, w = frame.shape[:2]
            if w != self.target_width or h != self.target_height:
                frame = cv2.resize(
                    frame,
                    (self.target_width, self.target_height),
                    interpolation=cv2.INTER_AREA,
                )

            timestamp = time.perf_counter()

            # Track FPS
            self.frame_times.append(timestamp)
            if len(self.frame_times) > 30:
                self.frame_times.pop(0)

            now = time.perf_counter()
            if now - self.last_fps_time >= 1.0 and len(self.frame_times) > 1:
                elapsed = self.frame_times[-1] - self.frame_times[0]
                fps = (len(self.frame_times) - 1) / elapsed if elapsed > 0 else 0
                self.fps_updated.emit(fps)
                self.last_fps_time = now

            self.frame_ready.emit(frame, timestamp)

        if self.cap:
            self.cap.release()

    def stop(self):
        """Stop the capture thread gracefully."""
        with QMutexLocker(self.mutex):
            self.running = False
        self.wait(2000)

    def set_resolution(self, width, height):
        """Change capture resolution on the fly."""
        self.target_width = width
        self.target_height = height
        if self.cap and self.cap.isOpened():
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)


def enumerate_cameras(max_test=10):
    """
    Find all available cameras and their capabilities.

    Returns list of dicts with camera info.
    """
    cameras = []

    for i in range(max_test):
        try:
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if not cap.isOpened():
                cap = cv2.VideoCapture(i)

            if cap.isOpened():
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)

                # Try to find max resolution
                max_w, max_h, max_fps = _probe_camera_max(cap)

                cameras.append({
                    "index": i,
                    "name": f"Camera {i}",
                    "default_resolution": (width, height),
                    "max_resolution": (max_w, max_h),
                    "max_fps": max_fps if max_fps > 0 else 30,
                })

                cap.release()
        except Exception:
            continue

    return cameras


def _probe_camera_max(cap):
    """Probe camera for maximum resolution."""
    import cv2

    test_resolutions = [
        (1920, 1080),
        (1280, 720),
        (960, 540),
        (640, 480),
        (320, 240),
    ]

    best_w, best_h, best_fps = 320, 240, 15

    for w, h in test_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)

        actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        if actual_w >= w * 0.9 and actual_h >= h * 0.9:
            best_w, best_h = actual_w, actual_h
            best_fps = actual_fps if actual_fps > 0 else 30
            break

    return best_w, best_h, best_fps
