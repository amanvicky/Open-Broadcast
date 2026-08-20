"""
OpenBroadcast — Virtual Camera Output

Outputs corrected video frames to a virtual camera
that can be used in Zoom, Teams, OBS, etc.

Requires:
- pyvirtualcam library
- OBS Studio with virtual camera (or UnityCapture)
"""

import numpy as np

try:
    import pyvirtualcam
    HAS_PYVIRTUALCAM = True
except ImportError:
    HAS_PYVIRTUALCAM = False


class VirtualCamera:
    """
    Outputs corrected frames to a virtual camera device.

    Usage:
        vcam = VirtualCamera(width=640, height=480, fps=30)
        vcam.start()
        vcam.send_frame(corrected_frame)
        vcam.stop()
    """

    def __init__(self, width=640, height=480, fps=30):
        self.width = width
        self.height = height
        self.fps = fps
        self.camera = None
        self.is_running = False

    def start(self):
        """Start the virtual camera."""
        if not HAS_PYVIRTUALCAM:
            raise RuntimeError(
                "pyvirtualcam is not installed.\n"
                "Install with: pip install pyvirtualcam\n"
                "Also requires OBS Studio virtual camera."
            )

        try:
            self.camera = pyvirtualcam.Camera(
                width=self.width,
                height=self.height,
                fps=self.fps,
                fmt=pyvirtualcam.PixelFormat.BGR,
            )
            self.is_running = True
            print(f"[VirtualCam] Started: {self.camera.device}")
            return True
        except Exception as e:
            print(f"[VirtualCam] Failed to start: {e}")
            return False

    def send_frame(self, frame):
        """
        Send a frame to the virtual camera.

        Args:
            frame: BGR numpy array (height, width, 3)
        """
        if not self.is_running or self.camera is None:
            return

        try:
            # Resize if needed
            h, w = frame.shape[:2]
            if w != self.width or h != self.height:
                import cv2
                frame = cv2.resize(frame, (self.width, self.height))

            self.camera.send(frame)
        except Exception as e:
            print(f"[VirtualCam] Frame send error: {e}")

    def stop(self):
        """Stop the virtual camera."""
        if self.camera is not None:
            try:
                self.camera.close()
            except Exception:
                pass
        self.is_running = False
        self.camera = None
        print("[VirtualCam] Stopped")

    @property
    def device_name(self):
        """Return the virtual camera device name."""
        if self.camera is not None:
            return str(self.camera.device)
        return "Not started"

    @staticmethod
    def is_available():
        """Check if virtual camera output is available."""
        return HAS_PYVIRTUALCAM
