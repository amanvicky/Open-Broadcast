"""
OpenBroadcast — Performance Monitoring

Tracks FPS and automatically adjusts quality to maintain target frame rate.
RAM-aware: always leaves headroom for other apps (Zoom, Teams, Chrome, etc.)

Memory budget strategy:
- 8GB system  → leave 2GB free (OpenBroadcast can use ~4-5GB max)
- 16GB system → leave 3GB free (can use more aggressively)
- 30GB+ system → leave 4GB free (plenty of room)
- If other apps spike memory → auto-degrade immediately
"""

import time
import psutil
from collections import deque


class FPSCounter:
    """Simple FPS counter using sliding window."""

    def __init__(self, window_size=30):
        self.frame_times = deque(maxlen=window_size)
        self.last_time = None
        self.fps = 0.0

    def update(self):
        """Call once per frame to update FPS calculation."""
        now = time.perf_counter()
        if self.last_time is not None:
            self.frame_times.append(now - self.last_time)
        self.last_time = now

        if len(self.frame_times) > 2:
            avg = sum(self.frame_times) / len(self.frame_times)
            self.fps = 1.0 / avg if avg > 0 else 0

    @property
    def frame_time_ms(self):
        """Average frame time in milliseconds."""
        if not self.frame_times:
            return 0
        return (sum(self.frame_times) / len(self.frame_times)) * 1000


class AdaptivePerformanceController:
    """
    Monitors FPS and RAM, automatically adjusts processing to maintain target.

    RAM strategy:
    - Always leave MIN_FREE_GB of RAM for other apps
    - Scale thresholds based on total system RAM
    - Monitor process RSS (how much OpenBroadcast itself uses)
    - If other apps consume memory → degrade before OOM
    """

    MODES = ["full", "reduced", "minimal"]

    # Minimum free RAM to always leave for other apps
    MIN_FREE_GB = {
        8: 2.0,     # 8GB system  → leave 2GB free
        16: 3.0,    # 16GB system → leave 3GB free
        32: 4.0,    # 30GB+ system → leave 4GB free
    }

    def __init__(self, target_fps=20, total_ram_gb=8):
        self.target_fps = target_fps
        self.total_ram_gb = total_ram_gb
        self.fps_counter = FPSCounter()
        self.current_mode = "full"
        self.adjustment_cooldown = 0
        self.memory_check_interval = 0

        # Determine free RAM threshold based on total RAM
        self.min_free_gb = 2.0  # Default
        for tier_gb, min_free in sorted(self.MIN_FREE_GB.items()):
            if total_ram_gb >= tier_gb:
                self.min_free_gb = min_free

        # Critical threshold = 80% of min_free (emergency degrade)
        self.critical_free_gb = self.min_free_gb * 0.5

        # Track our own process memory
        self.process = psutil.Process()
        self.our_rss_mb = 0

    def _get_min_free_gb(self):
        """Get the minimum free RAM to leave based on total system RAM."""
        for tier_gb, min_free in sorted(self.MIN_FREE_GB.items(), reverse=True):
            if self.total_ram_gb >= tier_gb:
                return min_free
        return 2.0  # Fallback for <8GB systems

    def update(self):
        """Call after processing each frame."""
        self.fps_counter.update()

        # Memory monitoring — check every 30 frames (~1 second at 30fps)
        self.memory_check_interval += 1
        if self.memory_check_interval >= 30:
            self.memory_check_interval = 0
            self._check_memory()

        if self.adjustment_cooldown > 0:
            self.adjustment_cooldown -= 1
            return

        fps = self.fps_counter.fps
        if fps <= 0:
            return

        # Need enough samples before adjusting
        if len(self.fps_counter.frame_times) < 15:
            return

        if fps < self.target_fps * 0.7:
            # Performance is bad — degrade
            idx = self.MODES.index(self.current_mode)
            if idx < len(self.MODES) - 1:
                self.current_mode = self.MODES[idx + 1]
                self.adjustment_cooldown = 30
                print(f"[Perf] FPS {fps:.1f} < {self.target_fps * 0.7:.1f} → mode: {self.current_mode}")

        elif fps > self.target_fps * 1.3:
            # Performance is good — try to improve
            idx = self.MODES.index(self.current_mode)
            if idx > 0:
                self.current_mode = self.MODES[idx - 1]
                self.adjustment_cooldown = 30
                print(f"[Perf] FPS {fps:.1f} > {self.target_fps * 1.3:.1f} → mode: {self.current_mode}")

    def _check_memory(self):
        """
        Monitor RAM usage with headroom for other apps.

        Strategy:
        1. Check system-wide available RAM
        2. Check how much OpenBroadcast itself uses (RSS)
        3. If available < min_free → degrade
        4. If available < critical → emergency minimal mode
        5. If memory pressure resolves → try to improve
        """
        try:
            # System-wide memory
            mem = psutil.virtual_memory()
            available_gb = mem.available / (1024**3)
            total_gb = mem.total / (1024**3)
            percent_used = mem.percent

            # Our process memory
            try:
                self.our_rss_mb = self.process.memory_info().rss / (1024**2)
            except Exception:
                self.our_rss_mb = 0

            min_free = self._get_min_free_gb()
            critical_free = min_free * 0.5

            # Emergency: very low memory — go minimal immediately
            if available_gb < critical_free:
                print(f"[Mem] EMERGENCY: {available_gb:.1f}GB free < {critical_free:.1f}GB min "
                      f"(total={total_gb:.0f}GB, our usage={self.our_rss_mb:.0f}MB) "
                      f"→ minimal mode")
                self.current_mode = "minimal"
                self.adjustment_cooldown = 60
                return

            # Warning: approaching memory limit — degrade one step
            if available_gb < min_free:
                if self.current_mode == "full":
                    print(f"[Mem] LOW: {available_gb:.1f}GB free < {min_free:.1f}GB target "
                          f"(total={total_gb:.0f}GB, our usage={self.our_rss_mb:.0f}MB) "
                          f"→ reduced mode")
                    self.current_mode = "reduced"
                    self.adjustment_cooldown = 30
                elif self.current_mode == "reduced":
                    # Already reduced but still low — go minimal
                    print(f"[Mem] STILL LOW: {available_gb:.1f}GB free → minimal mode")
                    self.current_mode = "minimal"
                    self.adjustment_cooldown = 60

            # Good: memory pressure resolved — try to improve
            elif available_gb > min_free + 1.0:
                if self.current_mode == "minimal" and available_gb > min_free + 2.0:
                    print(f"[Mem] RECOVERED: {available_gb:.1f}GB free → trying reduced mode")
                    self.current_mode = "reduced"
                    self.adjustment_cooldown = 30
                elif self.current_mode == "reduced" and available_gb > min_free + 3.0:
                    print(f"[Mem] HEALTHY: {available_gb:.1f}GB free → trying full mode")
                    self.current_mode = "full"
                    self.adjustment_cooldown = 30

            # Log memory state periodically
            print(f"[Mem] System: {available_gb:.1f}/{total_gb:.0f}GB free "
                  f"({percent_used:.0f}% used) | OB: {self.our_rss_mb:.0f}MB | "
                  f"Mode: {self.current_mode} | Threshold: {min_free:.1f}GB")

        except Exception:
            pass  # Don't crash on memory check failure

    @property
    def fps(self):
        return self.fps_counter.fps

    @property
    def frame_time_ms(self):
        return self.fps_counter.frame_time_ms

    @property
    def should_use_neural_model(self):
        return self.current_mode == "full"

    @property
    def should_reduce_resolution(self):
        return self.current_mode in ["reduced", "minimal"]

    @property
    def memory_info(self):
        """Return current memory info for status display."""
        try:
            mem = psutil.virtual_memory()
            return {
                "available_gb": round(mem.available / (1024**3), 1),
                "total_gb": round(mem.total / (1024**3), 1),
                "used_percent": mem.percent,
                "our_rss_mb": round(self.our_rss_mb),
                "min_free_gb": self.min_free_gb,
            }
        except Exception:
            return {}
