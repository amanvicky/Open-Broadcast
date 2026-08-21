"""OpenBroadcast — Camera Preview Widget."""

import cv2
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont
import math


class PreviewWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_frame = None
        self.overlay_frame = None
        self.fps = 0.0
        self.correction_enabled = True
        self.gaze_info = None
        self.eye_data = None
        self.compare_mode = False
        self.wizard_dot = None  # (x, y) normalized 0-1
        self.setMinimumSize(320, 240)

    def update_frame(self, frame, overlay=None):
        self.current_frame = frame
        self.overlay_frame = overlay
        self._frame_h, self._frame_w = frame.shape[:2] if frame is not None else (480, 640)
        self.update()

    def set_fps(self, fps):
        self.fps = fps

    def set_gaze_info(self, info):
        self.gaze_info = info

    def set_compare_mode(self, enabled):
        self.compare_mode = enabled
        self.update()

    def set_eye_data(self, eye_data):
        self.eye_data = eye_data

    def set_wizard_dot(self, pos):
        self.wizard_dot = pos
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        if self.current_frame is None:
            painter.setPen(QColor(100, 100, 100))
            painter.setFont(QFont("Segoe UI", 14))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter,
                           "Waiting for camera...")
            painter.end()
            return

        if self.compare_mode and self.overlay_frame is not None:
            self._draw_split(painter)
        else:
            display = self.overlay_frame if self.overlay_frame is not None else self.current_frame
            self._draw_frame(painter, display, self.rect())

        self._draw_fps(painter)
        self._draw_status(painter)
        if self.gaze_info:
            self._draw_gaze(painter)
        if self.eye_data and not self.compare_mode:
            self._draw_iris_overlay(painter)
        if self.wizard_dot:
            self._draw_wizard_dot(painter)

        painter.end()

    def _draw_split(self, painter):
        w, h = self.width(), self.height()
        half = w // 2

        left = self.rect()
        left.setWidth(half)
        self._draw_frame(painter, self.current_frame, left)

        right = self.rect()
        right.setLeft(half)
        right.setWidth(w - half)
        self._draw_frame(painter, self.overlay_frame, right)

        painter.setPen(QColor(255, 255, 255, 200))
        painter.drawLine(half, 0, half, h)

        self._draw_label(painter, "◀ ORIGINAL", 12, 45, QColor(255, 200, 0))
        self._draw_label(painter, "CORRECTED ▶", half + 12, 45, QColor(0, 255, 136))

    def _draw_frame(self, painter, frame, rect):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        q_img = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        scaled = pixmap.scaled(rect.size(), Qt.AspectRatioMode.KeepAspectRatio,
                              Qt.TransformationMode.SmoothTransformation)
        x = rect.x() + (rect.width() - scaled.width()) // 2
        y = rect.y() + (rect.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)

    def _draw_label(self, painter, text, x, y, color):
        font = QFont("Segoe UI", 11, QFont.Weight.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(text) + 16
        th = fm.height() + 8
        painter.setBrush(QColor(0, 0, 0, 160))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(x, y, tw, th, 4, 4)
        painter.setPen(color)
        painter.drawText(x + 8, y + fm.ascent() + 4, text)

    def _draw_fps(self, painter):
        text = f"FPS: {self.fps:.1f}"
        color = QColor(0, 255, 136) if self.fps >= 20 else QColor(255, 200, 0) if self.fps >= 15 else QColor(255, 50, 50)
        font = QFont("Consolas", 13, QFont.Weight.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(text) + 16
        th = fm.height() + 8
        rx = self.width() - tw - 10
        painter.setBrush(QColor(0, 0, 0, 160))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rx, 10, tw, th, 4, 4)
        painter.setPen(color)
        painter.drawText(rx + 8, 10 + fm.ascent() + 4, text)

    def _draw_status(self, painter):
        text = "CORRECTION: ON" if self.correction_enabled else "CORRECTION: OFF"
        color = QColor(0, 212, 255) if self.correction_enabled else QColor(150, 150, 150)
        font = QFont("Segoe UI", 11, QFont.Weight.Bold)
        painter.setFont(font)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(text) + 16
        th = fm.height() + 8
        painter.setBrush(QColor(0, 0, 0, 160))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(10, 10, tw, th, 4, 4)
        painter.setPen(color)
        painter.drawText(18, 10 + fm.ascent() + 4, text)

    def _draw_wizard_dot(self, painter):
        """Draw a moving calibration dot for the interactive wizard."""
        x, y = self.wizard_dot
        px = int(x * self.width())
        py = int(y * self.height())

        # Outer glow
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QColor(0, 200, 255, 60))
        painter.drawEllipse(px - 30, py - 30, 60, 60)

        # Inner dot
        painter.setBrush(QColor(0, 200, 255, 220))
        painter.drawEllipse(px - 12, py - 12, 24, 24)

        # Center
        painter.setBrush(QColor(255, 255, 255, 255))
        painter.drawEllipse(px - 4, py - 4, 8, 8)

    def _draw_iris_overlay(self, painter):
        """Draw iris position, target, and shift arrow."""
        for side in ("left", "right"):
            eye = self.eye_data.get(f"{side}_eye")
            if not eye:
                continue
            iris = eye["iris"]
            center = eye["center"]
            width = eye["width"]
            if width < 15:
                continue

            # Scale from frame coords to widget coords (use actual frame shape)
            frame_h = getattr(self, '_frame_h', 480)
            frame_w = getattr(self, '_frame_w', 640)
            widget_w, widget_h = self.width(), self.height()
            sx = widget_w / frame_w
            sy = widget_h / frame_h

            # Current iris position (red dot)
            ix = int(iris[0] * sx)
            iy = int(iris[1] * sy)
            painter.setPen(QColor(255, 50, 50))
            painter.setBrush(QColor(255, 50, 50, 180))
            painter.drawEllipse(ix - 5, iy - 5, 10, 10)

            # Target position (green dot)
            tx = int(center[0] * sx)
            ty = int(center[1] * sy + 0.21 * width * sy)
            painter.setPen(QColor(0, 255, 100))
            painter.setBrush(QColor(0, 255, 100, 180))
            painter.drawEllipse(tx - 5, ty - 5, 10, 10)

            # Shift arrow (yellow)
            dx = tx - ix
            dy = ty - iy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 2:
                painter.setPen(QColor(255, 255, 0, 200))
                painter.drawLine(ix, iy, tx, ty)
                # Shift label
                label = f"{dist:.0f}px"
                painter.setFont(QFont("Consolas", 9))
                painter.setPen(QColor(255, 255, 0))
                painter.drawText((ix + tx) // 2 + 5, (iy + ty) // 2 - 5, label)

    def _draw_gaze(self, painter):
        info = self.gaze_info
        if not info:
            return
        yaw = info.get("yaw", 0)
        pitch = info.get("pitch", 0)
        looking_at = info.get("is_looking_at_camera", True)

        text = f"Yaw: {yaw:+.1f}°  Pitch: {pitch:+.1f}°"
        if looking_at:
            text += "  [Looking at camera]"
            color = QColor(0, 255, 136)
        else:
            text += "  [Looking away]"
            color = QColor(255, 100, 100)

        font = QFont("Consolas", 10)
        painter.setFont(font)
        fm = painter.fontMetrics()
        tw = fm.horizontalAdvance(text) + 16
        th = fm.height() + 8
        ry = self.height() - th - 10
        painter.setBrush(QColor(0, 0, 0, 160))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(10, ry, tw, th, 4, 4)
        painter.setPen(color)
        painter.drawText(18, ry + fm.ascent() + 4, text)
