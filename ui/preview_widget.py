"""
OpenBroadcast — Camera Preview Widget

Displays camera feed with overlays, split-screen comparison,
and magnified iris-only zoom view.
"""

import cv2
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt, QRect
from PyQt6.QtGui import QImage, QPixmap, QPainter, QColor, QFont


class PreviewWidget(QWidget):
    """Displays camera feed with split-screen comparison and iris zoom."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.current_frame = None
        self.overlay_frame = None
        self.fps = 0.0
        self.correction_enabled = True
        self.show_landmarks = False
        self.gaze_info = None
        self.compare_mode = False
        self.eye_data = None
        self.setMinimumSize(320, 240)

    def update_frame(self, frame, overlay=None):
        self.current_frame = frame
        self.overlay_frame = overlay
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
            self._draw_split_screen(painter)
        else:
            display = self.overlay_frame if self.overlay_frame is not None else self.current_frame
            self._draw_frame(painter, display, self.rect())

        # Iris zoom view
        if self.eye_data and self.correction_enabled:
            self._draw_iris_zoom(painter)

        self._draw_fps_overlay(painter)
        self._draw_status_overlay(painter)
        if self.gaze_info:
            self._draw_gaze_overlay(painter)

        painter.end()

    def _draw_split_screen(self, painter):
        widget_w = self.width()
        widget_h = self.height()
        half_w = widget_w // 2

        left_rect = self.rect()
        left_rect.setWidth(half_w)
        self._draw_frame(painter, self.current_frame, left_rect)

        right_rect = self.rect()
        right_rect.setLeft(half_w)
        right_rect.setWidth(widget_w - half_w)
        self._draw_frame(painter, self.overlay_frame, right_rect)

        painter.setPen(QColor(255, 255, 255, 200))
        painter.drawLine(half_w, 0, half_w, widget_h)

        self._draw_label(painter, "◀ ORIGINAL", 12, 45, QColor(255, 200, 0))
        self._draw_label(painter, "CORRECTED ▶", half_w + 12, 45, QColor(0, 255, 136))

    def _draw_iris_zoom(self, painter):
        """Draw 4x magnified iris view — crops TIGHT around each iris
        so even small pixel shifts are clearly visible."""
        if not self.eye_data:
            return
        left_eye = self.eye_data.get("left_eye")
        right_eye = self.eye_data.get("right_eye")
        if not left_eye or not right_eye:
            return
        if self.current_frame is None:
            return

        h, w = self.current_frame.shape[:2]

        def crop_around_eye_center(eye):
            """Crop region around eye CENTER (socket), not iris.
            This ensures both original and corrected show the same region,
            so the iris shift is visible within the crop."""
            cx, cy = int(eye["center"][0]), int(eye["center"][1])
            r = int(eye["width"] * 0.9)
            x1 = max(0, cx - r)
            y1 = max(0, cy - r)
            x2 = min(w, cx + r)
            y2 = min(h, cy + r)
            return (x1, y1, x2, y2)

        left_box = crop_around_eye_center(left_eye)
        right_box = crop_around_eye_center(right_eye)

        def get_crop(box):
            return self.current_frame[box[1]:box[3], box[0]:box[2]]
        def get_corr_crop(box):
            if self.overlay_frame is not None:
                return self.overlay_frame[box[1]:box[3], box[0]:box[2]]
            return None

        left_crop = get_crop(left_box)
        right_crop = get_crop(right_box)
        corr_left = get_corr_crop(left_box)
        corr_right = get_corr_crop(right_box)

        if left_crop.size == 0 or right_crop.size == 0:
            return

        # Layout: 2 columns (orig / fixed) x 2 rows (left / right eye)
        zoom_w = 160
        zoom_h = 120
        gap = 6
        margin = 10
        panel_w = zoom_w * 2 + gap + 10
        panel_h = zoom_h * 2 + 50

        zx = self.width() - panel_w - margin
        zy = self.height() - panel_h - 40

        painter.setBrush(QColor(0, 0, 0, 210))
        painter.setPen(QColor(0, 212, 255, 180))
        painter.drawRoundedRect(zx - 5, zy - 5, panel_w + 10, panel_h + 5, 6, 6)

        font = QFont("Segoe UI", 9, QFont.Weight.Bold)
        painter.setFont(font)
        painter.setPen(QColor(0, 212, 255))
        painter.drawText(zx, zy + 12, "IRIS ZOOM (4x)")

        small_font = QFont("Segoe UI", 8)

        # Left eye
        row1_y = zy + 18
        self._draw_frame(painter, left_crop, QRect(zx, row1_y, zoom_w, zoom_h), True)
        if corr_left is not None and corr_left.size > 0:
            self._draw_frame(painter, corr_left, QRect(zx + zoom_w + gap, row1_y, zoom_w, zoom_h), True)
        painter.setFont(small_font)
        painter.setPen(QColor(255, 200, 0))
        painter.drawText(zx + 2, row1_y + zoom_h + 12, "L Orig")
        painter.setPen(QColor(0, 255, 136))
        painter.drawText(zx + zoom_w + gap + 2, row1_y + zoom_h + 12, "L Fixed")

        # Right eye
        row2_y = row1_y + zoom_h + 18
        self._draw_frame(painter, right_crop, QRect(zx, row2_y, zoom_w, zoom_h), True)
        if corr_right is not None and corr_right.size > 0:
            self._draw_frame(painter, corr_right, QRect(zx + zoom_w + gap, row2_y, zoom_w, zoom_h), True)
        painter.setPen(QColor(255, 200, 0))
        painter.drawText(zx + 2, row2_y + zoom_h + 12, "R Orig")
        painter.setPen(QColor(0, 255, 136))
        painter.drawText(zx + zoom_w + gap + 2, row2_y + zoom_h + 12, "R Fixed")

    def _draw_frame(self, painter, frame, target_rect, force_fill=False):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb.shape
        bytes_per_line = ch * w

        q_img = QImage(rgb.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(q_img)
        aspect = (Qt.AspectRatioMode.IgnoreAspectRatio if force_fill
                  else Qt.AspectRatioMode.KeepAspectRatio)
        scaled = pixmap.scaled(target_rect.size(), aspect,
                              Qt.TransformationMode.SmoothTransformation)
        x = target_rect.x() + (target_rect.width() - scaled.width()) // 2
        y = target_rect.y() + (target_rect.height() - scaled.height()) // 2
        painter.drawPixmap(x, y, scaled)

    def _draw_label(self, painter, text, x, y, color):
        font = QFont("Segoe UI", 11, QFont.Weight.Bold)
        painter.setFont(font)

        fm = painter.fontMetrics()
        text_width = fm.horizontalAdvance(text) + 16
        text_height = fm.height() + 8

        painter.setBrush(QColor(0, 0, 0, 160))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(x, y, text_width, text_height, 4, 4)

        painter.setPen(color)
        painter.drawText(x + 8, y + fm.ascent() + 4, text)

    def _draw_fps_overlay(self, painter):
        fps_text = f"FPS: {self.fps:.1f}"

        if self.fps >= 20:
            color = QColor(0, 255, 136)
        elif self.fps >= 15:
            color = QColor(255, 200, 0)
        else:
            color = QColor(255, 50, 50)

        font = QFont("Consolas", 13, QFont.Weight.Bold)
        painter.setFont(font)

        fm = painter.fontMetrics()
        text_width = fm.horizontalAdvance(fps_text) + 16
        text_height = fm.height() + 8

        rect_x = self.width() - text_width - 10
        rect_y = 10

        painter.setBrush(QColor(0, 0, 0, 160))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect_x, rect_y, text_width, text_height, 4, 4)

        painter.setPen(color)
        painter.drawText(rect_x + 8, rect_y + fm.ascent() + 4, fps_text)

    def _draw_status_overlay(self, painter):
        status = "CORRECTION: ON" if self.correction_enabled else "CORRECTION: OFF"
        color = QColor(0, 212, 255) if self.correction_enabled else QColor(150, 150, 150)

        font = QFont("Segoe UI", 11, QFont.Weight.Bold)
        painter.setFont(font)

        fm = painter.fontMetrics()
        text_width = fm.horizontalAdvance(status) + 16
        text_height = fm.height() + 8

        painter.setBrush(QColor(0, 0, 0, 160))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(10, 10, text_width, text_height, 4, 4)

        painter.setPen(color)
        painter.drawText(18, 10 + fm.ascent() + 4, status)

    def _draw_gaze_overlay(self, painter):
        info = self.gaze_info
        if not info:
            return

        yaw = info.get("yaw", 0)
        pitch = info.get("pitch", 0)
        looking_at = info.get("is_looking_at_camera", True)
        demo = info.get("demo_mode", False)

        text = f"Yaw: {yaw:+.1f}°  Pitch: {pitch:+.1f}°"
        if looking_at:
            text += "  [Looking at camera]"
            color = QColor(0, 255, 136)
        else:
            text += "  [Looking away]"
            color = QColor(255, 100, 100)

        if demo:
            text = "[DEMO 20°] " + text
            color = QColor(255, 165, 0)

        font = QFont("Consolas", 10)
        painter.setFont(font)

        fm = painter.fontMetrics()
        text_width = fm.horizontalAdvance(text) + 16
        text_height = fm.height() + 8

        rect_x = 10
        rect_y = self.height() - text_height - 10

        painter.setBrush(QColor(0, 0, 0, 160))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(rect_x, rect_y, text_width, text_height, 4, 4)

        painter.setPen(color)
        painter.drawText(rect_x + 8, rect_y + fm.ascent() + 4, text)

    def resizeEvent(self, event):
        super().resizeEvent(event)
