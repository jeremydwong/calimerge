"""
SkeletonViewWidget - renders a 3D skeleton projected onto the XY plane.

Receives list[np.ndarray | None] of 17 3D keypoints (COCO-17),
applies a stored view_transform, and paints them with QPainter.
"""

from __future__ import annotations

import numpy as np

from PySide6.QtWidgets import QWidget, QSizePolicy
from PySide6.QtCore import Qt
from PySide6.QtGui import QPainter, QColor, QPen, QFont


# COCO-17 skeleton connections
_SKELETON = [
    (0, 1), (0, 2), (1, 3), (2, 4),
    (5, 6), (5, 7), (7, 9), (6, 8), (8, 10),
    (5, 11), (6, 12), (11, 12),
    (11, 13), (13, 15), (12, 14), (14, 16),
]

# Per-keypoint colors (COCO-17): head, left limbs, right limbs
_KP_COLORS = [
    QColor(255, 255, 100),  # 0 nose
    QColor(200, 200, 80),   # 1 left_eye
    QColor(200, 200, 80),   # 2 right_eye
    QColor(180, 180, 60),   # 3 left_ear
    QColor(180, 180, 60),   # 4 right_ear
    QColor(100, 200, 120),  # 5 left_shoulder
    QColor(100, 160, 255),  # 6 right_shoulder
    QColor(100, 200, 120),  # 7 left_elbow
    QColor(100, 160, 255),  # 8 right_elbow
    QColor(100, 200, 120),  # 9 left_wrist
    QColor(100, 160, 255),  # 10 right_wrist
    QColor(100, 200, 120),  # 11 left_hip
    QColor(100, 160, 255),  # 12 right_hip
    QColor(100, 200, 120),  # 13 left_knee
    QColor(100, 160, 255),  # 14 right_knee
    QColor(100, 200, 120),  # 15 left_ankle
    QColor(100, 160, 255),  # 16 right_ankle
]


class SkeletonViewWidget(QWidget):
    """Displays a 3D skeleton projected onto the XY plane using QPainter."""

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumSize(200, 300)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._keypoints: list = []
        self._message: str = "No detection"
        self._view_transform: np.ndarray = np.eye(4)

    def update_keypoints(self, kps_3d: list) -> None:
        """Set new 3D keypoints and trigger a repaint.

        Args:
            kps_3d: list of np.ndarray (3,) or None, length 17 (COCO-17).
        """
        self._keypoints = list(kps_3d)
        self._message = ""
        self.update()

    def set_view_transform(self, mat: np.ndarray) -> None:
        """Set 4x4 view transform matrix and repaint."""
        self._view_transform = np.array(mat, dtype=float)
        self.update()

    def set_message(self, msg: str) -> None:
        """Show a status message (clears keypoints)."""
        self._keypoints = []
        self._message = msg
        self.update()

    def clear(self) -> None:
        """Clear keypoints and show default 'No detection' message."""
        self._keypoints = []
        self._message = "No detection"
        self.update()

    def get_keypoints(self) -> list:
        """Return the current list of keypoints (already transformed)."""
        return list(self._keypoints)

    # ── Painting ──

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w = self.width()
        h = self.height()

        # Background
        painter.fillRect(0, 0, w, h, QColor(30, 30, 30))

        has_kps = bool(self._keypoints) and any(k is not None for k in self._keypoints)

        if not has_kps:
            msg = self._message if self._message else "No detection"
            painter.setPen(QColor(100, 100, 100))
            painter.setFont(QFont("monospace", 10))
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, msg
            )
            painter.end()
            return

        # Project keypoints: apply view transform then use XY plane
        screen_pts: list[tuple[float, float] | None] = []
        raw_xy: list[tuple[float, float]] = []

        for kp in self._keypoints:
            if kp is None:
                screen_pts.append(None)
                continue
            pt = np.array(kp, dtype=float)
            if pt.shape != (3,):
                screen_pts.append(None)
                continue
            # Apply stored view transform
            pt4 = np.append(pt, 1.0)
            transformed = (self._view_transform @ pt4)[:3]
            raw_xy.append((transformed[0], transformed[1]))
            screen_pts.append((transformed[0], transformed[1]))

        if not raw_xy:
            painter.setPen(QColor(100, 100, 100))
            painter.setFont(QFont("monospace", 10))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No detection")
            painter.end()
            return

        # Compute bounding box for auto-scaling
        xs = [p[0] for p in raw_xy]
        ys = [p[1] for p in raw_xy]
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)

        margin = 20
        data_w = x_max - x_min or 1.0
        data_h = y_max - y_min or 1.0

        avail_w = w - 2 * margin
        avail_h = h - 2 * margin
        scale = min(avail_w / data_w, avail_h / data_h)

        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2

        def to_screen(xy: tuple[float, float]) -> tuple[int, int]:
            sx = int(w / 2 + (xy[0] - x_center) * scale)
            # Y-up: flip Y for screen coords
            sy = int(h / 2 - (xy[1] - y_center) * scale)
            return sx, sy

        # Build final screen coords
        final_pts: list[tuple[int, int] | None] = []
        for sp in screen_pts:
            if sp is None:
                final_pts.append(None)
            else:
                final_pts.append(to_screen(sp))

        # Draw skeleton limbs
        painter.setPen(QPen(QColor(220, 220, 220), 2))
        n = len(final_pts)
        for i, j in _SKELETON:
            if i >= n or j >= n:
                continue
            if final_pts[i] is None or final_pts[j] is None:
                continue
            painter.drawLine(
                final_pts[i][0], final_pts[i][1],
                final_pts[j][0], final_pts[j][1],
            )

        # Draw keypoints
        for idx, pt in enumerate(final_pts):
            if pt is None:
                continue
            color = _KP_COLORS[idx] if idx < len(_KP_COLORS) else QColor(200, 200, 200)
            painter.setBrush(color)
            painter.setPen(QPen(QColor(30, 30, 30), 1))
            painter.drawEllipse(pt[0] - 4, pt[1] - 4, 8, 8)

        painter.end()
