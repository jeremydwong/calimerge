"""
Multi-camera display grid widget.
"""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtWidgets import (
    QWidget,
    QGridLayout,
    QLabel,
    QVBoxLayout,
    QSizePolicy,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap

from ..frame_utils import bgr_to_pixmap
from ..colors import camera_color_hex


class CameraCell(QWidget):
    """Single camera display cell with label."""

    clicked = Signal(int)  # port

    def __init__(self, port: int, label: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.port = port
        self._last_frame_time: float = 0.0
        self._color_hex = camera_color_hex(port)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Camera label (colored to match camera's palette color)
        self.label = QLabel(label or f"Camera {port}")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet(
            f"font-weight: bold; font-size: 11px; color: {self._color_hex};"
        )
        layout.addWidget(self.label)

        # Frame display (border colored to match camera)
        # Sized for 4:3 aspect ratio cameras
        self.frame_label = QLabel()
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setMinimumSize(240, 180)  # 4:3 aspect ratio minimum
        self.frame_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.frame_label.setStyleSheet(
            f"background-color: #1a1a1a; border: 2px solid {self._color_hex};"
        )
        layout.addWidget(self.frame_label, stretch=1)

        # Status label (resolution + fps)
        self.status_label = QLabel("No signal")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.status_label)

    def set_frame(self, frame: np.ndarray) -> None:
        """Display a BGR frame."""
        if frame is None:
            self.frame_label.clear()
            self.status_label.setText("No signal")
            self._last_frame_time = 0.0
            return

        # Compute FPS from inter-frame period
        now = time.perf_counter()
        fps_str = ""
        if self._last_frame_time > 0:
            dt = now - self._last_frame_time
            if dt > 0:
                fps_str = f"  {1.0 / dt:.1f} fps"
        self._last_frame_time = now

        pixmap = bgr_to_pixmap(frame)
        if not pixmap.isNull():
            # Scale to fit label while preserving aspect ratio
            scaled = pixmap.scaled(
                self.frame_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.frame_label.setPixmap(scaled)
            h, w = frame.shape[:2]
            self.status_label.setText(f"{w}x{h}{fps_str}")

    def set_label(self, text: str) -> None:
        """Update camera label."""
        self.label.setText(text)

    def set_status(self, text: str) -> None:
        """Update status text."""
        self.status_label.setText(text)

    def mousePressEvent(self, event):
        """Emit clicked signal."""
        self.clicked.emit(self.port)
        super().mousePressEvent(event)


class CameraGrid(QWidget):
    """
    Grid layout for multiple camera displays.

    Automatically arranges cameras in optimal grid.
    Designed for 4:3 aspect ratio cameras in 2x2 grid layout.
    """

    camera_clicked = Signal(int)  # port

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.cells: dict[int, CameraCell] = {}

        self.grid_layout = QGridLayout(self)
        self.grid_layout.setSpacing(6)
        self.grid_layout.setContentsMargins(4, 4, 4, 4)
        # Minimum size for comfortable 2x2 viewing of 4:3 cameras
        self.setMinimumSize(520, 400)

    def set_cameras(self, camera_info: dict[int, str]) -> None:
        """
        Set up grid for cameras.

        Args:
            camera_info: dict of port -> display_name
        """
        # Clear existing
        for cell in self.cells.values():
            self.grid_layout.removeWidget(cell)
            cell.deleteLater()
        self.cells.clear()

        if not camera_info:
            return

        # Calculate grid dimensions
        n = len(camera_info)
        cols = _optimal_cols(n)
        rows = (n + cols - 1) // cols

        # Create cells
        for idx, (port, name) in enumerate(sorted(camera_info.items())):
            row = idx // cols
            col = idx % cols

            cell = CameraCell(port, name, self)
            cell.clicked.connect(self.camera_clicked.emit)
            self.cells[port] = cell
            self.grid_layout.addWidget(cell, row, col)

    def update_frame(self, port: int, frame: np.ndarray) -> None:
        """Update frame for a specific camera."""
        if port in self.cells:
            self.cells[port].set_frame(frame)

    def update_status(self, port: int, status: str) -> None:
        """Update status for a specific camera."""
        if port in self.cells:
            self.cells[port].set_status(status)

    def clear_all(self) -> None:
        """Clear all frame displays."""
        for cell in self.cells.values():
            cell.frame_label.clear()
            cell.set_status("No signal")


def _optimal_cols(n: int) -> int:
    """Calculate optimal column count for n cameras."""
    if n <= 1:
        return 1
    elif n <= 2:
        return 2
    elif n <= 4:
        return 2
    elif n <= 6:
        return 3
    elif n <= 9:
        return 3
    else:
        return 4
