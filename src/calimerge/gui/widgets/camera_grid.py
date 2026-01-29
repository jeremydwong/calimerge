"""
Multi-camera display grid widget.
"""

from __future__ import annotations

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


class CameraCell(QWidget):
    """Single camera display cell with label."""

    clicked = Signal(int)  # port

    def __init__(self, port: int, label: str = "", parent: QWidget | None = None):
        super().__init__(parent)
        self.port = port

        layout = QVBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(2)

        # Camera label
        self.label = QLabel(label or f"Camera {port}")
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setStyleSheet("font-weight: bold; font-size: 11px;")
        layout.addWidget(self.label)

        # Frame display
        self.frame_label = QLabel()
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setMinimumSize(160, 120)
        self.frame_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.frame_label.setStyleSheet(
            "background-color: #1a1a1a; border: 1px solid #333;"
        )
        layout.addWidget(self.frame_label, stretch=1)

        # Status label
        self.status_label = QLabel("No signal")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setStyleSheet("color: #888; font-size: 10px;")
        layout.addWidget(self.status_label)

    def set_frame(self, frame: np.ndarray) -> None:
        """Display a BGR frame."""
        if frame is None:
            self.frame_label.clear()
            self.status_label.setText("No signal")
            return

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
            self.status_label.setText(f"{w}x{h}")

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
    """

    camera_clicked = Signal(int)  # port

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.cells: dict[int, CameraCell] = {}

        self.grid_layout = QGridLayout(self)
        self.grid_layout.setSpacing(4)
        self.grid_layout.setContentsMargins(4, 4, 4, 4)

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
