#!/usr/bin/env python3
"""
Clock widget for visual camera synchronization verification.

Displays a large on-screen timer that updates every 10ms.
Point your cameras at this display, then capture synced frames.
The displayed time in each frame should match if cameras are synchronized.

Usage:
    python3 clock_widget.py
"""

import sys
import time
from PySide6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel
from PySide6.QtCore import QTimer, Qt
from PySide6.QtGui import QFont, QPalette, QColor


class ClockWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Calimerge Sync Clock")

        # Dark background for better camera visibility
        self.setAutoFillBackground(True)
        palette = self.palette()
        palette.setColor(QPalette.ColorRole.Window, QColor(0, 0, 0))
        self.setPalette(palette)

        # Layout
        layout = QVBoxLayout()
        layout.setContentsMargins(20, 20, 20, 20)

        # Main time label (large)
        self.time_label = QLabel(self)
        font = QFont("Courier", 120, QFont.Weight.Bold)
        self.time_label.setFont(font)
        self.time_label.setStyleSheet("color: #00FF00;")  # Green text
        self.time_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.time_label)

        # Milliseconds label (even larger for camera visibility)
        self.ms_label = QLabel(self)
        ms_font = QFont("Courier", 200, QFont.Weight.Bold)
        self.ms_label.setFont(ms_font)
        self.ms_label.setStyleSheet("color: #FFFF00;")  # Yellow text
        self.ms_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.ms_label)

        # Instructions
        self.info_label = QLabel("Point cameras at this display to verify sync")
        self.info_label.setStyleSheet("color: #888888; font-size: 18px;")
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.info_label)

        self.setLayout(layout)

        # Start time reference
        self.start_time = time.perf_counter()

        # Update timer (10ms interval)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_display)
        self.timer.start(10)

        # Set minimum size
        self.setMinimumSize(800, 600)

    def update_display(self):
        elapsed = time.perf_counter() - self.start_time

        # Full timestamp
        self.time_label.setText(f"{elapsed:10.4f}")

        # Just milliseconds (0-999) - easier to compare visually
        ms = int((elapsed * 1000) % 1000)
        self.ms_label.setText(f"{ms:03d}")


def main():
    app = QApplication(sys.argv)
    widget = ClockWidget()
    widget.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
