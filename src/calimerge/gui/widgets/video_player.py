"""
Video playback widget with scrubber.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSizePolicy,
)
from PySide6.QtCore import Qt, Signal, QTimer
from PySide6.QtGui import QPixmap

from ..frame_utils import bgr_to_pixmap


class VideoPlayer(QWidget):
    """
    Video playback widget with play/pause and frame scrubbing.
    """

    frame_changed = Signal(int, object)  # frame_index, np.ndarray

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)

        self.cap: cv2.VideoCapture | None = None
        self.video_path: Path | None = None
        self.total_frames: int = 0
        self.current_frame: int = 0
        self.fps: float = 30.0
        self.is_playing: bool = False

        self._init_ui()

        # Playback timer
        self.play_timer = QTimer(self)
        self.play_timer.timeout.connect(self._advance_frame)

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        # Frame display
        self.frame_label = QLabel()
        self.frame_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.frame_label.setMinimumSize(320, 240)
        self.frame_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.frame_label.setStyleSheet("background-color: #1a1a1a;")
        layout.addWidget(self.frame_label, stretch=1)

        # Controls
        controls = QHBoxLayout()
        controls.setContentsMargins(4, 4, 4, 4)

        self.play_button = QPushButton("Play")
        self.play_button.setFixedWidth(60)
        self.play_button.clicked.connect(self.toggle_play)
        self.play_button.setEnabled(False)
        controls.addWidget(self.play_button)

        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_slider_changed)
        controls.addWidget(self.slider, stretch=1)

        self.frame_label_info = QLabel("0 / 0")
        self.frame_label_info.setFixedWidth(80)
        self.frame_label_info.setAlignment(Qt.AlignmentFlag.AlignRight)
        controls.addWidget(self.frame_label_info)

        layout.addLayout(controls)

    def load_video(self, path: Path) -> bool:
        """Load a video file."""
        self.stop()

        if self.cap is not None:
            self.cap.release()

        self.cap = cv2.VideoCapture(str(path))
        if not self.cap.isOpened():
            self.cap = None
            self.video_path = None
            self._update_ui_state()
            return False

        self.video_path = path
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.current_frame = 0

        self.slider.setRange(0, max(0, self.total_frames - 1))
        self.slider.setValue(0)

        self._update_ui_state()
        self._show_frame(0)
        return True

    def unload(self):
        """Unload current video."""
        self.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.video_path = None
        self.total_frames = 0
        self.current_frame = 0
        self.frame_label.clear()
        self._update_ui_state()

    def play(self):
        """Start playback."""
        if self.cap is None or self.is_playing:
            return
        self.is_playing = True
        interval_ms = int(1000.0 / self.fps)
        self.play_timer.start(interval_ms)
        self._update_ui_state()

    def pause(self):
        """Pause playback."""
        self.is_playing = False
        self.play_timer.stop()
        self._update_ui_state()

    def stop(self):
        """Stop playback and reset to start."""
        self.pause()
        if self.cap is not None:
            self.seek(0)

    def toggle_play(self):
        """Toggle play/pause."""
        if self.is_playing:
            self.pause()
        else:
            self.play()

    def seek(self, frame_index: int):
        """Seek to specific frame."""
        if self.cap is None:
            return
        frame_index = max(0, min(frame_index, self.total_frames - 1))
        self._show_frame(frame_index)
        self.slider.blockSignals(True)
        self.slider.setValue(frame_index)
        self.slider.blockSignals(False)

    def get_current_frame(self) -> np.ndarray | None:
        """Get the current frame as BGR array."""
        if self.cap is None:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
        ret, frame = self.cap.read()
        return frame if ret else None

    def _show_frame(self, frame_index: int):
        """Display a specific frame."""
        if self.cap is None:
            return

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = self.cap.read()

        if ret:
            self.current_frame = frame_index
            pixmap = bgr_to_pixmap(frame)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    self.frame_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.frame_label.setPixmap(scaled)
            self.frame_label_info.setText(f"{frame_index + 1} / {self.total_frames}")
            self.frame_changed.emit(frame_index, frame)

    def _advance_frame(self):
        """Advance to next frame during playback."""
        if self.current_frame >= self.total_frames - 1:
            self.pause()
            return

        next_frame = self.current_frame + 1
        self._show_frame(next_frame)
        self.slider.blockSignals(True)
        self.slider.setValue(next_frame)
        self.slider.blockSignals(False)

    def _on_slider_changed(self, value: int):
        """Handle slider value change."""
        if not self.is_playing:
            self._show_frame(value)

    def _update_ui_state(self):
        """Update UI based on current state."""
        has_video = self.cap is not None
        self.play_button.setEnabled(has_video)
        self.slider.setEnabled(has_video)
        self.play_button.setText("Pause" if self.is_playing else "Play")

        if not has_video:
            self.frame_label_info.setText("0 / 0")

    def closeEvent(self, event):
        """Clean up on close."""
        self.unload()
        super().closeEvent(event)
