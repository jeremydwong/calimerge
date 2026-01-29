"""
Record tab - synchronized video capture.
"""

from __future__ import annotations

from pathlib import Path
from datetime import datetime

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QSpinBox,
    QDoubleSpinBox,
    QProgressBar,
    QTextEdit,
    QFileDialog,
)
from PySide6.QtCore import Signal
from PySide6.QtGui import QFont

from ..state import StateManager
from ..workers import RecordingWorker
from ..widgets.camera_grid import CameraGrid


class RecordTab(QWidget):
    """
    Synchronized recording tab.

    Allows users to:
    - Configure recording duration and FPS
    - Select output directory
    - Start/stop recording
    - View progress and results
    """

    status_message = Signal(str)

    def __init__(self, state_manager: StateManager, parent: QWidget | None = None):
        super().__init__(parent)
        self.state_manager = state_manager

        self.recording_worker: RecordingWorker | None = None
        self.opened_cameras: list = []
        self.output_path: Path | None = None
        self.base_output_path = Path("recordings")

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Settings group
        settings_group = QGroupBox("Recording Settings")
        settings_layout = QHBoxLayout(settings_group)

        settings_layout.addWidget(QLabel("Duration (sec):"))
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(1.0, 3600.0)
        self.duration_spin.setValue(10.0)
        self.duration_spin.setSingleStep(1.0)
        settings_layout.addWidget(self.duration_spin)

        settings_layout.addWidget(QLabel("FPS:"))
        self.fps_spin = QSpinBox()
        self.fps_spin.setRange(1, 120)
        self.fps_spin.setValue(30)
        settings_layout.addWidget(self.fps_spin)

        settings_layout.addStretch()

        self.browse_button = QPushButton("Output Folder...")
        self.browse_button.clicked.connect(self._browse_output)
        settings_layout.addWidget(self.browse_button)

        layout.addWidget(settings_group)

        # Output path display
        self.output_label = QLabel("Output: ./recordings/<timestamp>/")
        self.output_label.setFont(QFont("Courier", 10))
        layout.addWidget(self.output_label)

        # Preview grid
        preview_group = QGroupBox("Camera Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.camera_grid = CameraGrid()
        self.camera_grid.setMinimumHeight(200)
        preview_layout.addWidget(self.camera_grid)
        layout.addWidget(preview_group)

        # Controls
        controls_layout = QHBoxLayout()

        self.start_button = QPushButton("Start Recording")
        self.start_button.setMinimumHeight(50)
        self.start_button.setStyleSheet("font-size: 14px; font-weight: bold;")
        self.start_button.clicked.connect(self._start_recording)
        controls_layout.addWidget(self.start_button)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setMinimumHeight(50)
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_recording)
        controls_layout.addWidget(self.stop_button)

        layout.addLayout(controls_layout)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        layout.addWidget(self.progress_bar)

        # Log
        log_group = QGroupBox("Recording Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("Courier", 10))
        self.log_text.setMaximumHeight(150)
        log_layout.addWidget(self.log_text)
        layout.addWidget(log_group)

        # Results
        results_group = QGroupBox("Recording Results")
        results_layout = QVBoxLayout(results_group)
        self.results_label = QLabel("No recording yet")
        self.results_label.setFont(QFont("Courier", 11))
        results_layout.addWidget(self.results_label)
        layout.addWidget(results_group)

    def _connect_signals(self):
        self.state_manager.recording_changed.connect(self._on_recording_changed)

    def _browse_output(self):
        """Select output folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder")
        if folder:
            self.base_output_path = Path(folder)
            self.output_label.setText(f"Output: {folder}/<timestamp>/")

    def _log(self, message: str):
        """Append to log."""
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _start_recording(self):
        """Start recording."""
        from ..camera_binding import open_camera, close_camera

        cameras = self.state_manager.state.cameras

        # Check for enabled cameras
        enabled_cameras = [c for c in cameras.values() if c.enabled]
        if not enabled_cameras:
            self.status_message.emit("No cameras enabled")
            return

        # Open cameras
        self.opened_cameras = []
        camera_info = {}

        for port, cam_state in cameras.items():
            if not cam_state.enabled:
                continue
            try:
                open_camera(cam_state.info)
                self.opened_cameras.append(cam_state.info)
                camera_info[port] = cam_state.info.display_name
                self._log(f"Opened camera {port}: {cam_state.info.display_name}")
            except Exception as e:
                self._log(f"Failed to open camera {port}: {e}")

        if not self.opened_cameras:
            self.status_message.emit("No cameras could be opened")
            return

        # Set up preview grid
        self.camera_grid.set_cameras(camera_info)

        # Create output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_path = self.base_output_path / timestamp
        self.output_path.mkdir(parents=True, exist_ok=True)

        duration = self.duration_spin.value()
        fps = self.fps_spin.value()

        self._log(f"\n{'='*50}")
        self._log(f"Starting recording")
        self._log(f"  Duration: {duration}s")
        self._log(f"  FPS: {fps}")
        self._log(f"  Output: {self.output_path}")
        self._log(f"{'='*50}\n")

        # Update UI
        self.start_button.setEnabled(False)
        self.stop_button.setEnabled(True)
        self.duration_spin.setEnabled(False)
        self.fps_spin.setEnabled(False)
        self.browse_button.setEnabled(False)

        total_frames = int(duration * fps)
        self.progress_bar.setMaximum(total_frames)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        # Start worker
        self.recording_worker = RecordingWorker(
            self.opened_cameras, self.output_path, duration, fps
        )
        self.recording_worker.log_message.connect(self._log)
        self.recording_worker.progress_update.connect(self._on_progress)
        self.recording_worker.recording_finished.connect(self._on_finished)
        self.recording_worker.error.connect(self._on_error)
        self.recording_worker.start()

        self.state_manager.update_recording(is_recording=True, output_path=self.output_path)
        self.status_message.emit("Recording...")

    def _stop_recording(self):
        """Stop recording."""
        if self.recording_worker:
            self._log("\nStopping recording...")
            self.recording_worker.stop()

    def _on_progress(self, current: int, total: int):
        """Update progress bar."""
        self.progress_bar.setValue(current)
        self.state_manager.update_recording(current_frame=current, total_frames=total)

    def _on_finished(self, stats: dict):
        """Handle recording finished."""
        from ..camera_binding import close_camera

        # Close cameras
        for cam in self.opened_cameras:
            try:
                close_camera(cam)
            except Exception:
                pass
        self.opened_cameras = []

        # Update UI
        self.start_button.setEnabled(True)
        self.stop_button.setEnabled(False)
        self.duration_spin.setEnabled(True)
        self.fps_spin.setEnabled(True)
        self.browse_button.setEnabled(True)
        self.progress_bar.setVisible(False)

        self._log(f"\n{'='*50}")
        self._log("Recording complete!")
        self._log(f"  Sync frames: {stats['sync_count']}")
        self._log(f"  Duration: {stats['duration']:.2f}s")
        for cam_idx, cam_stats in stats["cameras"].items():
            self._log(
                f"  Camera {cam_idx}: {cam_stats['frame_count']} frames -> {cam_stats['video_file']}"
            )
        self._log(f"\nOutput: {self.output_path}")
        self._log(f"{'='*50}")

        # Update results
        result_lines = [
            f"Sync frames: {stats['sync_count']}",
            f"Duration: {stats['duration']:.2f}s",
            f"Output: {self.output_path}",
            "",
            "Files:",
        ]
        for cam_idx, cam_stats in stats["cameras"].items():
            result_lines.append(
                f"  {cam_stats['video_file']} ({cam_stats['frame_count']} frames)"
            )
        result_lines.append("  frame_time_history.csv")

        self.results_label.setText("\n".join(result_lines))

        self.state_manager.update_recording(is_recording=False)
        self.status_message.emit("Recording complete")

    def _on_error(self, error: str):
        """Handle recording error."""
        self._log(f"ERROR: {error}")
        self._on_finished({"sync_count": 0, "duration": 0, "cameras": {}})
        self.status_message.emit(f"Recording error: {error}")

    def _on_recording_changed(self, recording):
        """Handle recording state change."""
        pass  # UI already updated by callbacks
