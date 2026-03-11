"""
Process tab - tracking and triangulation.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QGroupBox,
    QComboBox,
    QFileDialog,
    QProgressBar,
    QSplitter,
    QTextEdit,
    QListWidget,
    QListWidgetItem,
    QSpinBox,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from ..state import StateManager
from ..workers import ProcessingWorker
from ..widgets.video_player import VideoPlayer


class ProcessTab(QWidget):
    """
    Processing tab for tracking and triangulation.

    Allows users to:
    - Load recording videos
    - Select tracker backend
    - Configure tracking parameters
    - Run 2D tracking and 3D triangulation
    - Export results
    """

    status_message = Signal(str)

    def __init__(self, state_manager: StateManager, parent: QWidget | None = None):
        super().__init__(parent)
        self.state_manager = state_manager

        self.processing_worker: ProcessingWorker | None = None
        self.video_paths: dict[int, Path] = {}
        self.output_path: Path | None = None
        self.frame_time_csv: Path | None = None

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Instructions
        instructions = QLabel(
            "Load recording videos and run tracking/triangulation pipeline. "
            "Extrinsic calibration must be completed first."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #888; margin-bottom: 10px;")
        layout.addWidget(instructions)

        # Settings group
        settings_group = QGroupBox("Processing Settings")
        settings_layout = QVBoxLayout(settings_group)

        # Row 1: Tracker and folder
        row1 = QHBoxLayout()

        row1.addWidget(QLabel("Tracker:"))
        self.tracker_combo = QComboBox()
        self.tracker_combo.addItems(["synthpose"])
        row1.addWidget(self.tracker_combo)

        row1.addStretch()

        self.load_folder_button = QPushButton("Load Recording Folder...")
        self.load_folder_button.clicked.connect(self._load_folder)
        row1.addWidget(self.load_folder_button)

        settings_layout.addLayout(row1)

        # Row 2: Synthpose parameters
        row2 = QHBoxLayout()

        row2.addWidget(QLabel("Max persons:"))
        self.max_persons_spin = QSpinBox()
        self.max_persons_spin.setRange(1, 10)
        self.max_persons_spin.setValue(2)
        row2.addWidget(self.max_persons_spin)

        row2.addWidget(QLabel("Device:"))
        self.device_combo = QComboBox()
        self.device_combo.addItems(["auto", "mps", "cuda", "cpu"])
        row2.addWidget(self.device_combo)

        row2.addWidget(QLabel("Batch size:"))
        self.batch_size_spin = QSpinBox()
        self.batch_size_spin.setRange(1, 64)
        self.batch_size_spin.setValue(8)
        row2.addWidget(self.batch_size_spin)

        row2.addWidget(QLabel("Skip frames:"))
        self.skip_frames_spin = QSpinBox()
        self.skip_frames_spin.setRange(1, 100)
        self.skip_frames_spin.setValue(1)
        self.skip_frames_spin.setToolTip("Process every Nth sync index (1 = all frames)")
        row2.addWidget(self.skip_frames_spin)

        row2.addStretch()
        settings_layout.addLayout(row2)

        layout.addWidget(settings_group)

        # Main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Video list
        videos_group = QGroupBox("Videos")
        videos_layout = QVBoxLayout(videos_group)

        self.video_list = QListWidget()
        self.video_list.itemClicked.connect(self._on_video_selected)
        videos_layout.addWidget(self.video_list)

        left_layout.addWidget(videos_group)

        # Processing controls
        processing_group = QGroupBox("Processing")
        processing_layout = QVBoxLayout(processing_group)

        self.process_button = QPushButton("Run Processing Pipeline")
        self.process_button.setMinimumHeight(40)
        self.process_button.clicked.connect(self._run_processing)
        self.process_button.setEnabled(False)
        processing_layout.addWidget(self.process_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        processing_layout.addWidget(self.progress_bar)

        self.step_label = QLabel("")
        self.step_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        processing_layout.addWidget(self.step_label)

        left_layout.addWidget(processing_group)

        # Log
        log_group = QGroupBox("Processing Log")
        log_layout = QVBoxLayout(log_group)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("monospace", 9))
        self.log_text.setMaximumHeight(200)
        log_layout.addWidget(self.log_text)

        left_layout.addWidget(log_group)

        # Export
        export_group = QGroupBox("Export")
        export_layout = QHBoxLayout(export_group)

        self.export_csv_button = QPushButton("Export CSV")
        self.export_csv_button.setEnabled(False)
        self.export_csv_button.clicked.connect(self._export_csv)
        export_layout.addWidget(self.export_csv_button)

        self.export_c3d_button = QPushButton("Export C3D")
        self.export_c3d_button.setEnabled(False)
        self.export_c3d_button.clicked.connect(self._export_c3d)
        export_layout.addWidget(self.export_c3d_button)

        export_layout.addStretch()

        left_layout.addWidget(export_group)

        # Live keypoint overlay (requires intrinsic + extrinsic calibration)
        live_group = QGroupBox("Live Overlay")
        live_layout = QVBoxLayout(live_group)

        self.live_overlay_button = QPushButton("Live Keypoint Projection")
        self.live_overlay_button.setToolTip(
            "Project 3D keypoints onto live camera feed (requires intrinsic + extrinsic calibration)"
        )
        self.live_overlay_button.setMinimumHeight(36)
        self.live_overlay_button.setEnabled(False)
        self.live_overlay_button.clicked.connect(self._toggle_live_overlay)
        live_layout.addWidget(self.live_overlay_button)

        self.live_status_label = QLabel("Requires intrinsic + extrinsic calibration")
        self.live_status_label.setStyleSheet("color: #888;")
        self.live_status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        live_layout.addWidget(self.live_status_label)

        left_layout.addWidget(live_group)

        splitter.addWidget(left_panel)

        # Right: video preview
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        preview_group = QGroupBox("Video Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.video_player = VideoPlayer()
        preview_layout.addWidget(self.video_player)
        right_layout.addWidget(preview_group)

        splitter.addWidget(right_panel)
        splitter.setSizes([400, 400])

        layout.addWidget(splitter)

    def _connect_signals(self):
        self.state_manager.processing_changed.connect(self._on_processing_changed)
        self.state_manager.calibration_changed.connect(self._on_calibration_changed)

    def _log(self, message: str):
        """Append to log."""
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _load_folder(self):
        """Load videos from recording folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Recording Folder")
        if not folder:
            return

        folder_path = Path(folder)
        self.output_path = folder_path
        self.video_paths.clear()
        self.video_list.clear()

        # Discover videos (handles both port{N}_{serial}.mp4 and port_N.mp4)
        from ..video_utils import discover_videos

        discovered = discover_videos(folder_path)
        for port, video_path in sorted(discovered.items()):
            self.video_paths[port] = video_path
            item = QListWidgetItem(f"Camera {port}: {video_path.name}")
            item.setData(Qt.ItemDataRole.UserRole, port)
            self.video_list.addItem(item)

        # Check for frame_time_history.csv
        csv_path = folder_path / "frame_time_history.csv"
        if csv_path.exists():
            self.frame_time_csv = csv_path
            self._log("Found frame_time_history.csv")
        else:
            self.frame_time_csv = None
            self._log("Warning: No frame_time_history.csv found")

        if self.video_paths:
            self.process_button.setEnabled(True)
            self.status_message.emit(f"Loaded {len(self.video_paths)} videos")
            self._log(f"Loaded folder: {folder_path}")
            for port, path in sorted(self.video_paths.items()):
                self._log(f"  Camera {port}: {path.name}")
        else:
            self.status_message.emit("No video files found in folder")

    def _on_video_selected(self, item: QListWidgetItem):
        """Handle video selection."""
        port = item.data(Qt.ItemDataRole.UserRole)
        if port in self.video_paths:
            self.video_player.load_video(self.video_paths[port])

    def _run_processing(self):
        """Run processing pipeline."""
        cal_state = self.state_manager.state.calibration

        # Check calibration
        if not cal_state.calibrated_cameras:
            self.status_message.emit("Extrinsic calibration required")
            return

        # Check we have videos for calibrated cameras
        missing = [
            port for port in cal_state.calibrated_cameras if port not in self.video_paths
        ]
        if missing:
            self.status_message.emit(f"Missing videos for cameras: {missing}")
            return

        # Check frame time CSV for synthpose
        tracker = self.tracker_combo.currentText()
        if tracker == "synthpose" and self.frame_time_csv is None:
            self.status_message.emit("frame_time_history.csv required for synthpose tracking")
            self._log("ERROR: frame_time_history.csv not found in recording folder")
            return

        self.process_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        self._log(f"\n{'='*50}")
        self._log("Starting processing pipeline")
        self._log(f"  Tracker: {tracker}")
        self._log(f"  Cameras: {list(self.video_paths.keys())}")
        self._log(f"  Max persons: {self.max_persons_spin.value()}")
        self._log(f"  Device: {self.device_combo.currentText()}")
        self._log(f"  Batch size: {self.batch_size_spin.value()}")
        self._log(f"  Skip frames: {self.skip_frames_spin.value()}")
        self._log(f"{'='*50}\n")

        self.processing_worker = ProcessingWorker(
            video_paths=self.video_paths.copy(),
            cameras=cal_state.calibrated_cameras.copy(),
            output_path=self.output_path,
            tracker_backend=tracker,
            frame_time_csv=self.frame_time_csv,
            device_name=self.device_combo.currentText(),
            max_persons=self.max_persons_spin.value(),
            batch_size=self.batch_size_spin.value(),
            skip_sync_indices=self.skip_frames_spin.value(),
        )
        self.processing_worker.log_message.connect(self._log)
        self.processing_worker.progress_update.connect(self._on_progress)
        self.processing_worker.processing_finished.connect(self._on_processing_done)
        self.processing_worker.error.connect(self._on_processing_error)
        self.processing_worker.finished.connect(self._on_worker_finished)
        self.processing_worker.start()

        self.state_manager.update_processing(is_processing=True)
        self.status_message.emit("Processing...")

    def _on_progress(self, step: str, progress: float):
        """Update progress."""
        self.step_label.setText(step)
        self.progress_bar.setValue(int(progress * 100))
        self.state_manager.update_processing(current_step=step, progress=progress)

    def _on_processing_done(self, output_file: Path):
        """Handle processing completion."""
        self._log("\nProcessing complete!")
        self._log(f"Output: {output_file}")
        self.export_csv_button.setEnabled(True)
        self.state_manager.update_processing(is_processing=False)
        self.status_message.emit("Processing complete")

    def _on_processing_error(self, error: str):
        """Handle processing error."""
        self._log(f"ERROR: {error}")
        self.status_message.emit(f"Processing error: {error}")

    def _on_worker_finished(self):
        """Clean up after worker."""
        self.process_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.step_label.setText("")
        self.state_manager.update_processing(is_processing=False)
        self.processing_worker = None

    def _on_processing_changed(self, processing):
        """Handle processing state change."""
        pass

    def _export_csv(self):
        """Export results as CSV."""
        if not self.output_path:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", str(self.output_path / "points_3d.csv"), "CSV (*.csv)"
        )
        if path:
            self._log(f"Exporting to {path}")
            # TODO: Implement actual export
            self.status_message.emit(f"Exported to {path}")

    def _export_c3d(self):
        """Export results as C3D."""
        if not self.output_path:
            return

        path, _ = QFileDialog.getSaveFileName(
            self, "Export C3D", str(self.output_path / "motion.c3d"), "C3D (*.c3d)"
        )
        if path:
            self._log(f"Exporting to {path}")
            # TODO: Implement C3D export
            self.status_message.emit(f"Exported to {path}")

    def _on_calibration_changed(self, cal_state):
        """Enable live overlay button when both intrinsic and extrinsic are done."""
        has_intrinsics = bool(cal_state.intrinsics)
        has_extrinsics = bool(cal_state.calibrated_cameras)

        ready = has_intrinsics and has_extrinsics
        self.live_overlay_button.setEnabled(ready)

        if ready:
            n_cams = len(cal_state.calibrated_cameras)
            self.live_status_label.setText(f"Ready ({n_cams} calibrated cameras)")
            self.live_status_label.setStyleSheet("color: #50c878;")
        elif has_intrinsics:
            self.live_status_label.setText("Intrinsics done, needs extrinsic calibration")
            self.live_status_label.setStyleSheet("color: #ffaa00;")
        else:
            self.live_status_label.setText("Requires intrinsic + extrinsic calibration")
            self.live_status_label.setStyleSheet("color: #888;")

    def _toggle_live_overlay(self):
        """Toggle live keypoint projection overlay."""
        # TODO: Implement live overlay - open camera, run pose estimation, project keypoints
        self._log("Live keypoint projection: not yet implemented")
        self.status_message.emit("Live overlay coming soon")
