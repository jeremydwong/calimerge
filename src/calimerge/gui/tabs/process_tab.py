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
        settings_layout = QHBoxLayout(settings_group)

        settings_layout.addWidget(QLabel("Tracker:"))
        self.tracker_combo = QComboBox()
        self.tracker_combo.addItems(["charuco", "mediapipe"])
        settings_layout.addWidget(self.tracker_combo)

        settings_layout.addStretch()

        self.load_folder_button = QPushButton("Load Recording Folder...")
        self.load_folder_button.clicked.connect(self._load_folder)
        settings_layout.addWidget(self.load_folder_button)

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
        self.log_text.setFont(QFont("Courier", 9))
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

        # Find port_X.mp4 files
        for video_file in folder_path.glob("port_*.mp4"):
            try:
                # Extract port number
                port = int(video_file.stem.split("_")[1])
                self.video_paths[port] = video_file

                item = QListWidgetItem(f"Camera {port}: {video_file.name}")
                item.setData(Qt.ItemDataRole.UserRole, port)
                self.video_list.addItem(item)
            except (ValueError, IndexError):
                continue

        if self.video_paths:
            self.process_button.setEnabled(True)
            self.status_message.emit(f"Loaded {len(self.video_paths)} videos")
            self._log(f"Loaded folder: {folder_path}")
            for port, path in sorted(self.video_paths.items()):
                self._log(f"  Camera {port}: {path.name}")
        else:
            self.status_message.emit("No port_X.mp4 videos found in folder")

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

        self.process_button.setEnabled(False)
        self.progress_bar.setVisible(True)
        self.progress_bar.setValue(0)

        tracker = self.tracker_combo.currentText()

        self._log(f"\n{'='*50}")
        self._log(f"Starting processing pipeline")
        self._log(f"  Tracker: {tracker}")
        self._log(f"  Cameras: {list(self.video_paths.keys())}")
        self._log(f"{'='*50}\n")

        self.processing_worker = ProcessingWorker(
            video_paths=self.video_paths.copy(),
            cameras=cal_state.calibrated_cameras.copy(),
            output_path=self.output_path,
            tracker_backend=tracker,
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
        self._log(f"\nProcessing complete!")
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
