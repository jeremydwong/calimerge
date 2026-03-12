"""
Extrinsic calibration tab - multi-camera spatial calibration.
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
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QFileDialog,
    QProgressBar,
    QSplitter,
    QTextEdit,
    QSpinBox,
    QDoubleSpinBox,
    QComboBox,
    QCheckBox,
    QSizePolicy,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from ..state import StateManager
from ..workers import ExtrinsicCalibrationWorker
from ..widgets.video_player import VideoPlayer
from ..frame_utils import bgr_to_pixmap
from ..colors import camera_color


class ExtrinsicTab(QWidget):
    """
    Extrinsic calibration tab.

    Allows users to:
    - Load synchronized calibration videos
    - Verify intrinsics are available
    - Configure ChArUco board and see a live preview
    - Run bundle adjustment
    - View and export camera positions
    """

    status_message = Signal(str)

    def __init__(self, state_manager: StateManager, parent: QWidget | None = None):
        super().__init__(parent)
        self.state_manager = state_manager

        self.calibration_worker: ExtrinsicCalibrationWorker | None = None
        self.video_paths: dict[int, Path] = {}
        self.frame_time_csv: Path | None = None
        # Collected detection frames: dict[sync_key] -> dict[port -> BGR frame with overlay]
        self._detection_groups: dict[int, dict[int, any]] = {}
        self._detection_order: list[int] = []  # ordered sync keys

        self._init_ui()
        self._connect_signals()
        self._update_charuco_preview()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Instructions
        instructions = QLabel(
            "Load synchronized videos of ChArUco board captured from multiple cameras. "
            "Intrinsic calibration must be completed first for all cameras."
        )
        instructions.setWordWrap(True)
        instructions.setStyleSheet("color: #888; margin-bottom: 10px;")
        layout.addWidget(instructions)

        # Top row: ChArUco settings + board preview
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left top: ChArUco settings
        charuco_group = QGroupBox("ChArUco Board (Extrinsic)")
        charuco_layout = QVBoxLayout(charuco_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Columns:"))
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(3, 20)
        self.cols_spin.setValue(4)
        self.cols_spin.valueChanged.connect(self._update_charuco_preview)
        row1.addWidget(self.cols_spin)

        row1.addWidget(QLabel("Rows:"))
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(3, 20)
        self.rows_spin.setValue(3)
        self.rows_spin.valueChanged.connect(self._update_charuco_preview)
        row1.addWidget(self.rows_spin)
        charuco_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Square (cm):"))
        self.square_spin = QDoubleSpinBox()
        self.square_spin.setRange(1.0, 30.0)
        self.square_spin.setValue(5.0)
        self.square_spin.setSingleStep(0.5)
        self.square_spin.setDecimals(1)
        self.square_spin.valueChanged.connect(self._update_charuco_preview)
        row2.addWidget(self.square_spin)

        row2.addWidget(QLabel("Dictionary:"))
        self.dict_combo = QComboBox()
        self.dict_combo.addItems([
            "DICT_4X4_50",
            "DICT_4X4_100",
            "DICT_5X5_50",
            "DICT_5X5_100",
            "DICT_6X6_50",
        ])
        self.dict_combo.setCurrentIndex(0)
        self.dict_combo.setEnabled(False)  # Fixed to DICT_4X4_50 for now
        self.dict_combo.setToolTip("Dictionary is fixed to DICT_4X4_50 for now")
        self.dict_combo.currentIndexChanged.connect(self._update_charuco_preview)
        row2.addWidget(self.dict_combo)
        charuco_layout.addLayout(row2)

        row3 = QHBoxLayout()
        self.inverted_checkbox = QCheckBox("Inverted (white markers on black)")
        self.inverted_checkbox.setToolTip(
            "Check if your ChArUco board has white markers on a black background"
        )
        self.inverted_checkbox.stateChanged.connect(self._update_charuco_preview)
        row3.addWidget(self.inverted_checkbox)
        row3.addStretch()
        charuco_layout.addLayout(row3)

        charuco_layout.addStretch()
        top_splitter.addWidget(charuco_group)

        # Right top: board preview
        preview_group = QGroupBox("Board Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.board_preview_label = QLabel()
        self.board_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.board_preview_label.setMinimumSize(200, 150)
        self.board_preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.board_preview_label.setStyleSheet("background-color: #1a1a1a;")
        preview_layout.addWidget(self.board_preview_label)
        top_splitter.addWidget(preview_group)

        top_splitter.setSizes([400, 300])

        # Outer vertical splitter: charuco row on top, main content below
        outer_splitter = QSplitter(Qt.Orientation.Vertical)
        outer_splitter.addWidget(top_splitter)

        # Main horizontal splitter: left panels | video preview
        main_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: vertical splitter for cameras / calibration / results
        left_splitter = QSplitter(Qt.Orientation.Vertical)

        # Camera table
        cameras_group = QGroupBox("Cameras (Enabled Only)")
        cameras_layout = QVBoxLayout(cameras_group)

        self.camera_table = QTableWidget()
        self.camera_table.setColumnCount(6)
        self.camera_table.setHorizontalHeaderLabels(
            ["", "Port", "Camera", "Intrinsics", "Video", "Status"]
        )
        header = self.camera_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(0, 30)  # Color
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(1, 40)  # Port
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)  # Camera name
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        header.resizeSection(3, 70)  # Intrinsics
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Interactive)
        header.resizeSection(4, 120)  # Video
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Interactive)
        header.resizeSection(5, 90)  # Status
        self.camera_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.camera_table.itemSelectionChanged.connect(self._on_camera_selected)
        cameras_layout.addWidget(self.camera_table)

        # Video controls
        video_controls = QHBoxLayout()

        self.load_folder_button = QPushButton("Load Video Folder...")
        self.load_folder_button.clicked.connect(self._load_video_folder)
        video_controls.addWidget(self.load_folder_button)

        self.load_single_button = QPushButton("Load Single Video...")
        self.load_single_button.clicked.connect(self._load_single_video)
        video_controls.addWidget(self.load_single_button)

        cameras_layout.addLayout(video_controls)
        left_splitter.addWidget(cameras_group)

        # Calibration controls
        calibration_group = QGroupBox("Calibration")
        calibration_layout = QVBoxLayout(calibration_group)

        self.calibrate_button = QPushButton("Run Extrinsic Calibration")
        self.calibrate_button.setMinimumHeight(40)
        self.calibrate_button.clicked.connect(self._run_calibration)
        calibration_layout.addWidget(self.calibrate_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        calibration_layout.addWidget(self.progress_bar)

        left_splitter.addWidget(calibration_group)

        # Results
        results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(results_group)

        self.results_text = QTextEdit()
        self.results_text.setReadOnly(True)
        self.results_text.setFont(QFont("monospace", 10))
        self.results_text.setPlaceholderText("Calibration results will appear here...")
        results_layout.addWidget(self.results_text)

        export_layout = QHBoxLayout()
        self.export_button = QPushButton("Export Camera Rig")
        self.export_button.setEnabled(False)
        self.export_button.clicked.connect(self._export_rig)
        export_layout.addWidget(self.export_button)
        export_layout.addStretch()

        results_layout.addLayout(export_layout)
        left_splitter.addWidget(results_group)

        left_splitter.setSizes([200, 60, 350])
        main_splitter.addWidget(left_splitter)

        # Right: video preview
        video_preview_group = QGroupBox("Extrinsic Cal Results")
        video_preview_layout = QVBoxLayout(video_preview_group)
        self.video_player = VideoPlayer()
        video_preview_layout.addWidget(self.video_player)

        main_splitter.addWidget(video_preview_group)
        main_splitter.setSizes([400, 400])

        outer_splitter.addWidget(main_splitter)
        outer_splitter.setSizes([150, 500])

        layout.addWidget(outer_splitter, stretch=1)

    def _connect_signals(self):
        self.state_manager.cameras_changed.connect(self._on_cameras_changed)
        self.state_manager.calibration_changed.connect(self._on_calibration_changed)

    # ── Project settings ──

    def apply_project_settings(self, settings: dict) -> None:
        """Apply loaded project settings to this tab's UI controls."""
        ext = settings.get("charuco_extrinsic", {})
        if "columns" in ext:
            self.cols_spin.setValue(int(ext["columns"]))
        if "rows" in ext:
            self.rows_spin.setValue(int(ext["rows"]))
        if "square_size_cm" in ext:
            self.square_spin.setValue(float(ext["square_size_cm"]))
        if "inverted" in ext:
            self.inverted_checkbox.setChecked(bool(ext["inverted"]))

    def get_project_settings(self) -> dict:
        """Return this tab's contribution to the project settings dict."""
        return {
            "charuco_extrinsic": {
                "columns": self.cols_spin.value(),
                "rows": self.rows_spin.value(),
                "square_size_cm": self.square_spin.value(),
                "inverted": self.inverted_checkbox.isChecked(),
            },
        }

    # ── ChArUco preview ──

    def _update_charuco_preview(self):
        """Regenerate and display the charuco board preview."""
        try:
            from ...calibration.charuco import generate_board_image

            config = self._get_charuco_config()
            cols, rows = config.columns, config.rows
            scale = 80
            img = generate_board_image(config, width=cols * scale, height=rows * scale)
            pixmap = bgr_to_pixmap(img)
            if not pixmap.isNull():
                scaled = pixmap.scaled(
                    self.board_preview_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.board_preview_label.setPixmap(scaled)
        except Exception:
            self.board_preview_label.setText("Preview unavailable")

    # ── Helpers ──

    @staticmethod
    def _camera_label(cam_state) -> str:
        """Format a camera label as 'nickname (serial)' or just 'serial'."""
        serial = cam_state.info.serial_number
        nick = getattr(cam_state, "nickname", "")
        if nick:
            return f"{nick} ({serial})"
        return serial

    # ── Camera table ──

    def _on_cameras_changed(self, cameras: dict):
        """Update camera table - only shows enabled cameras."""
        cal_state = self.state_manager.state.calibration

        # Filter to enabled cameras only
        enabled_cameras = {
            port: cam_state
            for port, cam_state in cameras.items()
            if cam_state.enabled
        }

        self.camera_table.setRowCount(len(enabled_cameras))

        for row, (port, cam_state) in enumerate(sorted(enabled_cameras.items())):
            # Color indicator (column 0)
            color = camera_color(port)
            color_widget = QWidget()
            color_widget.setFixedSize(20, 20)
            color_widget.setStyleSheet(
                f"background-color: {color.name()}; border-radius: 3px;"
            )
            color_container = QWidget()
            color_layout = QHBoxLayout(color_container)
            color_layout.addWidget(color_widget)
            color_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
            color_layout.setContentsMargins(4, 2, 4, 2)
            self.camera_table.setCellWidget(row, 0, color_container)

            # Port (column 1)
            port_item = QTableWidgetItem(str(port))
            port_item.setData(Qt.ItemDataRole.UserRole, port)
            port_item.setFlags(port_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.camera_table.setItem(row, 1, port_item)

            # Camera name + FULL serial (column 2)
            serial = cam_state.info.serial_number
            label = self._camera_label(cam_state)
            name_text = f"{label}\n{cam_state.info.display_name}"
            name_item = QTableWidgetItem(name_text)
            name_item.setData(Qt.ItemDataRole.UserRole, port)
            name_item.setData(Qt.ItemDataRole.UserRole + 1, serial)  # Store serial for lookup
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.camera_table.setItem(row, 2, name_item)

            # Intrinsics status (column 3) - use serial for lookup
            has_intrinsics = serial in cal_state.intrinsics
            intrinsics_text = "Ready" if has_intrinsics else "Missing"
            intrinsics_item = QTableWidgetItem(intrinsics_text)
            intrinsics_item.setFlags(intrinsics_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if not has_intrinsics:
                intrinsics_item.setForeground(Qt.GlobalColor.red)
            self.camera_table.setItem(row, 3, intrinsics_item)

            # Video path (column 4)
            video_path = self.video_paths.get(port)
            video_text = video_path.name if video_path else "Not loaded"
            video_item = QTableWidgetItem(video_text)
            video_item.setFlags(video_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.camera_table.setItem(row, 4, video_item)

            # Calibration status (column 5)
            if port in cal_state.calibrated_cameras:
                status = "Calibrated"
            else:
                status = "Pending"
            status_item = QTableWidgetItem(status)
            status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.camera_table.setItem(row, 5, status_item)

    def _on_calibration_changed(self, calibration):
        """Refresh table on calibration change."""
        cameras = self.state_manager.state.cameras
        if cameras:
            self._on_cameras_changed(cameras)

        # Update results
        if calibration.calibrated_cameras:
            self._show_results(calibration.calibrated_cameras, calibration.extrinsic_error)
            self.export_button.setEnabled(True)
        else:
            self.export_button.setEnabled(False)

    def _get_selected_port(self) -> int | None:
        """Get port of selected camera."""
        items = self.camera_table.selectedItems()
        if not items:
            return None
        row = items[0].row()
        # Port is stored in column 1
        item = self.camera_table.item(row, 1)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _on_camera_selected(self):
        """Handle camera selection."""
        port = self._get_selected_port()
        if port is None:
            return

        if port in self.video_paths:
            self.video_player.load_video(self.video_paths[port])
        else:
            self.video_player.unload()

    # ── Video loading ──

    def _load_video_folder(self):
        """Load videos from folder."""
        from ..video_utils import find_video_for_port

        folder = QFileDialog.getExistingDirectory(
            self, "Select Video Folder"
        )
        if not folder:
            return

        folder_path = Path(folder)
        cameras = self.state_manager.state.cameras
        loaded = 0

        for port, cam_state in cameras.items():
            serial = getattr(cam_state.info, "serial_number", None)
            video_path = find_video_for_port(folder_path, port, serial)
            if video_path:
                self.video_paths[port] = video_path
                loaded += 1

        # Check for frame_time_history.csv
        csv_path = folder_path / "frame_time_history.csv"
        if csv_path.exists():
            self.frame_time_csv = csv_path

        self._on_cameras_changed(cameras)
        self.status_message.emit(f"Loaded {loaded} videos from folder")

    def _load_single_video(self):
        """Load video for selected camera."""
        port = self._get_selected_port()
        if port is None:
            self.status_message.emit("Select a camera first")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Select Video", "", "Videos (*.mp4 *.avi *.mov)"
        )
        if path:
            self.video_paths[port] = Path(path)
            self.video_player.load_video(Path(path))
            self._on_cameras_changed(self.state_manager.state.cameras)
            cam = self.state_manager.state.cameras.get(port)
            label = self._camera_label(cam) if cam else str(port)
            self.status_message.emit(f"Loaded video for {label}")

    # ── Calibration ──

    def _get_charuco_config(self):
        """Get current ChArUco configuration for extrinsic calibration."""
        from ...types import CharucoConfig

        return CharucoConfig(
            columns=self.cols_spin.value(),
            rows=self.rows_spin.value(),
            square_size_cm=self.square_spin.value(),
            dictionary=self.dict_combo.currentText(),
            inverted=self.inverted_checkbox.isChecked(),
        )

    def _run_calibration(self):
        """Run extrinsic calibration."""
        cameras = self.state_manager.state.cameras
        cal_state = self.state_manager.state.calibration

        # Only calibrate enabled cameras
        enabled_ports = [port for port, cam in cameras.items() if cam.enabled]

        if len(enabled_ports) < 2:
            self.status_message.emit("Need at least 2 enabled cameras for extrinsic calibration")
            return

        # Check all enabled cameras have intrinsics (using serial number lookup)
        missing_intrinsics = []
        for port in enabled_ports:
            cam = cameras[port]
            if cam.info.serial_number not in cal_state.intrinsics:
                missing_intrinsics.append(self._camera_label(cam))
        if missing_intrinsics:
            self.status_message.emit(
                f"Missing intrinsics for: {', '.join(missing_intrinsics)}"
            )
            return

        # Check all enabled cameras have videos
        missing_videos = [
            self._camera_label(cameras[port])
            for port in enabled_ports if port not in self.video_paths
        ]
        if missing_videos:
            self.status_message.emit(f"Missing videos for: {', '.join(missing_videos)}")
            return

        charuco_config = self._get_charuco_config()

        self.calibrate_button.setEnabled(False)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)
        self.results_text.clear()
        self._detection_groups.clear()
        self._detection_order.clear()

        # Build port->intrinsics mapping from serial-based storage
        port_to_intrinsics = {}
        for port in enabled_ports:
            serial = cameras[port].info.serial_number
            if serial in cal_state.intrinsics:
                port_to_intrinsics[port] = cal_state.intrinsics[serial]

        self.calibration_worker = ExtrinsicCalibrationWorker(
            video_paths=self.video_paths.copy(),
            intrinsics=port_to_intrinsics,
            charuco_config=charuco_config,
            frame_time_csv=self.frame_time_csv,
        )
        self.calibration_worker.log_message.connect(
            lambda msg: self.results_text.append(msg)
        )
        self.calibration_worker.progress_update.connect(
            lambda p: self.progress_bar.setValue(int(p * 100))
        )
        self.calibration_worker.detection_frame.connect(self._on_detection_frame)
        self.calibration_worker.calibration_finished.connect(self._on_calibration_done)
        self.calibration_worker.error.connect(self._on_calibration_error)
        self.calibration_worker.finished.connect(self._on_worker_finished)
        self.calibration_worker.start()

        self.status_message.emit("Running extrinsic calibration...")

    def _on_detection_frame(self, port: int, frame_index: int, frame, corner_count: int):
        """Collect detection overlay frame grouped by sync key (frame_index)."""
        if frame_index not in self._detection_groups:
            self._detection_groups[frame_index] = {}
            self._detection_order.append(frame_index)
        self._detection_groups[frame_index][port] = frame.copy()

        # Show latest frame live during calibration
        pixmap = bgr_to_pixmap(frame)
        if pixmap is not None:
            scaled = pixmap.scaled(
                self.video_player.frame_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.video_player.frame_label.setPixmap(scaled)
            total_detections = sum(len(g) for g in self._detection_groups.values())
            self.video_player.frame_label_info.setText(
                f"{total_detections} detections"
            )

    def _on_calibration_done(self, cameras: dict, error: float):
        """Handle calibration completion - auto-saves to project directory."""
        self.state_manager.update_calibration(
            calibrated_cameras=cameras, extrinsic_error=error
        )

        # Auto-save to project directory if videos were loaded from a folder
        saved_path = None
        if self.video_paths:
            # Use the parent of any video path as the output directory
            first_video = next(iter(self.video_paths.values()))
            output_dir = first_video.parent
            calibration_file = output_dir / "calibration.toml"

            try:
                from ...config import save_calibration_to_toml
                save_calibration_to_toml(cameras, calibration_file)
                saved_path = calibration_file
                self.results_text.append(f"\nSaved extrinsic calibration to:\n  {calibration_file}")
            except Exception as e:
                self.results_text.append(f"\nFailed to save calibration: {e}")

        # Save charuco board PDF at exact dimensions
        self._save_charuco_pdf()

        if saved_path:
            self.status_message.emit(f"Extrinsic calibration complete (error: {error:.4f}), saved to {saved_path.name}")
        else:
            self.status_message.emit(f"Extrinsic calibration complete, error: {error:.4f}")

    def _save_charuco_pdf(self) -> None:
        """Save a PDF of the extrinsic charuco board at exact physical dimensions."""
        try:
            from ...calibration.charuco import create_charuco_pdf
            main_window = self.window()
            if hasattr(main_window, "cameras_tab"):
                folder = main_window.cameras_tab.base_output_path
            else:
                folder = None
            if folder is None:
                return
            folder.mkdir(parents=True, exist_ok=True)
            config = self._get_charuco_config()
            style = "W" if config.inverted else "K"
            size_str = str(config.square_size_cm).replace(".", "p")
            pdf_path = folder / f"charuco_c{config.columns}r{config.rows}s{size_str}cm_{style}.pdf"
            create_charuco_pdf(config, pdf_path)
            self.results_text.append(f"Saved charuco board PDF: {pdf_path}")
        except Exception as e:
            self.results_text.append(f"Failed to save charuco PDF: {e}")

    def _on_calibration_error(self, error: str):
        """Handle calibration error."""
        self.status_message.emit(f"Calibration failed: {error}")
        self.results_text.append(f"ERROR: {error}")

    def _on_worker_finished(self):
        """Clean up after worker and load detection frames into player."""
        self.calibrate_button.setEnabled(True)
        self.progress_bar.setVisible(False)
        self.calibration_worker = None

        # Build side-by-side composite frames from synced detection groups
        if self._detection_groups:
            self._load_detection_composites()

    def _load_detection_composites(self):
        """Compose per-sync-index grid of camera frames and load into video player."""
        import cv2
        import numpy as np
        import math

        composites = []
        ports = sorted({p for g in self._detection_groups.values() for p in g})
        n_cams = len(ports)
        if n_cams == 0:
            return

        # Grid layout: e.g. 2 cams = 1x2, 3 cams = 2x2, 4 cams = 2x2
        cols = math.ceil(math.sqrt(n_cams))
        rows = math.ceil(n_cams / cols)

        # Look up camera labels
        cameras = self.state_manager.state.cameras
        port_labels = {}
        for port in ports:
            cam = cameras.get(port)
            if cam:
                port_labels[port] = self._camera_label(cam)
            else:
                port_labels[port] = f"Port {port}"

        # Target tile size: scale each camera frame to a fixed size for the grid
        sample = next(iter(next(iter(self._detection_groups.values())).values()))
        tile_h = sample.shape[0]
        tile_w = sample.shape[1]

        pad = 4  # border between tiles

        for sync_key in self._detection_order:
            group = self._detection_groups[sync_key]
            if not group:
                continue

            grid_h = rows * tile_h + (rows - 1) * pad
            grid_w = cols * tile_w + (cols - 1) * pad
            grid = np.full((grid_h, grid_w, 3), 30, dtype=np.uint8)

            for idx, port in enumerate(ports):
                r = idx // cols
                c = idx % cols
                y = r * (tile_h + pad)
                x = c * (tile_w + pad)

                if port in group:
                    f = group[port].copy()
                else:
                    f = np.full((tile_h, tile_w, 3), 40, dtype=np.uint8)

                # Resize if needed
                if f.shape[0] != tile_h or f.shape[1] != tile_w:
                    f = cv2.resize(f, (tile_w, tile_h))

                # Camera label at top-left
                cv2.putText(
                    f, port_labels[port], (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1, cv2.LINE_AA,
                )

                # "no detection" for missing cameras
                if port not in group:
                    cv2.putText(
                        f, "no detection", (8, tile_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA,
                    )

                grid[y:y + tile_h, x:x + tile_w] = f

            composites.append(grid)

        if composites:
            self.video_player.load_frames(composites, fps=3.0)

    def _calibrated_camera_label(self, serial: str) -> str:
        """Look up nickname for a serial from live camera state, return 'nick (serial)' or 'serial'."""
        for cam_state in self.state_manager.state.cameras.values():
            if cam_state.info.serial_number == serial:
                nick = getattr(cam_state, "nickname", "")
                if nick:
                    return f"{nick} ({serial})"
                break
        return serial

    def _show_results(self, cameras: dict, error: float | None):
        """Display calibration results with camera positions and pairwise distances."""
        import numpy as np

        self.results_text.append("\n=== Extrinsic Calibration Results ===\n")

        if error is not None:
            self.results_text.append(f"Reprojection error: {error:.4f}")

        # Show camera positions
        self.results_text.append("\n--- Camera Positions (meters) ---")
        for port, cam in sorted(cameras.items()):
            t = cam.extrinsics.translation
            dist_from_origin = np.linalg.norm(t)
            label = self._calibrated_camera_label(cam.serial_number)
            self.results_text.append(
                f"{label}: "
                f"[{t[0]:.3f}, {t[1]:.3f}, {t[2]:.3f}] "
                f"({dist_from_origin:.2f}m from origin)"
            )

        # Show pairwise distances between cameras
        if len(cameras) >= 2:
            self.results_text.append("\n--- Camera Distances ---")
            sorted_ports = sorted(cameras.keys())
            for i, port_a in enumerate(sorted_ports):
                label_a = self._calibrated_camera_label(cameras[port_a].serial_number)
                for port_b in sorted_ports[i + 1:]:
                    label_b = self._calibrated_camera_label(cameras[port_b].serial_number)
                    t_a = cameras[port_a].extrinsics.translation
                    t_b = cameras[port_b].extrinsics.translation
                    distance = np.linalg.norm(t_a - t_b)
                    self.results_text.append(
                        f"  {label_a} <-> {label_b}: {distance:.3f}m"
                    )

    def _export_rig(self):
        """Export camera rig to file."""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Camera Rig", "camera_rig.toml", "TOML (*.toml)"
        )
        if not path:
            return

        cal_state = self.state_manager.state.calibration
        if not cal_state.calibrated_cameras:
            return

        try:
            import rtoml

            rig_data = {}
            for port, cam in cal_state.calibrated_cameras.items():
                rig_data[f"camera_{port}"] = {
                    "serial_number": cam.serial_number,
                    "translation": cam.extrinsics.translation.tolist(),
                    "rotation": cam.extrinsics.rotation.flatten().tolist(),
                }

            with open(path, "w") as f:
                rtoml.dump(rig_data, f)

            self.status_message.emit(f"Exported camera rig to {path}")
        except Exception as e:
            self.status_message.emit(f"Export failed: {e}")

    def resizeEvent(self, event):
        """Update charuco preview on resize."""
        super().resizeEvent(event)
        self._update_charuco_preview()
