"""
Intrinsic calibration tab - per-camera lens calibration.
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
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QFileDialog,
    QProgressBar,
    QSplitter,
    QComboBox,
    QSizePolicy,
    QTextEdit,
)
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont

from ..state import StateManager, CameraState
from ..workers import IntrinsicCalibrationWorker
from ..widgets.video_player import VideoPlayer
from ..frame_utils import bgr_to_pixmap
from ..colors import camera_color


class IntrinsicTab(QWidget):
    """
    Intrinsic calibration tab.

    Allows users to:
    - Configure ChArUco board parameters and see a preview
    - Load calibration videos per camera or from a folder
    - Run intrinsic calibration with live detection visualization
    - View and save results
    """

    status_message = Signal(str)

    def __init__(self, state_manager: StateManager, parent: QWidget | None = None):
        super().__init__(parent)
        self.state_manager = state_manager

        self.calibration_workers: dict[int, IntrinsicCalibrationWorker] = {}
        self.video_paths: dict[int, Path] = {}

        self._init_ui()
        self._connect_signals()
        self._update_charuco_preview()

    def _init_ui(self):
        layout = QVBoxLayout(self)

        # Outer vertical splitter: top config vs main content
        outer_splitter = QSplitter(Qt.Orientation.Vertical)

        # Top row: ChArUco settings + board preview (horizontal splitter)
        top_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: ChArUco settings
        charuco_group = QGroupBox("ChArUco Board Settings")
        charuco_layout = QVBoxLayout(charuco_group)

        row1 = QHBoxLayout()
        row1.addWidget(QLabel("Columns:"))
        self.cols_spin = QSpinBox()
        self.cols_spin.setRange(3, 20)
        self.cols_spin.setValue(7)
        self.cols_spin.valueChanged.connect(self._update_charuco_preview)
        row1.addWidget(self.cols_spin)

        row1.addWidget(QLabel("Rows:"))
        self.rows_spin = QSpinBox()
        self.rows_spin.setRange(3, 20)
        self.rows_spin.setValue(5)
        self.rows_spin.valueChanged.connect(self._update_charuco_preview)
        row1.addWidget(self.rows_spin)
        charuco_layout.addLayout(row1)

        row2 = QHBoxLayout()
        row2.addWidget(QLabel("Square (cm):"))
        self.square_spin = QDoubleSpinBox()
        self.square_spin.setRange(0.5, 20.0)
        self.square_spin.setValue(3.0)
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
        self.inverted_checkbox = QCheckBox("Inverted (white on black)")
        self.inverted_checkbox.setToolTip(
            "Check this if your printed board has white markers on a black background"
        )
        self.inverted_checkbox.stateChanged.connect(self._update_charuco_preview)
        row3.addWidget(self.inverted_checkbox)
        row3.addStretch()
        charuco_layout.addLayout(row3)

        charuco_layout.addStretch()
        top_splitter.addWidget(charuco_group)

        # Right: Board preview (resizable)
        preview_group = QGroupBox("Board Preview")
        preview_layout = QVBoxLayout(preview_group)
        self.board_preview_label = QLabel()
        self.board_preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.board_preview_label.setMinimumSize(160, 120)
        self.board_preview_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.board_preview_label.setStyleSheet("background-color: #1a1a1a;")
        preview_layout.addWidget(self.board_preview_label)
        top_splitter.addWidget(preview_group)

        top_splitter.setSizes([400, 300])
        outer_splitter.addWidget(top_splitter)

        # Main content splitter (horizontal: camera list vs preview)
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: camera list and controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)

        # Camera table
        cameras_group = QGroupBox("Cameras (Enabled Only)")
        cameras_layout = QVBoxLayout(cameras_group)

        self.camera_table = QTableWidget()
        self.camera_table.setColumnCount(7)
        self.camera_table.setHorizontalHeaderLabels(
            ["", "Port", "Camera", "Serial", "Video", "Status", "Error"]
        )
        header = self.camera_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(0, 30)  # Color
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(1, 40)  # Port
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.Interactive)
        header.resizeSection(2, 120)  # Camera name
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Interactive)
        header.resizeSection(3, 140)  # Serial
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)  # Video
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(5, 100)  # Status
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Fixed)
        header.resizeSection(6, 70)  # Error
        self.camera_table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        self.camera_table.itemSelectionChanged.connect(self._on_camera_selected)
        cameras_layout.addWidget(self.camera_table)

        # Camera controls
        camera_controls = QHBoxLayout()

        self.load_video_button = QPushButton("Load Video...")
        self.load_video_button.clicked.connect(self._load_video)
        camera_controls.addWidget(self.load_video_button)

        self.load_folder_button = QPushButton("Load Intrinsic Folder...")
        self.load_folder_button.setToolTip(
            "Load a folder with videos named port0_serial.mp4 or port_0.mp4"
        )
        self.load_folder_button.clicked.connect(self._load_folder)
        camera_controls.addWidget(self.load_folder_button)

        cameras_layout.addLayout(camera_controls)

        cal_controls = QHBoxLayout()
        self.calibrate_button = QPushButton("Calibrate Selected")
        self.calibrate_button.clicked.connect(self._calibrate_selected)
        cal_controls.addWidget(self.calibrate_button)

        self.calibrate_all_button = QPushButton("Calibrate All")
        self.calibrate_all_button.clicked.connect(self._calibrate_all)
        cal_controls.addWidget(self.calibrate_all_button)
        cameras_layout.addLayout(cal_controls)

        left_layout.addWidget(cameras_group)

        # Progress
        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        left_layout.addWidget(self.progress_bar)

        # Results
        results_group = QGroupBox("Calibration Results")
        results_layout = QVBoxLayout(results_group)
        self.results_label = QLabel("Select a camera to view results")
        self.results_label.setFont(QFont("monospace", 10))
        self.results_label.setWordWrap(True)
        results_layout.addWidget(self.results_label)

        self.save_button = QPushButton("Save to Database")
        self.save_button.setEnabled(False)
        self.save_button.clicked.connect(self._save_intrinsics)
        results_layout.addWidget(self.save_button)

        left_layout.addWidget(results_group)

        splitter.addWidget(left_panel)

        # Right: video preview + detection log
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        video_group = QGroupBox("Video / Detection Preview")
        video_layout = QVBoxLayout(video_group)

        # Detection frame display (shows frames with overlaid charuco corners)
        self.detection_label = QLabel()
        self.detection_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detection_label.setMinimumSize(320, 240)
        self.detection_label.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        self.detection_label.setStyleSheet("background-color: #1a1a1a;")
        video_layout.addWidget(self.detection_label, stretch=1)

        # Detection info label
        self.detection_info = QLabel("")
        self.detection_info.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.detection_info.setStyleSheet("color: #aaa; font-size: 11px;")
        video_layout.addWidget(self.detection_info)

        # Video player (for manual browsing when not calibrating)
        self.video_player = VideoPlayer()
        video_layout.addWidget(self.video_player)
        right_layout.addWidget(video_group)

        # Detection log
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFont(QFont("monospace", 9))
        self.log_text.setMaximumHeight(120)
        right_layout.addWidget(self.log_text)

        splitter.addWidget(right_panel)
        splitter.setSizes([350, 450])

        outer_splitter.addWidget(splitter)
        outer_splitter.setSizes([200, 500])

        layout.addWidget(outer_splitter, stretch=1)

    def _connect_signals(self):
        self.state_manager.cameras_changed.connect(self._on_cameras_changed)
        self.state_manager.calibration_changed.connect(self._on_calibration_changed)

    def showEvent(self, event):
        """Called when the tab becomes visible - load intrinsics from database."""
        super().showEvent(event)
        self._load_intrinsics_from_db()

    def _load_intrinsics_from_db(self):
        """Load intrinsics from ~/.calimerge/intrinsics.db for all enabled cameras."""
        cameras = self.state_manager.state.cameras
        if not cameras:
            return

        try:
            from ...config import load_intrinsics, get_default_intrinsics_db

            db_path = get_default_intrinsics_db()
            if not db_path.exists():
                return

            loaded_intrinsics = dict(self.state_manager.state.calibration.intrinsics)
            loaded_count = 0

            for port, cam_state in cameras.items():
                if not cam_state.enabled:
                    continue

                serial = cam_state.info.serial_number
                resolution = (cam_state.info.width, cam_state.info.height)

                # Skip if already loaded or resolution not set
                if serial in loaded_intrinsics:
                    continue
                if resolution[0] == 0 or resolution[1] == 0:
                    continue

                intrinsics = load_intrinsics(serial, resolution, db_path)
                if intrinsics is not None:
                    loaded_intrinsics[serial] = intrinsics
                    loaded_count += 1

            if loaded_count > 0:
                self.state_manager.update_calibration(intrinsics=loaded_intrinsics)
                self.status_message.emit(f"Loaded {loaded_count} intrinsics from database")
                # Refresh table to show updated status
                self._on_cameras_changed(cameras)
        except Exception as e:
            self.status_message.emit(f"Failed to load intrinsics: {e}")

    # ── ChArUco preview ──

    def _update_charuco_preview(self):
        """Regenerate and display the charuco board preview at fixed size."""
        try:
            from ...calibration.charuco import generate_board_image

            config = self._get_charuco_config()
            # Generate at a resolution proportional to board shape
            cols, rows = config.columns, config.rows
            scale = 60
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

    # ── Camera table ──

    def _on_cameras_changed(self, cameras: dict):
        """Update camera table when cameras change - only shows enabled cameras."""
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

            # Camera name + nickname (column 2)
            serial = cam_state.info.serial_number
            nickname = cam_state.nickname
            name_text = f"{nickname} - {cam_state.info.display_name}" if nickname else cam_state.info.display_name
            name_item = QTableWidgetItem(name_text)
            name_item.setData(Qt.ItemDataRole.UserRole, port)
            name_item.setData(Qt.ItemDataRole.UserRole + 1, serial)  # Store serial for lookup
            name_item.setFlags(name_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.camera_table.setItem(row, 2, name_item)

            # Serial (column 3) - dedicated column for readability
            serial_item = QTableWidgetItem(serial)
            serial_item.setFont(QFont("monospace", 9))
            serial_item.setFlags(serial_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.camera_table.setItem(row, 3, serial_item)

            # Video path (column 4)
            video_path = self.video_paths.get(port)
            video_text = video_path.name if video_path else "Not loaded"
            video_item = QTableWidgetItem(video_text)
            video_item.setFlags(video_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.camera_table.setItem(row, 4, video_item)

            # Status (column 5) - use serial for intrinsics lookup
            cal_state = self.state_manager.state.calibration
            camera_res = (cam_state.info.width, cam_state.info.height)

            if serial in cal_state.intrinsics:
                intr = cal_state.intrinsics[serial]
                # Check if intrinsics were scaled from a different resolution
                if intr.is_scaled:
                    # Show the original resolution they were scaled from
                    status = f"Scaled from {intr.scaled_from[0]}x{intr.scaled_from[1]}"
                else:
                    status = "Calibrated"
            elif port in cal_state.intrinsic_progress:
                progress = cal_state.intrinsic_progress[port]
                status = f"Processing {progress:.0%}"
            else:
                # Check database for this camera (any resolution)
                db_status = self._check_db_intrinsics_any_res(serial)
                status = db_status if db_status else "Pending"
            status_item = QTableWidgetItem(status)
            status_item.setFlags(status_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.camera_table.setItem(row, 5, status_item)

            # Error (column 6) - use serial for intrinsics lookup
            error_text = ""
            if serial in cal_state.intrinsics:
                error_text = f"{cal_state.intrinsics[serial].error:.4f}"
            error_item = QTableWidgetItem(error_text)
            error_item.setFlags(error_item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            self.camera_table.setItem(row, 6, error_item)

    def _check_db_intrinsics(self, info) -> str:
        """Check if intrinsics exist in database for this camera at current resolution."""
        try:
            from ...config import load_intrinsics, get_default_intrinsics_db

            if not info.serial_number or info.width == 0 or info.height == 0:
                return ""
            db_path = get_default_intrinsics_db()
            if not db_path.exists():
                return "No DB"
            result = load_intrinsics(
                info.serial_number,
                (info.width, info.height),
                db_path,
            )
            if result is not None:
                return f"Found (err {result.error:.4f})"
            return "Not found"
        except Exception:
            return ""

    def _check_db_intrinsics_status(self, serial: str, resolution: tuple[int, int]) -> str:
        """
        Check intrinsics availability for this camera at the given resolution.

        Returns status string indicating:
        - "Scalable from WxH" if same aspect ratio intrinsics exist
        - "In DB: WxH (wrong aspect)" if intrinsics exist but different aspect ratio
        - "" if no intrinsics in database
        """
        try:
            from ...config import check_intrinsics_availability, get_default_intrinsics_db

            db_path = get_default_intrinsics_db()
            if not db_path.exists():
                return ""

            status, source_res = check_intrinsics_availability(serial, resolution, db_path)

            if status == "exact":
                # This shouldn't happen - if exact match, it would be loaded
                return f"In DB: {source_res[0]}x{source_res[1]}"
            elif status == "scalable":
                return f"Scalable from {source_res[0]}x{source_res[1]}"
            elif status == "mismatch":
                return f"In DB: {source_res[0]}x{source_res[1]} (wrong aspect)"
            else:
                return ""
        except Exception:
            return ""

    def _check_db_intrinsics_any_res(self, serial: str) -> str:
        """Check if intrinsics exist in database for this camera at any resolution."""
        # Get camera resolution if available
        cameras = self.state_manager.state.cameras
        resolution = (0, 0)
        for port, cam_state in cameras.items():
            if cam_state.info.serial_number == serial:
                resolution = (cam_state.info.width, cam_state.info.height)
                break

        if resolution[0] > 0 and resolution[1] > 0:
            return self._check_db_intrinsics_status(serial, resolution)

        # Fallback: list all available resolutions
        try:
            from ...config import list_intrinsics, get_default_intrinsics_db

            db_path = get_default_intrinsics_db()
            if not db_path.exists():
                return ""

            all_intrinsics = list_intrinsics(db_path)
            matches = [
                (w, h, err) for sn, w, h, err in all_intrinsics
                if sn == serial
            ]

            if matches:
                res_list = [f"{w}x{h}" for w, h, _ in matches]
                return f"In DB: {', '.join(res_list)}"
            return ""
        except Exception:
            return ""

    def _on_calibration_changed(self, calibration):
        """Refresh table on calibration change."""
        cameras = self.state_manager.state.cameras
        if cameras:
            self._on_cameras_changed(cameras)

    def _get_selected_port(self) -> int | None:
        """Get port of selected camera."""
        items = self.camera_table.selectedItems()
        if not items:
            return None
        row = items[0].row()
        # Port is stored in column 1 or column 2 (both have UserRole data)
        item = self.camera_table.item(row, 1)
        return item.data(Qt.ItemDataRole.UserRole) if item else None

    def _get_selected_serial(self) -> str | None:
        """Get serial number of selected camera."""
        items = self.camera_table.selectedItems()
        if not items:
            return None
        row = items[0].row()
        # Serial is stored in column 2 as UserRole + 1
        item = self.camera_table.item(row, 2)
        return item.data(Qt.ItemDataRole.UserRole + 1) if item else None

    def _on_camera_selected(self):
        """Handle camera selection."""
        port = self._get_selected_port()
        serial = self._get_selected_serial()
        if port is None:
            return

        # Load video if available
        if port in self.video_paths:
            self.video_player.load_video(self.video_paths[port])
        else:
            self.video_player.unload()

        # Show calibration results if available (keyed by serial)
        cal_state = self.state_manager.state.calibration
        if serial and serial in cal_state.intrinsics:
            intrinsics = cal_state.intrinsics[serial]
            d = intrinsics.distortion
            dist_names = ["k1", "k2", "p1", "p2", "k3"]
            dist_lines = "\n".join(
                f"{name}: {d[i]:.6f}" for i, name in enumerate(dist_names) if i < len(d)
            )
            # Show scaled indicator if applicable
            res_text = f"{intrinsics.resolution[0]}x{intrinsics.resolution[1]}"
            if intrinsics.is_scaled:
                res_text += f" (scaled from {intrinsics.scaled_from[0]}x{intrinsics.scaled_from[1]})"
            self.results_label.setText(
                f"Serial: {intrinsics.serial_number}\n"
                f"Resolution: {res_text}\n"
                f"Error: {intrinsics.error:.4f}\n"
                f"Grid count: {intrinsics.grid_count}\n\n"
                f"Camera Matrix:\n"
                f"  fx: {intrinsics.matrix[0, 0]:.2f}\n"
                f"  fy: {intrinsics.matrix[1, 1]:.2f}\n"
                f"  cx: {intrinsics.matrix[0, 2]:.2f}\n"
                f"  cy: {intrinsics.matrix[1, 2]:.2f}\n\n"
                f"Distortion:\n{dist_lines}"
            )
            # Only allow saving if not scaled (scaled intrinsics are derived, not original)
            self.save_button.setEnabled(not intrinsics.is_scaled)
        else:
            self.results_label.setText("Not calibrated")
            self.save_button.setEnabled(False)

    # ── Video loading ──

    def _load_video(self):
        """Load calibration video for selected camera."""
        port = self._get_selected_port()
        if port is None:
            self.status_message.emit("Select a camera first")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "Select Calibration Video", "", "Videos (*.mp4 *.avi *.mov)"
        )
        if path:
            self.video_paths[port] = Path(path)
            self.video_player.load_video(Path(path))
            self._on_cameras_changed(self.state_manager.state.cameras)
            self.status_message.emit(f"Loaded video for camera {port}")

    def _load_folder(self):
        """Load intrinsic videos from a folder."""
        from ..video_utils import find_video_for_port, discover_videos

        folder = QFileDialog.getExistingDirectory(self, "Select Intrinsic Video Folder")
        if not folder:
            return

        folder_path = Path(folder)
        cameras = self.state_manager.state.cameras

        if cameras:
            loaded = 0
            for port, cam_state in cameras.items():
                serial = getattr(cam_state.info, "serial_number", None)
                video_path = find_video_for_port(folder_path, port, serial)
                if video_path:
                    self.video_paths[port] = video_path
                    loaded += 1
            self._on_cameras_changed(cameras)
            self.status_message.emit(f"Loaded {loaded} intrinsic video(s) from folder")
        else:
            # No cameras enumerated - create placeholder entries from video files
            self._load_folder_without_cameras(folder_path)

    def _load_folder_without_cameras(self, folder_path: Path):
        """Load videos from folder and create placeholder camera entries."""
        from ..video_utils import discover_videos, parse_video_filename

        loaded = 0
        camera_states = {}

        discovered = discover_videos(folder_path)
        for port, video_file in sorted(discovered.items()):
            self.video_paths[port] = video_file

            # Extract serial from filename if available
            parsed = parse_video_filename(video_file.name)
            serial = parsed[1] if parsed and parsed[1] else f"port_{port}"

            info = _PlaceholderCameraInfo(
                serial_number=serial,
                display_name=f"Camera {port} (from file)",
                device_index=port,
            )
            camera_states[port] = CameraState(info=info, enabled=True, is_open=False)
            loaded += 1

        if camera_states:
            self.state_manager.set_cameras(camera_states)

        self.status_message.emit(
            f"Loaded {loaded} video(s) from folder (no live cameras)"
        )

    # ── Calibration ──

    def _get_charuco_config(self):
        """Get current ChArUco configuration."""
        from ...types import CharucoConfig

        return CharucoConfig(
            columns=self.cols_spin.value(),
            rows=self.rows_spin.value(),
            square_size_cm=self.square_spin.value(),
            dictionary=self.dict_combo.currentText(),
            inverted=self.inverted_checkbox.isChecked(),
        )

    def _calibrate_selected(self):
        """Calibrate selected camera."""
        port = self._get_selected_port()
        if port is None:
            self.status_message.emit("Select a camera first")
            return

        if port not in self.video_paths:
            self.status_message.emit("Load a video first")
            return

        self._run_calibration(port)

    def _calibrate_all(self):
        """Calibrate all cameras with videos."""
        for port in self.video_paths:
            if port not in self.calibration_workers:
                self._run_calibration(port)

    def _run_calibration(self, port: int):
        """Run calibration for a camera."""
        cameras = self.state_manager.state.cameras
        cam_state = cameras.get(port)
        if not cam_state:
            return

        video_path = self.video_paths[port]
        charuco_config = self._get_charuco_config()
        serial_number = cam_state.info.serial_number

        worker = IntrinsicCalibrationWorker(
            video_path=video_path,
            serial_number=serial_number,
            charuco_config=charuco_config,
        )
        worker.log_message.connect(self._log)
        worker.progress_update.connect(
            lambda cur, tot, p=port: self._on_calibration_progress(p, cur, tot)
        )
        worker.detection_frame.connect(self._on_detection_frame)
        worker.calibration_finished.connect(
            lambda result, p=port: self._on_calibration_finished(p, result)
        )
        worker.error.connect(lambda err, p=port: self._on_calibration_error(p, err))
        worker.finished.connect(lambda p=port: self._on_worker_finished(p))

        self.calibration_workers[port] = worker

        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setVisible(True)

        # Clear detection display
        self.detection_label.clear()
        self.detection_info.setText("Scanning for charuco boards...")
        self.log_text.clear()

        worker.start()
        self._log(f"Calibrating camera {port}...")

    def _on_calibration_progress(self, port: int, current: int, total: int):
        """Update calibration progress."""
        progress = current / total if total > 0 else 0
        self.progress_bar.setValue(int(progress * 100))

        new_progress = {
            **self.state_manager.state.calibration.intrinsic_progress,
            port: progress,
        }
        self.state_manager.update_calibration(intrinsic_progress=new_progress)

    def _on_detection_frame(self, frame_index: int, frame, corner_count: int):
        """Show a frame with detected charuco corners overlaid."""
        pixmap = bgr_to_pixmap(frame)
        if not pixmap.isNull():
            scaled = pixmap.scaled(
                self.detection_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.detection_label.setPixmap(scaled)
        self.detection_info.setText(
            f"Frame {frame_index}: {corner_count} corners detected"
        )

    def _on_calibration_finished(self, port: int, intrinsics):
        """Handle calibration completion - auto-saves to database."""
        self.progress_bar.setVisible(False)
        self.detection_info.setText(
            f"Calibration complete! Error: {intrinsics.error:.4f}"
        )

        # Store by serial number (not port) so intrinsics follow the camera
        serial = intrinsics.serial_number
        new_intrinsics = {
            **self.state_manager.state.calibration.intrinsics,
            serial: intrinsics,
        }
        self.state_manager.update_calibration(intrinsics=new_intrinsics)

        # Auto-save to database
        try:
            from ...config import save_intrinsics, get_default_intrinsics_db

            db_path = get_default_intrinsics_db()
            save_intrinsics(intrinsics, db_path)
            self._log(
                f"Port {port} ({serial[-8:]}) calibrated (error: {intrinsics.error:.4f}), "
                f"saved to ~/.calimerge/intrinsics.db"
            )
        except Exception as e:
            self._log(
                f"Port {port} calibrated (error: {intrinsics.error:.4f}), "
                f"but failed to save: {e}"
            )

        # Refresh selection
        if self._get_selected_port() == port:
            self._on_camera_selected()

    def _on_calibration_error(self, port: int, error: str):
        """Handle calibration error."""
        self.progress_bar.setVisible(False)
        self.detection_info.setText(f"Calibration failed: {error}")
        self._log(f"Camera {port} calibration failed: {error}")

    def _on_worker_finished(self, port: int):
        """Clean up worker."""
        if port in self.calibration_workers:
            del self.calibration_workers[port]

    def _save_intrinsics(self):
        """Save intrinsics to database."""
        serial = self._get_selected_serial()
        if serial is None:
            return

        cal_state = self.state_manager.state.calibration
        if serial not in cal_state.intrinsics:
            return

        intrinsics = cal_state.intrinsics[serial]

        try:
            from ...config import save_intrinsics, get_default_intrinsics_db

            db_path = get_default_intrinsics_db()
            save_intrinsics(intrinsics, db_path)
            self._log(f"Saved intrinsics for {serial} to ~/.calimerge/intrinsics.db")
        except Exception as e:
            self._log(f"Failed to save: {e}")

    # ── Helpers ──

    def _log(self, message: str):
        self.log_text.append(message)
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
        self.status_message.emit(message)


class _PlaceholderCameraInfo:
    """Minimal camera info for file-based workflows without live cameras."""

    def __init__(self, serial_number: str, display_name: str, device_index: int):
        self.serial_number = serial_number
        self.display_name = display_name
        self.device_index = device_index
        self.width = 0
        self.height = 0
        self.fps = 0
        self.rotation = 0
        self.exposure = 0
        self.enabled = True
        self.supported_formats = []
        self._c_camera = None
