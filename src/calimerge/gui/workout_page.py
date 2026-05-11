"""
Workout recording page — default landing GUI.

Provides user login, camera initialization, live 3D pose detection,
workout selection, recording, and results display.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np

from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QLabel,
    QComboBox,
    QCheckBox,
    QPushButton,
    QGroupBox,
    QRadioButton,
    QButtonGroup,
    QDoubleSpinBox,
    QSpinBox,
    QSlider,
)
from PySide6.QtCore import Signal, Qt, QTimer
from PySide6.QtGui import QFont

from .state import StateManager, CameraState
from .workers import (
    CameraEnumerateWorker, CameraPreviewWorker,
    PoseDetectionWorker, CudaStreamDetectionWorker,
    MediaPipeHandsDetectionWorker, RecordingWorker,
)
from .widgets.camera_grid import CameraGrid
from .widgets.skeleton_view import SkeletonViewWidget


class WorkoutPage(QWidget):
    """Main workout recording page with login, workout selection, and results."""

    status_message = Signal(str)

    def __init__(self, state_manager: StateManager, parent=None):
        super().__init__(parent)
        self.state_manager = state_manager
        self._current_user_id: int | None = None
        self._current_username: str = ""
        self._calibration_available = False
        self._calibration_path: Path | None = None
        self._calibrated_cameras: dict | None = None
        self._calibration_session_id: int | None = None
        self._calibration_session_created_at: str | None = None
        # Zero-origin transform set by "Zero L-Ankle" button; persisted into
        # workouts.db at recording end (None until the user presses the btn).
        self._zero_origin_rotation: np.ndarray | None = None
        self._zero_origin_translation: np.ndarray | None = None

        # Camera state
        self.enumerate_worker: CameraEnumerateWorker | None = None
        self.preview_worker: CameraPreviewWorker | None = None
        self.detection_worker: PoseDetectionWorker | None = None
        self.recording_worker: RecordingWorker | None = None
        self._is_recording = False
        self._recording_keypoints: list[dict] = []  # collected during recording
        self._primary_person_index: int = 0  # person closest to calibrated origin
        self.opened_cameras: list = []
        self.opened_ports: list[int] = []
        self._last_frame_time: dict[int, float] = {}
        self._last_annotated: dict[int, object] = {}

        # View transform state (copied from cameras_tab)
        self._view_rotation = np.eye(4)
        self._view_has_origin = False

        # Per-serial camera preferences
        self._serial_prefs: dict[str, dict] = {}
        self._target_fps: int = 30

        # Active program state
        self._active_program: dict | None = None
        self._active_program_exercises: list[dict] = []
        self._current_program_exercise: dict | None = None

        # Auto-chain flag: after login, automatically start preview + detection
        self._auto_start_pipeline: bool = False

        # Coalescing buffers for the per-frame UI paints. Qt's queued signals
        # for cross-thread emits (`detection_ready`, `keypoints_3d_ready`)
        # pile up FIFO when the main thread is overloaded; processing that
        # backlog ends up painting frames from minutes ago. We drop-old by
        # stashing only the latest payload here and scheduling exactly one
        # paint via QTimer.singleShot. Recording-buffer fills happen in the
        # slot synchronously so no science data is lost.
        self._pending_grid_frames: dict[int, "np.ndarray"] = {}
        self._grid_paint_scheduled: bool = False
        self._pending_persons_3d: list | None = None
        self._kp3d_paint_scheduled: bool = False

        # Per-port arrival timestamps for the current recording. Cleared at
        # _on_record start, consumed in _on_record_finished to print
        # avg/median fps + max delta to stdout and the status bar.
        self._record_arrivals: dict[int, list[float]] = {}

        # 3 s pre-roll countdown before recording actually starts; lets the
        # operator walk into the capture volume. None when not counting.
        self._record_countdown_timer: QTimer | None = None
        self._record_countdown_remaining: int = 0

        self._init_ui()
        self._load_camera_prefs()
        # Restore last-used detection model/backend/confidence BEFORE
        # _load_view_transform — that function keys off
        # _current_model_key(), so the model dropdown must be on the
        # right entry first or the wrong preset gets applied to the
        # skeleton view at startup.
        self._restore_last_detect_state()
        self._check_calibration()
        self._load_view_transform()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(4, 4, 4, 4)

        # ── User bar ──
        user_group = QGroupBox("User")
        user_layout = QHBoxLayout(user_group)

        user_layout.addWidget(QLabel("Username:"))
        self.user_combo = QComboBox()
        self.user_combo.setEditable(True)
        self.user_combo.setMinimumWidth(150)
        self.user_combo.setToolTip("Select existing user or type new username")
        user_layout.addWidget(self.user_combo)

        self.login_btn = QPushButton("Log In")
        self.login_btn.clicked.connect(self._on_login)
        user_layout.addWidget(self.login_btn)

        user_layout.addSpacing(20)
        user_layout.addWidget(QLabel("Mass (kg):"))
        self.mass_spin = QDoubleSpinBox()
        self.mass_spin.setRange(10.0, 300.0)
        self.mass_spin.setValue(70.0)
        self.mass_spin.setDecimals(1)
        self.mass_spin.setSuffix(" kg")
        self.mass_spin.setEnabled(False)
        self.mass_spin.valueChanged.connect(self._on_mass_changed)
        user_layout.addWidget(self.mass_spin)

        self.user_status = QLabel("")
        self.user_status.setStyleSheet("color: #888;")
        user_layout.addWidget(self.user_status)
        user_layout.addSpacing(20)

        self.user_cal_status = QLabel("")
        self.user_cal_status.setStyleSheet("color: #888; font-size: 11px;")
        user_layout.addWidget(self.user_cal_status)

        user_layout.addSpacing(10)

        # Detection model + backend toggles (visible in user bar)
        self.detect_model_combo = QComboBox()
        # The body model is the SynthPose-trained VitPose (52 anatomical
        # keypoints), so the canonical model_key is "synthpose". The
        # user-facing label can stay "VitPose (Body)" since that's the
        # architecture, but every persisted reference uses "synthpose".
        self.detect_model_combo.addItem("VitPose / SynthPose (Body)", "synthpose")
        self.detect_model_combo.addItem("MediaPipe Hands", "mediapipe_hands")
        self.detect_model_combo.setToolTip("Detection model")
        self.detect_model_combo.setFixedWidth(130)
        self.detect_model_combo.currentIndexChanged.connect(self._on_model_changed)
        user_layout.addWidget(self.detect_model_combo)

        self.detect_backend_combo = QComboBox()
        self.detect_backend_combo.addItem("PyTorch", "pytorch")
        try:
            from ..tracking.cuda_stream_binding import is_available as _cuda_available
            if _cuda_available():
                self.detect_backend_combo.addItem("Hardware (CUDA)", "cuda")
        except Exception:
            pass
        try:
            from ..tracking.mps_stream_binding import is_available as _mps_available
            if _mps_available():
                self.detect_backend_combo.addItem("Hardware (MPS)", "mps")
        except Exception:
            pass
        self.detect_backend_combo.setToolTip(
            "Backend: PyTorch, CUDA TensorRT, or Apple MPS / CoreML"
        )
        self.detect_backend_combo.setFixedWidth(130)
        self.detect_backend_combo.currentIndexChanged.connect(self._on_model_changed)
        user_layout.addWidget(self.detect_backend_combo)

        # Person-detection confidence threshold slider. YOLO at low thresholds
        # spuriously fires on chairs/furniture; raising filters those out.
        # Restart of detection is required to take effect (handled below).
        user_layout.addSpacing(10)
        user_layout.addWidget(QLabel("Conf:"))
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setMinimum(10)        # 0.10
        self.conf_slider.setMaximum(95)        # 0.95
        self.conf_slider.setSingleStep(5)
        self.conf_slider.setPageStep(10)
        self.conf_slider.setValue(50)          # default 0.50
        self.conf_slider.setFixedWidth(120)
        self.conf_slider.setToolTip(
            "Person-detection confidence threshold (YOLO).\n"
            "Higher = fewer false positives (chairs/furniture)."
        )
        user_layout.addWidget(self.conf_slider)
        self.conf_value_label = QLabel("0.50")
        self.conf_value_label.setStyleSheet("color: #888; min-width: 32px;")
        user_layout.addWidget(self.conf_value_label)
        self.conf_slider.valueChanged.connect(self._on_confidence_changed)


        user_layout.addStretch()

        # Live capture FPS graph. Lives on the far right of the user bar:
        # last 5 s of values per camera, Y-axis pinned to recording_rate * 1.2
        # so over-target/under-target deviations are obvious at a glance.
        # buffer_size auto-resizes from set_target_fps() because we passed
        # time_window_s=5.0.
        from .tabs.cameras_tab import FpsGraphWidget
        self.fps_graph = FpsGraphWidget(
            buffer_size=int(self._target_fps * 5),
            y_max_factor=1.2,
            time_window_s=5.0,
        )
        self.fps_graph.setMinimumHeight(60)
        self.fps_graph.setMaximumHeight(70)
        self.fps_graph.setFixedWidth(240)
        user_layout.addWidget(self.fps_graph)

        layout.addWidget(user_group)

        # ── Camera + calibration + detection bar ──
        cam_group = QGroupBox("Cameras")
        cam_layout = QHBoxLayout(cam_group)

        self.init_cameras_btn = QPushButton("Initialize Cameras")
        self.init_cameras_btn.clicked.connect(self._on_init_cameras)
        cam_layout.addWidget(self.init_cameras_btn)

        self.preview_btn = QPushButton("Preview")
        self.preview_btn.setCheckable(True)
        self.preview_btn.setEnabled(False)
        self.preview_btn.clicked.connect(self._toggle_preview)
        cam_layout.addWidget(self.preview_btn)

        self.detect_checkbox = QCheckBox("Live Detection")
        self.detect_checkbox.setChecked(False)
        self.detect_checkbox.setToolTip("Overlay 2D pose detection + 3D skeleton")
        self.detect_checkbox.toggled.connect(self._on_detect_toggled)
        cam_layout.addWidget(self.detect_checkbox)

        self.camera_count_label = QLabel("No cameras")
        self.camera_count_label.setStyleSheet("color: #888;")
        cam_layout.addWidget(self.camera_count_label)

        cam_layout.addSpacing(20)
        self.cal_status = QLabel()
        cam_layout.addWidget(self.cal_status)
        cam_layout.addStretch()

        # Hidden — cameras auto-init on login. Widgets kept alive for
        # internal state management (auto-chain, status labels).
        cam_group.setVisible(False)
        layout.addWidget(cam_group)

        # ── Main vertical splitter: preview (top) | controls (mid) | analysis (bottom) ──
        main_splitter = QSplitter(Qt.Orientation.Vertical)

        # Top: camera grid + skeleton view (horizontal splitter)
        preview_splitter = QSplitter(Qt.Orientation.Horizontal)

        preview_widget = QWidget()
        preview_layout = QVBoxLayout(preview_widget)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        self.camera_grid = CameraGrid()
        self.camera_grid.setMinimumHeight(150)
        preview_layout.addWidget(self.camera_grid)
        preview_splitter.addWidget(preview_widget)

        skel_panel = QWidget()
        skel_layout = QVBoxLayout(skel_panel)
        skel_layout.setContentsMargins(4, 4, 4, 4)

        skel_header = QHBoxLayout()
        skel_label = QLabel("Live 3D Projection")
        skel_label.setFont(QFont("monospace", 9))
        skel_header.addWidget(skel_label)
        skel_header.addStretch()

        self.rotate_to_human_button = QPushButton("Rotate to Human")
        self.rotate_to_human_button.setEnabled(False)
        self.rotate_to_human_button.setToolTip(
            "Orient view: Y=up (head), X=foot-to-foot, Z=forward. Stand still, triggers in 3s."
        )
        self.rotate_to_human_button.clicked.connect(self._on_rotate_to_human)
        skel_header.addWidget(self.rotate_to_human_button)

        self.zero_origin_button = QPushButton("Zero at L_Ankle")
        self.zero_origin_button.setEnabled(False)
        self.zero_origin_button.setToolTip(
            "Set left ankle as floor origin (0,0,0). Stand still, triggers in 3s."
        )
        self.zero_origin_button.clicked.connect(self._on_zero_at_left_foot)
        skel_header.addWidget(self.zero_origin_button)

        skel_layout.addLayout(skel_header)
        self.skeleton_view = SkeletonViewWidget()
        skel_layout.addWidget(self.skeleton_view)
        preview_splitter.addWidget(skel_panel)

        preview_splitter.setSizes([600, 300])
        main_splitter.addWidget(preview_splitter)

        # Bottom row: horizontal splitter with three panes
        #   left (25%): today's plan (top) + record/sessions buttons (bottom)
        #   middle (50%): analysis + results
        #   right (25%): deadspace (reserved for future content)
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left pane: vertical stack of plan + buttons
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(4)

        from .todays_plan import TodaysPlanWidget
        self.todays_plan = TodaysPlanWidget()
        self.todays_plan.exercise_selected.connect(self._on_plan_exercise_selected)
        left_layout.addWidget(self.todays_plan, stretch=1)

        # Fallback manual selector (hidden when a program is loaded)
        self.manual_group = QGroupBox("Workout (manual)")
        manual_layout = QVBoxLayout(self.manual_group)

        self.workout_buttons = QButtonGroup(self)
        self.sts_radio = QRadioButton("Sit-to-Stand")
        self.sts_radio.setChecked(True)
        self.tug_radio = QRadioButton("Timed Up and Go")
        self.biceps_radio = QRadioButton("Biceps Curls")
        self.pushup_radio = QRadioButton("Pushups")
        self.pullup_radio = QRadioButton("Pullups")
        self.leg_raise_radio = QRadioButton("Leg Raises")
        self.tandem_radio = QRadioButton("Tandem Stance")
        self.stretch_radio = QRadioButton("Stretch")

        self.workout_buttons.addButton(self.sts_radio, 0)
        self.workout_buttons.addButton(self.tug_radio, 1)
        self.workout_buttons.addButton(self.biceps_radio, 2)
        self.workout_buttons.addButton(self.pushup_radio, 3)
        self.workout_buttons.addButton(self.pullup_radio, 4)
        self.workout_buttons.addButton(self.leg_raise_radio, 5)
        self.workout_buttons.addButton(self.tandem_radio, 6)
        self.workout_buttons.addButton(self.stretch_radio, 7)
        self.workout_buttons.buttonClicked.connect(self._on_workout_type_changed)

        manual_layout.addWidget(self.sts_radio)
        manual_layout.addWidget(self.tug_radio)
        manual_layout.addWidget(self.biceps_radio)
        manual_layout.addWidget(self.pushup_radio)
        manual_layout.addWidget(self.pullup_radio)
        manual_layout.addWidget(self.leg_raise_radio)
        manual_layout.addWidget(self.tandem_radio)
        manual_layout.addWidget(self.stretch_radio)

        dur_layout = QHBoxLayout()
        dur_layout.addWidget(QLabel("Duration:"))
        self.duration_spin = QSpinBox()
        self.duration_spin.setRange(5, 300)
        self.duration_spin.setValue(30)
        self.duration_spin.setSuffix(" s")
        dur_layout.addWidget(self.duration_spin)
        dur_layout.addStretch()
        manual_layout.addLayout(dur_layout)

        left_layout.addWidget(self.manual_group)

        # Record + View Sessions buttons in a sub-panel underneath plan
        btn_panel = QGroupBox("Session")
        btn_layout = QVBoxLayout(btn_panel)
        btn_layout.setSpacing(4)

        self.record_btn = QPushButton("Record Workout")
        self.record_btn.setMinimumHeight(50)
        self.record_btn.setStyleSheet(
            "QPushButton { font-size: 14px; font-weight: bold; }"
            "QPushButton:disabled { color: #666; }"
        )
        self.record_btn.setEnabled(False)
        self.record_btn.clicked.connect(self._on_record)
        btn_layout.addWidget(self.record_btn)

        self.view_sessions_btn = QPushButton("View Sessions")
        self.view_sessions_btn.clicked.connect(self._on_view_sessions)
        btn_layout.addWidget(self.view_sessions_btn)

        self.progress_btn = QPushButton("Progress Graph")
        self.progress_btn.clicked.connect(self._on_view_progress)
        btn_layout.addWidget(self.progress_btn)

        # CSV export controls
        self.generate_csv_checkbox = QCheckBox("Generate CSV after save")
        self.generate_csv_checkbox.setChecked(True)
        self.generate_csv_checkbox.setToolTip(
            "Checked: write keypoints_3d.csv synchronously after each "
            "recording. Unchecked: queue the job and process later via "
            "'Process Pending'."
        )
        self.generate_csv_checkbox.toggled.connect(self._on_csv_toggle_changed)
        # Avoid Qt's platform default of bold-on-checked-text in the global
        # stylesheet — keep the checkbox label at regular weight so it
        # matches the surrounding Gill Sans body text.
        self.generate_csv_checkbox.setStyleSheet("font-weight: normal;")
        btn_layout.addWidget(self.generate_csv_checkbox)

        # Pause-live-tracking toggle: when checked, the detection worker
        # is stopped for the duration of recording so video capture has the
        # full frame budget. Detection resumes automatically once recording
        # ends. Sits next to the CSV toggle since both are recording-time
        # behavioural switches.
        self.pause_tracking_during_record_checkbox = QCheckBox(
            "Pause live tracking during recording"
        )
        self.pause_tracking_during_record_checkbox.setChecked(True)
        self.pause_tracking_during_record_checkbox.setToolTip(
            "Checked: stop live 2D/3D tracking while recording so the\n"
            "cameras can hit the commanded fps. Tracking resumes\n"
            "automatically when the recording finishes. Raw video is\n"
            "always saved either way; you can re-process the videos\n"
            "later via the CUDA batch pipeline."
        )
        self.pause_tracking_during_record_checkbox.setStyleSheet(
            "font-weight: normal;"
        )
        btn_layout.addWidget(self.pause_tracking_during_record_checkbox)

        self.process_pending_btn = QPushButton("Process Pending CSVs")
        self.process_pending_btn.setToolTip(
            "Drain the queue of recorded sessions waiting for CSV export."
        )
        self.process_pending_btn.clicked.connect(self._on_process_pending_csvs)
        btn_layout.addWidget(self.process_pending_btn)

        # Initialise checkbox + label from app_settings
        self._init_csv_export_state()

        left_layout.addWidget(btn_panel)
        bottom_splitter.addWidget(left_pane)

        # Middle pane: analysis panel (the existing builder)
        self.results_group = QGroupBox("Results")
        results_layout = QVBoxLayout(self.results_group)
        self.results_label = QLabel("No results yet.")
        self.results_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.results_label.setStyleSheet("color: #888; font-size: 13px;")
        results_layout.addWidget(self.results_label)

        analysis_widget = self._build_analysis_panel()
        # Put the Results group UNDER the analysis plot inside the middle pane
        middle_pane = QWidget()
        middle_layout = QVBoxLayout(middle_pane)
        middle_layout.setContentsMargins(0, 0, 0, 0)
        middle_layout.setSpacing(4)
        middle_layout.addWidget(analysis_widget, stretch=3)
        middle_layout.addWidget(self.results_group, stretch=1)
        bottom_splitter.addWidget(middle_pane)

        # Right pane: long-term workout progress graph
        longterm_pane = self._build_longterm_graph_pane()
        bottom_splitter.addWidget(longterm_pane)

        bottom_splitter.setStretchFactor(0, 1)
        bottom_splitter.setStretchFactor(1, 2)
        bottom_splitter.setStretchFactor(2, 1)
        bottom_splitter.setSizes([250, 500, 250])

        main_splitter.addWidget(bottom_splitter)

        # Default sizes: 60% preview on top, 40% bottom row
        main_splitter.setSizes([500, 400])
        main_splitter.setCollapsible(0, False)

        layout.addWidget(main_splitter, stretch=1)

        # Bottom progress strip — used by the offline post-processing path
        # (run when 'Pause live tracking during recording' is on AND
        # 'Generate CSV after save' is on). Hidden when idle.
        from PySide6.QtWidgets import QProgressBar
        offline_row = QHBoxLayout()
        offline_row.setContentsMargins(6, 0, 6, 2)
        self.offline_status_label = QLabel("")
        self.offline_status_label.setStyleSheet("color: #aaa; font-size: 11px;")
        self.offline_progress_bar = QProgressBar()
        self.offline_progress_bar.setRange(0, 100)
        self.offline_progress_bar.setValue(0)
        self.offline_progress_bar.setTextVisible(True)
        self.offline_progress_bar.setFixedHeight(14)
        offline_row.addWidget(self.offline_status_label)
        offline_row.addWidget(self.offline_progress_bar, stretch=1)
        # Wrap in a container so we can hide both pieces in one toggle.
        self.offline_progress_container = QWidget()
        self.offline_progress_container.setLayout(offline_row)
        self.offline_progress_container.setVisible(False)
        layout.addWidget(self.offline_progress_container)

        # Start with manual fallback hidden — the Today's Plan widget
        # shows "log in and pick a program" empty state.
        self.manual_group.setVisible(False)

        self._update_cal_status()
        self._refresh_user_list()

    def _build_analysis_panel(self) -> QWidget:
        """Build the bottom analysis panel with pyqtgraph plot.

        Shows hip height (sit-to-stand) or head speed (TUG) depending on
        the currently selected workout type.
        """
        import pyqtgraph as pg

        pg.setConfigOption("background", "#1a1a1a")
        pg.setConfigOption("foreground", "#cccccc")

        panel = QGroupBox("Analysis")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(4, 4, 4, 4)

        # Controls row
        ctrl_row = QHBoxLayout()

        self.threshold_label = QLabel("Seated threshold (m):")
        ctrl_row.addWidget(self.threshold_label)
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.1, 2.0)
        self.threshold_spin.setValue(0.65)
        self.threshold_spin.setDecimals(3)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setToolTip(
            "Threshold for rep detection. Disabled until a workout has been recorded."
        )
        self.threshold_spin.setEnabled(False)
        self.threshold_spin.valueChanged.connect(self._on_threshold_changed)
        ctrl_row.addWidget(self.threshold_spin)

        self.speed_thresh_label = QLabel("Head speed threshold (m/s):")
        ctrl_row.addWidget(self.speed_thresh_label)
        self.speed_threshold_spin = QDoubleSpinBox()
        self.speed_threshold_spin.setRange(0.05, 3.0)
        self.speed_threshold_spin.setValue(0.3)
        self.speed_threshold_spin.setDecimals(3)
        self.speed_threshold_spin.setSingleStep(0.01)
        self.speed_threshold_spin.setToolTip(
            "Head speed above which subject is considered moving (TUG only)."
        )
        self.speed_threshold_spin.valueChanged.connect(self._on_speed_threshold_changed)
        ctrl_row.addWidget(self.speed_threshold_spin)

        self.analysis_summary_label = QLabel("No workout recorded yet")
        self.analysis_summary_label.setStyleSheet("color: #888;")
        ctrl_row.addSpacing(20)
        ctrl_row.addWidget(self.analysis_summary_label)
        ctrl_row.addStretch()

        layout.addLayout(ctrl_row)

        # pyqtgraph plot
        self.hip_plot = pg.PlotWidget()
        self.hip_plot.setLabel("left", "Hip Height (m)")
        self.hip_plot.setLabel("bottom", "Time (s)")
        self.hip_plot.showGrid(x=True, y=True, alpha=0.3)
        self.hip_plot.setMinimumHeight(120)

        # Traces for sit-to-stand (hip height)
        self.hip_trace_raw = self.hip_plot.plot(
            pen=pg.mkPen(color="#555555", width=1))
        self.hip_trace_smooth = self.hip_plot.plot(
            pen=pg.mkPen(color="#4CAF50", width=2))

        # Trace for TUG (head speed) — same plot, different y data
        self.head_speed_trace = self.hip_plot.plot(
            pen=pg.mkPen(color="#42A5F5", width=2))

        # Seated threshold line (draggable)
        self.threshold_line = pg.InfiniteLine(
            pos=0.65, angle=0, movable=True,
            pen=pg.mkPen(color="#FF5252", width=1, style=Qt.PenStyle.DashLine),
            label="seated",
            labelOpts={"position": 0.05, "color": "#FF5252"},
        )
        self.threshold_line.sigPositionChanged.connect(self._on_threshold_line_moved)
        self.hip_plot.addItem(self.threshold_line)

        # Head speed threshold line (draggable, TUG only)
        self.speed_threshold_line = pg.InfiniteLine(
            pos=0.3, angle=0, movable=True,
            pen=pg.mkPen(color="#42A5F5", width=1, style=Qt.PenStyle.DashLine),
            label="head speed",
            labelOpts={"position": 0.05, "color": "#42A5F5"},
        )
        self.speed_threshold_line.sigPositionChanged.connect(self._on_speed_line_moved)
        self.hip_plot.addItem(self.speed_threshold_line)

        # Scatter for rep peaks (sit-to-stand)
        self.rep_peaks_scatter = pg.ScatterPlotItem(
            size=12, brush=pg.mkBrush("#FFC107"),
            pen=pg.mkPen(color="#000", width=1),
            symbol="o",
        )
        self.hip_plot.addItem(self.rep_peaks_scatter)

        # Start/end marker lines for TUG
        self.tug_start_line = pg.InfiniteLine(
            pos=0, angle=90, movable=False,
            pen=pg.mkPen(color="#66BB6A", width=2),
            label="start",
            labelOpts={"position": 0.95, "color": "#66BB6A"},
        )
        self.tug_end_line = pg.InfiniteLine(
            pos=0, angle=90, movable=False,
            pen=pg.mkPen(color="#EF5350", width=2),
            label="end",
            labelOpts={"position": 0.95, "color": "#EF5350"},
        )
        self.hip_plot.addItem(self.tug_start_line)
        self.hip_plot.addItem(self.tug_end_line)

        layout.addWidget(self.hip_plot)

        # Buffer the latest analysis inputs for re-analysis on threshold change
        self._last_times: np.ndarray | None = None
        self._last_hip_z: np.ndarray | None = None
        self._last_head_xyz: np.ndarray | None = None   # (N, 3) for TUG
        self._last_elbow_angles: np.ndarray | None = None  # (N,) for biceps
        self._last_shoulder_z: np.ndarray | None = None    # (N,) for pushup
        self._last_head_z: np.ndarray | None = None         # (N,) for pullup
        self._last_knee_z: np.ndarray | None = None         # (N,) for leg raise
        self._last_hip_xy: np.ndarray | None = None          # (N, 2) for tandem
        self._last_workout_type: str = "sit_to_stand"
        self._last_session_id: int | None = None

        # Initial mode: sit-to-stand
        self._apply_plot_mode("sit_to_stand")

        return panel

    def _apply_plot_mode(self, workout_type: str):
        """Show/hide plot elements based on workout type and relabel controls."""
        is_sts = workout_type == "sit_to_stand"
        is_tug = workout_type == "timed_up_and_go"
        is_biceps = workout_type == "biceps_curl"
        is_pushup = workout_type == "pushup"
        is_pullup = workout_type == "pullup"
        is_leg = workout_type == "leg_raise"
        is_tandem = workout_type == "tandem_stance"
        is_stretch = workout_type == "stretch"
        is_trace = is_sts or is_biceps or is_pushup or is_pullup or is_leg or is_tandem or is_stretch

        self.hip_trace_raw.setVisible(is_trace)
        self.hip_trace_smooth.setVisible(is_trace)
        self.rep_peaks_scatter.setVisible(is_trace and not is_stretch and not is_tandem)

        self.head_speed_trace.setVisible(is_tug)
        self.speed_threshold_line.setVisible(is_tug)
        self.tug_start_line.setVisible(is_tug)
        self.tug_end_line.setVisible(is_tug)
        self.speed_threshold_spin.setVisible(is_tug)
        self.speed_thresh_label.setVisible(is_tug)

        # Show threshold line for every mode except stretch
        self.threshold_line.setVisible(not is_stretch)

        # Re-enable spin by default — stretch disables it explicitly below
        if self._last_times is not None:
            self.threshold_spin.setEnabled(not is_stretch)

        self.threshold_spin.blockSignals(True)
        self.threshold_line.blockSignals(True)

        if is_biceps:
            self.threshold_label.setText("Extended angle (\u00b0):")
            self.threshold_spin.setRange(0.0, 180.0)
            self.threshold_spin.setSingleStep(1.0)
            self.threshold_spin.setDecimals(1)
            if not (30.0 <= self.threshold_spin.value() <= 180.0):
                self.threshold_spin.setValue(150.0)
            self.threshold_line.setPos(self.threshold_spin.value())
            self.hip_plot.setLabel("left", "Elbow Angle (\u00b0)")
            self.threshold_line.label.setText("extended")

        elif is_pushup:
            self.threshold_label.setText("Top threshold (m):")
            self.threshold_spin.setRange(0.0, 2.0)
            self.threshold_spin.setSingleStep(0.01)
            self.threshold_spin.setDecimals(3)
            if self.threshold_spin.value() > 1.5:
                self.threshold_spin.setValue(0.30)
            self.threshold_line.setPos(self.threshold_spin.value())
            self.hip_plot.setLabel("left", "Shoulder Height (m)")
            self.threshold_line.label.setText("top")

        elif is_pullup:
            self.threshold_label.setText("Bar height (m):")
            self.threshold_spin.setRange(0.5, 3.0)
            self.threshold_spin.setSingleStep(0.01)
            self.threshold_spin.setDecimals(3)
            if not (0.5 <= self.threshold_spin.value() <= 3.0):
                self.threshold_spin.setValue(1.80)
            self.threshold_line.setPos(self.threshold_spin.value())
            self.hip_plot.setLabel("left", "Head Height (m)")
            self.threshold_line.label.setText("bar")

        elif workout_type == "leg_raise":
            self.threshold_label.setText("Lift threshold (m):")
            self.threshold_spin.setRange(0.0, 2.0)
            self.threshold_spin.setSingleStep(0.01)
            self.threshold_spin.setDecimals(3)
            if not (0.2 <= self.threshold_spin.value() <= 2.0):
                self.threshold_spin.setValue(0.60)
            self.threshold_line.setPos(self.threshold_spin.value())
            self.hip_plot.setLabel("left", "Knee Height (m)")
            self.threshold_line.label.setText("lifted")

        elif workout_type == "tandem_stance":
            self.threshold_label.setText("Sway threshold (m):")
            self.threshold_spin.setRange(0.005, 0.5)
            self.threshold_spin.setSingleStep(0.005)
            self.threshold_spin.setDecimals(3)
            if not (0.005 <= self.threshold_spin.value() <= 0.5):
                self.threshold_spin.setValue(0.05)
            self.threshold_line.setPos(self.threshold_spin.value())
            self.hip_plot.setLabel("left", "Horizontal Sway (m)")
            self.threshold_line.label.setText("stable")

        elif workout_type == "stretch":
            self.threshold_label.setText("(no threshold for stretch)")
            self.threshold_spin.setEnabled(False)
            self.hip_plot.setLabel("left", "Hip Height (m)")
            self.threshold_line.label.setText("")

        elif is_tug:
            self.threshold_label.setText("Seated threshold (m):")
            self.threshold_spin.setRange(0.1, 2.0)
            self.threshold_spin.setSingleStep(0.01)
            self.threshold_spin.setDecimals(3)
            if self.threshold_spin.value() > 3.0:
                self.threshold_spin.setValue(0.65)
            self.threshold_line.setPos(self.threshold_spin.value())
            self.hip_plot.setLabel("left", "Head Speed (m/s)  /  Hip (m)")
            self.threshold_line.label.setText("seated")

        else:  # sit_to_stand
            self.threshold_label.setText("Seated threshold (m):")
            self.threshold_spin.setRange(0.1, 2.0)
            self.threshold_spin.setSingleStep(0.01)
            self.threshold_spin.setDecimals(3)
            if self.threshold_spin.value() > 3.0:
                self.threshold_spin.setValue(0.65)
            self.threshold_line.setPos(self.threshold_spin.value())
            self.hip_plot.setLabel("left", "Hip Height (m)")
            self.threshold_line.label.setText("seated")

        self.threshold_spin.blockSignals(False)
        self.threshold_line.blockSignals(False)

    def _build_longterm_graph_pane(self) -> QWidget:
        """Build the right-pane long-term workout progress graph."""
        import pyqtgraph as pg

        pane = QGroupBox("Long-term Progress")
        layout = QVBoxLayout(pane)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        self.longterm_summary = QLabel("Select an exercise to view progress")
        self.longterm_summary.setStyleSheet("color: #888; font-size: 11px;")
        self.longterm_summary.setWordWrap(True)
        layout.addWidget(self.longterm_summary)

        self.longterm_plot = pg.PlotWidget()
        self.longterm_plot.setLabel("left", "Work (J)")
        self.longterm_plot.setLabel("bottom", "Day")
        self.longterm_plot.showGrid(x=True, y=True, alpha=0.3)
        self.longterm_plot.setMinimumHeight(100)

        self._longterm_week_regions: list = []

        self.longterm_scatter = pg.ScatterPlotItem(
            size=8, brush=pg.mkBrush("#42A5F5"),
            pen=pg.mkPen(color="#000", width=1),
        )
        self.longterm_plot.addItem(self.longterm_scatter)

        self.longterm_daily_trace = self.longterm_plot.plot(
            pen=pg.mkPen(color="#4CAF50", width=2),
        )

        self.longterm_target_trace = self.longterm_plot.plot(
            pen=pg.mkPen(color="#FFC107", width=1, style=Qt.PenStyle.DashLine),
        )

        layout.addWidget(self.longterm_plot, stretch=1)
        return pane

    def _refresh_longterm_graph(self):
        """Reload and redraw the long-term progress graph for the current exercise."""
        import pyqtgraph as pg

        for region in self._longterm_week_regions:
            self.longterm_plot.removeItem(region)
        self._longterm_week_regions.clear()
        self.longterm_scatter.clear()
        self.longterm_daily_trace.clear()
        self.longterm_target_trace.clear()

        if self._current_user_id is None or self._current_program_exercise is None:
            self.longterm_summary.setText("Select an exercise to view progress")
            return

        exercise = self._current_program_exercise
        workout_type = exercise["workout_type"]
        program_exercise_id = exercise["id"]
        sets_per_week_target = exercise.get("sets_per_day", 3) * exercise.get("days_per_week", 3)

        try:
            from ..workout_types import WORKOUT_TYPES
            wt_def = WORKOUT_TYPES.get(workout_type)
        except Exception:
            wt_def = None

        # Try work_per_rep_joules first, fall back to rep_count
        metric_name = "work_per_rep_joules"
        y_label = "Work (J)"
        points = self._load_longterm_points(program_exercise_id, metric_name)
        if not points:
            metric_name = "rep_count"
            y_label = "Repetitions"
            if wt_def and hasattr(wt_def, "primary_metric"):
                metric_name = wt_def.primary_metric
                y_label = getattr(wt_def, "primary_metric_label", "Value")
            points = self._load_longterm_points(program_exercise_id, metric_name)

        self.longterm_plot.setLabel("left", y_label)

        if not points:
            self.longterm_summary.setText(
                f"No sessions recorded for {exercise['display_name']} yet."
            )
            return

        from datetime import datetime, timedelta
        program_start = self._get_program_start_dt()
        day_origin = program_start if program_start else points[0][0]

        xs = [(dt - day_origin).total_seconds() / 86400.0 for dt, val in points]
        ys = [val for _, val in points]

        self.longterm_scatter.setData(x=xs, y=ys)

        by_day: dict[int, float] = {}
        for x, y in zip(xs, ys):
            day = int(x)
            if day not in by_day or y > by_day[day]:
                by_day[day] = y
        if by_day:
            days_sorted = sorted(by_day.keys())
            bests = [by_day[d] for d in days_sorted]
            self.longterm_daily_trace.setData(days_sorted, bests)

        # Week shading
        session_dates = [dt for dt, _ in points]
        total_days = max(1, int(max(xs)) + 7)

        for week_idx in range(total_days // 7 + 1):
            week_start_day = week_idx * 7
            week_end_day = week_start_day + 7
            sessions_in_week = sum(
                1 for x in xs if week_start_day <= x < week_end_day
            )
            met_target = sessions_in_week >= sets_per_week_target
            color = pg.mkBrush(80, 200, 120, 30) if met_target else pg.mkBrush(100, 100, 100, 15)
            region = pg.LinearRegionItem(
                values=(week_start_day, week_end_day),
                orientation="vertical",
                movable=False,
                brush=color,
                pen=pg.mkPen(None),
            )
            region.setZValue(-10)
            self.longterm_plot.addItem(region)
            self._longterm_week_regions.append(region)

        # Target improvement line
        improvement = getattr(wt_def, "weekly_improvement_factor", 1.02) if wt_def else 1.02
        if ys and improvement != 1.0:
            first_week_vals = [y for x, y in zip(xs, ys) if x < 7]
            baseline = float(np.mean(first_week_vals)) if first_week_vals else ys[0]
            max_day = int(max(xs)) + 7
            target_days = list(range(0, max_day + 1, 7))
            target_vals = [baseline * (improvement ** (d / 7.0)) for d in target_days]
            self.longterm_target_trace.setData(target_days, target_vals)

        peak = max(ys)
        self.longterm_summary.setText(
            f"{exercise['display_name']}: {len(points)} sessions, peak {peak:.1f}"
        )

    def _load_longterm_points(self, program_exercise_id: int,
                               metric_name: str) -> list[tuple]:
        """Load (datetime, metric_value) pairs for the given exercise + metric."""
        from datetime import datetime
        from ..config import workouts_db_path
        import sqlite3
        try:
            conn = sqlite3.connect(str(workouts_db_path()))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT s.created_at, r.metric_value "
                "FROM sessions s "
                "JOIN session_results r ON r.session_id = s.id "
                "WHERE s.user_id = ? AND s.program_exercise_id = ? "
                "  AND r.metric_name = ? "
                "ORDER BY s.created_at",
                (self._current_user_id, program_exercise_id, metric_name),
            ).fetchall()
            conn.close()
        except Exception:
            return []

        out = []
        for r in rows:
            try:
                ts = datetime.fromisoformat(str(r["created_at"]).replace(" ", "T"))
                out.append((ts, float(r["metric_value"])))
            except Exception:
                continue
        return out

    def _get_program_start_dt(self):
        from datetime import datetime
        from ..config import get_user_by_id
        try:
            user = get_user_by_id(self._current_user_id)
            started = user.get("program_started_at") if user else None
            if started:
                return datetime.fromisoformat(str(started).replace(" ", "T"))
        except Exception:
            pass
        return None

    def _on_threshold_changed(self, value: float):
        """Seated threshold spin box changed — update the line and reanalyse."""
        self.threshold_line.blockSignals(True)
        self.threshold_line.setPos(value)
        self.threshold_line.blockSignals(False)
        self._reanalyze_with_current_thresholds()

    def _on_threshold_line_moved(self):
        """Seated threshold line dragged — update the spinbox and reanalyse."""
        value = float(self.threshold_line.value())
        self.threshold_spin.blockSignals(True)
        self.threshold_spin.setValue(value)
        self.threshold_spin.blockSignals(False)
        self._reanalyze_with_current_thresholds()

    def _on_speed_threshold_changed(self, value: float):
        """Head speed threshold spin box changed — update line and reanalyse."""
        self.speed_threshold_line.blockSignals(True)
        self.speed_threshold_line.setPos(value)
        self.speed_threshold_line.blockSignals(False)
        self._reanalyze_with_current_thresholds()

    def _on_speed_line_moved(self):
        """Head speed threshold line dragged — update spinbox and reanalyse."""
        value = float(self.speed_threshold_line.value())
        self.speed_threshold_spin.blockSignals(True)
        self.speed_threshold_spin.setValue(value)
        self.speed_threshold_spin.blockSignals(False)
        self._reanalyze_with_current_thresholds()

    def _on_workout_type_changed(self, button):
        """Switch analysis panel mode when the workout type radio changes."""
        self._apply_plot_mode(self._selected_workout_type())

    def _selected_workout_type(self) -> str:
        """Return the string identifier for the currently selected workout.

        Priority order:
          1. The exercise the user picked from Today's Plan (program-driven,
             includes FGA tasks like 'fga_horizontal_head_turns'). Without
             this lookup, recording an FGA task silently saved as
             'sit_to_stand' and ran the wrong analyzer.
          2. The legacy radio buttons (sit-to-stand, push-up, etc.) from the
             pre-program-system UI.
          3. Sit-to-stand as a default of last resort.
        """
        ex = self._current_program_exercise
        if ex is not None:
            wt = ex.get("workout_type")
            if wt:
                return str(wt)

        if self.sts_radio.isChecked():
            return "sit_to_stand"
        if self.tug_radio.isChecked():
            return "timed_up_and_go"
        if self.biceps_radio.isChecked():
            return "biceps_curl"
        if self.pushup_radio.isChecked():
            return "pushup"
        if self.pullup_radio.isChecked():
            return "pullup"
        if self.leg_raise_radio.isChecked():
            return "leg_raise"
        if self.tandem_radio.isChecked():
            return "tandem_stance"
        if self.stretch_radio.isChecked():
            return "stretch"
        return "sit_to_stand"

    def _reanalyze_with_current_thresholds(self):
        """Re-run analysis with the current thresholds on buffered data.

        Any updated metrics are PERSISTED to the workouts database (overwriting
        the previous analysis for the same session).
        """
        if self._last_times is None:
            return

        wt = self._last_workout_type
        session_id = self._last_session_id

        if wt == "timed_up_and_go":
            if self._last_head_xyz is None or self._last_hip_z is None:
                return
            self._run_tug_analysis(
                self._last_times, self._last_hip_z, self._last_head_xyz, session_id,
            )

        elif wt == "biceps_curl":
            if self._last_elbow_angles is None:
                return
            self._run_biceps_analysis(
                self._last_times, self._last_elbow_angles, session_id,
            )

        elif wt == "pushup":
            if self._last_shoulder_z is None:
                return
            self._run_pushup_analysis(
                self._last_times, self._last_shoulder_z, session_id,
            )

        elif wt == "pullup":
            if self._last_head_z is None:
                return
            self._run_pullup_analysis(
                self._last_times, self._last_head_z, session_id,
            )

        elif wt == "leg_raise":
            if self._last_knee_z is None:
                return
            self._run_leg_raise_analysis(
                self._last_times, self._last_knee_z, session_id,
            )

        elif wt == "tandem_stance":
            if self._last_hip_xy is None:
                return
            self._run_tandem_analysis(
                self._last_times, self._last_hip_xy, session_id,
            )

        elif wt == "stretch":
            if self._last_hip_z is None:
                return
            self._run_stretch_analysis(
                self._last_times, self._last_hip_z, session_id,
            )

        else:  # sit_to_stand
            if self._last_hip_z is None:
                return
            self._run_sts_analysis(
                self._last_times, self._last_hip_z, session_id,
            )

    def _update_analysis_plot(self, times: np.ndarray, hip_z: np.ndarray, result):
        """Update the plot with new data and rep markers."""
        from ..analysis.sit_to_stand import _smooth
        smoothed = _smooth(hip_z, window=5)

        self.hip_trace_raw.setData(times, hip_z)
        self.hip_trace_smooth.setData(times, smoothed)

        # Rep peaks
        if result.rep_peak_times and result.rep_peak_heights:
            self.rep_peaks_scatter.setData(
                x=result.rep_peak_times, y=result.rep_peak_heights,
            )
        else:
            self.rep_peaks_scatter.clear()

        # Summary label
        if result.rep_count > 0:
            summary = (
                f"Reps: {result.rep_count}  |  "
                f"Total time: {result.total_time_seconds:.1f}s"
            )
            if result.avg_power_watts:
                summary += f"  |  Avg power: {result.avg_power_watts:.1f}W"
        else:
            summary = f"No reps detected (threshold = {result.seated_threshold_m:.2f}m)"
        self.analysis_summary_label.setText(summary)
        self.analysis_summary_label.setStyleSheet(
            "color: #4CAF50;" if result.rep_count > 0 else "color: #FFC107;"
        )

    def _update_pushup_plot(self, times: np.ndarray, shoulder_z: np.ndarray, result):
        """Update the plot for pushups: shoulder height trace + yellow dots at each rep's bottom."""
        from ..analysis._rep_common import smooth
        smoothed = smooth(shoulder_z, window=5)

        self.hip_trace_raw.setData(times, shoulder_z)
        self.hip_trace_smooth.setData(times, smoothed)

        # Place dots at the actual minimum of each rep (visible on the trace)
        if result.rep_min_times and result.rep_min_heights:
            self.rep_peaks_scatter.setData(
                x=result.rep_min_times, y=result.rep_min_heights,
            )
        else:
            self.rep_peaks_scatter.clear()

        if result.rep_count > 0:
            summary = (
                f"Pushups: {result.rep_count}  |  "
                f"Total time: {result.total_time_seconds:.1f}s  |  "
                f"Avg range: {result.avg_range_m * 100:.0f}cm"
            )
            self.analysis_summary_label.setStyleSheet("color: #4CAF50;")
        else:
            summary = f"No reps detected (threshold = {result.top_threshold_m:.2f}m)"
            self.analysis_summary_label.setStyleSheet("color: #FFC107;")
        self.analysis_summary_label.setText(summary)

    def _update_leg_raise_plot(self, times: np.ndarray, knee_z: np.ndarray, result):
        """Update the plot for leg raises: knee Z trace + dots at each peak."""
        from ..analysis._rep_common import smooth
        smoothed = smooth(knee_z, window=5)
        self.hip_trace_raw.setData(times, knee_z)
        self.hip_trace_smooth.setData(times, smoothed)

        if result.rep_peak_times and result.rep_peak_heights:
            self.rep_peaks_scatter.setData(
                x=result.rep_peak_times, y=result.rep_peak_heights,
            )
        else:
            self.rep_peaks_scatter.clear()

        if result.rep_count > 0:
            summary = (
                f"Leg raises: {result.rep_count}  |  "
                f"Total time: {result.total_time_seconds:.1f}s  |  "
                f"Avg lift: {result.avg_range_m * 100:.0f}cm"
            )
            self.analysis_summary_label.setStyleSheet("color: #4CAF50;")
        else:
            summary = f"No reps detected (threshold = {result.lift_threshold_m:.2f}m)"
            self.analysis_summary_label.setStyleSheet("color: #FFC107;")
        self.analysis_summary_label.setText(summary)

    def _update_tandem_plot(self, result):
        """Update the plot for tandem stance: sway trace + hold window highlight."""
        self.hip_trace_raw.setData(result.times, result.sway)
        self.hip_trace_smooth.setData(result.times, result.sway)
        self.rep_peaks_scatter.clear()

        summary = (
            f"Hold: {result.hold_seconds:.1f}s  |  "
            f"Total: {result.total_seconds:.1f}s  |  "
            f"Stable: {result.stability_fraction * 100:.0f}%  |  "
            f"Max sway: {result.max_sway_m * 100:.1f}cm"
        )
        if result.hold_seconds >= 10:
            self.analysis_summary_label.setStyleSheet("color: #4CAF50;")
        elif result.hold_seconds >= 5:
            self.analysis_summary_label.setStyleSheet("color: #FFC107;")
        else:
            self.analysis_summary_label.setStyleSheet("color: #FF5252;")
        self.analysis_summary_label.setText(summary)

    def _update_stretch_plot(self, times: np.ndarray, hip_z: np.ndarray, result):
        """Update the plot for stretch: hip Z trace over time."""
        self.hip_trace_raw.setData(times, hip_z)
        self.hip_trace_smooth.setData(result.times, result.hip_z)
        self.rep_peaks_scatter.clear()

        summary = (
            f"Held: {result.hold_seconds:.1f}s  |  "
            f"Steadiness: {result.steadiness * 100:.0f}%  |  "
            f"Range: {result.max_range_m * 100:.0f}cm"
        )
        if result.hold_seconds >= 20:
            self.analysis_summary_label.setStyleSheet("color: #4CAF50;")
        else:
            self.analysis_summary_label.setStyleSheet("color: #FFC107;")
        self.analysis_summary_label.setText(summary)

    def _update_pullup_plot(self, times: np.ndarray, head_z: np.ndarray, result):
        """Update the plot for pullups: head height trace + dots at each rep's top."""
        from ..analysis._rep_common import smooth
        smoothed = smooth(head_z, window=5)

        self.hip_trace_raw.setData(times, head_z)
        self.hip_trace_smooth.setData(times, smoothed)

        if result.rep_peak_times and result.rep_peak_heights:
            self.rep_peaks_scatter.setData(
                x=result.rep_peak_times, y=result.rep_peak_heights,
            )
        else:
            self.rep_peaks_scatter.clear()

        if result.rep_count > 0:
            summary = (
                f"Pullups: {result.rep_count}  |  "
                f"Total time: {result.total_time_seconds:.1f}s  |  "
                f"Avg range: {result.avg_range_m * 100:.0f}cm"
            )
            self.analysis_summary_label.setStyleSheet("color: #4CAF50;")
        else:
            summary = f"No reps detected (bar height = {result.top_threshold_m:.2f}m)"
            self.analysis_summary_label.setStyleSheet("color: #FFC107;")
        self.analysis_summary_label.setText(summary)

    def _update_angle_plot(self, times: np.ndarray, angles: np.ndarray, result):
        """Update the plot for angle-based workouts (biceps curls)."""
        from ..analysis._rep_common import smooth
        smoothed = smooth(angles, window=5)

        self.hip_trace_raw.setData(times, angles)
        self.hip_trace_smooth.setData(times, smoothed)

        if result.rep_peak_times:
            peak_heights = [result.extended_threshold_deg] * len(result.rep_peak_times)
            self.rep_peaks_scatter.setData(
                x=result.rep_peak_times, y=peak_heights,
            )
        else:
            self.rep_peaks_scatter.clear()

        if result.rep_count > 0:
            summary = (
                f"Reps: {result.rep_count}  |  "
                f"Total time: {result.total_time_seconds:.1f}s  |  "
                f"Avg range: {result.avg_range_deg:.0f}\u00b0"
            )
            self.analysis_summary_label.setStyleSheet("color: #4CAF50;")
        else:
            summary = (
                f"No reps detected (threshold = {result.extended_threshold_deg:.0f}\u00b0)"
            )
            self.analysis_summary_label.setStyleSheet("color: #FFC107;")
        self.analysis_summary_label.setText(summary)

    def _update_tug_plot(self, result):
        """Update the plot with TUG head speed trace and start/end markers."""
        # Hide sit-to-stand traces, show head speed
        self.head_speed_trace.setData(result.times, result.head_speed)

        # Clear rep scatter (not used for TUG)
        self.rep_peaks_scatter.clear()

        # Start/end vertical lines
        if result.start_valid and result.duration_seconds > 0:
            self.tug_start_line.setPos(result.start_time)
            self.tug_end_line.setPos(result.end_time)
            self.tug_start_line.setVisible(True)
            self.tug_end_line.setVisible(True)
            summary = (
                f"TUG Duration: {result.duration_seconds:.2f} s  |  "
                f"Start: {result.start_time:.2f}s  |  "
                f"End: {result.end_time:.2f}s  |  "
                f"Max head speed: {result.max_head_speed:.2f} m/s"
            )
            self.analysis_summary_label.setStyleSheet("color: #4CAF50;")
        else:
            self.tug_start_line.setVisible(False)
            self.tug_end_line.setVisible(False)
            summary = (
                f"TUG not detected \u2014 check thresholds "
                f"(seated={result.seated_threshold_m:.2f}m, "
                f"speed={result.speed_threshold_mps:.2f}m/s)"
            )
            self.analysis_summary_label.setStyleSheet("color: #FFC107;")
        self.analysis_summary_label.setText(summary)

    # ── User management ──

    def _refresh_user_list(self):
        try:
            from ..config import list_users
            users = list_users()
            self.user_combo.clear()
            for user in users:
                self.user_combo.addItem(user["username"], user["id"])
        except Exception:
            pass

    def _reset_for_user_switch(self):
        """Tear down workers + clear all per-session state in-memory.

        Called from _on_login when switching to a different user, so the
        new user gets a clean slate without restarting the application.
        Specifically this fixes the "ghost frames" bug: the camera grid,
        skeleton view, FPS graph, paint-coalescing buffers and analysis
        result arrays all kept the previous user's data otherwise, and
        you'd see stale annotated frames + skeleton until enough new
        frames came in to overwrite them.

        What is preserved (deliberately not user-scoped):
          - per-model rotate/zero presets in view_transforms.db
          - calibration (extrinsics, intrinsics)
          - app settings (last_detect_*, csv_export_immediate, ...)
          - detection model + backend selection in the dropdowns

        What is cleared:
          - active workers (recording, detection, preview) so the new
            session starts fresh — also forces the cross-frame person
            tracker in PoseDetectionWorker to forget previous track ids
          - opened cameras (re-opened by the auto-start that follows)
          - in-memory keypoint / FPS / arrival buffers
          - the camera grid, skeleton view, FPS graph displays
          - the analysis result arrays + plot UI
        """
        # 1. Stop active workers.
        if getattr(self, "recording_worker", None) is not None:
            try:
                self.recording_worker.running = False
                self.recording_worker.wait(2000)
            except Exception:
                pass
            self.recording_worker = None
        try:
            self._stop_detection()
        except Exception:
            pass
        if getattr(self, "preview_worker", None) is not None:
            try:
                self.preview_worker.stop()
                self.preview_worker.wait(2000)
            except Exception:
                pass
            self.preview_worker = None
        try:
            self._close_cameras()
        except Exception:
            pass

        # 2. Clear per-session data buffers.
        self._recording_keypoints = []
        self._record_arrivals = {}
        try:
            self._last_annotated.clear()
        except Exception:
            self._last_annotated = {}
        self._last_frame_time = {}
        self._pending_grid_frames = {}
        self._grid_paint_scheduled = False
        self._pending_persons_3d = None
        self._kp3d_paint_scheduled = False
        self._is_recording = False
        self._tracking_paused_for_recording = False
        self._offline_csv_pending = False
        self._primary_person_index = 0

        # 3. Clear analysis result state — last_* arrays drive the plot
        # widgets, so leaving them around shows the previous user's reps.
        for attr in (
            "_last_times", "_last_hip_z", "_last_hip_xy", "_last_head_xyz",
            "_last_head_z", "_last_elbow_angles", "_last_shoulder_z",
            "_last_knee_z",
        ):
            if hasattr(self, attr):
                setattr(self, attr, None)
        self._last_session_id = None

        # 4. Clear the visible widgets.
        try:
            self.skeleton_view.clear()           # drops persons (keeps R, t)
        except Exception:
            pass
        try:
            self.skeleton_view.clear_footsteps()
        except Exception:
            pass
        try:
            self.camera_grid.clear_all()
        except Exception:
            pass
        try:
            self.fps_graph.clear_all()
        except Exception:
            pass
        try:
            # Repaint the (now empty) plot panel so stale lines disappear.
            self._apply_plot_mode(self._selected_workout_type())
        except Exception:
            pass
        try:
            self.results_label.setText("No results yet.")
        except Exception:
            pass

    def _on_login(self):
        username = self.user_combo.currentText().strip()
        if not username:
            self.status_message.emit("Please enter a username")
            return

        try:
            from ..config import get_user, create_user
            user = get_user(username)
            if user is None:
                user = create_user(username)
                self.status_message.emit(f"Created new user: {username}")
                self._refresh_user_list()
                for i in range(self.user_combo.count()):
                    if self.user_combo.itemData(i) == user["id"]:
                        self.user_combo.setCurrentIndex(i)
                        break
            else:
                self.status_message.emit(f"Logged in as: {username}")

            # Detect user change BEFORE we overwrite _current_user_id, so
            # we can decide whether to reset session state. A first login
            # (current_user_id is None) doesn't need a reset — there's
            # nothing to reset. A re-login as the same user also doesn't
            # need it (saves a worker reload). Switching to a different
            # user does.
            user_changed = (
                self._current_user_id is not None
                and self._current_user_id != user["id"]
            )
            if user_changed:
                self._reset_for_user_switch()

            self._current_user_id = user["id"]
            self._current_username = user["username"]
            self.mass_spin.setEnabled(True)
            if user.get("mass_kg"):
                self.mass_spin.setValue(user["mass_kg"])
            self.user_status.setText(f"Logged in as {username}")
            self.user_status.setStyleSheet("color: #4CAF50;")

            # Load (or pick) the user's active program
            self._load_user_program(user)

            self._update_record_enabled()

            # Auto-start the full pipeline: enumerate → open → preview → detect.
            self._auto_start_pipeline = True
            self._on_init_cameras()
        except Exception as e:
            self.status_message.emit(f"Login failed: {e}")

    def _load_user_program(self, user: dict):
        """Load the user's active program, or open the picker if none is set."""
        from ..config import (
            get_user_by_id, get_program, get_program_exercises,
            list_programs, set_user_program,
        )

        # Reload fresh in case this user was just created
        fresh = get_user_by_id(user["id"]) or user
        program_id = fresh.get("active_program_id")

        if program_id is None:
            # No program yet — pop up the picker
            programs = list_programs()
            if not programs:
                self.status_message.emit("No programs available")
                self._show_manual_fallback(True)
                return
            exercises_by_program = {
                p["id"]: get_program_exercises(p["id"]) for p in programs
            }
            from .program_picker import ProgramPickerDialog
            dlg = ProgramPickerDialog(programs, exercises_by_program, parent=self)
            if dlg.exec() and dlg.selected_program_id is not None:
                set_user_program(fresh["id"], dlg.selected_program_id)
                program_id = dlg.selected_program_id
                self.status_message.emit("Program selected")
            else:
                # User cancelled — allow manual fallback
                self._show_manual_fallback(True)
                return

        program = get_program(program_id)
        exercises = get_program_exercises(program_id)
        if not program or not exercises:
            self._show_manual_fallback(True)
            return

        self._active_program = program
        self._active_program_exercises = exercises

        # Compute weekly progress for each exercise
        started_str = fresh.get("program_started_at")
        started_dt = None
        if started_str:
            try:
                from datetime import datetime
                started_dt = datetime.fromisoformat(started_str.replace(" ", "T"))
            except Exception:
                started_dt = None
        sets_week, sets_today = self._compute_weekly_sets(
            fresh["id"], exercises, started_dt)

        self.todays_plan.set_program(
            program, exercises, sets_week, sets_today, started_dt)
        self._show_manual_fallback(False)

        # Auto-pick the first exercise (or today's first one)
        if exercises:
            first = self.todays_plan.get_selected_exercise() or exercises[0]
            self._apply_program_exercise(first)

    def _show_manual_fallback(self, show: bool):
        self.manual_group.setVisible(show)
        self.todays_plan.setVisible(not show)

    def _compute_weekly_sets(self, user_id: int, exercises: list[dict],
                              program_started_at) -> tuple[dict[int, int], dict[int, int]]:
        """Count sessions per program_exercise for the current week + today.

        Returns (sets_done_week, sets_done_today) keyed by program_exercise_id.
        """
        from datetime import datetime, timedelta
        from ..config import count_sets_since

        if program_started_at is None:
            now = datetime.now()
            week_start = now - timedelta(days=now.isoweekday() - 1)
            week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
        else:
            now = datetime.now()
            delta_days = (now - program_started_at).days
            week_index = delta_days // 7
            week_start = program_started_at + timedelta(days=week_index * 7)

        today_start = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)

        week_since_str = week_start.strftime("%Y-%m-%d %H:%M:%S")
        today_since_str = today_start.strftime("%Y-%m-%d %H:%M:%S")

        sets_week = {
            ex["id"]: count_sets_since(user_id, ex["id"], week_since_str)
            for ex in exercises
        }
        sets_today = {
            ex["id"]: count_sets_since(user_id, ex["id"], today_since_str)
            for ex in exercises
        }
        return sets_week, sets_today

    def _on_plan_exercise_selected(self, exercise: dict):
        """User clicked an exercise in the Today's Plan widget."""
        self._apply_program_exercise(exercise)
        self._refresh_longterm_graph()

    def _apply_program_exercise(self, exercise: dict):
        """Switch the workout page over to the given program exercise."""
        self._current_program_exercise = exercise
        wt = exercise["workout_type"]

        # Auto-switch model based on workout type
        if wt == "hand_squeeze":
            idx = self.detect_model_combo.findData("mediapipe_hands")
            if idx >= 0:
                self.detect_model_combo.setCurrentIndex(idx)
        else:
            idx = self.detect_model_combo.findData("synthpose")
            if idx >= 0:
                self.detect_model_combo.setCurrentIndex(idx)

        # Mirror the choice into the hidden radio group so legacy
        # _selected_workout_type() keeps working.
        if wt == "sit_to_stand":
            self.sts_radio.setChecked(True)
        elif wt == "timed_up_and_go":
            self.tug_radio.setChecked(True)
        elif wt == "biceps_curl":
            self.biceps_radio.setChecked(True)
        elif wt == "pushup":
            self.pushup_radio.setChecked(True)
        elif wt == "pullup":
            self.pullup_radio.setChecked(True)
        elif wt == "leg_raise":
            self.leg_raise_radio.setChecked(True)
        elif wt == "tandem_stance":
            self.tandem_radio.setChecked(True)
        elif wt == "stretch":
            self.stretch_radio.setChecked(True)

        self._apply_plot_mode(wt)
        self._update_record_button_label()
        self._update_record_enabled()

    def _next_set_number_for_current_exercise(self) -> int:
        """Return the next set number (1-indexed) for the current program exercise."""
        if self._current_program_exercise is None or self._current_user_id is None:
            return 1
        try:
            from datetime import datetime, timedelta
            from ..config import count_sets_since, get_user_by_id
            user = get_user_by_id(self._current_user_id)
            started_str = user.get("program_started_at") if user else None
            if started_str:
                started_dt = datetime.fromisoformat(started_str.replace(" ", "T"))
                delta_days = (datetime.now() - started_dt).days
                week_start = started_dt + timedelta(days=(delta_days // 7) * 7)
            else:
                now = datetime.now()
                week_start = now - timedelta(days=now.isoweekday() - 1)
                week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
            done = count_sets_since(
                self._current_user_id,
                self._current_program_exercise["id"],
                week_start.strftime("%Y-%m-%d %H:%M:%S"),
            )
            return done + 1
        except Exception:
            return 1

    def _refresh_todays_plan(self):
        """Re-count weekly sets and rebuild the Today's Plan widget."""
        if self._current_user_id is None or not self._active_program_exercises:
            return
        try:
            from datetime import datetime
            from ..config import get_user_by_id
            user = get_user_by_id(self._current_user_id)
            started_str = user.get("program_started_at") if user else None
            started_dt = None
            if started_str:
                try:
                    started_dt = datetime.fromisoformat(started_str.replace(" ", "T"))
                except Exception:
                    started_dt = None
            sets_week, sets_today = self._compute_weekly_sets(
                self._current_user_id, self._active_program_exercises, started_dt,
            )
            self.todays_plan.set_program(
                self._active_program, self._active_program_exercises,
                sets_week, sets_today, started_dt,
            )
        except Exception as e:
            self.status_message.emit(f"Failed to refresh plan: {e}")

    def _update_record_button_label(self):
        """Refresh the record button text based on program exercise + set progress."""
        if self._is_recording:
            return
        ex = self._current_program_exercise
        if ex is None:
            self.record_btn.setText("Record Workout")
            return

        # How many sets have we done this week? Need next_set = done + 1.
        if self._current_user_id is not None:
            try:
                from datetime import datetime, timedelta
                from ..config import count_sets_since, get_user_by_id
                user = get_user_by_id(self._current_user_id)
                started_str = user.get("program_started_at") if user else None
                if started_str:
                    started_dt = datetime.fromisoformat(
                        started_str.replace(" ", "T"))
                    delta_days = (datetime.now() - started_dt).days
                    week_start = started_dt + timedelta(days=(delta_days // 7) * 7)
                else:
                    now = datetime.now()
                    week_start = now - timedelta(days=now.isoweekday() - 1)
                    week_start = week_start.replace(hour=0, minute=0, second=0, microsecond=0)
                done = count_sets_since(
                    self._current_user_id, ex["id"],
                    week_start.strftime("%Y-%m-%d %H:%M:%S"),
                )
                total = max(1, ex["sets_per_day"] * ex["days_per_week"])
                next_set = done + 1
                # Once the weekly target is met, "Set N of total" with
                # N > total is nonsense (e.g. "Set 2 of 1" for an
                # assessment). Show "Complete" so the operator knows
                # they've already finished, but keep the button clickable
                # in case they want to re-record an extra trial.
                if done >= total:
                    self.record_btn.setText(
                        f"Record {ex['display_name']} — Complete "
                        f"({done}/{total})"
                    )
                else:
                    self.record_btn.setText(
                        f"Record {ex['display_name']} — Set {next_set} of {total}"
                    )
                return
            except Exception:
                pass
        self.record_btn.setText(f"Record {ex['display_name']}")

    def _on_mass_changed(self, value: float):
        if self._current_user_id is None:
            return
        try:
            from ..config import update_user_mass
            update_user_mass(self._current_user_id, value)
        except Exception as e:
            self.status_message.emit(f"Failed to save mass: {e}")

    # ── Camera preferences ──

    def _load_camera_prefs(self):
        try:
            from ..config import load_app_settings, load_project_settings
            app = load_app_settings()
            folder = app.get("last_project_folder")
            if folder:
                settings = load_project_settings(Path(folder))
                self._serial_prefs = settings.get("cameras", {})
                fps = settings.get("fps")
                if fps:
                    self._target_fps = int(fps)
        except Exception:
            pass
        # Sync the FPS graph to the loaded recording rate. With time_window_s
        # set, this also resizes the rolling buffer to ~5 s of samples.
        if hasattr(self, "fps_graph"):
            self.fps_graph.set_target_fps(self._target_fps)

    # ── Camera initialization ──

    def _on_init_cameras(self):
        self.init_cameras_btn.setEnabled(False)
        self.status_message.emit("Enumerating cameras...")

        self.enumerate_worker = CameraEnumerateWorker()
        self.enumerate_worker.cameras_found.connect(self._on_cameras_found)
        self.enumerate_worker.error.connect(self._on_enumerate_error)
        self.enumerate_worker.finished.connect(
            lambda: self.init_cameras_btn.setEnabled(True)
        )
        self.enumerate_worker.start()

    def _on_cameras_found(self, cameras: list):
        from ..config import load_all_nicknames
        nicknames = load_all_nicknames()

        # Resolve calibration against the *exact* set of plugged-in cameras
        # before building CameraStates — port mapping below depends on it.
        self._resolve_calibration_for_serials(
            {cam.serial_number for cam in cameras}
        )

        # Seed per-serial preferences so saved resolutions/exposure flow into
        # CameraState (and through to intrinsic lookup) on first login.
        def _initial_res(serial: str):
            raw = self._serial_prefs.get(serial, {}).get("resolution")
            return tuple(raw) if raw else None

        cal_serial_to_port: dict[str, int] = {}
        if self._calibrated_cameras:
            for port, cal_cam in self._calibrated_cameras.items():
                cal_serial_to_port[cal_cam.serial_number] = port

        camera_states = {}
        matched = 0
        skipped = []
        for cam in cameras:
            if cam.serial_number in cal_serial_to_port:
                port = cal_serial_to_port[cam.serial_number]
                nickname = nicknames.get(cam.serial_number, "")
                camera_states[port] = CameraState(
                    info=cam, enabled=True, is_open=False, nickname=nickname,
                    selected_resolution=_initial_res(cam.serial_number),
                )
                matched += 1
            else:
                skipped.append(cam.display_name)

        if not cal_serial_to_port:
            for port, cam in enumerate(cameras):
                nickname = nicknames.get(cam.serial_number, "")
                camera_states[port] = CameraState(
                    info=cam, enabled=True, is_open=False, nickname=nickname,
                    selected_resolution=_initial_res(cam.serial_number),
                )
            matched = len(cameras)

        self.state_manager.set_cameras(camera_states)

        count = len(camera_states)
        self.camera_count_label.setText(f"{count} camera(s)")
        self.camera_count_label.setStyleSheet("color: #4CAF50;" if count > 0 else "color: #888;")
        self.preview_btn.setEnabled(count > 0)

        if skipped and cal_serial_to_port:
            self.status_message.emit(
                f"Found {matched} calibrated camera(s), skipped {len(skipped)} uncalibrated"
            )
        else:
            self.status_message.emit(f"Found {count} camera(s)")
        self._update_record_enabled()

        # Auto-chain: if this was triggered by login, also start preview + detection
        if self._auto_start_pipeline and count > 0:
            self._auto_start_pipeline = False
            self.preview_btn.setChecked(True)
            self._start_preview()
            if self._calibrated_cameras is not None:
                self.detect_checkbox.blockSignals(True)
                self.detect_checkbox.setChecked(True)
                self.detect_checkbox.blockSignals(False)
                self._start_detection()

    def _on_enumerate_error(self, error: str):
        self.status_message.emit(f"Camera enumeration failed: {error}")
        self.camera_count_label.setText("Error")
        self.camera_count_label.setStyleSheet("color: #FF5252;")

    def _toggle_preview(self):
        if self.preview_btn.isChecked():
            self._start_preview()
        else:
            self._stop_preview()

    def _start_preview(self):
        from ..camera_binding import open_camera, set_resolution, set_exposure

        cameras = self.state_manager.state.cameras
        self.opened_cameras = []
        self.opened_ports = []
        camera_info = {}
        opened_ports = []

        for port, cam_state in sorted(cameras.items()):
            if not cam_state.enabled:
                continue
            try:
                cam = cam_state.info
                open_camera(cam)
                self.opened_cameras.append(cam)
                self.opened_ports.append(port)
                opened_ports.append(port)
                nick = cam_state.nickname
                camera_info[port] = nick if nick else cam.display_name
            except Exception as e:
                self.status_message.emit(f"Failed to open camera {port}: {e}")

        if not self.opened_cameras:
            self.preview_btn.setChecked(False)
            self.status_message.emit("No cameras opened")
            return

        for port in opened_ports:
            cam_state = cameras[port]
            cam = cam_state.info
            prefs = self._serial_prefs.get(cam.serial_number, {})

            res = prefs.get("resolution")
            if res:
                try:
                    set_resolution(cam, int(res[0]), int(res[1]))
                except Exception as e:
                    self.status_message.emit(f"Port {port}: resolution failed: {e}")

            brightness = prefs.get("brightness", prefs.get("exposure"))
            if brightness is not None:
                try:
                    set_exposure(cam, int(brightness))
                except Exception as e:
                    self.status_message.emit(f"Port {port}: brightness failed: {e}")

        self.camera_grid.set_cameras(camera_info)
        self._last_frame_time.clear()
        self.fps_graph.set_cameras(camera_info)
        self.fps_graph.set_target_fps(self._target_fps)

        self.preview_worker = CameraPreviewWorker(
            self.opened_cameras, self.opened_ports, fps=self._target_fps
        )
        self.preview_worker.frame_captured.connect(self._on_frame_received)
        self.preview_worker.error.connect(self._on_preview_error)
        self.preview_worker.start()

        self.preview_btn.setText("Stop Preview")
        self.state_manager.update_state(is_previewing=True)
        self.status_message.emit(
            f"Preview started ({len(opened_ports)} cameras, {self._target_fps} fps)"
        )

    def _stop_preview(self):
        if self.preview_worker:
            self.preview_worker.stop()
            self.preview_worker.wait()
            self.preview_worker = None

        self._stop_detection()
        self.detect_checkbox.blockSignals(True)
        self.detect_checkbox.setChecked(False)
        self.detect_checkbox.blockSignals(False)

        self._close_cameras()
        self.camera_grid.clear_all()

        self.preview_btn.setText("Preview")
        self.preview_btn.setChecked(False)
        self.state_manager.update_state(is_previewing=False)
        self.status_message.emit("Preview stopped")

    def _close_cameras(self):
        from ..camera_binding import close_camera
        for cam in self.opened_cameras:
            try:
                close_camera(cam)
            except Exception:
                pass
        self.opened_cameras = []
        self.opened_ports = []

    def _on_frame_received(self, port: int, pixels):
        # When detection is running and we already have an annotated frame,
        # don't overwrite it with the raw frame — prevents flicker.
        if self.detection_worker is not None:
            if port not in self._last_annotated:
                # No annotated frame yet — show raw so the grid isn't blank
                self.camera_grid.update_frame(port, pixels)
            self.detection_worker.submit_frame(port, pixels)
        else:
            self.camera_grid.update_frame(port, pixels)

        # FPS graph update (cheap; Qt coalesces the repaint).
        now = time.perf_counter()
        prev = self._last_frame_time.get(port)
        if prev is not None:
            dt = now - prev
            if dt > 0:
                self.fps_graph.push_fps(port, 1.0 / dt)
        self._last_frame_time[port] = now

    def _on_preview_error(self, error: str):
        self.status_message.emit(f"Preview error: {error}")

    # ── Live Detection ──

    def _on_detect_toggled(self, checked: bool):
        # If the detection worker is already loaded, just pause/resume —
        # tearing it down would force a multi-second YOLO + VitPose reload
        # next time the user re-enables detection. Workers without
        # pause/resume fall through to the start/stop path so the toggle
        # always actually does something.
        worker = self.detection_worker
        if worker is not None and hasattr(worker, "pause") and hasattr(worker, "resume"):
            if checked:
                worker.resume()
                self.status_message.emit("Detection resumed")
            else:
                worker.pause()
                # Don't blank the skeleton view here — frozen-on-pause is
                # often what the user wants. _stop_detection clears it on
                # full teardown.
                self.status_message.emit("Detection paused")
            return

        if checked:
            self._start_detection()
        else:
            self._stop_detection()

    def _current_model_key(self) -> str:
        """Stable string id for the active detection model.

        Used as the key under which view-transform presets are stored in
        camera_rig.toml so each model can have its own saved
        rotate-to-human + zero-origin without overwriting siblings.
        """
        return self.detect_model_combo.currentData() or "synthpose"

    def _zero_point_for_model(self) -> tuple[int, str]:
        """Return (keypoint_index, label) for the active model's zero origin."""
        model = self._current_model_key()
        if model == "mediapipe_hands":
            return (4, "L_Thumb")  # MediaPipe hand landmark 4 = thumb tip
        return (15, "L_Ankle")    # SynthPose/COCO keypoint 15

    def _update_zero_button_label(self):
        _, label = self._zero_point_for_model()
        self.zero_origin_button.setText(f"Zero at {label}")

    def _restore_last_detect_state(self):
        """Pull last-used model/backend/confidence from app_settings into
        the GUI so the user lands in the same configuration as last
        session. Signals are blocked during the restore so we don't
        synthetically fire _on_model_changed → _start_detection before
        the rest of __init__ has finished running.
        """
        try:
            from ..config import load_app_settings
            settings = load_app_settings()
        except Exception:
            return

        # Older settings.json may still say "vitpose"; transparently
        # migrate to the canonical "synthpose" key.
        last_model = settings.get("last_detect_model", "synthpose")
        if last_model == "vitpose":
            last_model = "synthpose"
        last_backend = settings.get("last_detect_backend", "pytorch")
        last_conf = float(settings.get("last_detect_confidence", 0.50))

        for combo, key in (
            (self.detect_model_combo, last_model),
            (self.detect_backend_combo, last_backend),
        ):
            idx = combo.findData(key)
            if idx >= 0:
                combo.blockSignals(True)
                combo.setCurrentIndex(idx)
                combo.blockSignals(False)

        # Slider value is an int 10..95.
        slider_val = max(10, min(95, int(round(last_conf * 100))))
        self.conf_slider.blockSignals(True)
        self.conf_slider.setValue(slider_val)
        self.conf_slider.blockSignals(False)
        if hasattr(self, "conf_value_label"):
            self.conf_value_label.setText(f"{slider_val / 100.0:.2f}")

    def _persist_last_detect_state(self):
        """Save the current model/backend/confidence to app_settings.

        Called from _on_model_changed and _on_confidence_changed so the
        next launch picks up where the user left off.
        """
        try:
            from ..config import load_app_settings, save_app_settings
            settings = load_app_settings()
            settings["last_detect_model"] = (
                self.detect_model_combo.currentData() or "synthpose"
            )
            settings["last_detect_backend"] = (
                self.detect_backend_combo.currentData() or "pytorch"
            )
            settings["last_detect_confidence"] = (
                self.conf_slider.value() / 100.0
            )
            save_app_settings(settings)
        except Exception:
            pass

    def _on_model_changed(self, _index: int = 0):
        """Restart detection when the model or backend dropdown changes."""
        self._update_zero_button_label()
        self._persist_last_detect_state()
        # Switch to whichever rotate/zero preset was last saved for this
        # model (or reset to identity if none) so the user doesn't have
        # to re-zero every time they flip between models.
        self._load_view_transform()
        # Guard: don't restart during initial setup or if preview isn't running
        if not self.state_manager.state.is_previewing:
            return
        if self.detection_worker is not None:
            self._stop_detection()
            # Small delay to let the old thread fully release resources
            from PySide6.QtCore import QTimer
            QTimer.singleShot(200, self._start_detection)

    def _on_confidence_changed(self, value: int):
        """Slider tick (10–95) → confidence (0.10–0.95). Update label and live worker.

        For PyTorch the worker honors `confidence_threshold` on the next frame
        (no restart needed). For CUDA the threshold is fixed at pipeline init,
        so we restart detection so the new value takes effect.
        """
        conf = value / 100.0
        if hasattr(self, "conf_value_label"):
            self.conf_value_label.setText(f"{conf:.2f}")
        # Persist the new confidence so it's restored next launch.
        self._persist_last_detect_state()
        worker = self.detection_worker
        if worker is None:
            return
        # PyTorch worker reads this attr each frame.
        if isinstance(worker, PoseDetectionWorker):
            try:
                worker.confidence_threshold = conf
            except Exception:
                pass
            return
        # CUDA / MPS workers bake the threshold into the C config at create
        # time — restart so the new value takes effect.
        from .workers import MpsStreamDetectionWorker
        if isinstance(worker, (CudaStreamDetectionWorker, MpsStreamDetectionWorker)):
            from PySide6.QtCore import QTimer
            self._stop_detection()
            QTimer.singleShot(200, self._start_detection)

    def _current_person_confidence(self) -> float:
        return self.conf_slider.value() / 100.0 if hasattr(self, "conf_slider") else 0.50

    def _start_detection(self):
        if self.detection_worker is not None:
            return

        model = self.detect_model_combo.currentData() or "synthpose"

        if model == "mediapipe_hands":
            self._start_mediapipe_hands_detection()
            return

        cameras = self._calibrated_cameras
        if cameras is None:
            self.skeleton_view.set_message("No extrinsic calibration")

        backend = self.detect_backend_combo.currentData() or "pytorch"

        if backend == "cuda" and cameras is not None and self._calibration_available:
            self._start_cuda_detection(cameras)
        elif backend == "mps" and cameras is not None and self._calibration_available:
            self._start_mps_detection(cameras)
        else:
            self._start_pytorch_detection(cameras)

    def _normalize_calibrated_cameras(self, cameras):
        """Scale every camera's intrinsics to match the live recording resolution.

        Calibration on disk may store each camera's intrinsics at a different
        resolution (one cross-AR scaled, another an exact DB match). The 2D
        detections coming in are always at the live frame resolution, so any
        camera whose K matrix was calibrated at a different size produces a
        miscalibrated fundamental matrix → cross-camera matching mis-pairs
        people → triangulation produces 3D garbage even though 2D detections
        are correct.
        """
        if cameras is None:
            return None

        from ..types import CalibratedCamera, scale_intrinsics
        import dataclasses

        target_res = None
        live_cams = self.state_manager.state.cameras
        for port in sorted(cameras.keys()):
            cam_state = live_cams.get(port)
            if cam_state is not None:
                res = cam_state.selected_resolution or (
                    cam_state.info.width, cam_state.info.height
                )
                if res and res[0] > 0 and res[1] > 0:
                    target_res = (int(res[0]), int(res[1]))
                    break
        if target_res is None:
            first_port = sorted(cameras.keys())[0]
            target_res = tuple(cameras[first_port].intrinsics.resolution)

        normalized: dict = {}
        for port, cc in cameras.items():
            if tuple(cc.intrinsics.resolution) == target_res:
                normalized[port] = cc
            else:
                scaled = scale_intrinsics(cc.intrinsics, target_res)
                normalized[port] = dataclasses.replace(cc, intrinsics=scaled)
        return normalized

    def _start_pytorch_detection(self, cameras):
        """Start the Python-based detection worker (YOLO + VitPose via PyTorch)."""
        cameras = self._normalize_calibrated_cameras(cameras)
        self.detection_worker = PoseDetectionWorker(device_name="auto", cameras=cameras)
        self.detection_worker.confidence_threshold = self._current_person_confidence()
        self.detection_worker.detection_ready.connect(self._on_detection_ready)
        self.detection_worker.error.connect(self._on_detection_error)
        self.detection_worker.finished.connect(self._on_detection_finished)
        if cameras is not None:
            self.detection_worker.keypoints_3d_ready.connect(self._on_keypoints_3d)
        self.detection_worker.start()
        self.status_message.emit("Detection started (PyTorch)")

    def _start_cuda_detection(self, cameras):
        """Start the CUDA TensorRT streaming detection worker."""
        from .workers import CudaStreamDetectionWorker
        from ..config import (
            engine_cache_dir,
            models_dir,
            write_cuda_calibration_toml,
        )
        import tempfile

        # Normalize every camera's intrinsics to the live recording resolution
        # before handing off — see _normalize_calibrated_cameras for details.
        cameras = self._normalize_calibrated_cameras(cameras)

        # Write a CUDA-compatible calibration TOML (C parser needs intrinsics inline)
        cuda_cal_path = Path(tempfile.gettempdir()) / "calimerge_cuda_calibration.toml"
        write_cuda_calibration_toml(cameras, cuda_cal_path)

        # Resolve ONNX model paths under the app data dir, with a one-shot
        # fallback to the legacy <repo>/models/onnx/ location so long-time
        # users don't immediately break on the migration.
        onnx_dir = models_dir() / "onnx"
        legacy_onnx_dir = Path(__file__).resolve().parent.parent.parent.parent / "models" / "onnx"

        def _resolve_onnx(filename: str) -> str:
            primary = onnx_dir / filename
            if primary.exists():
                return str(primary)
            legacy = legacy_onnx_dir / filename
            if legacy.exists():
                return str(legacy)
            return str(primary)  # let the C side surface a clear missing-file error

        yolo_onnx = _resolve_onnx("yolo_v10s.onnx")
        vitpose_onnx = _resolve_onnx("vitpose_synthpose.onnx")

        # Engine cache is shared across recording sessions — engines depend on
        # (model, GPU, TRT version, precision), not the project folder.
        cache = engine_cache_dir()
        cache.mkdir(parents=True, exist_ok=True)
        engine_cache = str(cache)

        self.detection_worker = CudaStreamDetectionWorker(
            cameras=cameras,
            calibration_path=str(cuda_cal_path),
            yolo_onnx=yolo_onnx,
            vitpose_onnx=vitpose_onnx,
            engine_cache=engine_cache,
            max_persons=2,
            person_confidence=self._current_person_confidence(),
        )
        self.detection_worker.log_message.connect(
            lambda msg: self.status_message.emit(msg)
        )
        self.detection_worker.detection_ready.connect(self._on_detection_ready)
        self.detection_worker.error.connect(self._on_detection_error)
        self.detection_worker.finished.connect(self._on_detection_finished)
        self.detection_worker.keypoints_3d_ready.connect(self._on_keypoints_3d)
        self.detection_worker.start()
        self.status_message.emit("Detection started (CUDA TensorRT)")

    def _start_mps_detection(self, cameras):
        """Start the MPS / CoreML streaming detection worker (macOS only)."""
        from .workers import MpsStreamDetectionWorker
        from ..config import (
            models_dir,
            write_cuda_calibration_toml,
        )
        import tempfile

        # Normalise every camera's intrinsics to the live recording
        # resolution before handing off.  Same rationale as CUDA path:
        # mismatched K -> wrong fundamental matrix -> 3D garbage.
        cameras = self._normalize_calibrated_cameras(cameras)

        # The MPS pipeline reads the same calibration TOML format the
        # CUDA pipeline does (pt_calibration.c is platform-independent
        # under-the-hood), so we reuse write_cuda_calibration_toml.
        mps_cal_path = Path(tempfile.gettempdir()) / "calimerge_mps_calibration.toml"
        write_cuda_calibration_toml(cameras, mps_cal_path)

        # Resolve CoreML model paths under the app data dir, falling back
        # to <repo>/models/coreml/.  Matches the ONNX resolution policy
        # for the CUDA path.
        coreml_dir = models_dir() / "coreml"
        legacy_coreml_dir = (
            Path(__file__).resolve().parent.parent.parent.parent
            / "models" / "coreml"
        )

        def _resolve_coreml(filename: str) -> str:
            primary = coreml_dir / filename
            if primary.exists():
                return str(primary)
            legacy = legacy_coreml_dir / filename
            if legacy.exists():
                return str(legacy)
            return str(primary)  # let the .m side surface a clear missing-file error

        yolo_model = _resolve_coreml("yolo_v10s.mlpackage")
        vitpose_model = _resolve_coreml("vitpose_synthpose.mlpackage")

        self.detection_worker = MpsStreamDetectionWorker(
            cameras=cameras,
            calibration_path=str(mps_cal_path),
            yolo_model_path=yolo_model,
            vitpose_model_path=vitpose_model,
            max_persons=2,
            person_confidence=self._current_person_confidence(),
        )
        self.detection_worker.log_message.connect(
            lambda msg: self.status_message.emit(msg)
        )
        self.detection_worker.detection_ready.connect(self._on_detection_ready)
        self.detection_worker.error.connect(self._on_detection_error)
        self.detection_worker.finished.connect(self._on_detection_finished)
        self.detection_worker.keypoints_3d_ready.connect(self._on_keypoints_3d)
        self.detection_worker.start()
        self.status_message.emit("Detection started (MPS / CoreML)")

    def _start_mediapipe_hands_detection(self):
        """Start the MediaPipe Hands detection worker."""
        from .workers import MediaPipeHandsDetectionWorker

        self.detection_worker = MediaPipeHandsDetectionWorker(max_hands=2)
        self.detection_worker.detection_ready.connect(self._on_detection_ready)
        self.detection_worker.log_message.connect(
            lambda msg: self.status_message.emit(msg)
        )
        self.detection_worker.error.connect(self._on_detection_error)
        self.detection_worker.finished.connect(self._on_detection_finished)
        self.detection_worker.start()
        self.skeleton_view.set_message("Hand detection (2D only)")
        self.status_message.emit("Detection started (MediaPipe Hands)")

    def _stop_detection(self):
        # Mark this stop as intentional so _on_detection_finished (fired by
        # the QThread.finished signal once the worker thread exits) doesn't
        # uncheck the live-detection box. Without this flag, pausing
        # detection for a recording uncheck the box, and _on_record_finished
        # then sees `detect_checkbox.isChecked() == False` and never
        # restarts detection after the trial ends.
        self._detection_stopping_intentionally = True
        if self.detection_worker is not None:
            self.detection_worker.stop()
            if not self.detection_worker.wait(5000):
                self.status_message.emit("Detection worker taking long to stop...")
                self.detection_worker.wait(30000)
            self.detection_worker = None
            self._last_annotated.clear()
        self.skeleton_view.clear()
        self.rotate_to_human_button.setEnabled(False)
        self.zero_origin_button.setEnabled(False)

    def _on_detection_ready(self, port: int, annotated_frame):
        # Always cache the latest annotated frame — recording / re-render
        # paths read it off this dict, independent of whether the live grid
        # is currently being repainted.
        self._last_annotated[port] = annotated_frame.copy()
        # Coalesce: stash the freshest frame per port and schedule a single
        # paint. Multiple queued slot calls between paints collapse into one
        # repaint of the latest, so a backlog of stale `detection_ready`
        # signals can't replay minute-old footage when the UI thread is
        # overloaded.
        self._pending_grid_frames[port] = annotated_frame
        if not self._grid_paint_scheduled:
            self._grid_paint_scheduled = True
            QTimer.singleShot(0, self._paint_pending_grid_frames)

    def _paint_pending_grid_frames(self):
        self._grid_paint_scheduled = False
        latest = dict(self._pending_grid_frames)
        self._pending_grid_frames.clear()
        for port, frame in latest.items():
            self.camera_grid.update_frame(port, frame)

    def _on_detection_finished(self):
        # Intentional stop (e.g. pausing tracking for a recording, or model
        # change): leave the checkbox alone so the caller can restart later.
        if getattr(self, "_detection_stopping_intentionally", False):
            self._detection_stopping_intentionally = False
            return
        # Otherwise the worker died unexpectedly — uncheck the box so the
        # UI reflects reality.
        if self.detect_checkbox.isChecked():
            self.detect_checkbox.blockSignals(True)
            self.detect_checkbox.setChecked(False)
            self.detect_checkbox.blockSignals(False)
            self.detection_worker = None

    def _on_detection_error(self, error: str):
        self.status_message.emit(f"Detection error: {error}")
        self.detect_checkbox.blockSignals(True)
        self.detect_checkbox.setChecked(False)
        self.detect_checkbox.blockSignals(False)
        self._stop_detection()

    def _on_keypoints_3d(self, persons: list, primary_index: int = 0):
        clean_persons = []
        for kps_3d in persons:
            clean = [
                kp if (kp is not None and not np.isnan(kp).any()) else None
                for kp in kps_3d
            ]
            clean_persons.append(clean)

        # Coalesce queued emits (same drop-old story as the camera grid)
        # so a backlog can't replay minute-old skeletons. Recording-buffer
        # fills happen below, synchronously, so no science data is lost
        # even when the paint is coalesced.
        self._pending_persons_3d = clean_persons
        if not self._kp3d_paint_scheduled:
            self._kp3d_paint_scheduled = True
            QTimer.singleShot(0, self._paint_pending_skeleton)
        has_kps = any(any(k is not None for k in p) for p in clean_persons)
        self.rotate_to_human_button.setEnabled(has_kps)
        self.zero_origin_button.setEnabled(has_kps)

        # Track which person is primary (closest to calibrated origin)
        self._primary_person_index = primary_index

        # Buffer keypoints during recording (every emit, never coalesced —
        # this is the science data and must not drop frames).
        if self._is_recording and clean_persons:
            t = time.perf_counter() - self._recording_start_time
            self._recording_keypoints.append({
                "time": t,
                "persons": clean_persons,
                "primary_index": primary_index,
            })

    def _paint_pending_skeleton(self):
        self._kp3d_paint_scheduled = False
        persons = self._pending_persons_3d
        self._pending_persons_3d = None
        if persons is not None:
            self.skeleton_view.update_keypoints(persons)

    # ── Rotate to Human ──

    def _on_rotate_to_human(self):
        self._rotate_countdown = 5
        self.rotate_to_human_button.setEnabled(False)
        self.rotate_to_human_button.setText(f"Rotating in {self._rotate_countdown}s...")
        self._rotate_timer = QTimer()
        self._rotate_timer.timeout.connect(self._rotate_countdown_tick)
        self._rotate_timer.start(1000)

    def _rotate_countdown_tick(self):
        self._rotate_countdown -= 1
        if self._rotate_countdown > 0:
            self.rotate_to_human_button.setText(f"Rotating in {self._rotate_countdown}s...")
        else:
            self._rotate_timer.stop()
            self._rotate_timer = None
            self.rotate_to_human_button.setText("Rotate to Human")
            self._compute_rotate_to_human()

    def _compute_rotate_to_hand(self):
        """Compute a hand-frame rotation transform.

        Axes (per user spec):
          Y = normalize(wrist - middle_tip)  (long axis of hand, toward wrist)
          X = thumb axis, projected to be perpendicular to Y
          Z = normalize(cross(X, Y))         (re-orthogonalised)

        Stores rotation-only (no translation); the user re-zeros via
        "Zero at L_Thumb" to set translation. R is orthonormal by
        construction.

        MediaPipe hand landmarks: 0=wrist, 1=thumb_CMC (base), 4=thumb_tip,
        12=middle_tip.
        """
        kps = self.skeleton_view.get_keypoints()
        if not kps or not any(k is not None for k in kps):
            self.rotate_to_human_button.setEnabled(True)
            return

        def get_pt(idx):
            if idx < len(kps) and kps[idx] is not None:
                return np.array(kps[idx], dtype=float)
            return None

        wrist      = get_pt(0)
        thumb_base = get_pt(1)
        thumb_tip  = get_pt(4)
        middle_tip = get_pt(12)

        if wrist is None or middle_tip is None:
            self.rotate_to_human_button.setEnabled(True)
            return

        Y = wrist - middle_tip
        y_norm = np.linalg.norm(Y)
        if y_norm < 1e-3:
            self.rotate_to_human_button.setEnabled(True)
            return
        Y = Y / y_norm

        if thumb_base is not None and thumb_tip is not None:
            X_raw = thumb_tip - thumb_base
        else:
            X_raw = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(X_raw, Y)) > 0.9:
                X_raw = np.array([0.0, 0.0, 1.0])

        # Gram-Schmidt: drop the Y-component of X_raw.
        X = X_raw - np.dot(X_raw, Y) * Y
        x_norm = np.linalg.norm(X)
        if x_norm < 1e-4:
            self.rotate_to_human_button.setEnabled(True)
            return
        X = X / x_norm

        Z = np.cross(X, Y)
        Z = Z / np.linalg.norm(Z)

        R = np.column_stack([X, Y, Z])
        T = np.eye(4)
        T[:3, :3] = R.T

        self._view_rotation = T
        self._view_has_origin = False
        self.skeleton_view.set_view_transform(T, has_origin=False)
        self._save_view_transform(T, has_origin=False)
        self.rotate_to_human_button.setEnabled(True)

    def _compute_rotate_to_human(self):
        # Hand model uses a different axis convention (Y = tip→wrist,
        # X = thumb axis, Z = X×Y); body axes don't apply.
        if self._current_model_key() == "mediapipe_hands":
            self._compute_rotate_to_hand()
            return

        kps = self.skeleton_view.get_keypoints()
        if not kps or not any(k is not None for k in kps):
            self.rotate_to_human_button.setEnabled(True)
            return

        def get_pt(idx):
            if idx < len(kps) and kps[idx] is not None:
                return np.array(kps[idx], dtype=float)
            return None

        l_ankle = get_pt(15)
        r_ankle = get_pt(16)
        nose    = get_pt(0)
        l_hip   = get_pt(11)
        r_hip   = get_pt(12)
        l_sho   = get_pt(5)
        r_sho   = get_pt(6)

        # Z axis: avg_feet → head (body-up)
        foot_ref = (
            (l_ankle + r_ankle) / 2 if l_ankle is not None and r_ankle is not None
            else l_ankle if l_ankle is not None
            else r_ankle if r_ankle is not None
            else (l_hip + r_hip) / 2 if l_hip is not None and r_hip is not None
            else None
        )
        head_ref = (
            nose if nose is not None
            else (l_sho + r_sho) / 2 if l_sho is not None and r_sho is not None
            else (l_hip + r_hip) / 2 if l_hip is not None and r_hip is not None
            else None
        )
        if foot_ref is None or head_ref is None:
            self.rotate_to_human_button.setEnabled(True)
            return

        Z = head_ref - foot_ref
        z_norm = np.linalg.norm(Z)
        if z_norm < 0.01:
            self.rotate_to_human_button.setEnabled(True)
            return
        Z = Z / z_norm

        # X axis: L_Ankle → R_Ankle, orthogonalised against Z
        if l_ankle is not None and r_ankle is not None:
            X_raw = r_ankle - l_ankle
        elif l_hip is not None and r_hip is not None:
            X_raw = r_hip - l_hip
        else:
            X_raw = np.array([1.0, 0.0, 0.0])
            if abs(np.dot(X_raw, Z)) > 0.9:
                X_raw = np.array([0.0, 0.0, 1.0])

        X = X_raw - np.dot(X_raw, Z) * Z
        x_norm = np.linalg.norm(X)
        if x_norm < 1e-4:
            self.rotate_to_human_button.setEnabled(True)
            return
        X = X / x_norm

        Y = np.cross(Z, X)
        Y = Y / np.linalg.norm(Y)

        R = np.column_stack([X, Y, Z])
        T = np.eye(4)
        T[:3, :3] = R.T

        self._view_rotation = T
        self._view_has_origin = False
        self.skeleton_view.set_view_transform(T, has_origin=False)
        self._save_view_transform(T, has_origin=False)
        self.rotate_to_human_button.setEnabled(True)

    # ── Zero at Left Foot ──

    def _on_zero_at_left_foot(self):
        self._zero_countdown = 5
        self.zero_origin_button.setEnabled(False)
        self.zero_origin_button.setText(f"Zeroing in {self._zero_countdown}s...")
        self._zero_timer = QTimer()
        self._zero_timer.timeout.connect(self._zero_countdown_tick)
        self._zero_timer.start(1000)

    def _zero_countdown_tick(self):
        self._zero_countdown -= 1
        if self._zero_countdown > 0:
            self.zero_origin_button.setText(f"Zeroing in {self._zero_countdown}s...")
        else:
            self._zero_timer.stop()
            self._zero_timer = None
            self._update_zero_button_label()
            self._compute_zero_origin()

    def _compute_zero_origin(self):
        kps = self.skeleton_view.get_keypoints()
        kp_idx, label = self._zero_point_for_model()
        origin_pt = None
        if kps and len(kps) > kp_idx and kps[kp_idx] is not None:
            origin_pt = np.array(kps[kp_idx], dtype=float)

        if origin_pt is None:
            self.zero_origin_button.setEnabled(True)
            return

        R = self._view_rotation[:3, :3]
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = -R @ origin_pt

        self._view_has_origin = True
        self.skeleton_view.set_view_transform(T, has_origin=True)
        self._save_view_transform(T, has_origin=True)
        self._save_body_transform(R, origin_pt)

        # Stash for the next workout-save so the per-session row in workouts.db
        # captures the zero-origin transform that was active during recording.
        self._zero_origin_rotation = R.copy()
        self._zero_origin_translation = (-R @ origin_pt).copy()

        self.zero_origin_button.setEnabled(True)

    # ── View transform persistence (per-model preset DB) ──
    #
    # Storage is one row per model in a sqlite at
    # <app_data>/view_transforms.db. Switching detection model
    # automatically reloads the saved (R, t) for that model so the user
    # only has to press Rotate-to-Human + Zero once per model.
    #
    # The skeleton view's transform `T` is a 4x4 with R in the upper 3x3
    # and t in the last column. The persisted row stores R and t directly.

    def _get_camera_rig_path(self) -> Path | None:
        """Legacy camera_rig.toml path.

        Kept for the one-shot migration into the new view_transforms.db.
        New writes go through save_view_transform — the TOML is no longer
        updated.
        """
        try:
            from ..config import load_app_settings
            app = load_app_settings()
            folder = app.get("last_project_folder")
            if folder:
                return Path(folder) / "camera_rig.toml"
        except Exception:
            pass
        return None

    def _migrate_legacy_view_transform(self):
        """One-shot: import camera_rig.toml's [live_view] into the new
        view_transforms DB under model_key=vitpose, then forget the TOML.

        Idempotent — silently does nothing if the DB already has a row
        for vitpose, or if no legacy file exists.
        """
        try:
            from ..config import load_view_transform, save_view_transform
            from datetime import datetime
            if load_view_transform("synthpose", before=datetime.now().strftime("%Y-%m-%d %H:%M:%S")) is not None:
                return
            import rtoml
            rig_path = self._get_camera_rig_path()
            if rig_path is None or not rig_path.exists():
                return
            data = rtoml.load(rig_path)
            lv = data.get("live_view", {})
            if "transform" not in lv:
                return
            T = np.array(lv["transform"]).reshape(4, 4)
            R = T[:3, :3]
            t = T[:3, 3]
            save_view_transform(
                "synthpose", R, t, bool(lv.get("has_origin", False)),
                notes="legacy migration from camera_rig.toml [live_view]",
            )
        except Exception:
            pass

    def _save_view_transform(self, T: np.ndarray, has_origin: bool = False):
        """Persist the current transform under the active model key.

        Tags the row with the active extrinsic_session_id so future
        ``load_view_transform(model_key, extrinsic_session_id=N)``
        calls can resolve the correct preset for a given recording's
        calibration. The DB schema is append-only — every press of
        Rotate-to-Human / Zero-at-Ankle stores a new row, none are
        overwritten.
        """
        try:
            from ..config import save_view_transform
            R = T[:3, :3]
            t = T[:3, 3]
            save_view_transform(
                self._current_model_key(), R, t, has_origin,
                extrinsic_session_id=self._calibration_session_id,
            )
        except Exception:
            pass

    def _save_body_transform(self, R: np.ndarray, origin: np.ndarray):
        """No-op kept as a stub for callers.

        The body-frame transform used by `_run_workout_analysis` is now
        derived from the same per-model preset row written by
        `_save_view_transform` — there's no separate "body_transform"
        anymore.
        """
        return

    def _load_view_transform(self, model_key: str | None = None):
        """Apply the saved (R, t) for `model_key` to the live skeleton view.

        Falls back to identity (rotate-only with no transform) when no
        preset exists for that model — the user has to press Rotate /
        Zero once for any model they haven't trained yet.
        """
        if model_key is None:
            model_key = self._current_model_key()
        # Lazy migration so existing setups don't lose their preset.
        self._migrate_legacy_view_transform()
        try:
            from ..config import load_view_transform
            from datetime import datetime
            loaded = load_view_transform(
                model_key,
                before=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            )
        except Exception:
            loaded = None

        T = np.eye(4)
        has_origin = False
        if loaded is not None:
            R, t, has_origin = loaded
            T[:3, :3] = R
            T[:3, 3] = t
            if not has_origin:
                self._view_rotation = T
            else:
                # Restore the rotation-only matrix too so a subsequent
                # zero-press has the right base rotation to compose with.
                self._view_rotation = np.eye(4)
                self._view_rotation[:3, :3] = R
                # Carry forward zero info so the next session_save records
                # the active zero, not a stale one from a previous model.
                self._zero_origin_rotation = R.copy()
                self._zero_origin_translation = t.copy()
        else:
            self._view_rotation = np.eye(4)
            self._zero_origin_rotation = None
            self._zero_origin_translation = None

        self._view_has_origin = has_origin
        try:
            self.skeleton_view.set_view_transform(T, has_origin=has_origin)
        except Exception:
            pass

    # ── Calibration ──

    def _check_calibration(self):
        """Run the legacy TOML→DB migration and clear current calibration state.

        The actual extrinsic-session lookup is deferred to _resolve_calibration_for_serials,
        which runs once we know which cameras are physically plugged in (the
        match is by exact serial set).
        """
        self._calibration_available = False
        self._calibration_path = None
        self._calibrated_cameras = None
        self._calibration_session_id = None
        self._calibration_session_created_at = None
        self._calibration_serials_unmatched: set[str] | None = None

        try:
            from ..config import (
                load_app_settings,
                list_extrinsic_sessions,
                import_calibration_toml_into_db,
            )

            # One-shot migration: if the extrinsics DB is empty but a recent
            # session-folder calibration.toml exists, import the newest one
            # so the user has something to start with after the schema change.
            if not list_extrinsic_sessions():
                app = load_app_settings()
                folder = app.get("last_project_folder")
                if folder:
                    folder = Path(folder)
                    cal_files = sorted(folder.glob("*/calibration.toml"))
                    if cal_files:
                        try:
                            import_calibration_toml_into_db(
                                cal_files[-1],
                                notes=f"Migrated from {cal_files[-1]}",
                            )
                        except Exception:
                            # Migration failures should not block live load —
                            # the session-folder TOMLs are still on disk.
                            pass
        except Exception:
            pass

        self._update_cal_status()
        self._update_record_enabled()

        # Optimistic pre-login preview: if any extrinsic exists in the DB,
        # show its date instead of "No extrinsic — use Tools → Calibration".
        # The real binding (which session, which plugged-in cameras) is
        # resolved on login by _resolve_calibration_for_serials and will
        # overwrite this label.
        try:
            from ..config import load_latest_extrinsic_session
            latest = load_latest_extrinsic_session()
            if latest is not None:
                _, created_at, cameras = latest
                n = len(cameras) if cameras else 0
                date_part = ""
                time_part = ""
                if created_at and len(created_at) >= 19 and created_at[10] == " ":
                    date_part = created_at[:10]
                    time_part = created_at[11:19]
                if hasattr(self, "user_cal_status"):
                    if date_part:
                        text = (
                            f"Last extrinsic: {date_part} {time_part}  "
                            f"({n} cameras) — log in to bind"
                        )
                    else:
                        text = f"Last extrinsic ({n} cameras) — log in to bind"
                    self.user_cal_status.setText(text)
                    self.user_cal_status.setStyleSheet(
                        "color: #888; font-size: 11px;"
                    )
        except Exception:
            pass

    def _resolve_calibration_for_serials(self, serials):
        """Look up an extrinsic session for the plugged-in cameras.

        Matching strategy:
        1. Exact match — newest session whose camera-serial set equals
           `serials` exactly. Preferred when the user plugs in exactly the
           cameras they last calibrated.
        2. Subset fallback — newest session whose calibrated serials are
           ALL present among the plugged-in cameras. Lets the operator plug
           in extra (uncalibrated) cameras and still recover the most
           recent extrinsic for the calibrated subset. The plugged-in
           cameras outside the calibrated set are dropped from the
           recording set by `_on_cameras_found` (they aren't in
           `cal_serial_to_port`).
        """
        self._calibration_available = False
        self._calibration_path = None
        self._calibrated_cameras = None
        self._calibration_session_id = None
        self._calibration_session_created_at = None
        self._calibration_serials_unmatched = None

        target = {str(s) for s in serials if s}
        if not target:
            self._update_cal_status()
            self._update_record_enabled()
            return

        match = None
        try:
            from ..config import (
                find_extrinsic_session_by_serials,
                list_extrinsic_sessions,
                load_extrinsic_session,
            )
            match = find_extrinsic_session_by_serials(target)
            if match is None:
                # Subset fallback: walk sessions newest-first and pick the
                # first whose serials are all present in `target`.
                for sess in list_extrinsic_sessions():
                    loaded = load_extrinsic_session(int(sess["id"]))
                    if loaded is None:
                        continue
                    created_at, cameras = loaded
                    sess_serials = {c.serial_number for c in cameras.values()}
                    if sess_serials and sess_serials.issubset(target):
                        match = (int(sess["id"]), created_at, cameras)
                        break

            if match is not None:
                session_id, created_at, cameras = match
                self._calibration_available = True
                self._calibration_session_id = session_id
                self._calibration_session_created_at = created_at
                self._calibrated_cameras = cameras
            else:
                self._calibration_serials_unmatched = target
        except Exception:
            pass

        self._update_cal_status()
        self._update_record_enabled()

    def _update_cal_status(self):
        if self._calibration_available:
            n_cams = len(self._calibrated_cameras) if self._calibrated_cameras else 0
            date_part = ""
            time_part = ""
            # Prefer the DB-recorded created_at; fall back to parsing the
            # session-folder name for legacy paths.
            ts = getattr(self, "_calibration_session_created_at", None)
            if ts:
                # SQLite returns "YYYY-MM-DD HH:MM:SS"
                if len(ts) >= 19 and ts[10] == " ":
                    date_part = ts[:10]
                    time_part = ts[11:19]
            elif self._calibration_path:
                folder_name = self._calibration_path.parent.name
                if len(folder_name) >= 15 and "_" in folder_name:
                    d, t = folder_name.split("_", 1)
                    if len(d) == 8 and len(t) >= 6:
                        date_part = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
                        time_part = f"{t[:2]}:{t[2:4]}:{t[4:6]}"

            # Short version for the user bar (new location)
            if hasattr(self, "user_cal_status"):
                if date_part:
                    self.user_cal_status.setText(
                        f"Using extrinsic from date: {date_part}; time: {time_part}  "
                        f"({n_cams} cameras)"
                    )
                else:
                    self.user_cal_status.setText(f"Using extrinsic ({n_cams} cameras)")
                self.user_cal_status.setStyleSheet(
                    "color: #4CAF50; font-size: 11px; font-weight: bold;"
                )

            # Fallback: camera-bar label (may be hidden)
            if hasattr(self, "cal_status"):
                long_text = (
                    f"Using extrinsic from date: {date_part}; time: {time_part}"
                    if date_part else "Using extrinsic"
                )
                self.cal_status.setText(f"{long_text}  \u2014  {n_cams} cameras")
                self.cal_status.setStyleSheet("color: #4CAF50; font-weight: bold;")
        else:
            if hasattr(self, "user_cal_status"):
                self.user_cal_status.setText("No extrinsic \u2014 use Tools → Calibration")
                if hasattr(self, "_calibration_serials_unmatched") and self._calibration_serials_unmatched:
                    n = len(self._calibration_serials_unmatched)
                    self.user_cal_status.setText(
                        f"No extrinsic for this {n}-camera set"
                    )
                self.user_cal_status.setStyleSheet(
                    "color: #FF5252; font-size: 11px; font-weight: bold;"
                )
            if hasattr(self, "cal_status"):
                unmatched = getattr(self, "_calibration_serials_unmatched", None)
                if unmatched:
                    self.cal_status.setText(
                        f"No extrinsic for this {len(unmatched)}-camera set"
                    )
                else:
                    self.cal_status.setText("No extrinsic")
                self.cal_status.setStyleSheet("color: #FF5252; font-weight: bold;")

    def _update_record_enabled(self):
        has_user = self._current_user_id is not None
        has_cal = self._calibration_available
        has_cameras = len(self.state_manager.state.cameras) > 0
        self.record_btn.setEnabled(has_user and has_cal and has_cameras)

    # ── Recording ──

    def _get_workout_dir(self) -> Path | None:
        try:
            from ..config import load_app_settings
            app = load_app_settings()
            folder = app.get("last_project_folder")
            if folder:
                return Path(folder) / "workouts"
        except Exception:
            pass
        return None

    def _on_record(self):
        if self._is_recording:
            # Stop recording
            if self.recording_worker:
                self.recording_worker.running = False
            return

        # Cancel an in-progress countdown if the user clicks again.
        if getattr(self, "_record_countdown_timer", None) is not None:
            self._record_countdown_timer.stop()
            self._record_countdown_timer.deleteLater()
            self._record_countdown_timer = None
            if hasattr(self, "_record_btn_original_text"):
                self.record_btn.setText(self._record_btn_original_text)
            self.status_message.emit("Recording cancelled")
            return

        # Pre-roll countdown — gives the user time to walk into the capture
        # volume before the camera buffer actually starts. Click again to
        # cancel.
        self._record_btn_original_text = self.record_btn.text()
        self._record_countdown_remaining = 3
        self.record_btn.setText(
            f"Starting in {self._record_countdown_remaining}s... (click to cancel)"
        )
        self.status_message.emit(
            f"Recording starts in {self._record_countdown_remaining}s"
        )
        self._record_countdown_timer = QTimer(self)
        self._record_countdown_timer.timeout.connect(self._record_countdown_tick)
        self._record_countdown_timer.start(1000)

    def _record_countdown_tick(self):
        self._record_countdown_remaining -= 1
        if self._record_countdown_remaining > 0:
            self.record_btn.setText(
                f"Starting in {self._record_countdown_remaining}s... "
                f"(click to cancel)"
            )
            self.status_message.emit(
                f"Recording starts in {self._record_countdown_remaining}s"
            )
            return
        # Countdown done — drop the timer and kick off the real recording.
        if self._record_countdown_timer is not None:
            self._record_countdown_timer.stop()
            self._record_countdown_timer.deleteLater()
            self._record_countdown_timer = None
        if hasattr(self, "_record_btn_original_text"):
            self.record_btn.setText(self._record_btn_original_text)
        self._begin_recording_now()

    def _begin_recording_now(self):
        from datetime import datetime

        workout_dir = self._get_workout_dir()
        if workout_dir is None:
            self.status_message.emit("No workout directory configured")
            return

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workout_type = self._selected_workout_type()
        from ..session_naming import build_session_dir_name

        self._current_session_dir = workout_dir / build_session_dir_name(
            self._current_username,
            timestamp,
            workout_type,
        )
        self._current_session_dir.mkdir(parents=True, exist_ok=True)
        self._current_workout_type = workout_type

        duration = self.duration_spin.value()
        fps = self._target_fps

        # Ensure cameras are open
        if not self.opened_cameras:
            self._start_preview()
        if not self.opened_cameras:
            self.status_message.emit("No cameras available for recording")
            return

        # Pause preview worker — recording worker takes over capture
        if self.preview_worker:
            self.preview_worker.pause()

        self._is_recording = True
        self._recording_keypoints = []
        self._record_arrivals = {}
        self._recording_start_time = time.perf_counter()

        # Snapshot the active view transform so the npz written at the end
        # of this trial has detection points already rotated + zeroed
        # (downstream consumers want body-frame coords, not camera frame).
        # Snapshot at record-time, NOT at finish-time, so a mid-trial
        # re-zero wouldn't change what gets written. Re-orthonormalisation
        # happens inside write_raw_buffer.
        #
        # Source of truth: when zero has been pressed, the (R, t) we want
        # is _zero_origin_rotation/_zero_origin_translation — those are
        # the values _compute_zero_origin actually maintains AND what the
        # DB roundtrip preserves. _view_rotation only ever holds the
        # rotation-only matrix; its translation column stays zero even
        # after Zero is pressed, so reading translation from it gives the
        # wrong answer (rotated keypoints with no offset = ankles at z≈3).
        if self._view_has_origin and self._zero_origin_rotation is not None \
                and self._zero_origin_translation is not None:
            R_snap = np.asarray(self._zero_origin_rotation, dtype=np.float64).copy()
            t_snap = np.asarray(self._zero_origin_translation, dtype=np.float64).copy()
        else:
            R_snap = self._view_rotation[:3, :3].copy()
            t_snap = np.zeros(3)
        self._recording_view_R = R_snap
        self._recording_view_t = t_snap
        # Drop footstep history from the previous session so the overlay
        # reflects only the current recording.
        self.skeleton_view.clear_footsteps()
        self.record_btn.setText("Stop Recording")
        self.record_btn.setEnabled(True)
        self.status_message.emit(f"Recording {workout_type} for {duration}s...")

        # Pause-live-tracking-during-recording option: if the user wants
        # the cameras to have the full frame budget, stop the detection
        # worker now and remember to restart it on _on_record_finished.
        # We do NOT capture _recording_keypoints in this mode — the user
        # has already opted in to "save videos, process later" by checking
        # this box.
        self._tracking_paused_for_recording = bool(
            self.pause_tracking_during_record_checkbox.isChecked()
        )
        if self._tracking_paused_for_recording:
            # Pause the worker (don't tear it down) so the user gets the
            # full frame budget for the camera reads, but the multi-second
            # YOLO + VitPose load isn't paid again on resume. If the
            # worker class doesn't expose pause/resume (shouldn't happen
            # — all current backends do), fall back to a real stop so
            # detection actually halts; silently no-op'ing means the user
            # sees no perf benefit at all.
            worker = self.detection_worker
            if worker is not None:
                if hasattr(worker, "pause"):
                    worker.pause()
                else:
                    self._stop_detection()
        else:
            # Ensure detection is running for 3D keypoint collection
            if self.detection_worker is None and self._calibrated_cameras is not None:
                self._start_detection()

        self.recording_worker = RecordingWorker(
            cameras=self.opened_cameras,
            ports=self.opened_ports,
            output_path=self._current_session_dir,
            duration=duration,
            fps=fps,
            codec="hevc",
        )
        self.recording_worker.frame_captured.connect(self._on_record_frame)
        self.recording_worker.progress_update.connect(self._on_record_progress)
        self.recording_worker.recording_finished.connect(self._on_record_finished)
        self.recording_worker.error.connect(self._on_record_error)
        self.recording_worker.start()

    def _on_record_frame(self, port: int, pixels):
        """Update preview during recording + feed detection."""
        self.camera_grid.update_frame(port, pixels)
        if self.detection_worker is not None:
            self.detection_worker.submit_frame(port, pixels)

        # Track per-port arrival times for the post-recording stats and
        # push to the live FPS graph.
        now = time.perf_counter()
        self._record_arrivals.setdefault(port, []).append(now)
        prev = self._last_frame_time.get(port)
        if prev is not None:
            dt = now - prev
            if dt > 0:
                self.fps_graph.push_fps(port, 1.0 / dt)
        self._last_frame_time[port] = now

    def _on_record_progress(self, current: int, total: int):
        elapsed = current / max(self._target_fps, 1)
        remaining = (total - current) / max(self._target_fps, 1)
        self.record_btn.setText(f"Stop ({remaining:.0f}s left)")

    def _start_offline_processing(self):
        """Kick off the CUDA batch pipeline against the just-recorded videos.

        Used when the user paused live tracking during recording — the live
        keypoint buffer is empty so we re-run pose tracking offline. Shows
        progress in the bottom strip; on completion, writes
        keypoints_3d.raw.npz and per-person CSVs to the session dir.
        """
        if self._current_session_dir is None or self._calibrated_cameras is None:
            self._offline_csv_pending = False
            return
        # Build port -> video path. RecordingWorker writes
        # port_{N}_{sanitized_serial}.mp4; older sessions used port_{N}.mp4.
        # find_video_for_port checks both, so paused-tracking trials work
        # regardless of which generation produced the videos.
        from .video_utils import find_video_for_port
        session_dir = self._current_session_dir
        port_to_video = {}
        for port in self.opened_ports:
            cam = self.state_manager.state.cameras.get(port)
            serial = getattr(cam.info, "serial_number", None) if cam else None
            video = find_video_for_port(session_dir, port, serial=serial)
            if video is not None:
                port_to_video[port] = video
        if not port_to_video:
            msg = (
                f"Offline processing skipped: no port_*.mp4 found in "
                f"{session_dir.name}"
            )
            print(msg, flush=True)
            self.status_message.emit(msg)
            # Clear the pending flag — without this, the next successful
            # offline run would inherit stale state and fire CSV/analysis
            # against the wrong session.
            self._offline_csv_pending = False
            return

        frame_time_csv = session_dir / "frame_time_history.csv"
        if not frame_time_csv.exists():
            msg = (
                f"Offline processing skipped: frame_time_history.csv "
                f"missing in {session_dir.name}"
            )
            print(msg, flush=True)
            self.status_message.emit(msg)
            self._offline_csv_pending = False
            return

        cameras = self._normalize_calibrated_cameras(self._calibrated_cameras)

        # Pull batch_size from project settings (added in this PR; default 8).
        batch_size = 8
        try:
            from ..config import load_app_settings, load_project_settings
            app = load_app_settings()
            folder = app.get("last_project_folder")
            if folder:
                ps = load_project_settings(Path(folder))
                batch_size = int(ps.get("pose_batch_size", 8))
        except Exception:
            pass

        # Pick the unified-offline worker by default so the offline path
        # shares the live pipeline's primitives (and therefore its
        # tracker config + person-confidence default + fragmentation
        # behaviour). Setting CALIMERGE_LEGACY_OFFLINE=1 falls back to
        # the deprecated batch-mode worker — useful while the unified
        # path is still being validated against real recordings on
        # CUDA / MPS hosts. See unified_offline_worker.py header.
        import os
        use_legacy = bool(os.environ.get("CALIMERGE_LEGACY_OFFLINE"))

        if use_legacy:
            from .workers import OfflineProcessingWorker
            self._offline_worker = OfflineProcessingWorker(
                session_dir=session_dir,
                cameras=cameras,
                port_to_video=port_to_video,
                frame_time_csv=frame_time_csv,
                batch_size=batch_size,
                view_rotation=getattr(self, "_recording_view_R", None),
                view_translation=getattr(self, "_recording_view_t", None),
            )
        else:
            from .unified_offline_worker import UnifiedOfflineWorker
            # Use the same backend the user picked for the live path.
            # Offline-only fallback: when no extrinsics are available
            # _start_detection picks pytorch (the only backend that works
            # without calibration), so do the same here.
            backend = self.detect_backend_combo.currentData() or "pytorch"
            self._offline_worker = UnifiedOfflineWorker(
                session_dir=session_dir,
                cameras=cameras,
                port_to_video=port_to_video,
                frame_time_csv=frame_time_csv,
                backend=backend,
                view_rotation=getattr(self, "_recording_view_R", None),
                view_translation=getattr(self, "_recording_view_t", None),
                batch_size=batch_size,
                person_confidence=self._current_person_confidence(),
            )
        self._offline_worker.progress.connect(self._on_offline_progress)
        self._offline_worker.log_message.connect(self._on_offline_log)
        self._offline_worker.finished_ok.connect(self._on_offline_finished)
        self._offline_worker.failed.connect(self._on_offline_failed)

        self.offline_progress_container.setVisible(True)
        self.offline_progress_bar.setValue(0)
        self.offline_status_label.setText("Offline processing: starting...")
        self._offline_worker.start()

    def _on_offline_progress(self, step: str, fraction: float):
        pct = int(max(0.0, min(1.0, fraction)) * 100)
        self.offline_progress_bar.setValue(pct)
        self.offline_status_label.setText(f"Offline: {step}")

    def _on_offline_log(self, msg: str):
        self.status_message.emit(msg)

    def _on_offline_finished(self, session_dir):
        self.offline_progress_bar.setValue(100)
        self.offline_status_label.setText(
            f"Offline processing complete: {session_dir}"
        )
        # Auto-hide the progress strip after a short delay so it doesn't
        # linger forever; status_bar message lives on.
        QTimer.singleShot(4000, self._hide_offline_progress)
        self._offline_worker = None

        # If the trial that triggered this offline run had pause-tracking on
        # AND "Generate CSV after save" enabled, the in-memory keypoint
        # buffer is empty and _handle_csv_export already early-returned.
        # Now that the offline worker has written keypoints_3d.raw.npz, fan
        # out a CSV worker pointed at that file.
        if getattr(self, "_offline_csv_pending", False):
            self._offline_csv_pending = False
            sd = getattr(self, "_offline_csv_session_dir", None) or Path(session_dir)
            session_id = getattr(self, "_offline_csv_session_id", None)
            cal_path = getattr(self, "_offline_csv_calibration_path", None)
            backend, model_name = self._detection_model_label()
            try:
                self._spawn_csv_worker(
                    session_dir=sd,
                    buffer=None,  # forces the worker to read raw_buffer_path
                    session_id=session_id,
                    model_backend=backend,
                    model_name=model_name,
                    calibration_path=cal_path,
                )
                self.status_message.emit(
                    f"Generating CSV from offline keypoints for {Path(sd).name}"
                )
            except Exception as e:
                self.status_message.emit(f"CSV export from offline failed: {e}")

            # Run workout analysis on the keypoints just written. The npz
            # holds points in view frame (R+t already applied during write),
            # so pass already_in_view_frame=True to skip the body-transform
            # step inside _run_workout_analysis.
            try:
                from ..keypoint_export import read_raw_buffer, RAW_FILENAME
                buf = read_raw_buffer(Path(sd) / RAW_FILENAME)
                if buf:
                    self._run_workout_analysis(
                        session_id, buffer=buf, already_in_view_frame=True,
                    )
            except Exception as e:
                self.status_message.emit(
                    f"Analysis from offline keypoints failed: {e}"
                )

    def _on_offline_failed(self, error: str):
        self.offline_status_label.setText(f"Offline processing failed: {error[:200]}")
        self.offline_progress_bar.setValue(0)
        QTimer.singleShot(8000, self._hide_offline_progress)
        self._offline_worker = None
        # Don't leave the pending flag set — otherwise the next offline run
        # (different trial) would erroneously trigger a CSV from stale state.
        self._offline_csv_pending = False

    def _hide_offline_progress(self):
        self.offline_progress_container.setVisible(False)

    def _report_recording_stats(self):
        """Print + status-bar framerate stats for the just-finished trial.

        Source priority: prefer the per-port frame timestamps written by
        RecordingWorker into frame_time_history.csv — those reflect the
        actual capture loop, including any frames the loop dropped. The
        GUI-receipt timestamps (self._record_arrivals) are a fallback;
        they're measured at signal-delivery time on a rate-limited
        producer, so they look like a metronome (always ≈ target fps)
        and don't actually reveal capture problems.

        Also reports detection rate when a detection worker was active,
        since that's usually where slowdowns hit.
        """
        msg_parts: list[str] = []

        # ── Capture rate from frame_time_history.csv (authoritative) ──
        capture_block = self._capture_stats_from_csv()
        if capture_block is None:
            # Fall back to GUI-receipt timing only when the CSV is missing
            # (e.g. recording aborted before _save_frame_times).
            capture_block = self._capture_stats_from_arrivals()
        if capture_block is not None:
            msg_parts.append(capture_block)

        # ── Detection rate (if detection ran during the trial) ──
        if self._recording_keypoints:
            n = len(self._recording_keypoints)
            t0 = float(self._recording_keypoints[0].get("time", 0.0))
            t1 = float(self._recording_keypoints[-1].get("time", 0.0))
            elapsed = t1 - t0
            if elapsed > 0 and n > 1:
                det_fps = (n - 1) / elapsed
                msg_parts.append(
                    f"detection: {n} frames over {elapsed:.1f}s "
                    f"= {det_fps:.1f} fps"
                )
            else:
                msg_parts.append(f"detection: {n} frames")
        else:
            paused = bool(getattr(self, "_tracking_paused_for_recording", False))
            msg_parts.append(
                "detection: paused" if paused else "detection: 0 frames"
            )

        if not msg_parts:
            return
        long_msg = "Recording stats — " + "; ".join(msg_parts)
        print(long_msg, flush=True)
        self.status_message.emit(long_msg)

    def _capture_stats_from_csv(self) -> str | None:
        """Per-port capture-rate summary from frame_time_history.csv."""
        if self._current_session_dir is None:
            return None
        csv_path = self._current_session_dir / "frame_time_history.csv"
        if not csv_path.exists():
            return None
        # CSV columns (after a single comment line): sync_index, port,
        # frame_index, frame_time. frame_time = perf_counter offset from
        # recording start, so per-port np.diff(...) gives real inter-frame
        # gaps including any frame-loop overruns.
        try:
            import csv as _csv
            per_port: dict[int, list[float]] = {}
            with open(csv_path, "r", newline="") as f:
                # Skip leading "# cameras: ..." comment line.
                first = f.readline()
                if not first.startswith("#"):
                    f.seek(0)
                reader = _csv.reader(f)
                header = next(reader, None)
                if header is None:
                    return None
                for row in reader:
                    if len(row) < 4:
                        continue
                    try:
                        port = int(row[1])
                        ft = float(row[3])
                    except (ValueError, IndexError):
                        continue
                    per_port.setdefault(port, []).append(ft)
        except Exception:
            return None

        all_dts: list[float] = []
        per_port_summary: list[str] = []
        for port in sorted(per_port):
            times = np.asarray(per_port[port], dtype=float)
            if times.size < 2:
                continue
            dts = np.diff(times)
            if dts.size == 0:
                continue
            all_dts.extend(dts.tolist())
            per_port_summary.append(
                f"port {port}: {1.0 / dts.mean():.1f} fps "
                f"(max dt {dts.max() * 1000:.1f} ms)"
            )

        if not all_dts:
            return None
        a = np.asarray(all_dts, dtype=float)
        avg_dt = float(a.mean())
        median_dt = float(np.median(a))
        max_dt = float(a.max())
        avg_fps = 1.0 / avg_dt if avg_dt > 0 else 0.0
        median_fps = 1.0 / median_dt if median_dt > 0 else 0.0
        return (
            f"capture (from frame_time_history.csv) avg dt "
            f"{avg_dt * 1000:.1f} ms ({avg_fps:.1f} fps), median "
            f"{median_fps:.1f} fps, max dt {max_dt * 1000:.1f} ms; "
            + ", ".join(per_port_summary)
        )

    def _capture_stats_from_arrivals(self) -> str | None:
        """Fallback: use GUI-receipt timestamps when no CSV is available.

        These are measured on a rate-limited producer so they tend to
        report exactly the target rate even if real capture is unhealthy
        — which is why the CSV is preferred. We surface the source so the
        user knows which path produced the numbers.
        """
        all_dts: list[float] = []
        per_port_summary: list[str] = []
        for port in sorted(self._record_arrivals.keys()):
            arrs = self._record_arrivals[port]
            if len(arrs) < 2:
                continue
            arr = np.asarray(arrs, dtype=float)
            dts = np.diff(arr)
            if dts.size == 0:
                continue
            all_dts.extend(dts.tolist())
            per_port_summary.append(f"port {port}: {1.0 / dts.mean():.1f} fps")
        if not all_dts:
            return None
        a = np.asarray(all_dts, dtype=float)
        avg_dt = float(a.mean())
        median_dt = float(np.median(a))
        max_dt = float(a.max())
        return (
            f"capture (GUI-receipt fallback) avg dt {avg_dt * 1000:.1f} ms "
            f"({1.0 / avg_dt:.1f} fps), median {1.0 / median_dt:.1f} fps, "
            f"max dt {max_dt * 1000:.1f} ms; "
            + ", ".join(per_port_summary)
        )

    def _on_record_finished(self, stats: dict):
        self._is_recording = False
        self.recording_worker = None
        self.record_btn.setText("Record Workout")
        self._update_record_enabled()

        # Live capture stats — printed before any heavier post-processing so
        # the user gets immediate framerate feedback when iterating on
        # tracking performance. Stats are aggregated across all ports
        # (per-frame deltas concatenated) since cameras are sync-captured;
        # divergent ports show up as a single fat tail in max delta.
        self._report_recording_stats()

        # Resume preview
        if self.preview_worker:
            self.preview_worker.resume()

        # If detection was paused for the recording, bring it back so the
        # live skeleton view starts updating again. With the pause/resume
        # path the worker is still alive — just resume it; only fall back
        # to a full _start_detection if the worker is actually gone (e.g.
        # the user disabled detection entirely during the recording, or
        # the worker class lacks resume).
        was_paused = bool(getattr(self, "_tracking_paused_for_recording", False))
        if was_paused:
            self._tracking_paused_for_recording = False
            if self.detect_checkbox.isChecked():
                worker = self.detection_worker
                if worker is not None and hasattr(worker, "resume"):
                    worker.resume()
                else:
                    # Worker is gone OR doesn't support resume — start fresh.
                    if worker is None:
                        self._start_detection()

        # If tracking was paused during recording, the live keypoint buffer
        # is empty (or near-empty). Kick off offline post-processing on the
        # saved video files so the user still gets a CSV. Only fires when
        # the user has 'Generate CSV after save' on. The CSV worker is
        # spawned in _on_offline_finished, reading from the freshly
        # written keypoints_3d.raw.npz on disk (since the in-memory
        # buffer is empty in this code path).
        self._offline_csv_pending = False
        if was_paused and self.generate_csv_checkbox.isChecked():
            self._offline_csv_pending = True
            self._start_offline_processing()

        # Save 3D keypoints to binary file alongside the videos. The
        # record-time view transform snapshot is forwarded so this npz
        # matches the keypoints_3d.raw.npz the CSV worker writes — both
        # land in body/hand frame, with the inverse transform recorded
        # so consumers can recover camera coords if needed. Without
        # this, the notebook (test_output.ipynb) loaded the .npz file
        # and saw camera-frame coords (ankle z ≈ 2-3 m) while the live
        # display showed body-frame (ankle z ≈ 0).
        kps_file = None
        if self._recording_keypoints:
            try:
                from ..analysis.keypoints_io import save_keypoints_3d
                kps_file = self._current_session_dir / "keypoints_3d.npz"
                _save_backend, _save_model_name = self._detection_model_label()
                save_keypoints_3d(
                    kps_file, self._recording_keypoints,
                    primary_person_index=self._primary_person_index,
                    view_rotation=getattr(self, "_recording_view_R", None),
                    view_translation=getattr(self, "_recording_view_t", None),
                    model_backend=_save_backend,
                    model_name=_save_model_name,
                )
            except Exception as e:
                self.status_message.emit(f"Failed to save keypoints: {e}")
                kps_file = None

        # Pack the calibration config into a compact blob for reproducibility
        config_blob = None
        try:
            if self._calibrated_cameras:
                from ..config import pack_session_config
                config_blob = pack_session_config(self._calibrated_cameras)
        except Exception as e:
            self.status_message.emit(f"Failed to pack config: {e}")

        # Save session to database (with program linkage if applicable)
        session_id = None
        try:
            from ..config import create_session
            duration = stats.get("duration", 0)

            program_ex_id = None
            set_number = None
            if self._current_program_exercise is not None:
                program_ex_id = self._current_program_exercise["id"]
                # Count sets already recorded in the current program week
                set_number = self._next_set_number_for_current_exercise()

            _sess_backend, _sess_model_name = self._detection_model_label()
            session_id = create_session(
                user_id=self._current_user_id,
                workout_type=self._current_workout_type,
                duration_seconds=duration,
                recording_path=str(self._current_session_dir),
                calibration_path=str(self._calibration_path) if self._calibration_path else None,
                config_blob=config_blob,
                program_exercise_id=program_ex_id,
                set_number=set_number,
                extrinsic_session_id=self._calibration_session_id,
                extrinsic_calibrated_at=self._calibration_session_created_at,
                zero_origin_rotation=self._zero_origin_rotation,
                zero_origin_translation=self._zero_origin_translation,
                model_backend=_sess_backend,
                model_name=_sess_model_name,
            )
            sync_count = stats.get("sync_count", 0)
            self.status_message.emit(
                f"Recording complete: {sync_count} frames saved to {self._current_session_dir.name}"
            )
        except Exception as e:
            self.status_message.emit(f"Recording saved but DB write failed: {e}")

        # If we kicked off offline processing for this trial (paused-tracking
        # path), the offline worker is the one that produces the raw
        # keypoint buffer — so the CSV worker has to wait for it. Stash
        # session_id + session_dir + calibration_path here so
        # _on_offline_finished can spawn the CSV worker once the npz is on
        # disk.
        if getattr(self, "_offline_csv_pending", False):
            self._offline_csv_session_id = session_id
            self._offline_csv_session_dir = self._current_session_dir
            self._offline_csv_calibration_path = (
                str(self._calibration_path) if self._calibration_path else None
            )

        # Refresh the Today's Plan counts and button label
        self._refresh_todays_plan()
        self._update_record_button_label()
        self._refresh_longterm_graph()

        # CSV export of 3D keypoints (immediate or queued)
        try:
            self._handle_csv_export(session_id)
        except Exception as e:
            self.status_message.emit(f"CSV export setup failed: {e}")

        # Run analysis on collected keypoints
        self._run_workout_analysis(session_id)

    def _on_record_error(self, error: str):
        self._is_recording = False
        self.recording_worker = None
        self.record_btn.setText("Record Workout")
        self._update_record_enabled()
        self.status_message.emit(f"Recording error: {error}")
        if self.preview_worker:
            self.preview_worker.resume()

    # ── CSV export of 3D keypoints ──

    def _init_csv_export_state(self):
        """Restore the immediate/queued toggle and refresh the pending count."""
        try:
            from ..config import load_app_settings
            settings = load_app_settings()
            immediate = bool(settings.get("csv_export_immediate", True))
            self.generate_csv_checkbox.blockSignals(True)
            self.generate_csv_checkbox.setChecked(immediate)
            self.generate_csv_checkbox.blockSignals(False)
        except Exception:
            pass
        self._refresh_pending_csv_label()

    def _on_csv_toggle_changed(self, checked: bool):
        try:
            from ..config import load_app_settings, save_app_settings
            settings = load_app_settings()
            settings["csv_export_immediate"] = bool(checked)
            save_app_settings(settings)
        except Exception as e:
            self.status_message.emit(f"Failed to save CSV preference: {e}")
        self._refresh_pending_csv_label()

    def _refresh_pending_csv_label(self):
        try:
            from ..config import load_app_settings
            jobs = load_app_settings().get("pending_csv_jobs", []) or []
        except Exception:
            jobs = []
        n = len(jobs)
        if n == 0:
            self.process_pending_btn.setText("Process Pending CSVs")
            self.process_pending_btn.setEnabled(False)
        else:
            self.process_pending_btn.setText(f"Process Pending CSVs ({n})")
            self.process_pending_btn.setEnabled(True)

    def _detection_model_label(self) -> tuple[str | None, str | None]:
        """Return (backend, model_name) for meta.json provenance."""
        backend = None
        model_name = None
        try:
            backend = self.detect_backend_combo.currentData()
        except Exception:
            backend = None
        try:
            base = self.detect_model_combo.currentData()
        except Exception:
            base = None
        if base == "vitpose":
            model_name = "vitpose_synthpose"
        elif base == "mediapipe_hands":
            model_name = "mediapipe_hands"
        else:
            model_name = base
        return backend, model_name

    def _handle_csv_export(self, session_id: int | None):
        """Either run the export worker now, or queue the job."""
        if not self._recording_keypoints:
            return

        from ..config import load_app_settings, save_app_settings
        from ..keypoint_export import (
            write_raw_buffer,
            make_job_descriptor,
            RAW_FILENAME,
            FRAME_TIME_HISTORY_FILENAME,
        )

        session_dir = self._current_session_dir
        if session_dir is None:
            return

        backend, model_name = self._detection_model_label()
        # Always persist the raw buffer alongside the videos. This lets a
        # queued job survive a process restart, AND gives the immediate
        # path a recovery point if the worker crashes. The snapshotted
        # view transform (R, t) is forwarded so the keypoints in the npz
        # are already in body/hand frame and the inverse transform is
        # recorded for consumers that want camera coords.
        history_path = session_dir / FRAME_TIME_HISTORY_FILENAME
        try:
            write_raw_buffer(
                session_dir / RAW_FILENAME,
                self._recording_keypoints,
                frame_time_history_path=(
                    history_path if history_path.exists() else None
                ),
                view_rotation=getattr(self, "_recording_view_R", None),
                view_translation=getattr(self, "_recording_view_t", None),
                model_backend=backend,
                model_name=model_name,
            )
        except Exception as e:
            self.status_message.emit(f"Failed to dump raw keypoints: {e}")

        settings = load_app_settings()
        immediate = bool(settings.get("csv_export_immediate", True))

        cal_path = str(self._calibration_path) if self._calibration_path else None

        if immediate:
            # Pass buffer=None so the CSV worker reads from the
            # raw_buffer_path npz we just wrote — that file has the
            # view transform (rotate-to-human + zero) ALREADY applied
            # by write_raw_buffer. The in-memory _recording_keypoints
            # is still in CAMERA frame (the detection worker emits raw
            # 3D, _on_keypoints_3d appends it without transforming),
            # so handing that buffer to the worker would write a CSV
            # in camera frame while the npz was in view frame — that's
            # how ankle showed up at ~2.4 m in the CSV but ~0 m in
            # the live view + npz.
            self._spawn_csv_worker(
                session_dir=session_dir,
                buffer=None,
                session_id=session_id,
                model_backend=backend,
                model_name=model_name,
                calibration_path=cal_path,
            )
        else:
            job = make_job_descriptor(
                session_dir=session_dir,
                session_id=session_id,
                model_backend=backend,
                model_name=model_name,
                calibration_path=cal_path,
            )
            jobs = list(settings.get("pending_csv_jobs", []) or [])
            jobs.append(job)
            settings["pending_csv_jobs"] = jobs
            try:
                save_app_settings(settings)
            except Exception as e:
                self.status_message.emit(f"Failed to enqueue CSV job: {e}")
                return
            self.status_message.emit(
                f"CSV queued for {Path(session_dir).name} "
                f"({len(jobs)} pending)"
            )
            self._refresh_pending_csv_label()

    def _spawn_csv_worker(
        self,
        *,
        session_dir: Path,
        buffer: list[dict] | None,
        session_id: int | None,
        model_backend: str | None,
        model_name: str | None,
        calibration_path: str | None,
        on_done=None,
    ):
        from .csv_export_worker import CsvExportWorker
        from ..config import DEFAULT_INTRINSICS_DB
        from ..keypoint_export import RAW_FILENAME

        worker = CsvExportWorker(
            session_dir=session_dir,
            recording_keypoints=buffer,
            calibrated_cameras=self._calibrated_cameras,
            model_backend=model_backend,
            model_name=model_name,
            num_keypoints=52,
            session_id=session_id,
            intrinsics_db_path=DEFAULT_INTRINSICS_DB,
            raw_buffer_path=Path(session_dir) / RAW_FILENAME,
            extra_meta={"calibration_path": calibration_path},
        )
        # Track active workers so they aren't garbage-collected mid-run.
        if not hasattr(self, "_csv_workers"):
            self._csv_workers: list = []
        self._csv_workers.append(worker)

        def _ok(info: dict):
            self.status_message.emit(
                f"CSV saved: {Path(info['csv_path']).name} "
                f"({info.get('rows', 0)} rows)"
            )
            if on_done is not None:
                on_done(True, info)
            self._cleanup_csv_worker(worker)

        def _fail(session_dir_str: str, msg: str):
            first = msg.splitlines()[0] if msg else "<unknown>"
            self.status_message.emit(
                f"CSV export failed for {Path(session_dir_str).name}: {first}"
            )
            if on_done is not None:
                on_done(False, {"session_dir": session_dir_str, "error": msg})
            self._cleanup_csv_worker(worker)

        def _progress(msg: str):
            self.status_message.emit(msg)

        worker.finished_ok.connect(_ok)
        worker.failed.connect(_fail)
        worker.progress.connect(_progress)
        # Belt-and-braces cleanup: the worker holds a full keypoint
        # buffer + (potentially) detector model refs, so if it raises
        # before either finished_ok or failed fires, the entry pinned
        # in self._csv_workers leaks the entire blob. Hook QThread's
        # built-in finished signal too — that fires unconditionally
        # when run() returns, success or exception.
        worker.finished.connect(lambda w=worker: self._cleanup_csv_worker(w))
        worker.start()

    def _cleanup_csv_worker(self, worker):
        try:
            if hasattr(self, "_csv_workers") and worker in self._csv_workers:
                self._csv_workers.remove(worker)
        except Exception:
            pass
        # Schedule the QThread for deletion so its event loop + ref to
        # the keypoint buffer go away. deleteLater is the Qt-safe way:
        # the object stays alive until the next event-loop iteration,
        # then dies.
        try:
            worker.deleteLater()
        except Exception:
            pass

    def _on_process_pending_csvs(self):
        """Drain the queue. Each job runs sequentially via the worker."""
        from ..config import load_app_settings, save_app_settings
        from ..keypoint_export import iter_jobs

        settings = load_app_settings()
        jobs = list(settings.get("pending_csv_jobs", []) or [])
        runnable = list(iter_jobs(jobs))
        if not runnable:
            # Even if there were stale entries (sessions deleted), purge them.
            settings["pending_csv_jobs"] = []
            save_app_settings(settings)
            self.status_message.emit("No pending CSV jobs.")
            self._refresh_pending_csv_label()
            return

        # Pop the first runnable job, run it, then chain into the next.
        self._drain_csv_queue(runnable)

    def _drain_csv_queue(self, queue: list[dict]):
        if not queue:
            self.status_message.emit("Pending CSVs processed.")
            self._refresh_pending_csv_label()
            return
        job = queue[0]
        rest = queue[1:]
        sd = Path(job["session_dir"])

        def _after(success: bool, info: dict):
            # Remove the persisted entry whether it succeeded or not (a
            # failure shouldn't trap the user in an infinite retry loop).
            self._remove_persisted_job(job)
            self._refresh_pending_csv_label()
            self._drain_csv_queue(rest)

        self._spawn_csv_worker(
            session_dir=sd,
            buffer=None,  # force the worker to load from raw_buffer_path
            session_id=job.get("session_id"),
            model_backend=job.get("model_backend"),
            model_name=job.get("model_name"),
            calibration_path=job.get("calibration_path"),
            on_done=_after,
        )

    def _remove_persisted_job(self, job: dict):
        try:
            from ..config import load_app_settings, save_app_settings
            settings = load_app_settings()
            jobs = list(settings.get("pending_csv_jobs", []) or [])
            target_dir = job.get("session_dir")
            jobs = [j for j in jobs if j.get("session_dir") != target_dir]
            settings["pending_csv_jobs"] = jobs
            save_app_settings(settings)
        except Exception:
            pass

    # ── Session history ──

    def _on_view_progress(self):
        """Open the progress graph dialog for the active program."""
        if self._current_user_id is None:
            self.status_message.emit("Log in first to view progress")
            return
        if self._active_program is None or not self._active_program_exercises:
            self.status_message.emit("No active program to show progress for")
            return
        from .progress_graph import ProgressGraphDialog
        dlg = ProgressGraphDialog(
            self._current_user_id,
            self._active_program,
            self._active_program_exercises,
            parent=self,
        )
        dlg.exec()

    def _on_view_sessions(self):
        """Show a dialog listing past workout sessions."""
        from PySide6.QtWidgets import (
            QDialog, QVBoxLayout, QHBoxLayout, QTableWidget, QTableWidgetItem,
            QHeaderView, QDialogButtonBox, QPushButton,
        )

        if self._current_user_id is None:
            self.status_message.emit("Log in first to view sessions")
            return

        try:
            from ..config import get_sessions_for_user
            sessions = get_sessions_for_user(self._current_user_id)
        except Exception as e:
            self.status_message.emit(f"Failed to load sessions: {e}")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle(f"Workout Sessions — {self._current_username}")
        dlg.setMinimumSize(800, 500)
        layout = QVBoxLayout(dlg)

        table = QTableWidget()
        table.setColumnCount(5)
        table.setHorizontalHeaderLabels(["Date", "Workout", "Duration (s)", "Recording", "Calibration"])
        header = table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        table.setRowCount(len(sessions))

        row_to_session: dict[int, dict] = {}
        for row, session in enumerate(sessions):
            row_to_session[row] = session
            table.setItem(row, 0, QTableWidgetItem(str(session.get("created_at", ""))))
            table.setItem(row, 1, QTableWidgetItem(session.get("workout_type", "")))
            dur = session.get("duration_seconds")
            table.setItem(row, 2, QTableWidgetItem(f"{dur:.1f}" if dur else ""))
            rec_path = session.get("recording_path", "")
            if rec_path:
                rec_path = Path(rec_path).name
            table.setItem(row, 3, QTableWidgetItem(rec_path))
            cal_path = session.get("calibration_path", "")
            if cal_path:
                cal_path = Path(cal_path).parent.name
            table.setItem(row, 4, QTableWidgetItem(cal_path))

        table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        layout.addWidget(table)

        # Action buttons
        action_layout = QHBoxLayout()
        watch_btn = QPushButton("Watch Workout")
        analyze_btn = QPushButton("Show Analysis")
        open_folder_btn = QPushButton("Open Folder")
        action_layout.addWidget(watch_btn)
        action_layout.addWidget(analyze_btn)
        action_layout.addWidget(open_folder_btn)
        action_layout.addStretch()
        layout.addLayout(action_layout)

        def _selected_session() -> dict | None:
            rows = table.selectionModel().selectedRows()
            if not rows:
                self.status_message.emit("Select a session first")
                return None
            return row_to_session.get(rows[0].row())

        def _watch():
            sess = _selected_session()
            if sess:
                self._open_session_videos(sess)

        def _analyze():
            sess = _selected_session()
            if sess:
                self._show_session_analysis(sess)

        def _open_folder():
            sess = _selected_session()
            if sess and sess.get("recording_path"):
                import os, subprocess, sys
                path = sess["recording_path"]
                try:
                    if sys.platform == "win32":
                        os.startfile(path)
                    elif sys.platform == "darwin":
                        subprocess.run(["open", path])
                    else:
                        subprocess.run(["xdg-open", path])
                except Exception as e:
                    self.status_message.emit(f"Failed to open folder: {e}")

        watch_btn.clicked.connect(_watch)
        analyze_btn.clicked.connect(_analyze)
        open_folder_btn.clicked.connect(_open_folder)
        table.doubleClicked.connect(lambda: _watch())

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(dlg.close)
        layout.addWidget(buttons)

        dlg.exec()

    def _open_session_videos(self, session: dict):
        """Open the unified playback dialog (videos + 3D replay)."""
        rec_path = session.get("recording_path")
        if not rec_path:
            self.status_message.emit("No recording path for this session")
            return
        if not Path(rec_path).is_dir():
            self.status_message.emit(f"Recording folder not found: {rec_path}")
            return

        from .workout_playback import WorkoutPlaybackDialog
        dlg = WorkoutPlaybackDialog(session, parent=self)
        dlg.show()

    def _show_session_analysis(self, session: dict):
        """Load stored metrics for a session and show them in the results panel."""
        from PySide6.QtWidgets import QMessageBox
        try:
            import sqlite3
            from ..config import workouts_db_path
            conn = sqlite3.connect(str(workouts_db_path()))
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT metric_name, metric_value, metadata FROM session_results "
                "WHERE session_id = ?",
                (session["id"],),
            ).fetchall()
            conn.close()
        except Exception as e:
            self.status_message.emit(f"Failed to load results: {e}")
            return

        if not rows:
            QMessageBox.information(
                self, "No Analysis",
                "This session has no stored analysis results.\n\n"
                "Re-running analysis from recorded videos is not yet implemented.",
            )
            return

        results = {}
        for r in rows:
            name = r["metric_name"]
            val = r["metric_value"]
            if name == "per_rep_times" and r["metadata"]:
                import json
                try:
                    results["per_rep_times"] = json.loads(r["metadata"])
                except Exception:
                    pass
            elif name == "rep_count":
                results["rep_count"] = int(val)
            elif name == "total_time":
                results["total_time_seconds"] = val
            elif name == "work_per_rep_joules":
                results["work_per_rep_joules"] = val
            elif name == "avg_power_watts":
                results["avg_power_watts"] = val

        self.show_results(results)
        rec_name = Path(session["recording_path"]).name if session.get("recording_path") else ""
        self.status_message.emit(f"Loaded analysis for {rec_name}")

    def _run_workout_analysis(
        self,
        session_id: int | None,
        buffer: list[dict] | None = None,
        already_in_view_frame: bool = False,
    ):
        """Analyze collected 3D keypoints after recording.

        Parameters
        ----------
        session_id : int | None
            Workouts.db row id to attach metrics to.
        buffer : list[dict] | None
            Override input. Defaults to ``self._recording_keypoints``;
            pass an explicit buffer when running on data loaded from a
            just-written keypoints_3d.raw.npz (paused-tracking offline
            path).
        already_in_view_frame : bool
            When True, skip the body_R/body_origin transform — points
            are already in the user-zeroed view frame (the offline npz
            stores them that way after the view-transform-in-write
            change). When False (in-memory live buffer), apply the
            body transform as before.
        """
        if buffer is None:
            buffer = self._recording_keypoints
        if not buffer:
            self.status_message.emit("No 3D keypoints collected — was Live Detection enabled?")
            return

        # Body-centred coordinates: when the buffer is in camera frame
        # (live path), apply the active model's saved (R, t) from the
        # view-transform DB. When already_in_view_frame is True, the
        # buffer is already there — leave the transform as identity so
        # we don't double-apply.
        body_R = None
        body_origin = None
        if not already_in_view_frame:
            try:
                from ..config import load_view_transform
                from datetime import datetime
                preset = load_view_transform(
                    self._current_model_key(),
                    before=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                )
                if preset is not None:
                    R_view, t_view, has_origin = preset
                    if has_origin:
                        # _run_workout_analysis expects the form
                        #     p_body = R @ (p - origin)
                        # whereas the DB stores
                        #     p_view = R @ p + t
                        # i.e. origin = -R^T @ t. Convert.
                        body_R = R_view
                        body_origin = -R_view.T @ t_view
            except Exception:
                pass

        # Extract per-frame signals
        # COCO-17: 0=nose, 5=L_sho, 6=R_sho, 7=L_elb, 8=R_elb, 9=L_wri, 10=R_wri,
        #          11=L_hip, 12=R_hip
        from ..analysis._rep_common import average_elbow_angle

        times = []
        hip_z = []
        hip_xy = []
        head_xyz = []
        head_z_only = []
        elbow_angles = []
        shoulder_z = []
        knee_z_max = []
        # Determine which person is the primary subject (closest to origin).
        # Use per-frame primary_index stored by the tracker; fall back to 0.
        for frame in buffer:
            t = frame["time"]
            persons = frame["persons"]
            if not persons:
                continue
            p_idx = frame.get("primary_index", 0)
            if p_idx >= len(persons):
                p_idx = 0
            kps = persons[p_idx]
            l_hip = kps[11] if len(kps) > 11 else None
            r_hip = kps[12] if len(kps) > 12 else None
            l_sho = kps[5]  if len(kps) > 5  else None
            r_sho = kps[6]  if len(kps) > 6  else None
            l_knee = kps[13] if len(kps) > 13 else None
            r_knee = kps[14] if len(kps) > 14 else None
            nose  = kps[0]  if len(kps) > 0  else None

            # Hip position
            if l_hip is None and r_hip is None:
                hip_val = np.nan
                hip_xy_val = np.array([np.nan, np.nan])
            else:
                if l_hip is not None and r_hip is not None:
                    hip = (np.array(l_hip) + np.array(r_hip)) / 2
                elif l_hip is not None:
                    hip = np.array(l_hip)
                else:
                    hip = np.array(r_hip)
                if body_R is not None and body_origin is not None:
                    hip = body_R @ (hip - body_origin)
                hip_val = hip[2]
                hip_xy_val = np.array([hip[0], hip[1]])

            # Knee max Z (leg raise)
            knee_val = np.nan
            knee_values = []
            for kn in (l_knee, r_knee):
                if kn is not None:
                    kn_arr = np.array(kn, dtype=float)
                    if body_R is not None and body_origin is not None:
                        kn_arr = body_R @ (kn_arr - body_origin)
                    knee_values.append(kn_arr[2])
            if knee_values:
                knee_val = float(np.nanmax(knee_values))

            # Shoulder midpoint (pushup)
            if l_sho is None and r_sho is None:
                shoulder_val = np.nan
            else:
                if l_sho is not None and r_sho is not None:
                    sho = (np.array(l_sho) + np.array(r_sho)) / 2
                elif l_sho is not None:
                    sho = np.array(l_sho)
                else:
                    sho = np.array(r_sho)
                if body_R is not None and body_origin is not None:
                    sho = body_R @ (sho - body_origin)
                shoulder_val = sho[2]

            # Head position (TUG + pullup)
            if nose is not None:
                head = np.array(nose, dtype=float)
                if body_R is not None and body_origin is not None:
                    head = body_R @ (head - body_origin)
                head_z_val = float(head[2])
            else:
                head = np.array([np.nan, np.nan, np.nan])
                head_z_val = float("nan")

            # Elbow angle (biceps) — pass primary person at index 0
            angle = average_elbow_angle([kps])

            if (np.isnan(hip_val) and np.isnan(head).all()
                    and np.isnan(angle) and np.isnan(shoulder_val)):
                continue

            times.append(t)
            hip_z.append(hip_val)
            hip_xy.append(hip_xy_val)
            head_xyz.append(head)
            head_z_only.append(head_z_val)
            elbow_angles.append(angle)
            shoulder_z.append(shoulder_val)
            knee_z_max.append(knee_val)

        if len(times) < 10:
            self.status_message.emit(f"Only {len(times)} usable frames detected — not enough for analysis")
            return

        times_arr = np.array(times)
        hip_z_arr = np.array(hip_z)
        hip_xy_arr = np.array(hip_xy)
        head_arr = np.array(head_xyz)
        head_z_arr = np.array(head_z_only)
        angle_arr = np.array(elbow_angles)
        shoulder_z_arr = np.array(shoulder_z)
        knee_z_arr = np.array(knee_z_max)

        self._last_times = times_arr
        self._last_hip_z = hip_z_arr
        self._last_hip_xy = hip_xy_arr
        self._last_head_xyz = head_arr
        self._last_head_z = head_z_arr
        self._last_elbow_angles = angle_arr
        self._last_shoulder_z = shoulder_z_arr
        self._last_knee_z = knee_z_arr
        self._last_workout_type = self._current_workout_type
        self._last_session_id = session_id

        self._apply_plot_mode(self._current_workout_type)
        # Now that we have buffered data, enable the threshold spin (except stretch)
        if self._current_workout_type != "stretch":
            self.threshold_spin.setEnabled(True)

        wt = self._current_workout_type
        if wt == "timed_up_and_go":
            self._run_tug_analysis(times_arr, hip_z_arr, head_arr, session_id)
        elif wt == "biceps_curl":
            self._run_biceps_analysis(times_arr, angle_arr, session_id)
        elif wt == "pushup":
            self._run_pushup_analysis(times_arr, shoulder_z_arr, session_id)
        elif wt == "pullup":
            self._run_pullup_analysis(times_arr, head_z_arr, session_id)
        elif wt == "leg_raise":
            self._run_leg_raise_analysis(times_arr, knee_z_arr, session_id)
        elif wt == "tandem_stance":
            self._run_tandem_analysis(times_arr, hip_xy_arr, session_id)
        elif wt == "stretch":
            self._run_stretch_analysis(times_arr, hip_z_arr, session_id)
        else:
            self._run_sts_analysis(times_arr, hip_z_arr, session_id)

    def _persist_metrics(self, session_id: int | None, metrics: dict):
        """Delete old session_results for this session and write the new metrics.

        `metrics` is a dict of {metric_name: value_or_(value, metadata_str)}.
        None values are skipped. Called from every _run_*_analysis so live
        threshold changes overwrite the stored analysis.
        """
        if session_id is None:
            return
        try:
            from ..config import delete_session_results, save_session_result
            delete_session_results(session_id)
            for name, val in metrics.items():
                if val is None:
                    continue
                if isinstance(val, tuple):
                    value, metadata = val
                else:
                    value, metadata = val, None
                if value is None:
                    continue
                save_session_result(session_id, name, float(value), metadata=metadata)
        except Exception as e:
            self.status_message.emit(f"Failed to save results: {e}")

    def _run_sts_analysis(self, times_arr, hip_z_arr, session_id):
        import json
        from ..analysis.sit_to_stand import analyze_sit_to_stand
        mass = self.mass_spin.value() if self.mass_spin.isEnabled() else None
        threshold = self.threshold_spin.value()
        result = analyze_sit_to_stand(
            hip_z_arr, times_arr, mass_kg=mass, seated_threshold_m=threshold,
        )

        per_rep_meta = json.dumps(result.per_rep_times) if result.per_rep_times else None
        self._persist_metrics(session_id, {
            "rep_count": result.rep_count,
            "total_time": result.total_time_seconds,
            "seated_threshold_m": result.seated_threshold_m,
            "com_displacement_m": result.com_displacement_m if result.com_displacement_m > 0 else None,
            "work_per_rep_joules": result.work_per_rep_joules,
            "avg_power_watts": result.avg_power_watts,
            "per_rep_times": (0.0, per_rep_meta) if per_rep_meta else None,
        })

        self._update_analysis_plot(times_arr, hip_z_arr, result)
        self.show_results(result.to_dict())

    def _run_biceps_analysis(self, times_arr, angle_arr, session_id):
        from ..analysis.biceps_curl import analyze_biceps_curl
        result = analyze_biceps_curl(
            angle_arr, times_arr,
            extended_threshold_deg=self.threshold_spin.value(),
        )

        self._persist_metrics(session_id, {
            "rep_count": result.rep_count,
            "total_time": result.total_time_seconds,
            "extended_threshold_deg": result.extended_threshold_deg,
            "avg_range_deg": result.avg_range_deg,
        })

        self._update_angle_plot(times_arr, angle_arr, result)
        self.show_results(result.to_dict())

    def _run_pushup_analysis(self, times_arr, shoulder_z_arr, session_id):
        from ..analysis.pushup import analyze_pushup
        result = analyze_pushup(
            shoulder_z_arr, times_arr,
            top_threshold_m=self.threshold_spin.value(),
        )

        self._persist_metrics(session_id, {
            "rep_count": result.rep_count,
            "total_time": result.total_time_seconds,
            "top_threshold_m": result.top_threshold_m,
            "avg_range_m": result.avg_range_m,
        })

        self._update_pushup_plot(times_arr, shoulder_z_arr, result)
        self.show_results(result.to_dict())

    def _run_pullup_analysis(self, times_arr, head_z_arr, session_id):
        from ..analysis.pullup import analyze_pullup
        result = analyze_pullup(
            head_z_arr, times_arr,
            top_threshold_m=self.threshold_spin.value(),
        )

        self._persist_metrics(session_id, {
            "rep_count": result.rep_count,
            "total_time": result.total_time_seconds,
            "top_threshold_m": result.top_threshold_m,
            "avg_range_m": result.avg_range_m,
        })

        self._update_pullup_plot(times_arr, head_z_arr, result)
        self.show_results(result.to_dict())

    def _run_leg_raise_analysis(self, times_arr, knee_z_arr, session_id):
        from ..analysis.leg_raise import analyze_leg_raise
        result = analyze_leg_raise(
            knee_z_arr, times_arr,
            lift_threshold_m=self.threshold_spin.value(),
        )

        self._persist_metrics(session_id, {
            "rep_count": result.rep_count,
            "total_time": result.total_time_seconds,
            "lift_threshold_m": result.lift_threshold_m,
            "avg_range_m": result.avg_range_m,
        })

        self._update_leg_raise_plot(times_arr, knee_z_arr, result)
        self.show_results(result.to_dict())

    def _run_tandem_analysis(self, times_arr, hip_xy_arr, session_id):
        from ..analysis.tandem_stance import analyze_tandem_stance
        result = analyze_tandem_stance(
            hip_xy_arr, times_arr,
            sway_threshold_m=self.threshold_spin.value(),
        )

        self._persist_metrics(session_id, {
            "hold_seconds": result.hold_seconds,
            "total_seconds": result.total_seconds,
            "stability_fraction": result.stability_fraction,
            "sway_threshold_m": result.sway_threshold_m,
        })

        self._update_tandem_plot(result)
        self.show_results(result.to_dict())

    def _run_stretch_analysis(self, times_arr, hip_z_arr, session_id):
        from ..analysis.stretch import analyze_stretch
        result = analyze_stretch(hip_z_arr, times_arr)

        self._persist_metrics(session_id, {
            "hold_seconds": result.hold_seconds,
            "steadiness": result.steadiness,
            "max_range_m": result.max_range_m,
        })

        self._update_stretch_plot(times_arr, hip_z_arr, result)
        self.show_results(result.to_dict())

    def _run_tug_analysis(self, times_arr, hip_z_arr, head_arr, session_id):
        from ..analysis.tug import analyze_tug
        result = analyze_tug(
            hip_z_arr, head_arr, times_arr,
            seated_threshold_m=self.threshold_spin.value(),
            speed_threshold_mps=self.speed_threshold_spin.value(),
        )

        self._persist_metrics(session_id, {
            "tug_duration": result.duration_seconds,
            "tug_start_time": result.start_time,
            "tug_end_time": result.end_time,
            "tug_max_head_speed": result.max_head_speed,
            "seated_threshold_m": result.seated_threshold_m,
            "speed_threshold_mps": result.speed_threshold_mps,
        })

        self._update_tug_plot(result)
        self.show_results(result.to_dict())

    def show_results(self, results: dict):
        # Each metric appears in the result dict whether or not it was
        # actually computed \u2014 analysers leave fields they couldn't fill
        # set to None (e.g. sit-to-stand on a level-walk recording can't
        # measure work/power because COM displacement is zero, so
        # avg_power_watts comes back None). Guarding only on key
        # presence used to crash with
        #     TypeError: unsupported format string passed to NoneType.__format__
        # whenever the user re-ran analysis on a recording that didn't
        # match the active analyser.
        def _v(key):
            """Return the numeric value at `key` if present and not None."""
            v = results.get(key)
            return v if v is not None else None

        lines = []
        if (v := _v("duration_seconds")) is not None:
            lines.append(f"TUG duration: {v:.2f} s")
        if (v := _v("max_head_speed")) is not None:
            lines.append(f"Max head speed: {v:.2f} m/s")
        if (v := _v("rep_count")) is not None:
            lines.append(f"Repetitions: {v}")
        if (v := _v("total_time_seconds")) is not None:
            lines.append(f"Total time: {v:.1f} s")
        prt = results.get("per_rep_times")
        if prt:  # truthy: non-empty list
            avg = sum(prt) / len(prt)
            lines.append(f"Avg rep time: {avg:.2f} s")
        if (v := _v("avg_range_deg")) is not None:
            lines.append(f"Avg range: {v:.0f}\u00b0")
        if (v := _v("avg_range_m")) is not None:
            lines.append(f"Avg range: {v * 100:.0f} cm")
        if (v := _v("avg_power_watts")) is not None:
            lines.append(f"Avg power: {v:.1f} W")
        if (v := _v("work_per_rep_joules")) is not None:
            lines.append(f"Work per rep: {v:.1f} J")
        # Duration-based (tandem, stretch)
        if (v := _v("hold_seconds")) is not None:
            lines.append(f"Hold: {v:.1f} s")
        if (v := _v("stability_fraction")) is not None:
            lines.append(f"Stable: {v * 100:.0f}%")
        if (v := _v("steadiness")) is not None:
            lines.append(f"Steadiness: {v * 100:.0f}%")

        self.results_label.setText("\n".join(lines) if lines else "No results")
        self.results_label.setStyleSheet("color: #FFF; font-size: 16px;")
