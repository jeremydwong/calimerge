"""
Main window for the Calimerge unified GUI.

Default landing page is the WorkoutPage. Calibration tools are
accessible via Tools → Calibration menu.
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QStatusBar,
    QLabel,
    QMessageBox,
    QFileDialog,
)
from PySide6.QtGui import QAction
from .state import StateManager
from .workout_page import WorkoutPage
from .calibration_dialog import CalibrationDialog
from .. import __version__


class MainWindow(QMainWindow):
    """Main application window with workout landing page."""

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Calimerge - Multi-Camera Motion Capture")
        self.setMinimumSize(1000, 700)

        # State manager
        self.state_manager = StateManager(self)

        # Calibration dialog (lazy — created on first open)
        self._cal_dialog: CalibrationDialog | None = None

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        # ── Menu bar ──
        menubar = self.menuBar()

        file_menu = menubar.addMenu("File")
        workdir_action = QAction("Workout Directory...", self)
        workdir_action.setToolTip("Set the directory for recordings, calibrations, and databases")
        workdir_action.triggered.connect(self._browse_workout_dir)
        file_menu.addAction(workdir_action)

        tools_menu = menubar.addMenu("Tools")
        cal_action = QAction("Calibration...", self)
        cal_action.setToolTip("Open camera calibration tools (Record, Intrinsic, Extrinsic, Process)")
        cal_action.triggered.connect(self._open_calibration)
        tools_menu.addAction(cal_action)

        # ── Central widget: workout page ──
        self.workout_page = WorkoutPage(self.state_manager)
        self.setCentralWidget(self.workout_page)

        # ── Status bar ──
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        version_label = QLabel(f"v{__version__}")
        version_label.setStyleSheet("color: #888; padding-right: 8px;")
        self.status_bar.addPermanentWidget(version_label)

    def _connect_signals(self):
        self.state_manager.status_message.connect(self._show_status)
        self.state_manager.error_occurred.connect(self._show_error)
        self.workout_page.status_message.connect(self._show_status)

    def _browse_workout_dir(self):
        """Let the user choose the workout/project directory."""
        from ..config import load_app_settings, save_app_settings

        app = load_app_settings()
        current = app.get("last_project_folder", "")

        folder = QFileDialog.getExistingDirectory(
            self, "Select Workout Directory", current)
        if not folder:
            return

        app["last_project_folder"] = folder
        save_app_settings(app)
        self._show_status(f"Workout directory: {folder}")

        # Update calibration dialog if open
        if self._cal_dialog:
            from pathlib import Path
            self._cal_dialog.cameras_tab.base_output_path = Path(folder)
            self._on_project_folder_changed(Path(folder))

        # Refresh calibration status on workout page
        self.workout_page._check_calibration()

    def _open_calibration(self):
        """Open the calibration tools dialog."""
        if self._cal_dialog is None:
            self._cal_dialog = CalibrationDialog(self.state_manager, parent=self)
            self._cal_dialog.status_message.connect(self._show_status)

            # Wire up project folder / settings persistence
            ct = self._cal_dialog.cameras_tab
            ct.project_folder_changed.connect(self._on_project_folder_changed)
            ct.save_settings_requested.connect(self._save_project_settings)

            # Load startup settings into calibration tabs
            self._load_startup_settings()

        self._cal_dialog.show()
        self._cal_dialog.raise_()
        self._cal_dialog.activateWindow()

    def _on_project_folder_changed(self, folder):
        try:
            from ..config import load_project_settings
            settings = load_project_settings(folder)
            if self._cal_dialog:
                self._cal_dialog.apply_project_settings(settings)
            self._show_status(f"Project settings loaded from {folder}")
            # Refresh calibration status on workout page
            self.workout_page._check_calibration()
        except Exception as e:
            self._show_status(f"Could not load project settings: {e}")

    def _collect_project_settings(self) -> dict:
        if self._cal_dialog:
            return self._cal_dialog.get_project_settings()
        return {}

    def _save_project_settings(self) -> None:
        try:
            from ..config import save_project_settings, load_app_settings, save_app_settings
            if not self._cal_dialog:
                return
            folder = self._cal_dialog.cameras_tab.base_output_path
            settings = self._collect_project_settings()
            save_project_settings(settings, folder)
            app = load_app_settings()
            resolved = str(folder.resolve())
            if app.get("last_project_folder") != resolved:
                app["last_project_folder"] = resolved
                save_app_settings(app)
        except Exception:
            pass

    def _load_startup_settings(self) -> None:
        try:
            from ..config import load_project_settings
            if not self._cal_dialog:
                return
            folder = self._cal_dialog.cameras_tab.base_output_path
            if folder.is_dir():
                settings = load_project_settings(folder)
                self._cal_dialog.apply_project_settings(settings)
        except Exception:
            pass

    def _show_status(self, message: str):
        self.status_bar.showMessage(message, 5000)

    def _show_error(self, message: str):
        QMessageBox.warning(self, "Error", message)

    def closeEvent(self, event):
        self._save_project_settings()

        # Stop all workout page workers before closing
        try:
            wp = self.workout_page
            wp._stop_detection()
            wp._stop_preview()
            if wp.recording_worker is not None:
                wp.recording_worker.running = False
                wp.recording_worker.wait(2000)
        except Exception:
            pass

        if self._cal_dialog:
            self._cal_dialog.close()

        try:
            from ..camera_binding import shutdown
            shutdown()
        except Exception:
            pass

        event.accept()


def main():
    """Entry point for the GUI application."""
    app = QApplication(sys.argv)

    app.setApplicationName("Calimerge")
    app.setOrganizationName("Calimerge")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
