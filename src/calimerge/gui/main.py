"""
Main window for the Calimerge unified GUI.

Default landing page is the WorkoutPage. Calibration tools are
accessible via Tools → Calibration menu.
"""

from __future__ import annotations

import sys
from pathlib import Path

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QStatusBar,
    QLabel,
    QMessageBox,
    QFileDialog,
)
from PySide6.QtGui import QAction, QFont
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
        workdir_action.setToolTip("Set the directory for recordings and per-project calibration files")
        workdir_action.triggered.connect(self._browse_workout_dir)
        file_menu.addAction(workdir_action)

        datadir_action = QAction("App Data Directory...", self)
        datadir_action.setToolTip("Set where Calimerge caches model files, databases, and app settings")
        datadir_action.triggered.connect(self._browse_data_dir)
        file_menu.addAction(datadir_action)

        file_menu.addSeparator()

        add_person_action = QAction("Add Person...", self)
        add_person_action.setToolTip("Create a new user and assign them to a workout program")
        add_person_action.triggered.connect(self._add_person)
        file_menu.addAction(add_person_action)

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

    def _browse_data_dir(self):
        """Let the user choose the app data directory (models, DBs, settings)."""
        from ..config import data_dir, set_data_dir

        current = str(data_dir())
        folder = QFileDialog.getExistingDirectory(
            self, "Select App Data Directory", current)
        if not folder:
            return

        try:
            set_data_dir(Path(folder))
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Could not set data directory: {e}")
            return

        QMessageBox.information(
            self,
            "App Data Directory Updated",
            f"Calimerge will use this directory on next launch:\n\n{folder}\n\n"
            "Move your existing model files and databases there manually if you want "
            "to keep them. Restart the app to pick up the change.",
        )
        self._show_status(f"App data directory set to: {folder} (restart required)")

    def _add_person(self):
        """Open the Add Person dialog to create a user + assign a program."""
        from .add_person_dialog import AddPersonDialog

        dlg = AddPersonDialog(parent=self)
        if dlg.exec() != AddPersonDialog.DialogCode.Accepted:
            return
        if dlg.created_user is None:
            return

        username = dlg.created_user["username"]
        # Refresh the workout page's user dropdown so the new person is
        # immediately selectable without needing a restart.
        try:
            self.workout_page._refresh_user_list()
            for i in range(self.workout_page.user_combo.count()):
                if self.workout_page.user_combo.itemText(i) == username:
                    self.workout_page.user_combo.setCurrentIndex(i)
                    break
        except Exception:
            pass

        prog_msg = ""
        if dlg.selected_program_id is not None:
            try:
                from ..config import get_program
                prog = get_program(dlg.selected_program_id)
                if prog:
                    prog_msg = f" — program: {prog['display_name']}"
            except Exception:
                pass
        self._show_status(f"Added person: {username}{prog_msg}")

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

    # Application-wide font: Gill Sans, with platform fallbacks so we
    # degrade gracefully on machines that don't have it. Explicit
    # `setFont(QFont("monospace", ...))` calls in widgets that show
    # code/log/numeric output remain unaffected — those are intentional.
    # Explicit Normal weight: Qt's platform style on Windows can pick up a
    # bolder Segoe UI variant that reads as 'fat' if we don't pin this.
    _ui_font = QFont()
    _ui_font.setFamilies([
        "Gill Sans Nova Light",   # Win11 ships Nova family with Light/Cond Light faces
        "Gill Sans Nova",         # request weight pin (ExtraLight/Light) below
        "Gill Sans MT Light",     # explicit Light variant from Office bundle
        "Gill Sans Light",        # generic Light face name
        "Gill Sans MT",           # Windows (Office bundled)
        "Gill Sans",              # macOS / Windows (if installed)
        "Gill Sans Std",          # Adobe-installed variant
        "Helvetica Neue",
        "Helvetica",
        "Segoe UI Light",         # Windows fallback (lighter than Segoe UI Normal)
        "Segoe UI",
        "Arial",
    ])
    _ui_font.setWeight(QFont.Weight.ExtraLight)  # 200 — try thinner; falls back to Light if unavailable
    _ui_font.setPointSize(10)  # slight bump from the platform default (~9pt on Windows)
    _ui_font.setStyleStrategy(QFont.StyleStrategy.PreferAntialias)
    app.setFont(_ui_font)

    # Print the actually-resolved family + weight so the user can verify
    # whether Gill Sans Light was found or if Qt fell through to a fallback.
    from PySide6.QtGui import QFontInfo
    info = QFontInfo(_ui_font)
    print(
        f"[ui-font] requested Gill Sans Light, resolved to: {info.family()!r} "
        f"(weight={info.weight()})",
        flush=True,
    )

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
