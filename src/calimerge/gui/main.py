"""
Main window for the Calimerge unified GUI.
"""

from __future__ import annotations

import sys

from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QTabWidget,
    QStatusBar,
    QWidget,
    QVBoxLayout,
    QMessageBox,
)
from .state import StateManager
from .tabs import CamerasTab, IntrinsicTab, ExtrinsicTab, ProcessTab
from .. import __version__


class MainWindow(QMainWindow):
    """
    Main application window with tabbed workflow.

    Tabs:
    1. Record - Camera detection, preview, configuration, recording
    2. Intrinsic - Per-camera lens calibration
    3. Extrinsic - Multi-camera spatial calibration
    4. Process - Tracking and triangulation
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Calimerge - Multi-Camera Motion Capture")
        self.setMinimumSize(1000, 700)

        # State manager
        self.state_manager = StateManager(self)

        self._init_ui()
        self._connect_signals()
        self._load_startup_settings()

    def _init_ui(self):
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Create tabs
        self.cameras_tab = CamerasTab(self.state_manager)
        self.intrinsic_tab = IntrinsicTab(self.state_manager)
        self.extrinsic_tab = ExtrinsicTab(self.state_manager)
        self.process_tab = ProcessTab(self.state_manager)

        self.tabs.addTab(self.cameras_tab, "1. Record")
        self.tabs.addTab(self.intrinsic_tab, "2. Intrinsic")
        self.tabs.addTab(self.extrinsic_tab, "3. Extrinsic")
        self.tabs.addTab(self.process_tab, "4. Process")

        layout.addWidget(self.tabs)
        self.setCentralWidget(central)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

        # Version label (permanent, right side)
        from PySide6.QtWidgets import QLabel
        version_label = QLabel(f"v{__version__}")
        version_label.setStyleSheet("color: #888; padding-right: 8px;")
        self.status_bar.addPermanentWidget(version_label)

    def _connect_signals(self):
        self.cameras_tab.status_message.connect(self._show_status)
        self.intrinsic_tab.status_message.connect(self._show_status)
        self.extrinsic_tab.status_message.connect(self._show_status)
        self.process_tab.status_message.connect(self._show_status)

        self.state_manager.status_message.connect(self._show_status)
        self.state_manager.error_occurred.connect(self._show_error)

        # When project folder changes, load and apply its settings
        self.cameras_tab.project_folder_changed.connect(self._on_project_folder_changed)
        # Save settings whenever cameras tab requests it (e.g. resolution change)
        self.cameras_tab.save_settings_requested.connect(self._save_project_settings)

    def _on_project_folder_changed(self, folder):
        try:
            from ..config import load_project_settings
            settings = load_project_settings(folder)
            self.setstate(settings)
            self._show_status(f"Project settings loaded from {folder}")
        except Exception as e:
            self._show_status(f"Could not load project settings: {e}")

    def setstate(self, settings: dict) -> None:
        """
        Apply a project settings dict to all tabs.

        settings keys (all optional):
          fps, codec               → cameras tab recording settings
          cameras                  → {serial: {enabled, resolution, exposure}}
          intrinsic_max_frames     → intrinsic tab frame-count slider
          charuco_intrinsic        → intrinsic tab board config
          charuco_extrinsic        → extrinsic tab board config
        """
        self.cameras_tab.apply_project_settings(settings)
        self.intrinsic_tab.apply_project_settings(settings)
        self.extrinsic_tab.apply_project_settings(settings)

    def _collect_project_settings(self) -> dict:
        """Gather current settings from all tabs into one dict."""
        settings = {}
        settings.update(self.cameras_tab.get_project_settings())
        settings.update(self.intrinsic_tab.get_project_settings())
        settings.update(self.extrinsic_tab.get_project_settings())
        return settings

    def _save_project_settings(self) -> None:
        """Save current settings to the active project folder."""
        try:
            from ..config import save_project_settings, load_app_settings, save_app_settings
            folder = self.cameras_tab.base_output_path
            settings = self._collect_project_settings()
            save_project_settings(settings, folder)
            # Ensure app_settings always knows the active project folder
            app = load_app_settings()
            resolved = str(folder.resolve())
            if app.get("last_project_folder") != resolved:
                app["last_project_folder"] = resolved
                save_app_settings(app)
        except Exception:
            pass

    def _on_tab_changed(self, index: int):
        self.state_manager.update_state(current_tab=index)

    def _show_status(self, message: str):
        self.status_bar.showMessage(message, 5000)

    def _show_error(self, message: str):
        QMessageBox.warning(self, "Error", message)

    def _load_startup_settings(self) -> None:
        """Load project settings from the last-used project folder on startup."""
        try:
            from ..config import load_project_settings
            folder = self.cameras_tab.base_output_path
            if folder.is_dir():
                settings = load_project_settings(folder)
                self.setstate(settings)
        except Exception:
            pass

    def closeEvent(self, event):
        # Save project settings before exit
        self._save_project_settings()

        # Stop preview/recording
        if hasattr(self.cameras_tab, "stop_preview"):
            self.cameras_tab.stop_preview()

        # Shutdown camera subsystem
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
