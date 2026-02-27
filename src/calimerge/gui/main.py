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
from PySide6.QtCore import Qt

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

    def _on_tab_changed(self, index: int):
        self.state_manager.update_state(current_tab=index)

    def _show_status(self, message: str):
        self.status_bar.showMessage(message, 5000)

    def _show_error(self, message: str):
        QMessageBox.warning(self, "Error", message)

    def closeEvent(self, event):
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
