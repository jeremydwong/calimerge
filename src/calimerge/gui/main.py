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
from .tabs import CamerasTab, RecordTab, IntrinsicTab, ExtrinsicTab, ProcessTab


class MainWindow(QMainWindow):
    """
    Main application window with tabbed workflow.

    Tabs:
    1. Cameras - Detection, preview, configuration
    2. Record - Synchronized video capture
    3. Intrinsic - Per-camera lens calibration
    4. Extrinsic - Multi-camera spatial calibration
    5. Process - Tracking and triangulation
    """

    def __init__(self):
        super().__init__()

        self.setWindowTitle("Calimerge - Multi-Camera Motion Capture")
        self.setMinimumSize(1000, 700)

        # State manager (thin coordinator, not god object)
        self.state_manager = StateManager(self)

        self._init_ui()
        self._connect_signals()

    def _init_ui(self):
        # Central widget with tabs
        central = QWidget()
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)

        self.tabs = QTabWidget()
        self.tabs.currentChanged.connect(self._on_tab_changed)

        # Create tabs
        self.cameras_tab = CamerasTab(self.state_manager)
        self.record_tab = RecordTab(self.state_manager)
        self.intrinsic_tab = IntrinsicTab(self.state_manager)
        self.extrinsic_tab = ExtrinsicTab(self.state_manager)
        self.process_tab = ProcessTab(self.state_manager)

        self.tabs.addTab(self.cameras_tab, "1. Cameras")
        self.tabs.addTab(self.record_tab, "2. Record")
        self.tabs.addTab(self.intrinsic_tab, "3. Intrinsic")
        self.tabs.addTab(self.extrinsic_tab, "4. Extrinsic")
        self.tabs.addTab(self.process_tab, "5. Process")

        layout.addWidget(self.tabs)
        self.setCentralWidget(central)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")

    def _connect_signals(self):
        # Connect tab status signals to status bar
        self.cameras_tab.status_message.connect(self._show_status)
        self.record_tab.status_message.connect(self._show_status)
        self.intrinsic_tab.status_message.connect(self._show_status)
        self.extrinsic_tab.status_message.connect(self._show_status)
        self.process_tab.status_message.connect(self._show_status)

        # Connect state manager signals
        self.state_manager.status_message.connect(self._show_status)
        self.state_manager.error_occurred.connect(self._show_error)

    def _on_tab_changed(self, index: int):
        """Handle tab change."""
        self.state_manager.update_state(current_tab=index)

    def _show_status(self, message: str):
        """Show message in status bar."""
        self.status_bar.showMessage(message, 5000)

    def _show_error(self, message: str):
        """Show error dialog."""
        QMessageBox.warning(self, "Error", message)

    def closeEvent(self, event):
        """Handle window close."""
        # Stop any active preview/recording
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

    # Set application metadata
    app.setApplicationName("Calimerge")
    app.setOrganizationName("Calimerge")

    # Create and show main window
    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
