"""
Calibration tools dialog — wraps the existing 4-tab calibration workflow.

Opened via Tools → Calibration from the main window.
"""

from __future__ import annotations

from PySide6.QtWidgets import QDialog, QVBoxLayout, QTabWidget
from PySide6.QtCore import Signal

from .state import StateManager
from .tabs import CamerasTab, IntrinsicTab, ExtrinsicTab, ProcessTab


class CalibrationDialog(QDialog):
    """Non-modal dialog containing the calibration tab workflow."""

    status_message = Signal(str)

    def __init__(self, state_manager: StateManager, parent=None):
        super().__init__(parent)
        self.state_manager = state_manager
        self.setWindowTitle("Calibration Tools")
        self.setMinimumSize(1100, 750)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        self.tabs = QTabWidget()

        self.cameras_tab = CamerasTab(state_manager)
        self.intrinsic_tab = IntrinsicTab(state_manager)
        self.extrinsic_tab = ExtrinsicTab(state_manager)
        self.process_tab = ProcessTab(state_manager)

        self.tabs.addTab(self.cameras_tab, "1. Record")
        self.tabs.addTab(self.intrinsic_tab, "2. Intrinsic")
        self.tabs.addTab(self.extrinsic_tab, "3. Extrinsic")
        self.tabs.addTab(self.process_tab, "4. Process")

        layout.addWidget(self.tabs)

        # Forward status messages
        self.cameras_tab.status_message.connect(self.status_message.emit)
        self.intrinsic_tab.status_message.connect(self.status_message.emit)
        self.extrinsic_tab.status_message.connect(self.status_message.emit)
        self.process_tab.status_message.connect(self.status_message.emit)

    def apply_project_settings(self, settings: dict) -> None:
        self.cameras_tab.apply_project_settings(settings)
        self.intrinsic_tab.apply_project_settings(settings)
        self.extrinsic_tab.apply_project_settings(settings)

    def get_project_settings(self) -> dict:
        settings = {}
        settings.update(self.cameras_tab.get_project_settings())
        settings.update(self.intrinsic_tab.get_project_settings())
        settings.update(self.extrinsic_tab.get_project_settings())
        return settings

    def closeEvent(self, event):
        # Stop preview if running
        if hasattr(self.cameras_tab, "stop_preview"):
            self.cameras_tab.stop_preview()
        event.accept()
