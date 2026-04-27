"""
Add Person dialog — File menu entry to create a new user and assign them
to a workout program in one step.

Distinct from program_picker.py: that dialog is used inside the workout-page
login flow when a *known* user has no active program. This dialog is the
operator's path for adding a brand-new person to the system.
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QFormLayout,
    QLabel,
    QLineEdit,
    QDoubleSpinBox,
    QComboBox,
    QPushButton,
    QDialogButtonBox,
    QMessageBox,
)


class AddPersonDialog(QDialog):
    """Create a new user and (optionally) assign a workout program to them."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add Person")
        self.setMinimumWidth(360)

        from ..config import list_programs

        self._programs = list_programs()
        self.created_user: dict | None = None  # populated on accept
        self.selected_program_id: int | None = None

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        header = QLabel("Add a new person and assign a workout program.")
        header.setStyleSheet("font-weight: bold; color: #4CAF50;")
        layout.addWidget(header)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignRight)

        self.username_edit = QLineEdit()
        self.username_edit.setPlaceholderText("e.g. jane.doe")
        self.username_edit.setMaxLength(64)
        form.addRow("Username:", self.username_edit)

        self.mass_spin = QDoubleSpinBox()
        self.mass_spin.setRange(0.0, 300.0)
        self.mass_spin.setDecimals(1)
        self.mass_spin.setSuffix(" kg")
        self.mass_spin.setValue(0.0)
        self.mass_spin.setSpecialValueText("(unknown)")
        form.addRow("Mass:", self.mass_spin)

        self.program_combo = QComboBox()
        self.program_combo.addItem("(no program)", None)
        for prog in self._programs:
            self.program_combo.addItem(prog["display_name"], prog["id"])
        if self._programs:
            # Pre-select the first program so the common path is one click.
            self.program_combo.setCurrentIndex(1)
        form.addRow("Program:", self.program_combo)

        layout.addLayout(form)

        self._program_blurb = QLabel("")
        self._program_blurb.setWordWrap(True)
        self._program_blurb.setStyleSheet("color: #888; font-size: 11px;")
        layout.addWidget(self._program_blurb)
        self.program_combo.currentIndexChanged.connect(self._refresh_blurb)
        self._refresh_blurb()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._on_accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _refresh_blurb(self):
        prog_id = self.program_combo.currentData()
        if prog_id is None:
            self._program_blurb.setText(
                "No program assigned. The user can pick one at first login."
            )
            return
        prog = next((p for p in self._programs if p["id"] == prog_id), None)
        desc = (prog or {}).get("description") or ""
        self._program_blurb.setText(desc)

    def _on_accept(self):
        from ..config import create_user, get_user, set_user_program

        username = self.username_edit.text().strip()
        if not username:
            QMessageBox.warning(self, "Add Person", "Username is required.")
            return

        # Refuse to silently overwrite an existing user — the operator
        # should consciously decide whether they want a fresh person.
        existing = get_user(username)
        if existing is not None:
            QMessageBox.warning(
                self,
                "Add Person",
                f"User '{username}' already exists. Pick a different username "
                f"or use the workout page to log in as them.",
            )
            return

        mass = self.mass_spin.value()
        mass_kg = mass if mass > 0 else None

        try:
            user = create_user(username, mass_kg=mass_kg)
        except Exception as e:
            QMessageBox.critical(self, "Add Person", f"Failed to create user: {e}")
            return

        program_id = self.program_combo.currentData()
        if program_id is not None:
            try:
                set_user_program(user["id"], program_id)
            except Exception as e:
                QMessageBox.warning(
                    self,
                    "Add Person",
                    f"User created, but failed to assign program: {e}",
                )

        self.created_user = user
        self.selected_program_id = program_id
        self.accept()
