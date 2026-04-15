"""
Program picker dialog — shown at login if the user has no active program.

Presents the available program templates with a short description each and
lets the user choose one. Selection is persisted via set_user_program().
"""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QRadioButton,
    QButtonGroup,
    QGroupBox,
    QDialogButtonBox,
)


class ProgramPickerDialog(QDialog):
    """Dialog letting the user choose an active program."""

    def __init__(self, programs: list[dict], program_exercises: dict[int, list[dict]],
                 parent=None):
        """
        programs: list of dicts as returned by list_programs()
        program_exercises: dict mapping program_id -> list of exercise dicts
        """
        super().__init__(parent)
        self.setWindowTitle("Choose Your Program")
        self.setMinimumSize(600, 400)

        self._programs = programs
        self._program_exercises = program_exercises
        self.selected_program_id: int | None = None

        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        header = QLabel("Pick a workout program to follow.")
        header.setStyleSheet("font-size: 14px; font-weight: bold; color: #4CAF50;")
        layout.addWidget(header)

        self.button_group = QButtonGroup(self)

        for i, prog in enumerate(programs):
            group = QGroupBox()
            group_layout = QVBoxLayout(group)

            radio = QRadioButton(prog["display_name"])
            radio.setStyleSheet("font-size: 13px; font-weight: bold;")
            if i == 0:
                radio.setChecked(True)
                self.selected_program_id = prog["id"]
            radio.toggled.connect(self._on_radio_toggled)
            self.button_group.addButton(radio, prog["id"])
            group_layout.addWidget(radio)

            desc = QLabel(prog.get("description") or "")
            desc.setWordWrap(True)
            desc.setStyleSheet("color: #aaa;")
            group_layout.addWidget(desc)

            # Exercise summary
            exercises = program_exercises.get(prog["id"], [])
            if exercises:
                ex_lines = []
                for e in exercises:
                    if e["target_reps"] is not None:
                        target = f"{e['target_reps']} reps"
                    elif e["target_duration_seconds"] is not None:
                        target = f"{int(e['target_duration_seconds'])} s hold"
                    else:
                        target = "—"
                    ex_lines.append(
                        f"  • {e['display_name']}: {e['sets_per_day']} sets × "
                        f"{e['days_per_week']}/wk ({target})"
                    )
                ex_label = QLabel("\n".join(ex_lines))
                ex_label.setStyleSheet("color: #ccc; font-family: monospace;")
                group_layout.addWidget(ex_label)

            layout.addWidget(group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _on_radio_toggled(self, checked: bool):
        if checked:
            btn = self.sender()
            prog_id = self.button_group.id(btn)
            if prog_id >= 0:
                self.selected_program_id = prog_id
