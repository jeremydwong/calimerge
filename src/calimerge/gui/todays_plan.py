"""
Today's Plan widget — shown on the workout page for users following a program.

Lists each exercise in the user's active program with a "X/Y sets this week"
status, highlights exercises suggested for today, and emits a signal when the
user picks one to work on. Designed to replace the workout-type radio button
block on the workout page.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QGroupBox,
)

from ..programs import parse_suggested_days


class ExerciseRow(QFrame):
    """One row in the Today's Plan list.

    Colouring:
      - Dull red (default): sets done today are less than sets_per_day
      - Green: today's quota is met
      - A "TODAY" badge appears if the exercise is suggested for the current weekday.
    """

    clicked = Signal(dict)

    def __init__(self, exercise: dict, sets_done_week: int, sets_done_today: int,
                 is_today: bool, parent=None):
        super().__init__(parent)
        self.exercise = exercise
        self.setFrameShape(QFrame.Shape.StyledPanel)
        self.setCursor(Qt.CursorShape.PointingHandCursor)

        sets_per_day = exercise["sets_per_day"]
        completed_today = sets_done_today >= sets_per_day

        # Calmer dusty-red while incomplete, fresh green when today's quota
        # is met. The previous reds (#3a1e1e bg / #8B3A3A border) read as
        # alarm; these are muted enough to live next to ten stacked rows
        # without making the panel feel angry.
        if completed_today:
            bg_color = "#1e3a1e"
            border_color = "#4CAF50"
            status_color = "#4CAF50"
        else:
            bg_color = "#2f2426"   # muted dark, slight rose tint
            border_color = "#8a5d5d"  # dusty rose
            status_color = "#FFC107"

        # Tight padding — earlier 4px CSS pad + 6px layout margins burned
        # ~20 px per row, which made a 10-task FGA plan unreadable except
        # at fullscreen. Drop to 1 px each for a near-flat row stack.
        self.setStyleSheet(
            f"QFrame {{ background: {bg_color}; "
            f"border: 1px solid {border_color}; border-radius: 3px; padding: 1px; }}"
            f"QFrame:hover {{ border: 1px solid #4CAF50; }}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(6, 1, 6, 1)
        layout.setSpacing(10)

        # Name + target on a single row (rather than stacked) — cramped
        # widths weren't surfacing the second line readably anyway. Regular
        # weight (was bold); the row already has its own border and bg
        # contrast, so additional weight just made the panel shouty.
        name_label = QLabel(exercise["display_name"])
        name_label.setStyleSheet("font-size: 12px;")
        layout.addWidget(name_label, stretch=1)

        # Suppress the rep string for assessment-shape rows (sets=1, reps=1)
        # — reads as awkward boilerplate ("1 reps × 1 sets"). Keep it for
        # rep-based programs where it actually communicates a target.
        is_assessment_shape = (
            sets_per_day == 1
            and exercise.get("target_reps") in (1, None)
            and exercise.get("target_duration_seconds") is None
        )
        if not is_assessment_shape:
            if exercise["target_reps"] is not None:
                target_str = f"{exercise['target_reps']} reps × {sets_per_day} sets"
            elif exercise["target_duration_seconds"] is not None:
                target_str = (
                    f"{int(exercise['target_duration_seconds'])}s × {sets_per_day} sets"
                )
            else:
                target_str = f"{sets_per_day} sets"
            target_label = QLabel(target_str)
            target_label.setStyleSheet("font-size: 10px; color: #aaa;")
            layout.addWidget(target_label)

        # Today's progress + weekly tail. Regular weight to match the name
        # — colour is doing the highlighting.
        total_sets_week = sets_per_day * exercise["days_per_week"]
        status_text = f"{sets_done_today}/{sets_per_day} today"
        status_label = QLabel(status_text)
        status_label.setStyleSheet(
            f"color: {status_color}; font-size: 11px;"
        )
        layout.addWidget(status_label)

        week_label = QLabel(f"({sets_done_week}/{total_sets_week} week)")
        week_label.setStyleSheet("color: #888; font-size: 9px;")
        layout.addWidget(week_label)

        if is_today:
            today_badge = QLabel("TODAY")
            today_badge.setStyleSheet(
                "background: #4CAF50; color: #000; font-weight: bold; "
                "padding: 1px 4px; border-radius: 3px; font-size: 9px;"
            )
            layout.addWidget(today_badge)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self.exercise)
        super().mousePressEvent(event)


class TodaysPlanWidget(QGroupBox):
    """Today's plan panel.

    - Displays the active program name
    - Lists exercises with weekly progress
    - Emits exercise_selected(dict) when the user clicks one
    """

    exercise_selected = Signal(dict)

    def __init__(self, parent=None):
        super().__init__("Today's Plan")
        self._program: dict | None = None
        self._exercises: list[dict] = []
        self._sets_done_week: dict[int, int] = {}
        self._sets_done_today: dict[int, int] = {}
        self._selected_id: int | None = None
        self._program_started_at: datetime | None = None

        layout = QVBoxLayout(self)
        layout.setSpacing(4)
        layout.setContentsMargins(6, 6, 6, 6)

        self.header = QLabel("No program active")
        self.header.setStyleSheet("font-size: 12px; color: #888;")
        layout.addWidget(self.header)

        self.rows_container = QWidget()
        self.rows_layout = QVBoxLayout(self.rows_container)
        self.rows_layout.setSpacing(1)   # was 4 — tighter inter-row stack
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.rows_container)

        self.empty_label = QLabel(
            "Log in and select a program to see your plan here."
        )
        self.empty_label.setStyleSheet("color: #888; padding: 10px;")
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.empty_label)

    def set_program(self, program: dict | None, exercises: list[dict],
                    sets_done_week: dict[int, int],
                    sets_done_today: dict[int, int] | None = None,
                    program_started_at: datetime | None = None) -> None:
        """Populate the widget with a program's exercises and progress counts."""
        self._program = program
        self._exercises = list(exercises)
        self._sets_done_week = dict(sets_done_week)
        self._sets_done_today = dict(sets_done_today or {})
        self._program_started_at = program_started_at
        self._rebuild()

    def get_selected_exercise(self) -> dict | None:
        for ex in self._exercises:
            if ex["id"] == self._selected_id:
                return ex
        return None

    def _rebuild(self):
        while self.rows_layout.count():
            item = self.rows_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)

        if not self._program or not self._exercises:
            self.header.setText("No program active")
            self.empty_label.setVisible(True)
            return

        self.empty_label.setVisible(False)
        week_num = self._program_relative_week()
        header_text = self._program["display_name"]
        if week_num is not None:
            header_text += f"  —  Week {week_num}"

        # Is every exercise suggested-for-today done for today?
        today_iso = datetime.now().isoweekday()
        all_done = True
        any_today = False
        for ex in self._exercises:
            suggested = parse_suggested_days(ex.get("suggested_days"))
            is_today = (not suggested) or (today_iso in suggested)
            if is_today:
                any_today = True
                if self._sets_done_today.get(ex["id"], 0) < ex["sets_per_day"]:
                    all_done = False

        if any_today and all_done:
            header_text += "  ✓ complete"
            self.header.setStyleSheet(
                "font-size: 12px; color: #4CAF50; font-weight: bold;"
            )
            self.setStyleSheet(
                "QGroupBox { border: 2px solid #4CAF50; border-radius: 4px; "
                "margin-top: 6px; } "
                "QGroupBox::title { color: #4CAF50; }"
            )
        else:
            self.header.setStyleSheet("font-size: 12px; color: #ccc;")
            self.setStyleSheet("")

        self.header.setText(header_text)

        for ex in self._exercises:
            sets_week = self._sets_done_week.get(ex["id"], 0)
            sets_today = self._sets_done_today.get(ex["id"], 0)
            suggested = parse_suggested_days(ex.get("suggested_days"))
            is_today = (not suggested) or (today_iso in suggested)
            row = ExerciseRow(ex, sets_week, sets_today, is_today)
            row.clicked.connect(self._on_row_clicked)
            self.rows_layout.addWidget(row)

        if self._selected_id is None and self._exercises:
            self._selected_id = self._exercises[0]["id"]

    def _on_row_clicked(self, exercise: dict):
        self._selected_id = exercise["id"]
        self.exercise_selected.emit(exercise)

    def _program_relative_week(self) -> int | None:
        """Return 1-based week since program_started_at, or None."""
        if self._program_started_at is None:
            return None
        delta = datetime.now() - self._program_started_at
        return int(delta.days // 7) + 1
