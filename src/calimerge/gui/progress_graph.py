"""
Progress graph dialog — per-exercise improvement over the program weeks.

For each exercise in the user's active program, plots the chosen metric
(typically rep_count) over calendar days and overlays a target line at
1.2× the subject's current peak, so it's easy to see whether you're
trending toward your next goal.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QComboBox,
    QDialogButtonBox,
    QDoubleSpinBox,
)

import pyqtgraph as pg


# Primary metric for each exercise type — shown on the Y axis.
EXERCISE_METRIC: dict[str, tuple[str, str]] = {
    "sit_to_stand":     ("rep_count", "Repetitions"),
    "pushup":           ("rep_count", "Repetitions"),
    "pullup":           ("rep_count", "Repetitions"),
    "biceps_curl":      ("rep_count", "Repetitions"),
    "leg_raise":        ("rep_count", "Repetitions"),
    "timed_up_and_go":  ("tug_duration", "Duration (s, lower is better)"),
    "tandem_stance":    ("hold_seconds", "Hold (s)"),
    "stretch":          ("hold_seconds", "Hold (s)"),
}


class ProgressGraphDialog(QDialog):
    """Dialog showing per-exercise progress over time with a target line."""

    def __init__(self, user_id: int, program: dict, exercises: list[dict], parent=None):
        super().__init__(parent)
        self.user_id = user_id
        self.program = program
        self.exercises = exercises

        self.setWindowTitle(f"Progress — {program['display_name']}")
        self.setMinimumSize(800, 500)

        pg.setConfigOption("background", "#1a1a1a")
        pg.setConfigOption("foreground", "#cccccc")

        layout = QVBoxLayout(self)
        layout.setSpacing(6)

        # Controls row
        ctrl_row = QHBoxLayout()
        ctrl_row.addWidget(QLabel("Exercise:"))
        self.exercise_combo = QComboBox()
        for ex in exercises:
            self.exercise_combo.addItem(ex["display_name"], ex)
        self.exercise_combo.currentIndexChanged.connect(self._refresh_plot)
        ctrl_row.addWidget(self.exercise_combo)

        ctrl_row.addSpacing(20)
        ctrl_row.addWidget(QLabel("Target multiplier:"))
        self.target_mult_spin = QDoubleSpinBox()
        self.target_mult_spin.setRange(1.0, 5.0)
        self.target_mult_spin.setSingleStep(0.1)
        self.target_mult_spin.setValue(1.2)
        self.target_mult_spin.valueChanged.connect(self._refresh_plot)
        ctrl_row.addWidget(self.target_mult_spin)

        self.summary_label = QLabel("")
        self.summary_label.setStyleSheet("color: #ccc;")
        ctrl_row.addSpacing(20)
        ctrl_row.addWidget(self.summary_label)
        ctrl_row.addStretch()
        layout.addLayout(ctrl_row)

        # Main plot
        self.plot = pg.PlotWidget()
        self.plot.setLabel("bottom", "Day")
        self.plot.setLabel("left", "Repetitions")
        self.plot.showGrid(x=True, y=True, alpha=0.3)

        # Individual session points
        self.session_scatter = pg.ScatterPlotItem(
            size=10, brush=pg.mkBrush("#42A5F5"),
            pen=pg.mkPen(color="#000"),
        )
        self.plot.addItem(self.session_scatter)

        # Daily best trace
        self.daily_best_trace = self.plot.plot(
            pen=pg.mkPen(color="#4CAF50", width=2),
            name="Daily best",
        )

        # Current peak horizontal line
        self.peak_line = pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen(color="#FFC107", width=1, style=Qt.PenStyle.DashLine),
            label="current peak",
            labelOpts={"position": 0.05, "color": "#FFC107"},
        )
        self.plot.addItem(self.peak_line)

        # Target horizontal line
        self.target_line = pg.InfiniteLine(
            pos=0, angle=0,
            pen=pg.mkPen(color="#EF5350", width=2, style=Qt.PenStyle.DashLine),
            label="target",
            labelOpts={"position": 0.05, "color": "#EF5350"},
        )
        self.plot.addItem(self.target_line)

        layout.addWidget(self.plot, stretch=1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        buttons.rejected.connect(self.close)
        layout.addWidget(buttons)

        self._refresh_plot()

    def _refresh_plot(self):
        exercise = self.exercise_combo.currentData()
        if exercise is None:
            return

        workout_type = exercise["workout_type"]
        metric_name, y_label = EXERCISE_METRIC.get(
            workout_type, ("rep_count", "Repetitions")
        )
        self.plot.setLabel("left", y_label)

        # Query the database for all sessions of this exercise + their metric
        points = self._load_exercise_points(exercise["id"], metric_name)

        if not points:
            self.session_scatter.clear()
            self.daily_best_trace.clear()
            self.peak_line.setPos(0)
            self.target_line.setPos(0)
            self.summary_label.setText("No sessions recorded for this exercise yet.")
            return

        # Normalise timestamps to day-index relative to program start
        program_started = self._get_program_start()
        day_origin = program_started if program_started else points[0][0]

        xs = [(dt - day_origin).total_seconds() / 86400.0 for dt, _ in points]
        ys = [v for _, v in points]

        self.session_scatter.setData(x=xs, y=ys)

        # Daily best trace
        by_day: dict[int, float] = {}
        for x, y in zip(xs, ys):
            day = int(x)
            if day not in by_day or y > by_day[day]:
                by_day[day] = y
        days_sorted = sorted(by_day.keys())
        bests = [by_day[d] for d in days_sorted]
        self.daily_best_trace.setData(days_sorted, bests)

        # Peak + target
        peak = float(max(ys))
        target_mult = self.target_mult_spin.value()
        target = peak * target_mult
        self.peak_line.setPos(peak)
        self.target_line.setPos(target)

        self.summary_label.setText(
            f"Sessions: {len(points)}  |  "
            f"Current peak: {peak:.1f}  |  "
            f"Target: {target:.1f} ({target_mult:.1f}×)"
        )

    def _load_exercise_points(self, program_exercise_id: int,
                               metric_name: str) -> list[tuple[datetime, float]]:
        """Return [(session_created_at, metric_value), ...] for this exercise."""
        from ..config import DEFAULT_WORKOUTS_DB
        import sqlite3
        conn = sqlite3.connect(str(DEFAULT_WORKOUTS_DB))
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT s.created_at, r.metric_value "
            "FROM sessions s "
            "JOIN session_results r ON r.session_id = s.id "
            "WHERE s.user_id = ? AND s.program_exercise_id = ? "
            "  AND r.metric_name = ? "
            "ORDER BY s.created_at",
            (self.user_id, program_exercise_id, metric_name),
        ).fetchall()
        conn.close()

        out: list[tuple[datetime, float]] = []
        for r in rows:
            try:
                ts = datetime.fromisoformat(str(r["created_at"]).replace(" ", "T"))
                out.append((ts, float(r["metric_value"])))
            except Exception:
                continue
        return out

    def _get_program_start(self) -> datetime | None:
        from ..config import get_user_by_id
        try:
            user = get_user_by_id(self.user_id)
            started = user.get("program_started_at") if user else None
            if started:
                return datetime.fromisoformat(str(started).replace(" ", "T"))
        except Exception:
            pass
        return None
