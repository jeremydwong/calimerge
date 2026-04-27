"""Tests for the Today's Plan widget layout — specifically the "name and
reps shouldn't be vertically stacked" fix and the assessment-shape special
case (FGA's `1 reps × 1 sets` rendering).

These need a QApplication, but pytest-qt's `qtbot` fixture handles that.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pytestqt")  # skip if pytest-qt isn't installed


def _ex(display_name: str, sets_per_day: int = 1, target_reps=None,
        target_duration_seconds=None, days_per_week: int = 1,
        suggested_days: str | None = None, exercise_id: int = 1) -> dict:
    return {
        "id": exercise_id,
        "display_name": display_name,
        "sets_per_day": sets_per_day,
        "target_reps": target_reps,
        "target_duration_seconds": target_duration_seconds,
        "days_per_week": days_per_week,
        "suggested_days": suggested_days,
        "workout_type": display_name.lower().replace(" ", "_"),
    }


def _row_target_label_text(row) -> str | None:
    """Return the text of any rep/duration target label in the row, else None.

    The fix replaces a vertical name-then-target stack with a single horizontal
    row, so the target label (when present) is a sibling of the name label.
    """
    from PySide6.QtWidgets import QLabel
    labels = [
        w.text() for w in row.findChildren(QLabel)
        if "reps" in w.text() or " sets" in w.text() or w.text().endswith("s × 1 sets")
    ]
    return labels[0] if labels else None


def test_assessment_shape_suppresses_target_string(qtbot):
    """FGA tasks have sets_per_day=1, target_reps=1 — that string read as
    awkward boilerplate; we suppress it."""
    from calimerge.gui.todays_plan import ExerciseRow

    row = ExerciseRow(_ex("Gait on Level Surface", sets_per_day=1, target_reps=1),
                      sets_done_week=0, sets_done_today=0, is_today=True)
    qtbot.addWidget(row)
    assert _row_target_label_text(row) is None


def test_rep_program_shows_target_string(qtbot):
    """A real rep-based program (e.g. push-ups: 3 sets × 8 reps) should still
    display the target."""
    from calimerge.gui.todays_plan import ExerciseRow

    row = ExerciseRow(_ex("Push-ups", sets_per_day=3, target_reps=8),
                      sets_done_week=0, sets_done_today=0, is_today=True)
    qtbot.addWidget(row)
    text = _row_target_label_text(row)
    assert text is not None
    assert "8 reps" in text
    assert "3 sets" in text


def test_duration_program_shows_target_string(qtbot):
    """Duration-based exercise: e.g. plank for 30 s × 3 sets."""
    from calimerge.gui.todays_plan import ExerciseRow

    row = ExerciseRow(_ex("Plank", sets_per_day=3, target_duration_seconds=30),
                      sets_done_week=0, sets_done_today=0, is_today=True)
    qtbot.addWidget(row)
    text = _row_target_label_text(row)
    assert text is not None
    assert "30s" in text
    assert "3 sets" in text


def test_row_uses_horizontal_layout_top_level(qtbot):
    """The fix is to drop the name+target QVBoxLayout and put both on the
    top-level QHBoxLayout. Verify the top-level layout has no QVBoxLayout
    child stacking name+target."""
    from PySide6.QtWidgets import QHBoxLayout, QVBoxLayout
    from calimerge.gui.todays_plan import ExerciseRow

    row = ExerciseRow(_ex("Push-ups", sets_per_day=3, target_reps=8),
                      sets_done_week=0, sets_done_today=0, is_today=True)
    qtbot.addWidget(row)
    top = row.layout()
    assert isinstance(top, QHBoxLayout)
    # No nested QVBoxLayout (which is what stacked name + target before).
    for i in range(top.count()):
        item = top.itemAt(i)
        assert not isinstance(item.layout(), QVBoxLayout), (
            "name+reps should not be stacked vertically"
        )
