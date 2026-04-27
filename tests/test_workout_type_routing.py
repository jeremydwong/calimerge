"""Tests for two related bugs:

1. `_selected_workout_type()` now consults `_current_program_exercise` first
   (so FGA tasks save under their own workout_type instead of falling through
   to "sit_to_stand").

2. The record-button label clamps "Set N of total" once the weekly target
   is met — previously read as "Set 2 of 1" for assessments, which is
   nonsense.
"""

from __future__ import annotations

import pytest

pytest.importorskip("pytestqt")


@pytest.fixture
def page(qtbot):
    from calimerge.gui.workout_page import WorkoutPage
    from calimerge.gui.state import StateManager

    sm = StateManager()
    p = WorkoutPage(sm)
    qtbot.addWidget(p)
    return p


# ── Routing: program-exercise overrides legacy radios ────────────────


def test_program_exercise_overrides_legacy_radios(page):
    """If the user picked an FGA task from Today's Plan, the saver must use
    that exercise's workout_type, not the radio-button fallback."""
    page.sts_radio.setChecked(True)  # legacy radio still selected
    page._current_program_exercise = {
        "id": 99,
        "workout_type": "fga_horizontal_head_turns",
        "display_name": "Gait with Horizontal Head Turns",
        "sets_per_day": 1,
        "days_per_week": 1,
        "target_reps": 1,
    }

    assert page._selected_workout_type() == "fga_horizontal_head_turns"


def test_falls_back_to_radios_when_no_program_exercise(page):
    """When no program exercise is active, the legacy radios drive routing.
    This is the pre-program-system path — must not regress."""
    page._current_program_exercise = None
    page.sts_radio.setChecked(True)
    assert page._selected_workout_type() == "sit_to_stand"

    page.biceps_radio.setChecked(True)
    assert page._selected_workout_type() == "biceps_curl"


def test_program_exercise_with_empty_workout_type_falls_back(page):
    """Defensive: if a program row somehow has an empty workout_type, fall
    back to the radio buttons rather than recording with an empty string."""
    page._current_program_exercise = {
        "id": 100,
        "workout_type": "",
        "display_name": "Bad Row",
        "sets_per_day": 1,
        "days_per_week": 1,
    }
    page.sts_radio.setChecked(True)
    assert page._selected_workout_type() == "sit_to_stand"


# ── Record-button label: no more "Set 2 of 1" nonsense ───────────────


def test_record_button_label_during_first_set(page, monkeypatch):
    """Before any set is recorded, button should read 'Set 1 of 1' for an
    FGA assessment. (Sanity baseline.)"""
    monkeypatch.setattr(
        "calimerge.config.count_sets_since", lambda *a, **kw: 0
    )
    monkeypatch.setattr(
        "calimerge.config.get_user_by_id",
        lambda *a, **kw: {"id": 1, "program_started_at": None},
    )
    page._current_user_id = 1
    page._current_program_exercise = {
        "id": 99,
        "workout_type": "fga_horizontal_head_turns",
        "display_name": "Gait with Horizontal Head Turns",
        "sets_per_day": 1,
        "days_per_week": 1,
    }
    page._update_record_button_label()
    assert "Set 1 of 1" in page.record_btn.text()


def test_record_button_label_after_target_met_says_complete(page, monkeypatch):
    """The bug: after one FGA recording, button used to read 'Set 2 of 1'.
    Should say 'Complete (1/1)' instead."""
    monkeypatch.setattr(
        "calimerge.config.count_sets_since", lambda *a, **kw: 1
    )
    monkeypatch.setattr(
        "calimerge.config.get_user_by_id",
        lambda *a, **kw: {"id": 1, "program_started_at": None},
    )
    page._current_user_id = 1
    page._current_program_exercise = {
        "id": 99,
        "workout_type": "fga_horizontal_head_turns",
        "display_name": "Gait with Horizontal Head Turns",
        "sets_per_day": 1,
        "days_per_week": 1,
    }
    page._update_record_button_label()
    label = page.record_btn.text()
    assert "Complete" in label
    assert "1/1" in label
    assert "Set 2 of 1" not in label


def test_record_button_label_within_target_uses_set_n_of_total(page, monkeypatch):
    """A multi-set rep program should still show 'Set 3 of 9' style during
    progress. Only the post-target case is special-cased."""
    monkeypatch.setattr(
        "calimerge.config.count_sets_since", lambda *a, **kw: 2
    )
    monkeypatch.setattr(
        "calimerge.config.get_user_by_id",
        lambda *a, **kw: {"id": 1, "program_started_at": None},
    )
    page._current_user_id = 1
    page._current_program_exercise = {
        "id": 5,
        "workout_type": "biceps_curl",
        "display_name": "Biceps Curl",
        "sets_per_day": 3,
        "days_per_week": 3,  # total = 9
        "target_reps": 8,
    }
    page._update_record_button_label()
    assert "Set 3 of 9" in page.record_btn.text()
