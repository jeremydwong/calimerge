"""
Tests for the Functional Gait Assessment (FGA) program template.

Verifies:
  - FGA seeds into a fresh workouts.db alongside existing programs.
  - All 10 standard FGA tasks are present in canonical order.
  - Each task has a unique workout_type, a non-empty display_name, and a
    description carrying the test instructions.
  - Existing programs (Vivifrail, Calisthenics) still seed correctly.

These tests use a temp DB so they never touch the user's real workouts.db.
"""

from __future__ import annotations

import pytest

from calimerge import programs as programs_module
from calimerge.config import (
    get_program_exercises,
    init_workouts_db,
    list_programs,
)


# Canonical FGA task order from Wrisley et al. 2004.
EXPECTED_FGA_ORDER = [
    "fga_level_gait",
    "fga_speed_change",
    "fga_horizontal_head_turns",
    "fga_vertical_head_turns",
    "fga_pivot_turn",
    "fga_step_over_obstacle",
    "fga_narrow_base",
    "fga_eyes_closed",
    "fga_backwards",
    "fga_steps",
]


@pytest.fixture
def fresh_workouts_db(temp_dir):
    """Initialize a fresh workouts.db in a temp dir and return its path."""
    db_path = temp_dir / "workouts.db"
    init_workouts_db(db_path)
    return db_path


def _get_program(programs: list[dict], name: str) -> dict:
    matches = [p for p in programs if p["name"] == name]
    assert len(matches) == 1, f"Expected exactly one program named {name!r}"
    return matches[0]


class TestFGADefinition:
    """Tests on the in-memory FGA dict (no DB required)."""

    def test_fga_in_default_programs(self):
        names = [p["name"] for p in programs_module.DEFAULT_PROGRAMS]
        assert "fga" in names

    def test_existing_programs_still_present(self):
        names = [p["name"] for p in programs_module.DEFAULT_PROGRAMS]
        # Sanity: don't accidentally drop the existing programs.
        assert "vivifrail" in names
        assert "calisthenics" in names

    def test_fga_has_ten_tasks(self):
        fga = _get_program(programs_module.DEFAULT_PROGRAMS, "fga")
        assert len(fga["exercises"]) == 10

    def test_fga_workout_types_are_unique(self):
        fga = _get_program(programs_module.DEFAULT_PROGRAMS, "fga")
        types = [ex["workout_type"] for ex in fga["exercises"]]
        assert len(set(types)) == len(types)


class TestFGASeeding:
    """Tests that FGA round-trips through the workouts.db seeder."""

    def test_fga_program_seeded(self, fresh_workouts_db):
        all_progs = list_programs(fresh_workouts_db)
        names = [p["name"] for p in all_progs]
        assert "fga" in names

    def test_existing_programs_still_seed(self, fresh_workouts_db):
        all_progs = list_programs(fresh_workouts_db)
        names = [p["name"] for p in all_progs]
        # FGA must not break the prior seeding.
        assert "vivifrail" in names
        assert "calisthenics" in names

    def test_fga_has_ten_exercises_in_db(self, fresh_workouts_db):
        all_progs = list_programs(fresh_workouts_db)
        fga = _get_program(all_progs, "fga")
        exs = get_program_exercises(fga["id"], fresh_workouts_db)
        assert len(exs) == 10

    def test_fga_tasks_in_canonical_order(self, fresh_workouts_db):
        all_progs = list_programs(fresh_workouts_db)
        fga = _get_program(all_progs, "fga")
        exs = get_program_exercises(fga["id"], fresh_workouts_db)
        actual_order = [e["workout_type"] for e in exs]
        assert actual_order == EXPECTED_FGA_ORDER

    def test_each_task_has_unique_workout_type(self, fresh_workouts_db):
        all_progs = list_programs(fresh_workouts_db)
        fga = _get_program(all_progs, "fga")
        exs = get_program_exercises(fga["id"], fresh_workouts_db)
        types = [e["workout_type"] for e in exs]
        assert len(set(types)) == len(types)

    def test_each_task_has_nonempty_display_name(self, fresh_workouts_db):
        all_progs = list_programs(fresh_workouts_db)
        fga = _get_program(all_progs, "fga")
        exs = get_program_exercises(fga["id"], fresh_workouts_db)
        for e in exs:
            assert e["display_name"]
            assert isinstance(e["display_name"], str)
            assert e["display_name"].strip() != ""

    def test_each_task_has_description(self, fresh_workouts_db):
        """Per the FGA spec, each task carries its instructions in description."""
        all_progs = list_programs(fresh_workouts_db)
        fga = _get_program(all_progs, "fga")
        exs = get_program_exercises(fga["id"], fresh_workouts_db)
        for e in exs:
            assert "description" in e, "program_exercises row missing description col"
            assert e["description"], (
                f"FGA task {e['workout_type']} has empty description"
            )
            assert len(e["description"]) > 20  # non-trivial instruction text

    def test_assessment_shape_is_one_set_one_rep(self, fresh_workouts_db):
        """FGA tasks are performed once each, not as workout reps."""
        all_progs = list_programs(fresh_workouts_db)
        fga = _get_program(all_progs, "fga")
        exs = get_program_exercises(fga["id"], fresh_workouts_db)
        for e in exs:
            assert e["sets_per_day"] == 1
            assert e["target_reps"] == 1

    def test_seeding_is_idempotent(self, fresh_workouts_db):
        """Re-running init must not duplicate FGA rows."""
        # Simulate a second app launch.
        init_workouts_db(fresh_workouts_db)

        all_progs = list_programs(fresh_workouts_db)
        # Exactly one fga program row.
        fga_rows = [p for p in all_progs if p["name"] == "fga"]
        assert len(fga_rows) == 1

        exs = get_program_exercises(fga_rows[0]["id"], fresh_workouts_db)
        assert len(exs) == 10

    def test_existing_program_exercises_preserved(self, fresh_workouts_db):
        """Vivifrail's exercises must still seed correctly alongside FGA."""
        all_progs = list_programs(fresh_workouts_db)
        vivi = _get_program(all_progs, "vivifrail")
        vivi_exs = get_program_exercises(vivi["id"], fresh_workouts_db)
        # Vivifrail has 4 exercises in programs.py.
        assert len(vivi_exs) == 4
        types = [e["workout_type"] for e in vivi_exs]
        assert "sit_to_stand" in types
        assert "tandem_stance" in types
