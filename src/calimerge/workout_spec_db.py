"""Workout spec database — single source of truth for exercises.

Today the workout/exercise/analysis routing is split across three places:

  - ``calimerge.programs`` (Python dicts seeded into workouts.db at first run)
  - ``WorkoutPage._apply_plot_mode`` (per-workout-type threshold spin
    widget config: range, default value, label, decimals)
  - ``WorkoutPage._run_*_analysis`` (per-workout-type call into the
    matching analyser in ``calimerge.analysis.*``)

This module migrates that to a sqlite ``workout_spec.db``. The runtime
GUI is NOT touched yet — this is just the storage + the populator.

Schema (kept narrow on purpose — easier to extend later than to drop
columns):

    programs(
        id INTEGER PRIMARY KEY,
        name TEXT UNIQUE,
        display_name TEXT,
        description TEXT
    )

    workout_specs(
        workout_type TEXT PRIMARY KEY,        -- e.g. "sit_to_stand"
        display_name TEXT,
        description TEXT,
        recording_duration_seconds REAL,      -- override the duration spin
        analysis_module TEXT,                 -- "calimerge.analysis.sit_to_stand"
        analysis_function TEXT,               -- "analyze_sit_to_stand"
        threshold_label TEXT,                 -- shown next to the spin
        threshold_unit TEXT,                  -- "m", "deg", "m/s", ...
        threshold_default REAL,
        threshold_min REAL,
        threshold_max REAL,
        threshold_step REAL,
        threshold_decimals INTEGER
    )

    program_exercises(
        program_name TEXT NOT NULL REFERENCES programs(name),
        workout_type TEXT NOT NULL REFERENCES workout_specs(workout_type),
        order_index INTEGER NOT NULL,
        sets_per_day INTEGER,
        target_reps INTEGER,
        target_duration_seconds REAL,
        days_per_week INTEGER,
        suggested_days TEXT,
        break_seconds INTEGER,
        per_exercise_description TEXT,        -- FGA carries per-task instructions here
        PRIMARY KEY (program_name, workout_type)
    )

Migration is intentionally NOT wired up yet. To smoke-test the populator,
run::

    uv run python tests/manual/build_workout_spec_db.py

"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from .config import data_dir
from .programs import DEFAULT_PROGRAMS


def workout_spec_db_path() -> Path:
    """Single-file location for the spec DB.

    Lives alongside ``view_transforms.db`` under ``<app_data>/models/``
    rather than ``workouts.db`` because (a) it is content-addressable
    factory data, not user-generated, and (b) workouts.db is in the
    user's project tree where it gets backed up — this DB is regenerable.
    """
    return data_dir() / "models" / "workout_spec.db"


# ── Per-workout-type analyser + threshold metadata ─────────────────────
#
# Mined from WorkoutPage._apply_plot_mode + the _run_*_analysis methods
# on 2026-04-28. When the GUI eventually reads from workout_spec.db at
# runtime, this table goes away — until then, this is the canonical
# source for those values.

WORKOUT_SPECS: dict[str, dict] = {
    "sit_to_stand": {
        "display_name": "Sit-to-Stand",
        "description": (
            "Stand up and sit down repeatedly. Hip height crosses the "
            "seated threshold once per rep."
        ),
        "recording_duration_seconds": 30.0,
        "analysis_module": "calimerge.analysis.sit_to_stand",
        "analysis_function": "analyze_sit_to_stand",
        "threshold_label": "Seated threshold",
        "threshold_unit": "m",
        "threshold_default": 0.65,
        "threshold_min": 0.10,
        "threshold_max": 2.00,
        "threshold_step": 0.01,
        "threshold_decimals": 3,
    },
    "biceps_curl": {
        "display_name": "Biceps Curls",
        "description": (
            "Elbow angle oscillates between flexed (small angle) and "
            "extended (~150 deg). One rep = curl + return."
        ),
        "recording_duration_seconds": 30.0,
        "analysis_module": "calimerge.analysis.biceps_curl",
        "analysis_function": "analyze_biceps_curl",
        "threshold_label": "Extended angle",
        "threshold_unit": "deg",
        "threshold_default": 150.0,
        "threshold_min": 0.0,
        "threshold_max": 180.0,
        "threshold_step": 1.0,
        "threshold_decimals": 1,
    },
    "pushup": {
        "display_name": "Pushups",
        "description": (
            "Shoulder height oscillates between top (~30 cm) and bottom. "
            "One rep = up + down."
        ),
        "recording_duration_seconds": 30.0,
        "analysis_module": "calimerge.analysis.pushup",
        "analysis_function": "analyze_pushup",
        "threshold_label": "Top threshold",
        "threshold_unit": "m",
        "threshold_default": 0.30,
        "threshold_min": 0.0,
        "threshold_max": 2.0,
        "threshold_step": 0.01,
        "threshold_decimals": 3,
    },
    "pullup": {
        "display_name": "Pullups",
        "description": (
            "Head height oscillates around bar level. One rep = "
            "head-above-bar then back below."
        ),
        "recording_duration_seconds": 30.0,
        "analysis_module": "calimerge.analysis.pullup",
        "analysis_function": "analyze_pullup",
        "threshold_label": "Bar height",
        "threshold_unit": "m",
        "threshold_default": 1.80,
        "threshold_min": 0.5,
        "threshold_max": 3.0,
        "threshold_step": 0.01,
        "threshold_decimals": 3,
    },
    "leg_raise": {
        "display_name": "Leg Raises",
        "description": "Knee Z crosses the lift threshold once per rep.",
        "recording_duration_seconds": 30.0,
        "analysis_module": "calimerge.analysis.leg_raise",
        "analysis_function": "analyze_leg_raise",
        "threshold_label": "Lift threshold",
        "threshold_unit": "m",
        "threshold_default": 0.60,
        "threshold_min": 0.0,
        "threshold_max": 2.0,
        "threshold_step": 0.01,
        "threshold_decimals": 3,
    },
    "tandem_stance": {
        "display_name": "Tandem Stance",
        "description": (
            "Heel-to-toe stance. Score = fraction of time horizontal "
            "sway stays below the threshold."
        ),
        "recording_duration_seconds": 20.0,
        "analysis_module": "calimerge.analysis.tandem_stance",
        "analysis_function": "analyze_tandem_stance",
        "threshold_label": "Sway threshold",
        "threshold_unit": "m",
        "threshold_default": 0.05,
        "threshold_min": 0.005,
        "threshold_max": 0.5,
        "threshold_step": 0.005,
        "threshold_decimals": 3,
    },
    "stretch": {
        "display_name": "Chair Stretches",
        "description": "Held position; no rep counting, no threshold.",
        "recording_duration_seconds": 30.0,
        "analysis_module": "calimerge.analysis.stretch",
        "analysis_function": "analyze_stretch",
        "threshold_label": None,
        "threshold_unit": None,
        "threshold_default": None,
        "threshold_min": None,
        "threshold_max": None,
        "threshold_step": None,
        "threshold_decimals": None,
    },
    "timed_up_and_go": {
        "display_name": "Timed Up and Go",
        "description": (
            "Stand up, walk 3 m, turn, walk back, sit. Total time."
        ),
        "recording_duration_seconds": 30.0,
        "analysis_module": "calimerge.analysis.tug",
        "analysis_function": "analyze_tug",
        "threshold_label": "Seated threshold",
        "threshold_unit": "m",
        "threshold_default": 0.65,
        "threshold_min": 0.10,
        "threshold_max": 2.00,
        "threshold_step": 0.01,
        "threshold_decimals": 3,
    },
    "spinner": {
        "display_name": "Spinner",
        "description": (
            "Single-trial head-motion tracker. The subject does whatever "
            "the trial is (e.g. spin in place, head-turn drill); we "
            "summarise total path length, peak/mean head speed, and the "
            "x/y/z bounding-box of the head over the trial."
        ),
        "recording_duration_seconds": 15.0,
        "analysis_module": "calimerge.analysis.spinner",
        "analysis_function": "analyze_spinner",
        "threshold_label": None,
        "threshold_unit": None,
        "threshold_default": None,
        "threshold_min": None,
        "threshold_max": None,
        "threshold_step": None,
        "threshold_decimals": None,
    },
    "hand_squeeze": {
        "display_name": "Hand Squeeze",
        "description": (
            "Open / close fist. Inter-fingertip distance crosses the "
            "closed threshold once per rep."
        ),
        "recording_duration_seconds": 30.0,
        "analysis_module": "calimerge.analysis.hand_squeeze",
        "analysis_function": "analyze_hand_squeeze",
        "threshold_label": "Closed threshold",
        "threshold_unit": "m",
        "threshold_default": 0.05,
        "threshold_min": 0.005,
        "threshold_max": 0.5,
        "threshold_step": 0.005,
        "threshold_decimals": 3,
    },
    # ── FGA tasks ──
    # No analyser exists yet — the GUI currently routes these through
    # the sit-to-stand path which was producing nonsense (and crashed
    # show_results until the None-safe fix). Recording duration is set
    # per task to roughly cover a 6 m walk at normal pace + buffer.
    "fga_level_gait": {
        "display_name": "1. Gait on Level Surface",
        "description": "Walk 6 m at normal speed.",
        "recording_duration_seconds": 12.0,
        "analysis_module": None,
        "analysis_function": None,
        "threshold_label": None,
        "threshold_unit": None,
        "threshold_default": None,
        "threshold_min": None,
        "threshold_max": None,
        "threshold_step": None,
        "threshold_decimals": None,
    },
    "fga_speed_change": {
        "display_name": "2. Change in Gait Speed",
        "description": "Walk 1.5 m at normal pace, then fast, then slow.",
        "recording_duration_seconds": 15.0,
        "analysis_module": None,
        "analysis_function": None,
        "threshold_label": None, "threshold_unit": None,
        "threshold_default": None, "threshold_min": None,
        "threshold_max": None, "threshold_step": None,
        "threshold_decimals": None,
    },
    "fga_horizontal_head_turns": {
        "display_name": "3. Gait with Horizontal Head Turns",
        "description": "Walk 6 m, turning head right then left every 3 steps.",
        "recording_duration_seconds": 15.0,
        "analysis_module": None,
        "analysis_function": None,
        "threshold_label": None, "threshold_unit": None,
        "threshold_default": None, "threshold_min": None,
        "threshold_max": None, "threshold_step": None,
        "threshold_decimals": None,
    },
    "fga_vertical_head_turns": {
        "display_name": "4. Gait with Vertical Head Turns",
        "description": "Walk 6 m, tilting head up then down every 3 steps.",
        "recording_duration_seconds": 15.0,
        "analysis_module": None,
        "analysis_function": None,
        "threshold_label": None, "threshold_unit": None,
        "threshold_default": None, "threshold_min": None,
        "threshold_max": None, "threshold_step": None,
        "threshold_decimals": None,
    },
    "fga_pivot_turn": {
        "display_name": "5. Gait and Pivot Turn",
        "description": "On 'turn and stop', pivot 180 deg quickly.",
        "recording_duration_seconds": 12.0,
        "analysis_module": None,
        "analysis_function": None,
        "threshold_label": None, "threshold_unit": None,
        "threshold_default": None, "threshold_min": None,
        "threshold_max": None, "threshold_step": None,
        "threshold_decimals": None,
    },
    "fga_step_over_obstacle": {
        "display_name": "6. Step Over Obstacle",
        "description": "Step over a ~22 cm obstacle while walking normally.",
        "recording_duration_seconds": 12.0,
        "analysis_module": None,
        "analysis_function": None,
        "threshold_label": None, "threshold_unit": None,
        "threshold_default": None, "threshold_min": None,
        "threshold_max": None, "threshold_step": None,
        "threshold_decimals": None,
    },
    "fga_narrow_base": {
        "display_name": "7. Gait with Narrow Base of Support",
        "description": "Walk heel-to-toe, max 10 steps over 3.6 m.",
        "recording_duration_seconds": 15.0,
        "analysis_module": None,
        "analysis_function": None,
        "threshold_label": None, "threshold_unit": None,
        "threshold_default": None, "threshold_min": None,
        "threshold_max": None, "threshold_step": None,
        "threshold_decimals": None,
    },
    "fga_eyes_closed": {
        "display_name": "8. Gait with Eyes Closed",
        "description": "Walk 6 m at normal speed with eyes closed.",
        "recording_duration_seconds": 15.0,
        "analysis_module": None,
        "analysis_function": None,
        "threshold_label": None, "threshold_unit": None,
        "threshold_default": None, "threshold_min": None,
        "threshold_max": None, "threshold_step": None,
        "threshold_decimals": None,
    },
    "fga_backwards": {
        "display_name": "9. Ambulating Backwards",
        "description": "Walk backward 6 m at normal speed.",
        "recording_duration_seconds": 15.0,
        "analysis_module": None,
        "analysis_function": None,
        "threshold_label": None, "threshold_unit": None,
        "threshold_default": None, "threshold_min": None,
        "threshold_max": None, "threshold_step": None,
        "threshold_decimals": None,
    },
    "fga_steps": {
        "display_name": "10. Steps",
        "description": "Walk up the stairs, turn, walk down.",
        "recording_duration_seconds": 30.0,
        "analysis_module": None,
        "analysis_function": None,
        "threshold_label": None, "threshold_unit": None,
        "threshold_default": None, "threshold_min": None,
        "threshold_max": None, "threshold_step": None,
        "threshold_decimals": None,
    },
}


# ── Schema + populator ─────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS programs (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,
    display_name TEXT NOT NULL,
    description TEXT
);

CREATE TABLE IF NOT EXISTS workout_specs (
    workout_type TEXT PRIMARY KEY,
    display_name TEXT NOT NULL,
    description TEXT,
    recording_duration_seconds REAL,
    analysis_module TEXT,
    analysis_function TEXT,
    threshold_label TEXT,
    threshold_unit TEXT,
    threshold_default REAL,
    threshold_min REAL,
    threshold_max REAL,
    threshold_step REAL,
    threshold_decimals INTEGER
);

CREATE TABLE IF NOT EXISTS program_exercises (
    program_name TEXT NOT NULL,
    workout_type TEXT NOT NULL,
    order_index INTEGER NOT NULL,
    sets_per_day INTEGER,
    target_reps INTEGER,
    target_duration_seconds REAL,
    days_per_week INTEGER,
    suggested_days TEXT,
    break_seconds INTEGER,
    per_exercise_description TEXT,
    PRIMARY KEY (program_name, workout_type),
    FOREIGN KEY (program_name) REFERENCES programs(name),
    FOREIGN KEY (workout_type) REFERENCES workout_specs(workout_type)
);
"""


def init_workout_spec_db(db_path: Path | None = None) -> None:
    if db_path is None:
        db_path = workout_spec_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def populate_workout_spec_db(
    db_path: Path | None = None,
    overwrite: bool = True,
) -> Path:
    """Populate workout_spec.db from programs.py + WORKOUT_SPECS.

    With ``overwrite=True`` (default) any pre-existing rows are deleted
    first — this is a content-addressable factory DB that should always
    match the current source. With ``overwrite=False`` the populate is
    additive (insert-or-ignore) so manual edits survive.
    """
    if db_path is None:
        db_path = workout_spec_db_path()
    init_workout_spec_db(db_path)

    conn = sqlite3.connect(str(db_path))
    try:
        if overwrite:
            conn.execute("DELETE FROM program_exercises")
            conn.execute("DELETE FROM programs")
            conn.execute("DELETE FROM workout_specs")

        # 1. workout_specs.
        for wt, spec in WORKOUT_SPECS.items():
            conn.execute(
                """
                INSERT OR REPLACE INTO workout_specs (
                    workout_type, display_name, description,
                    recording_duration_seconds,
                    analysis_module, analysis_function,
                    threshold_label, threshold_unit,
                    threshold_default, threshold_min, threshold_max,
                    threshold_step, threshold_decimals
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    wt,
                    spec["display_name"],
                    spec.get("description"),
                    spec.get("recording_duration_seconds"),
                    spec.get("analysis_module"),
                    spec.get("analysis_function"),
                    spec.get("threshold_label"),
                    spec.get("threshold_unit"),
                    spec.get("threshold_default"),
                    spec.get("threshold_min"),
                    spec.get("threshold_max"),
                    spec.get("threshold_step"),
                    spec.get("threshold_decimals"),
                ),
            )

        # 2. programs + 3. program_exercises.
        for prog in DEFAULT_PROGRAMS:
            conn.execute(
                "INSERT OR REPLACE INTO programs (name, display_name, description) "
                "VALUES (?, ?, ?)",
                (prog["name"], prog["display_name"], prog.get("description", "")),
            )
            for ex in prog.get("exercises", []):
                wt = ex["workout_type"]
                if wt not in WORKOUT_SPECS:
                    # Caller will see this in their dump — better to surface
                    # missing-spec issues loudly than to silently insert a
                    # row that points nowhere.
                    raise KeyError(
                        f"workout_type {wt!r} from program {prog['name']!r} "
                        f"has no entry in WORKOUT_SPECS"
                    )
                conn.execute(
                    """
                    INSERT OR REPLACE INTO program_exercises (
                        program_name, workout_type, order_index,
                        sets_per_day, target_reps, target_duration_seconds,
                        days_per_week, suggested_days, break_seconds,
                        per_exercise_description
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        prog["name"],
                        wt,
                        int(ex.get("order_index", 0)),
                        int(ex["sets_per_day"]),
                        ex.get("target_reps"),
                        ex.get("target_duration_seconds"),
                        int(ex["days_per_week"]),
                        ex.get("suggested_days"),
                        int(ex.get("break_seconds", 60)),
                        ex.get("description"),
                    ),
                )
        conn.commit()
    finally:
        conn.close()
    return db_path


# ── Read helpers (handy for the GUI migration later) ───────────────────


def load_workout_spec(workout_type: str, db_path: Path | None = None) -> dict | None:
    if db_path is None:
        db_path = workout_spec_db_path()
    if not db_path.exists():
        return None
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT * FROM workout_specs WHERE workout_type = ?",
            (workout_type,),
        ).fetchone()
    finally:
        conn.close()
    return dict(row) if row else None


def load_program_exercises(
    program_name: str,
    db_path: Path | None = None,
) -> list[dict]:
    if db_path is None:
        db_path = workout_spec_db_path()
    if not db_path.exists():
        return []
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT pe.*, ws.display_name AS spec_display_name, "
            "       ws.recording_duration_seconds, ws.analysis_module, "
            "       ws.analysis_function, ws.threshold_default, "
            "       ws.threshold_label, ws.threshold_unit "
            "FROM program_exercises pe "
            "JOIN workout_specs ws ON ws.workout_type = pe.workout_type "
            "WHERE pe.program_name = ? "
            "ORDER BY pe.order_index",
            (program_name,),
        ).fetchall()
    finally:
        conn.close()
    return [dict(r) for r in rows]
