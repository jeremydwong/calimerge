"""Build the per-machine workout_spec.db from programs.py + WORKOUT_SPECS.

Usage:
    uv run python tests/manual/build_workout_spec_db.py

Idempotent: blows away and rebuilds the tables every run, so editing
WORKOUT_SPECS or programs.py and re-running gives you the latest snapshot.

The runtime GUI does NOT read from this DB yet — this is the migration
prep step the user requested. Once the GUI is migrated to read from this
DB, the Python dicts in programs.py and the per-workout-type branches in
WorkoutPage._apply_plot_mode become redundant.
"""

from __future__ import annotations

import sqlite3

from calimerge.workout_spec_db import (
    populate_workout_spec_db,
    workout_spec_db_path,
)


def _dump(db_path):
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        n_specs = conn.execute("SELECT COUNT(*) FROM workout_specs").fetchone()[0]
        n_progs = conn.execute("SELECT COUNT(*) FROM programs").fetchone()[0]
        n_exs = conn.execute("SELECT COUNT(*) FROM program_exercises").fetchone()[0]
        print(f"workout_specs:     {n_specs}")
        print(f"programs:          {n_progs}")
        print(f"program_exercises: {n_exs}")
        print()

        print("-- programs " + "-" * 50)
        for r in conn.execute("SELECT name, display_name FROM programs ORDER BY id"):
            print(f"  {r['name']:14s}  {r['display_name']}")
        print()

        print("-- workout_specs (analysis routing + thresholds) " + "-" * 16)
        rows = conn.execute(
            "SELECT workout_type, display_name, recording_duration_seconds, "
            "       analysis_function, threshold_label, threshold_default, "
            "       threshold_unit "
            "FROM workout_specs ORDER BY workout_type"
        ).fetchall()
        for r in rows:
            dur = r["recording_duration_seconds"]
            dur_s = f"{dur:.0f}s" if dur is not None else "  -"
            ana = r["analysis_function"] or "(none)"
            if r["threshold_default"] is not None:
                th = f"{r['threshold_label']}={r['threshold_default']:g} {r['threshold_unit'] or ''}"
            else:
                th = "(no threshold)"
            print(f"  {r['workout_type']:30s}  {dur_s:>4s}  {ana:24s}  {th}")
        print()

        print("-- program_exercises (program -> exercise order) " + "-" * 16)
        for prog in ("vivifrail", "calisthenics", "fga"):
            print(f"  [{prog}]")
            ex_rows = conn.execute(
                "SELECT order_index, workout_type, sets_per_day, target_reps, "
                "       target_duration_seconds, days_per_week "
                "FROM program_exercises WHERE program_name = ? "
                "ORDER BY order_index",
                (prog,),
            ).fetchall()
            for r in ex_rows:
                reps = r["target_reps"] if r["target_reps"] is not None else "-"
                dur = (
                    f"{r['target_duration_seconds']:.0f}s"
                    if r["target_duration_seconds"] is not None else "-"
                )
                print(
                    f"    {r['order_index']:>2d}. {r['workout_type']:30s}  "
                    f"{r['sets_per_day']}x{reps} reps OR {dur}, "
                    f"{r['days_per_week']}x/week"
                )
            print()
    finally:
        conn.close()


def main() -> int:
    path = populate_workout_spec_db(overwrite=True)
    print(f"Wrote {path}\n")
    _dump(path)
    print(
        "Note: the runtime GUI does NOT yet read from this DB. The migration\n"
        "(point WorkoutPage at it instead of the hardcoded Python dicts) is\n"
        "deferred per your request.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
