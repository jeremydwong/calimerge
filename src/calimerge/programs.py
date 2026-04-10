"""
Default workout program definitions.

These templates are seeded into the workouts database on first run. Users
can follow one at a time; their selection is stored as active_program_id
on the users table. Each program has an ordered list of exercises with
sets-per-day, days-per-week, suggested weekdays, and break durations.
"""

from __future__ import annotations


# ISO weekday numbers: Mon=1, Tue=2, Wed=3, Thu=4, Fri=5, Sat=6, Sun=7
VIVIFRAIL = {
    "name": "vivifrail",
    "display_name": "Vivifrail (Older Adults)",
    "description": (
        "Multicomponent exercise program designed to prevent and reverse "
        "frailty in older adults. Combines resistance, balance, and flexibility."
    ),
    "exercises": [
        {
            "workout_type": "sit_to_stand",
            "display_name": "Sit-to-Stand",
            "sets_per_day": 3,
            "target_reps": 10,
            "target_duration_seconds": None,
            "days_per_week": 3,
            "suggested_days": "1,3,5",
            "break_seconds": 60,
            "order_index": 0,
        },
        {
            "workout_type": "leg_raise",
            "display_name": "Leg Raises",
            "sets_per_day": 3,
            "target_reps": 10,
            "target_duration_seconds": None,
            "days_per_week": 3,
            "suggested_days": "1,3,5",
            "break_seconds": 60,
            "order_index": 1,
        },
        {
            "workout_type": "tandem_stance",
            "display_name": "Tandem Stance (Feet in Line)",
            "sets_per_day": 3,
            "target_reps": None,
            "target_duration_seconds": 20.0,
            "days_per_week": 3,
            "suggested_days": "1,3,5",
            "break_seconds": 30,
            "order_index": 2,
        },
        {
            "workout_type": "stretch",
            "display_name": "Chair Stretches",
            "sets_per_day": 3,
            "target_reps": None,
            "target_duration_seconds": 30.0,
            "days_per_week": 3,
            "suggested_days": "1,3,5",
            "break_seconds": 15,
            "order_index": 3,
        },
    ],
}


CALISTHENICS = {
    "name": "calisthenics",
    "display_name": "Calisthenics",
    "description": (
        "Bodyweight strength training with pushups, pullups, and biceps curls. "
        "Split across the week to allow upper-body recovery."
    ),
    "exercises": [
        {
            "workout_type": "pushup",
            "display_name": "Pushups",
            "sets_per_day": 3,
            "target_reps": 15,
            "target_duration_seconds": None,
            "days_per_week": 3,
            "suggested_days": "1,3,5",
            "break_seconds": 90,
            "order_index": 0,
        },
        {
            "workout_type": "pullup",
            "display_name": "Pullups",
            "sets_per_day": 3,
            "target_reps": 8,
            "target_duration_seconds": None,
            "days_per_week": 3,
            "suggested_days": "1,3,5",
            "break_seconds": 120,
            "order_index": 1,
        },
        {
            "workout_type": "biceps_curl",
            "display_name": "Biceps Curls",
            "sets_per_day": 3,
            "target_reps": 12,
            "target_duration_seconds": None,
            "days_per_week": 2,
            "suggested_days": "2,4",
            "break_seconds": 60,
            "order_index": 2,
        },
    ],
}


DEFAULT_PROGRAMS = [VIVIFRAIL, CALISTHENICS]


def parse_suggested_days(days_csv: str | None) -> set[int]:
    """Parse a '1,3,5' string into a set of ISO weekday ints. Empty → all days."""
    if not days_csv:
        return set()
    out: set[int] = set()
    for tok in days_csv.split(","):
        tok = tok.strip()
        if tok.isdigit():
            out.add(int(tok))
    return out
