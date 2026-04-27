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


# ----------------------------------------------------------------------------
# Functional Gait Assessment (FGA)
# ----------------------------------------------------------------------------
# 10-item clinical gait assessment (Wrisley et al., Phys Ther 2004;84:906-918).
# Walkway: 6 m (20 ft) long x 30.48 cm (12 in) wide. Each item is scored 0-3
# (0=severe, 3=normal); total /30. Community-dwelling cutoff for fall risk:
# <=22/30 (Wrisley & Kumar 2010).
#
# Unlike Vivifrail / Calisthenics, FGA is an ASSESSMENT — each task is
# performed once, not a workout you do reps of. We model that by setting
# sets_per_day=1, target_reps=1, days_per_week=1. The per-task `description`
# field carries the test instructions so the operator running the assessment
# can see them in-app.
FGA = {
    "name": "fga",
    "display_name": "Functional Gait Assessment (FGA)",
    "description": (
        "Wrisley 10-item clinical gait assessment. Each task scored 0-3; "
        "total /30. Cutoff <=22/30 indicates fall risk in community-dwelling "
        "older adults. Walkway: 6 m long, 30.48 cm wide."
    ),
    "exercises": [
        {
            "workout_type": "fga_level_gait",
            "display_name": "1. Gait on Level Surface",
            "description": (
                "Walk 6 m at normal speed. Score 3: time <5.5 s, no aid, "
                "deviation <=15.24 cm, no imbalance, normal pattern."
            ),
            "sets_per_day": 1,
            "target_reps": 1,
            "target_duration_seconds": None,
            "days_per_week": 1,
            "suggested_days": None,
            "break_seconds": 30,
            "order_index": 1,
        },
        {
            "workout_type": "fga_speed_change",
            "display_name": "2. Change in Gait Speed",
            "description": (
                "Walk 1.5 m at normal pace, then 1.5 m fast on command, then "
                "1.5 m slow on command. Score 3: smooth changes, significant "
                "fast/slow difference, deviation <=15.24 cm."
            ),
            "sets_per_day": 1,
            "target_reps": 1,
            "target_duration_seconds": None,
            "days_per_week": 1,
            "suggested_days": None,
            "break_seconds": 30,
            "order_index": 2,
        },
        {
            "workout_type": "fga_horizontal_head_turns",
            "display_name": "3. Gait with Horizontal Head Turns",
            "description": (
                "Walk 6 m, turning head right then left every 3 steps "
                "(2 reps each direction). Score 3: smooth turns, no change "
                "in gait, deviation <=15.24 cm."
            ),
            "sets_per_day": 1,
            "target_reps": 1,
            "target_duration_seconds": None,
            "days_per_week": 1,
            "suggested_days": None,
            "break_seconds": 30,
            "order_index": 3,
        },
        {
            "workout_type": "fga_vertical_head_turns",
            "display_name": "4. Gait with Vertical Head Turns",
            "description": (
                "Walk 6 m, tilting head up then down every 3 steps "
                "(2 reps each direction). Score 3: no change in gait, "
                "deviation <=15.24 cm."
            ),
            "sets_per_day": 1,
            "target_reps": 1,
            "target_duration_seconds": None,
            "days_per_week": 1,
            "suggested_days": None,
            "break_seconds": 30,
            "order_index": 4,
        },
        {
            "workout_type": "fga_pivot_turn",
            "display_name": "5. Gait and Pivot Turn",
            "description": (
                "On 'turn and stop', pivot 180 deg quickly to face the "
                "opposite direction and stop. Score 3: pivot safely "
                "within 3 s, stops quickly, no loss of balance."
            ),
            "sets_per_day": 1,
            "target_reps": 1,
            "target_duration_seconds": None,
            "days_per_week": 1,
            "suggested_days": None,
            "break_seconds": 30,
            "order_index": 5,
        },
        {
            "workout_type": "fga_step_over_obstacle",
            "display_name": "6. Step Over Obstacle",
            "description": (
                "Step (not walk around) over an obstacle while walking "
                "normally. Use shoe boxes: 4.5 in single, 9 in stacked. "
                "Score 3: clears 22.86 cm without contact or speed change."
            ),
            "sets_per_day": 1,
            "target_reps": 1,
            "target_duration_seconds": None,
            "days_per_week": 1,
            "suggested_days": None,
            "break_seconds": 30,
            "order_index": 6,
        },
        {
            "workout_type": "fga_narrow_base",
            "display_name": "7. Gait with Narrow Base of Support",
            "description": (
                "Walk heel-to-toe (tandem) with arms folded across chest, "
                "max 10 steps over 3.6 m. Score 3: 10 valid steps with "
                "no staggering."
            ),
            "sets_per_day": 1,
            "target_reps": 1,
            "target_duration_seconds": None,
            "days_per_week": 1,
            "suggested_days": None,
            "break_seconds": 30,
            "order_index": 7,
        },
        {
            "workout_type": "fga_eyes_closed",
            "display_name": "8. Gait with Eyes Closed",
            "description": (
                "Walk 6 m at normal speed with eyes closed. Score 3: time "
                "<7 s, no aid, deviation <=15.24 cm, no imbalance, normal "
                "pattern."
            ),
            "sets_per_day": 1,
            "target_reps": 1,
            "target_duration_seconds": None,
            "days_per_week": 1,
            "suggested_days": None,
            "break_seconds": 30,
            "order_index": 8,
        },
        {
            "workout_type": "fga_backwards",
            "display_name": "9. Ambulating Backwards",
            "description": (
                "Walk backward 6 m at normal speed. Score 3: no aid, good "
                "speed, deviation <=15.24 cm, no imbalance, normal pattern."
            ),
            "sets_per_day": 1,
            "target_reps": 1,
            "target_duration_seconds": None,
            "days_per_week": 1,
            "suggested_days": None,
            "break_seconds": 30,
            "order_index": 9,
        },
        {
            "workout_type": "fga_steps",
            "display_name": "10. Steps",
            "description": (
                "Walk up the stairs (use rail if needed), turn, walk down. "
                "Score 3: alternating feet, no rail used."
            ),
            "sets_per_day": 1,
            "target_reps": 1,
            "target_duration_seconds": None,
            "days_per_week": 1,
            "suggested_days": None,
            "break_seconds": 30,
            "order_index": 10,
        },
    ],
}


DEFAULT_PROGRAMS = [VIVIFRAIL, CALISTHENICS, FGA]


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
