"""
Workout type definitions — standardized struct for all exercise types.

Each WorkoutType describes how an exercise is detected, analysed, and
displayed. Both the Vivifrail (older adults) and Calisthenics programs
reference the same WorkoutType registry by name key.

The registry is WORKOUT_TYPES: dict[str, WorkoutType].
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WorkoutType:
    """Standardized definition of a workout/exercise type."""

    name: str                              # e.g. "sit_to_stand"
    display_name: str                      # e.g. "Sit-to-Stand"
    signal_source: str                     # "hip_z", "shoulder_z", "head_z",
                                           # "elbow_angle", "knee_z", "hip_sway",
                                           # "head_speed", "duration_only"
    detection_direction: str               # "up" (crossing upward) or "down"
                                           # (crossing downward) or "duration"
                                           # or "speed"
    init_threshold: float                  # threshold for rejecting initial
                                           # garbage (e.g. 0.65m seated)
    rep_threshold: float                   # threshold for counting reps
    default_duration_seconds: int          # default recording duration
    analysis_function: str                 # name of the analysis function
    analysis_module: str                   # module path
    threshold_label: str                   # GUI label for the threshold spinbox
    threshold_unit: str                    # "m", "deg", "m/s"
    threshold_range: tuple[float, float]   # min/max for the spinbox
    y_axis_label: str                      # plot Y axis label
    weekly_improvement_factor: float       # e.g. 1.02 for 2%/week
    primary_metric: str                    # metric name stored in session_results
    primary_metric_label: str              # human-readable label for the metric


WORKOUT_TYPES: dict[str, WorkoutType] = {
    "sit_to_stand": WorkoutType(
        name="sit_to_stand",
        display_name="Sit-to-Stand",
        signal_source="hip_z",
        detection_direction="up",
        init_threshold=0.65,
        rep_threshold=0.65,
        default_duration_seconds=30,
        analysis_function="analyze_sit_to_stand",
        analysis_module="calimerge.analysis.sit_to_stand",
        threshold_label="Seated threshold (m):",
        threshold_unit="m",
        threshold_range=(0.1, 2.0),
        y_axis_label="Hip Height (m)",
        weekly_improvement_factor=1.01,      # Vivifrail: ~1% / week
        primary_metric="work_joules",
        primary_metric_label="Work (J)",
    ),
    "timed_up_and_go": WorkoutType(
        name="timed_up_and_go",
        display_name="Timed Up and Go",
        signal_source="head_speed",
        detection_direction="speed",
        init_threshold=0.3,
        rep_threshold=0.3,
        default_duration_seconds=30,
        analysis_function="analyze_tug",
        analysis_module="calimerge.analysis.tug",
        threshold_label="Head speed threshold (m/s):",
        threshold_unit="m/s",
        threshold_range=(0.05, 3.0),
        y_axis_label="Head Speed (m/s)",
        weekly_improvement_factor=0.99,      # lower is better; 1% faster / week
        primary_metric="tug_duration",
        primary_metric_label="Duration (s)",
    ),
    "biceps_curl": WorkoutType(
        name="biceps_curl",
        display_name="Biceps Curls",
        signal_source="elbow_angle",
        detection_direction="up",
        init_threshold=150.0,
        rep_threshold=150.0,
        default_duration_seconds=30,
        analysis_function="analyze_biceps_curl",
        analysis_module="calimerge.analysis.biceps_curl",
        threshold_label="Extended angle (\u00b0):",
        threshold_unit="\u00b0",
        threshold_range=(0.0, 180.0),
        y_axis_label="Elbow Angle (\u00b0)",
        weekly_improvement_factor=1.02,      # Calisthenics: ~2% / week
        primary_metric="work_joules",
        primary_metric_label="Work (J)",
    ),
    "pushup": WorkoutType(
        name="pushup",
        display_name="Pushups",
        signal_source="shoulder_z",
        detection_direction="down",
        init_threshold=0.30,
        rep_threshold=0.30,
        default_duration_seconds=60,
        analysis_function="analyze_pushup",
        analysis_module="calimerge.analysis.pushup",
        threshold_label="Top threshold (m):",
        threshold_unit="m",
        threshold_range=(0.0, 2.0),
        y_axis_label="Shoulder Height (m)",
        weekly_improvement_factor=1.02,      # Calisthenics: ~2% / week
        primary_metric="work_joules",
        primary_metric_label="Work (J)",
    ),
    "pullup": WorkoutType(
        name="pullup",
        display_name="Pullups",
        signal_source="head_z",
        detection_direction="up",
        init_threshold=1.80,
        rep_threshold=1.80,
        default_duration_seconds=60,
        analysis_function="analyze_pullup",
        analysis_module="calimerge.analysis.pullup",
        threshold_label="Bar height (m):",
        threshold_unit="m",
        threshold_range=(0.5, 3.0),
        y_axis_label="Head Height (m)",
        weekly_improvement_factor=1.02,      # Calisthenics: ~2% / week
        primary_metric="work_joules",
        primary_metric_label="Work (J)",
    ),
    "leg_raise": WorkoutType(
        name="leg_raise",
        display_name="Leg Raises",
        signal_source="knee_z",
        detection_direction="up",
        init_threshold=0.60,
        rep_threshold=0.60,
        default_duration_seconds=30,
        analysis_function="analyze_leg_raise",
        analysis_module="calimerge.analysis.leg_raise",
        threshold_label="Lift threshold (m):",
        threshold_unit="m",
        threshold_range=(0.0, 2.0),
        y_axis_label="Knee Height (m)",
        weekly_improvement_factor=1.01,      # Vivifrail: ~1% / week
        primary_metric="rep_count",
        primary_metric_label="Repetitions",
    ),
    "tandem_stance": WorkoutType(
        name="tandem_stance",
        display_name="Tandem Stance",
        signal_source="hip_sway",
        detection_direction="duration",
        init_threshold=0.05,
        rep_threshold=0.05,
        default_duration_seconds=30,
        analysis_function="analyze_tandem_stance",
        analysis_module="calimerge.analysis.tandem_stance",
        threshold_label="Sway threshold (m):",
        threshold_unit="m",
        threshold_range=(0.005, 0.5),
        y_axis_label="Horizontal Sway (m)",
        weekly_improvement_factor=1.01,      # Vivifrail: ~1% / week
        primary_metric="hold_seconds",
        primary_metric_label="Hold (s)",
    ),
    "stretch": WorkoutType(
        name="stretch",
        display_name="Stretch",
        signal_source="duration_only",
        detection_direction="duration",
        init_threshold=0.0,
        rep_threshold=0.0,
        default_duration_seconds=30,
        analysis_function="analyze_stretch",
        analysis_module="calimerge.analysis.stretch",
        threshold_label="(no threshold for stretch)",
        threshold_unit="m",
        threshold_range=(0.0, 1.0),
        y_axis_label="Hip Height (m)",
        weekly_improvement_factor=1.005,     # Flexibility: ~0.5% / week
        primary_metric="hold_seconds",
        primary_metric_label="Hold (s)",
    ),
}


def get_workout_type(name: str) -> WorkoutType:
    """Look up a WorkoutType by name key. Raises KeyError if not found."""
    return WORKOUT_TYPES[name]
