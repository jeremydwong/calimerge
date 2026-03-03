"""
Re-run the coord_3x1_3 pipeline with auto device (CUDA if available),
time it, and plot foot marker positions vs reference data.
"""

import os
import sys
import tempfile
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Ensure posetrack is importable
POSETRACK_SRC = os.path.join(os.path.dirname(__file__), "..", "src")
sys.path.insert(0, POSETRACK_SRC)

from posetrack.process_synced_poses import process_synced_mwc_frames_multi_person_perf
from posetrack.pose_detector import LOCAL_DET_DIR, LOCAL_SP_DIR

# Paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
POSETRACK_ROOT = os.path.dirname(SCRIPT_DIR)
COORD_3X1_DIR = os.path.join(POSETRACK_ROOT, "tests", "caliscope", "coord_3x1_3")
REF_CSV = os.path.join(
    POSETRACK_ROOT, "output", "caliscope", "coord_3x1_3",
    "output_3d_poses_tracked.csv_person0.csv"
)
OUTPUT_DIR = os.path.join(POSETRACK_ROOT, "scripts", "output")


def run_pipeline():
    """Run the pipeline with auto device and return (output_csv_path, elapsed_seconds)."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path = os.path.join(OUTPUT_DIR, "output_3d_poses_tracked.csv")

    print(f"Input dir:  {COORD_3X1_DIR}")
    print(f"Models:     {LOCAL_SP_DIR}")
    print(f"Detector:   {LOCAL_DET_DIR}")
    print(f"Output:     {output_path}")
    print(f"Reference:  {REF_CSV}")
    print()

    t0 = time.time()
    process_synced_mwc_frames_multi_person_perf(
        frame_history_csv_path=os.path.join(COORD_3X1_DIR, "frame_time_history.csv"),
        calibration_path=os.path.join(COORD_3X1_DIR, "config.toml"),
        video_dir=COORD_3X1_DIR,
        output_path=output_path,
        model_dir=LOCAL_SP_DIR,
        detector_dir=LOCAL_DET_DIR,
        calib_type="mwc",
        skip_sync_indices=1,
        person_confidence=0.8,
        keypoint_confidence=0.1,
        device_name="auto",
        max_persons=1,
        batch_size=32,
    )
    elapsed = time.time() - t0

    person0_path = output_path + "_person0.csv"
    print(f"\nPipeline finished in {elapsed:.1f}s")
    return person0_path, elapsed


def plot_foot_markers(new_csv, ref_csv, elapsed):
    """Plot foot marker X/Y/Z over sync_index, new vs reference."""
    new = pd.read_csv(new_csv)
    ref = pd.read_csv(ref_csv)

    # Foot markers to plot
    markers = ["L_Ankle", "R_Ankle", "l_big_toe", "r_big_toe"]
    axes = ["X", "Y", "Z"]

    fig, axs = plt.subplots(len(markers), len(axes), figsize=(18, 3.5 * len(markers)),
                            sharex=True)

    for row, marker in enumerate(markers):
        for col, axis in enumerate(axes):
            col_name = f"{marker}_{axis}"
            ax = axs[row][col]

            if col_name in ref.columns:
                ax.plot(ref["sync_index"], ref[col_name],
                        label="reference", color="steelblue", alpha=0.7, linewidth=1)
            if col_name in new.columns:
                ax.plot(new["sync_index"], new[col_name],
                        label="new (auto)", color="orangered", alpha=0.7,
                        linewidth=1, linestyle="--")

            ax.set_ylabel(f"{col_name} (m)")
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
            if row == len(markers) - 1:
                ax.set_xlabel("sync_index")
            ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Foot Marker Comparison: Reference vs New (device=auto, {elapsed:.1f}s)",
        fontsize=14, y=1.01,
    )
    fig.tight_layout()

    out_png = os.path.join(OUTPUT_DIR, "foot_markers_comparison.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    print(f"Plot saved to: {out_png}")
    return out_png


if __name__ == "__main__":
    person0_csv, elapsed = run_pipeline()
    plot_foot_markers(person0_csv, REF_CSV, elapsed)
