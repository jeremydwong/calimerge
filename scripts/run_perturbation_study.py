"""
Extrinsic perturbation sensitivity study.

Usage:
    uv run python scripts/run_perturbation_study.py [--config CONFIG] [--reference REF_CSV] [--output OUTPUT_DIR] [--camera PORT]

Defaults use the coord_3x1_3 test dataset.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Extrinsic perturbation study")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("posetrack/tests/caliscope/coord_3x1_3/config.toml"),
    )
    parser.add_argument(
        "--reference",
        type=Path,
        default=Path(
            "posetrack/tests/caliscope/coord_3x1_3/tracking_output/"
            "output_3d_poses_tracked.csv_person0.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("analysis_output/perturbation_study"),
    )
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--mc-trials", type=int, default=50)
    args = parser.parse_args()

    from calimerge.analysis.perturbation_study import run_study

    run_study(args.config, args.reference, args.output, args.camera, args.mc_trials)

    # Import and run plots if matplotlib available
    try:
        from calimerge.analysis.plot_perturbation import generate_all_plots

        generate_all_plots(args.output)
        print(f"Plots saved to {args.output}")
    except ImportError:
        print("matplotlib not available, skipping plots")


if __name__ == "__main__":
    main()
