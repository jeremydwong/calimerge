"""Overlay online vs offline ankle traces from the same recording.

Loads the user's online-recorded npz (from <last_project_folder>/workouts/
<recording>/keypoints_3d.npz) and the offline pipeline's npz (from
tests/data/<recording>/keypoints_3d.npz produced by
run_offline_pipeline_on_test_data.py) and plots them on top of each other.

Three stacked subplots (x, y, z) per ankle, with online drawn solid and
offline drawn dashed. The user wants to spot:
  * Whether the offline run has the same t=11s "snap to static object"
    artefact the online run has.
  * Whether the offline run is generally cleaner / dirtier.
  * Whether raising person_confidence in the offline run kills the snap.

Usage:
    VIRTUAL_ENV= ~/.local/bin/uv run python tests/manual/compare_online_vs_offline_npz.py \\
        zelda_20260428_152104_fga_horizontal_head_turns

Output: tests/data/<recording>/online_vs_offline_ankles.png
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]

L_ANKLE = 15
R_ANKLE = 16


def _load_app_settings():
    from calimerge.config import load_app_settings
    return load_app_settings()


def _find_online_npz(recording_name: str) -> Path | None:
    """Find the online-recorded keypoints_3d.npz under
    <last_project_folder>/workouts/<recording_name>/."""
    settings = _load_app_settings()
    folder = settings.get("last_project_folder")
    candidates: list[Path] = []
    if folder:
        candidates.append(Path(folder) / "workouts" / recording_name / "keypoints_3d.npz")
    # Also walk OneDrive default location, where the user has their data.
    candidates.append(
        Path("~/OneDrive/Documents/calimerge/recordings/workouts").expanduser()
        / recording_name / "keypoints_3d.npz"
    )
    for c in candidates:
        if c.exists():
            return c
    return None


def _union_ankle_traces(npz_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (times, left_xyz_union, right_xyz_union).

    Picks first-valid person slot per frame to recover continuous
    motion when the C tracker fragmented one subject across slots.
    """
    data = np.load(str(npz_path))
    times = data["timestamps"]
    kps = data["keypoints_3d"]                 # (T, P, K, 3)
    n_frames, max_persons, n_kps, _ = kps.shape

    left = np.full((n_frames, 3), np.nan, dtype=np.float64)
    right = np.full((n_frames, 3), np.nan, dtype=np.float64)
    for i in range(n_frames):
        for p in range(max_persons):
            la = kps[i, p, L_ANKLE]
            if np.all(np.isfinite(la)) and not np.all(np.isfinite(left[i])):
                left[i] = la
            ra = kps[i, p, R_ANKLE]
            if np.all(np.isfinite(ra)) and not np.all(np.isfinite(right[i])):
                right[i] = ra
    return np.asarray(times, dtype=np.float64), left, right


def _peak_to_peak_summary(label: str, times: np.ndarray,
                          left: np.ndarray, right: np.ndarray):
    print(f"[{label}] {label} ankle ranges (m):")
    print(f"{'axis':>5} {'L_min':>10} {'L_max':>10} {'L_mean':>10} "
          f"{'L_p2p':>10}   {'R_min':>10} {'R_max':>10} {'R_mean':>10} "
          f"{'R_p2p':>10}")
    for ax_i, name in enumerate(("x", "y", "z")):
        la = left[:, ax_i]
        ra = right[:, ax_i]
        n_l = int(np.sum(np.isfinite(la)))
        n_r = int(np.sum(np.isfinite(ra)))
        if n_l == 0 or n_r == 0:
            print(f"  {name}: no valid frames")
            continue
        print(
            f"{name:>5} "
            f"{float(np.nanmin(la)):>+10.3f} "
            f"{float(np.nanmax(la)):>+10.3f} "
            f"{float(np.nanmean(la)):>+10.3f} "
            f"{float(np.nanmax(la) - np.nanmin(la)):>10.3f}   "
            f"{float(np.nanmin(ra)):>+10.3f} "
            f"{float(np.nanmax(ra)):>+10.3f} "
            f"{float(np.nanmean(ra)):>+10.3f} "
            f"{float(np.nanmax(ra) - np.nanmin(ra)):>10.3f}"
        )


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument(
        "recording",
        help="Recording subfolder name (the same one passed to "
             "run_offline_pipeline_on_test_data.py)",
    )
    p.add_argument(
        "--online-npz",
        type=Path, default=None,
        help="Override the auto-detected online npz path.",
    )
    args = p.parse_args()

    offline_npz = REPO_ROOT / "tests" / "data" / args.recording / "keypoints_3d.npz"
    if not offline_npz.exists():
        print(f"ERROR: offline npz not found at {offline_npz}\n"
              f"Run: VIRTUAL_ENV= ~/.local/bin/uv run python "
              f"tests/manual/run_offline_pipeline_on_test_data.py "
              f"{args.recording}", flush=True)
        return 2

    online_npz = args.online_npz or _find_online_npz(args.recording)
    if online_npz is None or not online_npz.exists():
        print(f"ERROR: online npz not found for {args.recording!r}. "
              f"Pass --online-npz <path> if it lives somewhere unusual.",
              flush=True)
        return 2

    print(f"online npz:  {online_npz}")
    print(f"offline npz: {offline_npz}")
    print()

    on_t, on_left, on_right = _union_ankle_traces(online_npz)
    of_t, of_left, of_right = _union_ankle_traces(offline_npz)

    on_data = np.load(str(online_npz))
    of_data = np.load(str(offline_npz))
    if "view_transform_R" in on_data.files:
        print(f"online view_transform_t = "
              f"{on_data['view_transform_t'].tolist()}")
    else:
        print("online npz has NO view_transform_R — camera frame.")
    if "view_transform_R" in of_data.files:
        print(f"offline view_transform_t = "
              f"{of_data['view_transform_t'].tolist()}")
    print()
    _peak_to_peak_summary("online", on_t, on_left, on_right)
    print()
    _peak_to_peak_summary("offline", of_t, of_left, of_right)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"ERROR: matplotlib unavailable: {e}", flush=True)
        return 3

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    L_ON, R_ON = "#1f77b4", "#d62728"   # online: solid blue / red
    L_OF, R_OF = "#56baff", "#ff8a8a"   # offline: light blue / light red

    for ax, ax_i, name in zip(axes, range(3), ("x", "y", "z")):
        # Online traces — solid, full-saturation, on top.
        ax.plot(on_t, on_left[:, ax_i], color=L_ON, linewidth=1.6,
                label="L_Ankle online" if ax_i == 0 else None)
        ax.plot(on_t, on_right[:, ax_i], color=R_ON, linewidth=1.6,
                label="R_Ankle online" if ax_i == 0 else None)
        # Offline — dashed, lighter, behind.
        ax.plot(of_t, of_left[:, ax_i], color=L_OF, linewidth=1.2,
                linestyle="--",
                label="L_Ankle offline" if ax_i == 0 else None)
        ax.plot(of_t, of_right[:, ax_i], color=R_OF, linewidth=1.2,
                linestyle="--",
                label="R_Ankle offline" if ax_i == 0 else None)
        ax.axhline(0, color="#999", linewidth=0.5, linestyle=":")
        # Highlight the user-flagged "snap" window.
        ax.axvspan(10.5, 11.5, color="#ffe080", alpha=0.25, zorder=-1)
        ax.set_ylabel(f"{name} (m)")
        ax.grid(True, alpha=0.3)
    axes[0].set_title(
        f"online vs offline ankle positions — {args.recording}\n"
        f"shaded band = user-flagged static-object snap at t~11s"
    )
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("time (s)")

    out = REPO_ROOT / "tests" / "data" / args.recording / "online_vs_offline_ankles.png"
    fig.tight_layout()
    fig.savefig(str(out), dpi=120)
    plt.close(fig)
    print()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
