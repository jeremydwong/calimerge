"""Plot ankle positions over time from the offline-pipeline npz.

Run:
    VIRTUAL_ENV= ~/.local/bin/uv run python tests/manual/plot_ankle_positions.py

Reads tests/data/<recording>/keypoints_3d.npz produced by
run_offline_pipeline_on_test_data.py and writes
tests/data/<recording>/ankle_positions.png.

Three stacked subplots: ankle x, y, z over time. Both feet on the same
axes so it's easy to see whether one is staying put while the other
moves. NaN frames are gapped — matplotlib handles that natively.

Diagnostic block at the end prints peak-to-peak per axis per ankle, plus
the whole-trial mean. The user expects:
  * mean(y) ~ 0 (subject walking back and forth around a centre)
  * peak abs(y) ~ 6 m (length of the walk)
  * z near floor (small range)
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RECORDING = "zelda_20260428_151934_fga_horizontal_head_turns"

# Allow positional arg: recording subfolder name under tests/data/.
_recording = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_RECORDING
RECORDING_DIR = REPO_ROOT / "tests" / "data" / _recording

# COCO-17 ankle indices (also valid in SynthPose-52)
L_ANKLE = 15
R_ANKLE = 16


def main() -> int:
    npz_path = RECORDING_DIR / "keypoints_3d.npz"
    if not npz_path.exists():
        print(f"ERROR: missing {npz_path}. Run "
              f"tests/manual/run_offline_pipeline_on_test_data.py first.",
              flush=True)
        return 2

    data = np.load(str(npz_path))
    times = data["timestamps"]                      # (T,)
    kps = data["keypoints_3d"]                      # (T, P, K, 3)
    counts = data["person_count"]                   # (T,)
    primary = data.get("primary_person_index",
                       np.zeros(len(times), dtype=np.int32))

    n_frames, max_persons, n_kps, _ = kps.shape
    print(f"loaded {npz_path.name}")
    print(f"  frames={n_frames}  persons<={max_persons}  kps={n_kps}")
    print(f"  duration={float(times[-1] - times[0]):.2f}s")
    if "view_transform_R" in data.files:
        R = data["view_transform_R"]
        t = data["view_transform_t"]
        det = float(np.linalg.det(R))
        print(f"  view_transform_R det={det:+.6f}  (1.0 = proper rotation)")
        print(f"  view_transform_t = [{t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f}]")
    else:
        print("  WARNING: npz has no view_transform_R; "
              "this looks like camera-frame data.")

    # Build per-person ankle traces. With fragmentation the C tracker
    # spreads a single subject across multiple person slots — only one
    # slot has data at any given frame — so plotting just primary=0
    # would show only the first fragment. Instead we collect every
    # person slot's full ankle history and plot them all.
    per_person_left: dict[int, np.ndarray] = {}
    per_person_right: dict[int, np.ndarray] = {}
    for p in range(max_persons):
        ll = np.full((n_frames, 3), np.nan, dtype=np.float64)
        rr = np.full((n_frames, 3), np.nan, dtype=np.float64)
        any_left = False
        any_right = False
        for i in range(n_frames):
            la = kps[i, p, L_ANKLE]
            ra = kps[i, p, R_ANKLE]
            if np.all(np.isfinite(la)):
                ll[i] = la
                any_left = True
            if np.all(np.isfinite(ra)):
                rr[i] = ra
                any_right = True
        if any_left or any_right:
            per_person_left[p] = ll
            per_person_right[p] = rr
            n_l = int(np.sum(np.isfinite(ll[:, 0])))
            n_r = int(np.sum(np.isfinite(rr[:, 0])))
            print(f"  person[{p}]: L_Ankle valid={n_l}, R_Ankle valid={n_r}")

    # Also union the ankle data across persons (one trace per side, by
    # picking the first-valid slot at each frame). Useful when the
    # tracks are non-overlapping in time — that is, when fragmentation
    # made the same subject appear in different slots at different
    # times — because the union recovers the continuous trajectory.
    left_union = np.full((n_frames, 3), np.nan, dtype=np.float64)
    right_union = np.full((n_frames, 3), np.nan, dtype=np.float64)
    for i in range(n_frames):
        for p in range(max_persons):
            la = kps[i, p, L_ANKLE]
            if np.all(np.isfinite(la)) and not np.all(np.isfinite(left_union[i])):
                left_union[i] = la
            ra = kps[i, p, R_ANKLE]
            if np.all(np.isfinite(ra)) and not np.all(np.isfinite(right_union[i])):
                right_union[i] = ra

    n_left = int(np.sum(np.isfinite(left_union[:, 0])))
    n_right = int(np.sum(np.isfinite(right_union[:, 0])))
    print(f"  union L_Ankle valid frames: {n_left}/{n_frames}")
    print(f"  union R_Ankle valid frames: {n_right}/{n_frames}")
    left = left_union
    right = right_union

    # Per-axis stats (whole-trial peak-to-peak + mean), so the user can
    # compare against their expectation that y peaks at ~6 m and mean(y)
    # ~ 0 for a back-and-forth walk.
    axis_names = ("x", "y", "z")
    print()
    print(f"{'axis':>5} {'L_min':>10} {'L_max':>10} {'L_mean':>10} "
          f"{'L_p2p':>10}   {'R_min':>10} {'R_max':>10} {'R_mean':>10} "
          f"{'R_p2p':>10}")
    for ax_i, name in enumerate(axis_names):
        la = left[:, ax_i]
        ra = right[:, ax_i]
        l_min = float(np.nanmin(la)) if n_left else float("nan")
        l_max = float(np.nanmax(la)) if n_left else float("nan")
        l_mean = float(np.nanmean(la)) if n_left else float("nan")
        l_p2p = (l_max - l_min) if n_left else float("nan")
        r_min = float(np.nanmin(ra)) if n_right else float("nan")
        r_max = float(np.nanmax(ra)) if n_right else float("nan")
        r_mean = float(np.nanmean(ra)) if n_right else float("nan")
        r_p2p = (r_max - r_min) if n_right else float("nan")
        print(f"{name:>5} {l_min:>+10.3f} {l_max:>+10.3f} {l_mean:>+10.3f} "
              f"{l_p2p:>10.3f}   "
              f"{r_min:>+10.3f} {r_max:>+10.3f} {r_mean:>+10.3f} "
              f"{r_p2p:>10.3f}")

    # Plot.
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"ERROR: matplotlib not available: {e}")
        return 3

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), sharex=True)
    PERSON_COLORS = ["#5099ff", "#ff8c00", "#22aa44", "#cc44cc",
                     "#cc6644", "#449988", "#aaaa44", "#aa44aa"]
    for ax, ax_i, name in zip(axes, range(3), axis_names):
        # Per-person traces, dotted (left) + dashed (right). Each
        # person fragment gets its own colour so it's obvious where
        # the C tracker fragmented mid-trial.
        for p, ll in sorted(per_person_left.items()):
            color = PERSON_COLORS[p % len(PERSON_COLORS)]
            ax.plot(times, ll[:, ax_i], color=color, linewidth=1.0,
                    alpha=0.55, label=f"P{p} L" if ax_i == 0 else None)
            rr = per_person_right.get(p)
            if rr is not None:
                ax.plot(times, rr[:, ax_i], color=color, linewidth=1.0,
                        linestyle="--", alpha=0.55,
                        label=f"P{p} R" if ax_i == 0 else None)
        # Union trace on top, thicker, so the user can see the
        # consensus motion across all fragments.
        ax.plot(times, left[:, ax_i], color="#000000", linewidth=2.0,
                label="union L" if ax_i == 0 else None)
        ax.plot(times, right[:, ax_i], color="#444444", linewidth=2.0,
                linestyle="--",
                label="union R" if ax_i == 0 else None)
        ax.axhline(0, color="#888", linewidth=0.5, linestyle=":")
        ax.set_ylabel(f"{name} (m)")
        ax.grid(True, alpha=0.3)
    axes[0].set_title(
        f"Ankle positions over time — {RECORDING_DIR.name}\n"
        f"all persons + union, body frame (after rotate-to-human + zero-at-L_ankle)"
    )
    axes[0].legend(loc="upper right", fontsize=8, ncol=2)
    axes[-1].set_xlabel("time (s)")

    out = RECORDING_DIR / "ankle_positions.png"
    fig.tight_layout()
    fig.savefig(str(out), dpi=120)
    plt.close(fig)
    print()
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
