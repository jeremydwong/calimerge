"""
Plot offline-pipeline diagnostics for one recording:

  * hip-trace.png — per-track Hip-COM (x, y, z) vs time, one subplot
    per axis. Lets you see whether the subject is staying still, walking
    around, or jumping to bogus positions (the ghost-track signature).
  * fps-trace.png — instantaneous FPS over time, derived from
    frame_time_history.csv. Spikes/dropouts = sync hiccups upstream of
    pose estimation. Mean ± std reported in the title.

Outputs are written into <recording>/annotated/, alongside the
overlay mp4s from annotate_offline_npz.py.

Run:
    uv run python3 tests/manual/plot_offline_diagnostics.py
    uv run python3 tests/manual/plot_offline_diagnostics.py --npz keypoints_3d.mps.npz
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless — write PNG, no GUI

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402


REPO = Path(__file__).resolve().parents[2]
RECORDING_NAME = "zelda_20260428_151934_fga_horizontal_head_turns"
ZELDA = REPO / "tests" / "data" / RECORDING_NAME

L_HIP = 11
R_HIP = 12

# Same colour cycle the annotator uses, in matplotlib RGB normalised
# (annotator uses BGR for OpenCV).
TRACK_COLORS = [
    (0.31, 0.78, 0.47),  # green
    (0.39, 0.63, 1.00),  # blue
    (1.00, 0.71, 0.31),  # orange
    (0.86, 0.39, 0.86),  # purple
    (1.00, 0.39, 0.39),  # red
    (0.39, 0.86, 0.86),  # cyan
    (1.00, 0.86, 0.31),  # yellow
    (0.71, 0.55, 1.00),  # lavender
]


def _hip_com(kps_slot: np.ndarray) -> np.ndarray:
    """(N, K, 3) → (N, 3) hip COM with NaN where neither hip is finite."""
    n = kps_slot.shape[0]
    out = np.full((n, 3), np.nan, dtype=np.float32)
    for i in range(n):
        l = kps_slot[i, L_HIP]
        r = kps_slot[i, R_HIP]
        ok_l = np.isfinite(l).all()
        ok_r = np.isfinite(r).all()
        if ok_l and ok_r:
            out[i] = (l + r) * 0.5
        elif ok_l:
            out[i] = l
        elif ok_r:
            out[i] = r
    return out


def _read_fps_series(frame_time_csv: Path) -> tuple[np.ndarray, np.ndarray] | None:
    """Returns (frame_centre_times_s, instantaneous_fps) or None on parse failure."""
    by_sync: dict[int, list[float]] = {}
    with open(frame_time_csv, newline="") as f:
        cleaned = (line for line in f if not line.startswith("#"))
        reader = csv.DictReader(cleaned)
        for row in reader:
            try:
                sync = int(row["sync_index"])
                t = float(row["frame_time"])
            except (KeyError, ValueError, TypeError):
                continue
            by_sync.setdefault(sync, []).append(t)

    syncs = sorted(by_sync.keys())
    if len(syncs) < 2:
        return None
    times = np.array([np.median(by_sync[s]) for s in syncs], dtype=np.float64)
    times -= times[0]
    dt = np.diff(times)
    fps = 1.0 / np.where(dt > 1e-9, dt, np.nan)
    centres = (times[:-1] + times[1:]) * 0.5
    return centres, fps


def _plot_hip_trace(
    npz_path: Path,
    out_path: Path,
    fps_series: tuple[np.ndarray, np.ndarray] | None = None,
) -> None:
    d = np.load(npz_path)
    kps = d["keypoints_3d"]            # (N, P, K, 3)
    timestamps = d["timestamps"]       # (N,) seconds
    backend = str(d["model_backend"]) if "model_backend" in d.files else "?"

    n_axes = 4 if fps_series is not None else 3
    fig, axes = plt.subplots(n_axes, 1, figsize=(11, 2.4 * n_axes), sharex=True)
    if n_axes == 1:
        axes = [axes]
    axis_names = ("x", "y", "z")

    legend_handles: list = []
    for slot in range(kps.shape[1]):
        com = _hip_com(kps[:, slot, :, :])
        valid = np.isfinite(com).all(axis=-1)
        if not valid.any():
            continue
        n_valid = int(valid.sum())
        color = TRACK_COLORS[slot % len(TRACK_COLORS)]
        label = f"slot {slot}  ({n_valid} frames)"
        for ax_i in range(3):
            line, = axes[ax_i].plot(
                timestamps[valid], com[valid, ax_i],
                color=color, lw=1.2, label=label if ax_i == 0 else None,
            )
            if ax_i == 0:
                legend_handles.append(line)

    for ax_i, name in enumerate(axis_names):
        axes[ax_i].set_ylabel(f"hip {name} (m)")
        axes[ax_i].grid(alpha=0.3)
        axes[ax_i].axhline(0, color="0.5", lw=0.5)

    if fps_series is not None:
        centres, fps = fps_series
        finite = np.isfinite(fps)
        ax = axes[3]
        ax.plot(centres, fps, lw=0.7, color="0.25")
        if finite.any():
            mean_fps = float(np.nanmean(fps[finite]))
            p50 = float(np.nanpercentile(fps[finite], 50))
            ax.axhline(mean_fps, color="C1", lw=1, ls="--", alpha=0.8,
                       label=f"mean = {mean_fps:.2f}")
            ax.axhline(p50, color="C2", lw=1, ls=":", alpha=0.8,
                       label=f"median = {p50:.2f}")
            ax.legend(loc="lower right", fontsize=8)
        ax.set_ylabel("FPS")
        ax.grid(alpha=0.3)

    axes[-1].set_xlabel("time (s)")

    fig.suptitle(
        f"Hip-COM trace — {RECORDING_NAME}  [backend={backend}]",
        fontsize=11,
    )
    if legend_handles:
        axes[0].legend(handles=legend_handles, loc="upper right", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  hip-trace (with fps) → {out_path}")


def _plot_fps_trace(frame_time_csv: Path, out_path: Path) -> None:
    """FPS = 1/dt where dt is the median per-port time between consecutive sync indices."""
    by_sync: dict[int, list[float]] = {}
    with open(frame_time_csv, newline="") as f:
        # frame_time_history.csv prefixes a `# cameras: ...` comment line
        # before the actual header. Skip leading `#` lines so DictReader
        # uses the real `sync_index,port,frame_index,frame_time` row.
        cleaned = (line for line in f if not line.startswith("#"))
        reader = csv.DictReader(cleaned)
        for row in reader:
            try:
                sync = int(row["sync_index"])
                t = float(row["frame_time"])
            except (KeyError, ValueError, TypeError):
                continue
            by_sync.setdefault(sync, []).append(t)

    syncs = sorted(by_sync.keys())
    if len(syncs) < 2:
        print(f"  fps-trace skipped: only {len(syncs)} sync indices in CSV")
        return

    times = np.array([np.median(by_sync[s]) for s in syncs], dtype=np.float64)
    times = times - times[0]                # seconds since first frame
    dt = np.diff(times)                     # (N-1,)
    fps = 1.0 / np.where(dt > 1e-9, dt, np.nan)
    centers = (times[:-1] + times[1:]) * 0.5

    finite = np.isfinite(fps)
    mean_fps = float(np.nanmean(fps[finite])) if finite.any() else float("nan")
    std_fps = float(np.nanstd(fps[finite])) if finite.any() else float("nan")
    p5, p50, p95 = (
        float(np.nanpercentile(fps[finite], q)) for q in (5, 50, 95)
    ) if finite.any() else (float("nan"),) * 3

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(centers, fps, lw=0.7, color="0.25")
    ax.axhline(mean_fps, color="C1", lw=1, ls="--", alpha=0.8,
               label=f"mean = {mean_fps:.2f}")
    ax.axhline(p50, color="C2", lw=1, ls=":", alpha=0.8,
               label=f"median = {p50:.2f}")
    ax.set_ylabel("instantaneous FPS")
    ax.set_xlabel("time (s)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(
        f"FPS trace — {RECORDING_NAME}\n"
        f"mean={mean_fps:.2f}  std={std_fps:.2f}  "
        f"p5/p50/p95 = {p5:.1f}/{p50:.1f}/{p95:.1f}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  fps-trace → {out_path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--npz", type=str, default="keypoints_3d.npz",
        help="filename inside the recording dir; default keypoints_3d.npz",
    )
    parser.add_argument(
        "--out-stem", type=str, default=None,
        help="override output filename stem (default derived from --npz)",
    )
    parser.add_argument(
        "--out-dir", type=Path, default=None,
        help="Output directory; default is <recording>/annotated/.",
    )
    args = parser.parse_args(argv)

    npz_path = ZELDA / args.npz
    if not npz_path.exists():
        print(f"missing: {npz_path}")
        return 1
    out_dir = args.out_dir if args.out_dir is not None else ZELDA / "annotated"
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.out_stem or Path(args.npz).stem  # e.g. keypoints_3d, keypoints_3d.mps

    csv_path = ZELDA / "frame_time_history.csv"
    fps_series = None
    if csv_path.exists():
        fps_series = _read_fps_series(csv_path)
        if fps_series is None:
            print(f"  fps not parseable from {csv_path}")
        else:
            # standalone close-up still useful for fps-only inspection
            _plot_fps_trace(csv_path, out_dir / "fps_trace.png")
    else:
        print(f"  fps unavailable: missing {csv_path}")

    _plot_hip_trace(npz_path, out_dir / f"{stem}_hip_trace.png", fps_series=fps_series)

    return 0


if __name__ == "__main__":
    sys.exit(main())
