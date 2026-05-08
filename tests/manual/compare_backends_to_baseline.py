"""
Backend regression test: compare pytorch / mps / cuda offline pipeline
outputs against the git-tracked baseline npz for the zelda fixture.

The git-committed `tests/data/zelda_*/keypoints_3d.npz` (commit e4c84b5)
is the reference Windows-pipeline output: 106 frames, 4 tracked persons
in slots 0..3, SynthPose-52 keypoints.

This script:
  1. Saves the git-HEAD baseline to /tmp so it survives the runs that
     overwrite the fixture file.
  2. For each available backend, runs the offline pipeline
     (run_offline_pipeline_on_test_data.py) and snapshots the resulting
     dense npz into a per-backend file inside the recording dir.
  3. Restores the original baseline files (keypoints_3d.npz +
     keypoints_3d.raw.npz) from git so the working tree stays clean.
  4. Loads baseline + per-backend snapshots and prints a comparison
     table: track count, per-slot frame range, per-slot Hip-COM
     trajectory range, and (where slots align by frame range) the
     mean/max Euclidean Hip-COM distance per overlapping frame.

Run:
    uv run python3 tests/manual/compare_backends_to_baseline.py

Useful flags:
    --backends pytorch,mps      # which to run; default = all available
    --skip-run                  # only re-load existing per-backend snapshots
                                # (use after a slow run to re-display the table)
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
RECORDING_NAME = "zelda_20260428_151934_fga_horizontal_head_turns"
ZELDA = REPO / "tests" / "data" / RECORDING_NAME
BASELINE_TMP = Path(tempfile.gettempdir()) / f"baseline_{RECORDING_NAME}_keypoints_3d.npz"
RUNNER = REPO / "tests" / "manual" / "run_offline_pipeline_on_test_data.py"

# Hip indices for SynthPose-52 (= COCO-17 indices for hips)
L_HIP = 11
R_HIP = 12

# Tracked fixture files we must restore after the runs trample them.
FIXTURE_FILES = ("keypoints_3d.npz", "keypoints_3d.raw.npz")


def _extract_baseline_from_git() -> bool:
    target = f"tests/data/{RECORDING_NAME}/keypoints_3d.npz"
    cmd = ["git", "-C", str(REPO), "show", f"HEAD:{target}"]
    with open(BASELINE_TMP, "wb") as f:
        rc = subprocess.run(cmd, stdout=f).returncode
    if rc != 0 or BASELINE_TMP.stat().st_size == 0:
        print(f"[baseline] FAILED to extract from git ({cmd})")
        return False
    print(f"[baseline] saved git HEAD copy -> {BASELINE_TMP} "
          f"({BASELINE_TMP.stat().st_size} bytes)")
    return True


def _stage_baseline_from_path(src: Path) -> bool:
    if not src.exists():
        print(f"[baseline] FAILED: --baseline-path {src} does not exist")
        return False
    shutil.copy2(src, BASELINE_TMP)
    print(f"[baseline] copied {src} -> {BASELINE_TMP} "
          f"({BASELINE_TMP.stat().st_size} bytes)")
    return True


def _backend_available(name: str) -> bool:
    if name == "pytorch":
        return True  # always available if calimerge is importable
    if name == "mps":
        try:
            from calimerge.tracking.mps_offline_binding import is_available
            return is_available()
        except Exception:
            return False
    if name == "cuda":
        try:
            from calimerge.tracking.cuda_binding import is_available
            return is_available()
        except Exception:
            return False
    return False


def _run_backend(backend: str) -> bool:
    """Run the offline pipeline for one backend; copy outputs into per-backend files."""
    print(f"\n[run] backend={backend} (full recording, no sync cap)")
    cmd = [
        sys.executable, str(RUNNER),
        "--unified-backend", backend,
    ]
    rc = subprocess.run(cmd, cwd=str(REPO)).returncode
    if rc != 0:
        print(f"[run] backend={backend} returned exit code {rc}")
        return False

    # Snapshot the just-produced npz before the next backend overwrites it.
    src = ZELDA / "keypoints_3d.npz"
    dst = ZELDA / f"keypoints_3d.{backend}.npz"
    if not src.exists():
        print(f"[run] expected {src} after run, missing")
        return False
    shutil.copy2(src, dst)
    print(f"[run] snapshot -> {dst.name}")
    return True


def _restore_fixtures() -> None:
    rel_files = [f"tests/data/{RECORDING_NAME}/{name}" for name in FIXTURE_FILES]
    cmd = ["git", "-C", str(REPO), "checkout", "HEAD", "--"] + rel_files
    subprocess.run(cmd, check=False)
    print(f"[restore] git checkout HEAD -- {' '.join(FIXTURE_FILES)}")


def _hip_com(kps: np.ndarray) -> np.ndarray:
    """Per-frame Hip-COM (xyz) for one slot. NaN where neither hip is valid."""
    n_frames = kps.shape[0]
    out = np.full((n_frames, 3), np.nan, dtype=np.float32)
    for i in range(n_frames):
        l = kps[i, L_HIP]
        r = kps[i, R_HIP]
        l_ok = np.isfinite(l).all()
        r_ok = np.isfinite(r).all()
        if l_ok and r_ok:
            out[i] = (l + r) * 0.5
        elif l_ok:
            out[i] = l
        elif r_ok:
            out[i] = r
    return out


def _slot_summary(kps: np.ndarray) -> list[dict]:
    """One dict per non-empty slot: {slot, n_frames, first, last, com_min, com_max}."""
    out = []
    finite = np.isfinite(kps).all(axis=-1)  # (frames, persons, kps)
    for p in range(kps.shape[1]):
        slot_valid = finite[:, p, :].any(axis=-1)
        if not slot_valid.any():
            continue
        idx = np.where(slot_valid)[0]
        com = _hip_com(kps[:, p, :, :])
        com_finite = com[np.isfinite(com).all(axis=-1)]
        if com_finite.size > 0:
            com_min = com_finite.min(axis=0)
            com_max = com_finite.max(axis=0)
        else:
            com_min = com_max = np.array([np.nan, np.nan, np.nan])
        out.append({
            "slot": p,
            "n_frames": int(slot_valid.sum()),
            "first": int(idx[0]),
            "last": int(idx[-1]),
            "com_min": com_min,
            "com_max": com_max,
            "com": com,  # (n_frames, 3) for diff computation
        })
    return out


def _compare_slots(baseline_slots: list[dict], backend_slots: list[dict]) -> list[dict]:
    """Greedy IoU-based track matching across baseline and backend slots.

    Track slot indices are not stable across runs (the C tracker assigns
    them in detection order). We instead match by frame-overlap: for each
    pair (b, x), compute overlap-IoU on the per-frame valid mask, then
    pick pairs greedily highest-IoU first.
    """
    # Build per-slot valid-frame masks aligned to a common frame axis.
    n_frames = 0
    for s in (*baseline_slots, *backend_slots):
        n_frames = max(n_frames, s["com"].shape[0])

    def _mask(s: dict) -> np.ndarray:
        m = np.zeros(n_frames, dtype=bool)
        valid = np.isfinite(s["com"]).all(axis=-1)
        m[: valid.shape[0]] = valid
        return m

    b_masks = [_mask(s) for s in baseline_slots]
    x_masks = [_mask(s) for s in backend_slots]

    pairs: list[tuple[float, int, int]] = []
    for i, bm in enumerate(b_masks):
        for j, xm in enumerate(x_masks):
            inter = int((bm & xm).sum())
            union = int((bm | xm).sum())
            if union == 0:
                continue
            iou = inter / union
            if inter > 0:
                pairs.append((iou, i, j))

    pairs.sort(reverse=True)  # highest IoU first
    matched_b: set[int] = set()
    matched_x: set[int] = set()
    matches: list[tuple[int, int, float]] = []  # (b_idx, x_idx, iou)
    for iou, i, j in pairs:
        if i in matched_b or j in matched_x:
            continue
        matched_b.add(i)
        matched_x.add(j)
        matches.append((i, j, iou))

    results = []
    # Matched pairs first
    for b_idx, x_idx, iou in matches:
        b = baseline_slots[b_idx]
        x = backend_slots[x_idx]
        n = min(b["com"].shape[0], x["com"].shape[0])
        both = (
            np.isfinite(b["com"][:n]).all(axis=-1)
            & np.isfinite(x["com"][:n]).all(axis=-1)
        )
        result = {
            "kind": "match",
            "iou": iou,
            "baseline": b,
            "backend": x,
            "overlap_frames": int(both.sum()),
        }
        if both.any():
            diff = b["com"][:n][both] - x["com"][:n][both]
            dist = np.linalg.norm(diff, axis=-1)
            result["mean_dist_m"] = float(dist.mean())
            result["max_dist_m"] = float(dist.max())
        results.append(result)

    # Orphans
    for i, b in enumerate(baseline_slots):
        if i not in matched_b:
            results.append({"kind": "baseline_only", "baseline": b, "backend": None})
    for j, x in enumerate(backend_slots):
        if j not in matched_x:
            results.append({"kind": "backend_only", "baseline": None, "backend": x})
    return results


_PARAM_KEYS = ("person_confidence", "max_track_distance", "track_patience")


def _read_params(path: Path) -> dict:
    """Pull recorded params + provenance from an npz, tolerating missing keys."""
    d = np.load(path)
    out = {}
    for k in _PARAM_KEYS + ("model_backend", "model_name"):
        if k in d.files:
            v = d[k]
            try:
                out[k] = v.item() if hasattr(v, "item") else v
            except Exception:
                out[k] = v
    return out


def _check_param_invariant(baseline: dict, backend: dict) -> bool:
    """Return True if numeric tracker params match within tight tolerance."""
    ok = True
    for k in _PARAM_KEYS:
        b = baseline.get(k)
        x = backend.get(k)
        if b is None or x is None:
            print(f"  [param] {k:<22} baseline={b!r:>10}  backend={x!r:>10}  (missing)")
            ok = False
            continue
        # Floats: tight float32 tolerance; ints: equality.
        if k == "track_patience":
            same = int(b) == int(x)
        else:
            same = abs(float(b) - float(x)) < 1e-4
        marker = "OK " if same else "!! "
        print(f"  [param] {k:<22} baseline={float(b):>10.4f}  backend={float(x):>10.4f}  {marker}")
        if not same:
            ok = False
    return ok


def _print_report(label: str, baseline_path: Path, backend_path: Path) -> None:
    print(f"\n=== {label} (vs baseline) ===")
    if not backend_path.exists():
        print(f"  {backend_path.name}: missing — run skipped or failed")
        return

    baseline_params = _read_params(baseline_path)
    backend_params = _read_params(backend_path)
    print(f"  baseline:  backend={baseline_params.get('model_backend','?')}  model={baseline_params.get('model_name','?')}")
    print(f"  backend:   backend={backend_params.get('model_backend','?')}  model={backend_params.get('model_name','?')}")
    params_ok = _check_param_invariant(baseline_params, backend_params)
    if not params_ok:
        print("  [WARN] tracker params diverge between baseline and backend -- distance metrics below are not apples-to-apples.")

    base = np.load(baseline_path)["keypoints_3d"]
    back = np.load(backend_path)["keypoints_3d"]
    print(f"  baseline shape: {base.shape}  backend shape: {back.shape}")

    bs = _slot_summary(base)
    xs = _slot_summary(back)
    print(f"  baseline tracks: {len(bs)}  backend tracks: {len(xs)}")

    rows = _compare_slots(bs, xs)
    print(
        f"  {'kind':>13} | {'baseline':>30} | {'backend':>30} | "
        f"{'iou':>5} | {'overlap_n':>9} | {'mean_dist_m':>11} | {'max_dist_m':>10}"
    )
    print(
        f"  {'-'*13} | {'-'*30} | {'-'*30} | {'-'*5} | {'-'*9} | {'-'*11} | {'-'*10}"
    )
    for r in rows:
        b = r["baseline"]
        x = r["backend"]
        b_desc = (
            f"slot{b['slot']} {b['n_frames']}f [{b['first']}..{b['last']}]"
            if b else "—"
        )
        x_desc = (
            f"slot{x['slot']} {x['n_frames']}f [{x['first']}..{x['last']}]"
            if x else "—"
        )
        iou = f"{r.get('iou', 0):.2f}" if r["kind"] == "match" else "—"
        overlap = r.get("overlap_frames", 0)
        if overlap > 0 and "mean_dist_m" in r:
            m = f"{r['mean_dist_m']:.4f}"
            M = f"{r['max_dist_m']:.4f}"
        else:
            m = M = "—"
        print(
            f"  {r['kind']:>13} | {b_desc:>30} | {x_desc:>30} | "
            f"{iou:>5} | {overlap:>9} | {m:>11} | {M:>10}"
        )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--backends", default="pytorch,mps,cuda",
        help="Comma-separated backends to run (skipped if unavailable).",
    )
    parser.add_argument(
        "--skip-run", action="store_true",
        help="Skip running backends; just compare existing per-backend snapshots.",
    )
    parser.add_argument(
        "--baseline-path", type=Path, default=None,
        help="Local npz to use as the baseline (overrides git extraction). "
             "Useful when the new baseline hasn't been committed yet.",
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="After comparison, run annotate_offline_npz.py and "
             "plot_offline_diagnostics.py for each backend, writing output "
             "to <recordings_dir>/<recording_name>/annotated/.",
    )
    args = parser.parse_args(argv)

    if args.baseline_path is not None:
        if not _stage_baseline_from_path(args.baseline_path):
            return 2
    else:
        if not _extract_baseline_from_git():
            return 2

    requested = [b.strip() for b in args.backends.split(",") if b.strip()]
    available = [b for b in requested if _backend_available(b)]
    skipped = [b for b in requested if b not in available]
    if skipped:
        print(f"[backends] skipping unavailable: {skipped}")
    print(f"[backends] running: {available}")

    if not args.skip_run:
        for b in available:
            ok = _run_backend(b)
            if not ok:
                print(f"[backends] {b} run failed; continuing")
        if args.baseline_path is not None:
            # User-provided baseline: restore that exact file back into the
            # recording dir, since git HEAD is stale relative to the new
            # baseline.
            shutil.copy2(BASELINE_TMP, ZELDA / "keypoints_3d.npz")
            print(f"[restore] restored {ZELDA.name}/keypoints_3d.npz from "
                  f"--baseline-path snapshot")
        else:
            _restore_fixtures()

    baseline_path = BASELINE_TMP
    for b in available:
        _print_report(f"{b}", baseline_path, ZELDA / f"keypoints_3d.{b}.npz")

    if args.visualize and available:
        from calimerge.config import workouts_db_path
        out_dir = workouts_db_path().parent / RECORDING_NAME / "annotated"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n[visualize] output dir: {out_dir}")
        annotate = REPO / "tests" / "manual" / "annotate_offline_npz.py"
        diag = REPO / "tests" / "manual" / "plot_offline_diagnostics.py"
        for b in available:
            npz_name = f"keypoints_3d.{b}.npz"
            if not (ZELDA / npz_name).exists():
                continue
            print(f"\n[visualize] annotating {b}...")
            subprocess.run(
                [sys.executable, str(annotate),
                 "--npz", npz_name, "--out-dir", str(out_dir)],
                cwd=str(REPO),
            )
            print(f"[visualize] plotting diagnostics {b}...")
            subprocess.run(
                [sys.executable, str(diag),
                 "--npz", npz_name, "--out-dir", str(out_dir)],
                cwd=str(REPO),
            )

    print("\n[done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
