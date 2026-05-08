"""
Regenerate the zelda PyTorch baseline with the canonical params.

End-to-end:
  1. Drives run_offline_pipeline_on_test_data.py with PyTorch backend,
     no sync cap, explicit (person_confidence=0.1, max_track_distance=0.5,
     track_patience=60).
  2. Snapshots the resulting `tests/data/<recording>/keypoints_3d.npz`
     to `/tmp/pytorch_baseline.npz` so the comparator's --baseline-path
     pickup is one path away.
  3. Inspects the regenerated npz: shape, embedded params, per-slot
     coverage. Lets us verify the plumbing actually took effect (e.g.
     `max_track_distance` written into the npz now matches what the
     tracker used internally).

Run:
    uv run python3 tests/manual/regen_pytorch_baseline.py
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np


REPO = Path(__file__).resolve().parents[2]
RECORDING_NAME = "zelda_20260428_151934_fga_horizontal_head_turns"
ZELDA = REPO / "tests" / "data" / RECORDING_NAME
NPZ = ZELDA / "keypoints_3d.npz"
BASELINE_TMP = Path("/tmp/pytorch_baseline.npz")
RUNNER = REPO / "tests" / "manual" / "run_offline_pipeline_on_test_data.py"

PERSON_CONFIDENCE = 0.1
MAX_TRACK_DISTANCE = 0.5
TRACK_PATIENCE = 60


def _run_pipeline(extrinsic_session_id: int | None) -> int:
    """Run the offline pipeline; return rc. Output is streamed to stdout."""
    cmd = [
        "uv", "run", "python3", str(RUNNER),
        "--unified-backend", "pytorch",
        "--person-confidence", str(PERSON_CONFIDENCE),
        "--max-track-distance", str(MAX_TRACK_DISTANCE),
        "--track-patience", str(TRACK_PATIENCE),
    ]
    if extrinsic_session_id is not None:
        cmd += ["--extrinsic-session-id", str(extrinsic_session_id)]
    print(f"[regen] running: {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=str(REPO)).returncode


def _snapshot_baseline() -> bool:
    if not NPZ.exists():
        print(f"[regen] expected {NPZ} after pipeline run, missing.")
        return False
    shutil.copy2(NPZ, BASELINE_TMP)
    print(f"[regen] snapshot → {BASELINE_TMP} ({BASELINE_TMP.stat().st_size} bytes)")
    return True


def _inspect_npz() -> int:
    if not NPZ.exists():
        return 1
    d = np.load(NPZ)
    print(f"\n=== {NPZ} ===")
    print(f"keys: {list(d.keys())}")
    print(f"shape: {d['keypoints_3d'].shape}")
    for k in (
        "model_backend", "model_name",
        "person_confidence", "max_track_distance", "track_patience",
    ):
        if k in d.files:
            v = d[k]
            print(f"  {k}: {v}")

    finite = np.isfinite(d["keypoints_3d"]).all(axis=-1)
    print("\nper-slot coverage:")
    for p in range(d["keypoints_3d"].shape[1]):
        valid = finite[:, p, :].any(axis=-1)
        if valid.any():
            idx = np.where(valid)[0]
            print(f"  slot {p}: {valid.sum()} frames (idx {idx[0]}..{idx[-1]})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--extrinsic-session-id", type=int, default=None,
        help="Force a specific extrinsic_session id (passed through to runner).",
    )
    args = parser.parse_args()

    rc = _run_pipeline(args.extrinsic_session_id)
    if rc != 0:
        print(f"[regen] pipeline returned exit code {rc}")
        return rc
    if not _snapshot_baseline():
        return 2
    return _inspect_npz()


if __name__ == "__main__":
    sys.exit(main())
