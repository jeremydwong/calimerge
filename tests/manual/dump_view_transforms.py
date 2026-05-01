"""Headless dump of <app_data>/models/view_transforms.db.

Lists every saved per-model rotate-to-human + zero-origin preset so you
can verify what the GUI is actually persisting (and pick up
inconsistencies between PyTorch and CUDA backends — the underlying
storage is the same regardless of backend, and is keyed on
detect_model_combo's `currentData()` value, not the backend).

Usage:
    uv run python tests/manual/dump_view_transforms.py
"""

from __future__ import annotations

import numpy as np

from calimerge.config import (
    view_transforms_db_path,
    load_view_transform,
)


KNOWN_MODEL_KEYS = ("vitpose", "mediapipe_hands")


def _det(R: np.ndarray) -> float:
    return float(np.linalg.det(R))


def _orthonormality_error(R: np.ndarray) -> float:
    return float(np.linalg.norm(R.T @ R - np.eye(3)))


def main() -> int:
    db = view_transforms_db_path()
    print(f"DB path: {db}")
    print(f"DB exists: {db.exists()}")
    print()

    if not db.exists():
        print(
            "DB has not been created yet — neither Rotate-to-Human nor "
            "Zero-at-* has been pressed for any model."
        )
        return 0

    # Walk the well-known model keys. Anything else (future models) won't
    # be listed here, but the table itself is keyed by string so the
    # GUI's lookup never relies on this list.
    for key in KNOWN_MODEL_KEYS:
        loaded = load_view_transform(key)
        print(f"-- model_key = {key!r} " + "-" * 40)
        if loaded is None:
            print("  (no row - Rotate-to-Human / Zero has not been pressed "
                  "for this model)")
            print()
            continue
        R, t, has_origin = loaded
        print(f"  has_origin: {has_origin}")
        print(f"  det(R): {_det(R):+.9f}  (should be ~ +1.0)")
        print(f"  ||R^T R - I||: {_orthonormality_error(R):.3e}  "
              f"(should be ~ 0.0)")
        print(f"  R =")
        for row in R:
            print(f"    [{row[0]:+.6f} {row[1]:+.6f} {row[2]:+.6f}]")
        print(f"  t = [{t[0]:+.6f} {t[1]:+.6f} {t[2]:+.6f}]")
        if has_origin:
            # X0 = -R^T t in camera/world frame.
            X0 = -R.T @ t
            print(f"  X0 (origin in camera frame, = -R^T t): "
                  f"[{X0[0]:+.6f} {X0[1]:+.6f} {X0[2]:+.6f}]")
        print()

    print(
        "Note: this preset is applied identically by the PyTorch and CUDA\n"
        "backends - the snapshot is taken at record-time inside\n"
        "WorkoutPage._begin_recording_now and forwarded to write_raw_buffer\n"
        "regardless of which detection worker produced the keypoints.\n"
        "If the rotation/zero looks wrong only on CUDA, check\n"
        "WorkoutPage._on_model_changed: backend changes used to call\n"
        "skeleton_view.clear() which silently reset the view transform,\n"
        "so the live skeleton DISPLAY reverted to camera frame even though\n"
        "the saved npz was correctly rotated. That bug is fixed in\n"
        "skeleton_view.clear().\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
