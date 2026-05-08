"""
Skeleton3D construction — densify triangulated XYZPoints into a time-stacked array.

The offline pipeline produces one XYZPoints per sync index with a sparse
(point_ids, xyz) pair. Analyzers want a dense (N, P, K, 3) array with a
parallel timestamps (N,). This module is the bridge.

Compatible on-disk with the .npz format in analysis/keypoints_io.py, so
analyzers that already read .npz can consume Skeleton3D without changes.
"""

from __future__ import annotations

import csv
from pathlib import Path
from statistics import median

import numpy as np

from .tracking.registry import SYNTHPOSE_SCHEMA  # re-export for back-compat
from .types import KeypointSchema, Skeleton3D, XYZPoints

__all__ = (
    "SYNTHPOSE_SCHEMA",
    "Skeleton3D",
    "XYZPoints",
    "KeypointSchema",
    "build_skeleton_3d",
    "load_skeleton_3d_npz",
    "load_sync_index_times",
    "save_skeleton_3d_npz",
)


def load_sync_index_times(frame_time_csv: Path) -> dict[int, float]:
    """
    Read frame_time_history.csv → {sync_index: median frame_time across ports}.

    The CSV has one row per (sync_index, port). Cameras are synced, so per-port
    times within a sync index are near-identical — median is robust to outliers.
    """
    by_sync: dict[int, list[float]] = {}
    with open(frame_time_csv, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            sync = int(row["sync_index"])
            by_sync.setdefault(sync, []).append(float(row["frame_time"]))
    return {s: median(ts) for s, ts in by_sync.items()}


def build_skeleton_3d(
    xyz_points: list[XYZPoints],
    schema: KeypointSchema,
    frame_time_csv: Path | None = None,
    fps: float = 30.0,
    track_id: int = 0,
) -> Skeleton3D:
    """
    Stack a list of per-frame XYZPoints into a dense Skeleton3D.

    Parameters
    ----------
    xyz_points
        Ordered list of XYZPoints, one per sync index. Single-person for now;
        see `build_skeleton_3d_multi` for the multi-track form.
    schema
        KeypointSchema whose length determines the K axis of the output.
    frame_time_csv
        Path to frame_time_history.csv. If None, timestamps fall back to
        `sync_index / fps` (coarse but usable for quick tests).
    fps
        Fallback frame rate when frame_time_csv is unavailable.
    track_id
        Stable identifier for this person in the output (default 0).

    Returns
    -------
    Skeleton3D with xyz shape (N, 1, K, 3), NaN where keypoints are missing.
    """
    if not xyz_points:
        return Skeleton3D(
            xyz=np.zeros((0, 1, schema.K, 3), dtype=np.float32),
            timestamps=np.zeros(0, dtype=np.float64),
            sync_indices=np.zeros(0, dtype=np.int64),
            track_ids=(track_id,),
            schema=schema,
        )

    N = len(xyz_points)
    K = schema.K
    xyz = np.full((N, 1, K, 3), np.nan, dtype=np.float32)
    sync_indices = np.empty(N, dtype=np.int64)

    for i, pts in enumerate(xyz_points):
        sync_indices[i] = pts.sync_index
        if pts.point_ids is None or len(pts.point_ids) == 0:
            continue
        ids = np.asarray(pts.point_ids, dtype=np.int64)
        in_range = (ids >= 0) & (ids < K)
        xyz[i, 0, ids[in_range], :] = np.asarray(pts.xyz, dtype=np.float32)[in_range]

    if frame_time_csv is not None:
        time_by_sync = load_sync_index_times(Path(frame_time_csv))
        absolute = np.array(
            [time_by_sync.get(int(s), np.nan) for s in sync_indices],
            dtype=np.float64,
        )
        first_valid = np.argmax(~np.isnan(absolute)) if np.any(~np.isnan(absolute)) else 0
        t0 = absolute[first_valid] if not np.isnan(absolute[first_valid]) else 0.0
        timestamps = absolute - t0
    else:
        timestamps = sync_indices.astype(np.float64) / float(fps)
        timestamps -= timestamps[0]

    return Skeleton3D(
        xyz=xyz,
        timestamps=timestamps,
        sync_indices=sync_indices,
        track_ids=(track_id,),
        schema=schema,
    )


def save_skeleton_3d_npz(path: Path, skeleton: Skeleton3D) -> None:
    """
    Write Skeleton3D to the .npz format used by analysis/keypoints_io.py.

    Round-trips with load_keypoints_3d(). Schema names are stored alongside
    so downstream code doesn't have to assume SynthPose-52.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    person_count = np.any(~np.isnan(skeleton.xyz), axis=(2, 3)).sum(axis=1).astype(np.int32)
    np.savez_compressed(
        path,
        timestamps=skeleton.timestamps,
        keypoints_3d=skeleton.xyz,
        person_count=person_count,
        sync_indices=skeleton.sync_indices,
        track_ids=np.array(skeleton.track_ids, dtype=np.int32),
        schema_names=np.array(skeleton.schema.names),
    )


def load_skeleton_3d_npz(path: Path) -> Skeleton3D:
    """Inverse of save_skeleton_3d_npz, with a fallback for older .npz files."""
    data = np.load(path, allow_pickle=False)
    if "schema_names" in data.files:
        schema = KeypointSchema(names=tuple(str(n) for n in data["schema_names"]))
    else:
        schema = SYNTHPOSE_SCHEMA
    sync_indices = (
        data["sync_indices"] if "sync_indices" in data.files
        else np.arange(len(data["timestamps"]), dtype=np.int64)
    )
    track_ids = (
        tuple(int(t) for t in data["track_ids"]) if "track_ids" in data.files
        else tuple(range(data["keypoints_3d"].shape[1]))
    )
    return Skeleton3D(
        xyz=data["keypoints_3d"].astype(np.float32),
        timestamps=data["timestamps"].astype(np.float64),
        sync_indices=np.asarray(sync_indices, dtype=np.int64),
        track_ids=track_ids,
        schema=schema,
    )
