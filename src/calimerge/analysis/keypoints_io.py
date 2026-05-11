"""
Binary serialization for 3D keypoint time series.

Format: NumPy .npz file with the following arrays:
    timestamps    : (N,)           float64  — seconds since recording start
    keypoints_3d  : (N, P, K, 3)   float32  — per-person per-keypoint xyz (NaN if missing)
    person_count  : (N,)           int32    — number of persons present in each frame

Where:
    N = number of frames recorded
    P = maximum persons per frame (padded, trailing entries may be all-NaN)
    K = number of keypoints per person (17 for COCO-17)

This keeps the file dense and directly loadable as numpy arrays for analysis.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


# SynthPose uses 52 keypoints. COCO uses 17. We store whatever the
# model produces — the default here should match the active model.
DEFAULT_NUM_KEYPOINTS = 52


def _orthonormalize_rotation(R: np.ndarray) -> np.ndarray:
    """Project a near-rotation 3x3 matrix to the closest proper rotation.

    Same Kabsch projection used by keypoint_export.write_raw_buffer so the
    two npz files agree to floating-point precision when given the same
    (R, t).
    """
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    U, _, Vt = np.linalg.svd(R)
    R_ortho = U @ Vt
    if np.linalg.det(R_ortho) < 0:
        U[:, -1] *= -1
        R_ortho = U @ Vt
    return R_ortho


def save_keypoints_3d(
    path: Path,
    frames: list[dict],
    num_keypoints: int = DEFAULT_NUM_KEYPOINTS,
    max_persons: int = 4,
    primary_person_index: int = 0,
    view_rotation: np.ndarray | None = None,
    view_translation: np.ndarray | None = None,
    model_backend: str | None = None,
    model_name: str | None = None,
    person_confidence: float | None = None,
    max_track_distance: float | None = None,
    track_patience: int | None = None,
    extrinsic_session_id: int | None = None,
    extrinsic_created_at: str | None = None,
) -> None:
    """
    Save a list of per-frame keypoint records to a .npz file.

    Parameters
    ----------
    path : Path
        Output .npz path.
    frames : list of dict
        Each dict has:
            "time": float     — seconds since recording start
            "persons": list   — list of persons; each person is a list of
                                 (x, y, z) tuples or np.ndarray(3,) or None
            "primary_index": int (optional) — per-frame primary person index
        Persons here are in CAMERA frame — the live keypoint emitter
        (PoseDetectionWorker / CudaStreamDetectionWorker) writes raw
        triangulated points.
    num_keypoints : int
        Number of keypoints per person (default 52 for SynthPose).
    max_persons : int
        Maximum persons to store per frame. Extras are dropped.
    primary_person_index : int
        Index of the person closest to the calibrated origin (the exercise
        subject).  Stored as a scalar in the npz for downstream analysis.
    view_rotation, view_translation : np.ndarray | None, optional
        Camera->view transform snapshotted at record-time. When provided,
        every keypoint is mapped via ``p_view = R @ p_cam + t`` before
        being saved, and the transform is recorded in the npz under
        ``view_transform_R`` (3x3) and ``view_transform_t`` (3,) so a
        consumer that wants camera coords can invert via
        ``p_cam = R.T @ (p_view - t)``. R is re-orthonormalised via SVD
        before being applied + saved. When omitted, keypoints are
        written as-is and an identity transform is recorded — needed for
        the test_output.ipynb notebook's ankle plot to match the live
        body-frame readout.
    """
    n_frames = len(frames)
    if n_frames == 0:
        return

    timestamps = np.zeros(n_frames, dtype=np.float64)
    keypoints = np.full(
        (n_frames, max_persons, num_keypoints, 3),
        np.nan, dtype=np.float32,
    )
    person_counts = np.zeros(n_frames, dtype=np.int32)
    primary_indices = np.zeros(n_frames, dtype=np.int32)

    # Resolve the view transform once. Default = identity so legacy
    # callers that pass nothing get the old camera-frame behaviour.
    if view_rotation is not None:
        R_view = _orthonormalize_rotation(view_rotation).astype(np.float64)
    else:
        R_view = np.eye(3, dtype=np.float64)
    if view_translation is not None:
        t_view = np.asarray(view_translation, dtype=np.float64).reshape(3)
    else:
        t_view = np.zeros(3, dtype=np.float64)
    apply_transform = not (
        np.allclose(R_view, np.eye(3)) and np.allclose(t_view, 0.0)
    )

    for i, frame in enumerate(frames):
        timestamps[i] = frame.get("time", 0.0)
        persons = frame.get("persons", [])
        n_persons = min(len(persons), max_persons)
        person_counts[i] = n_persons
        primary_indices[i] = frame.get("primary_index", primary_person_index)

        for p_idx in range(n_persons):
            person = persons[p_idx]
            if person is None:
                continue
            for k_idx in range(min(len(person), num_keypoints)):
                kp = person[k_idx]
                if kp is None:
                    continue
                try:
                    arr = np.asarray(kp, dtype=np.float32)
                except Exception:
                    continue
                if arr.shape != (3,):
                    continue
                if apply_transform:
                    pt = R_view @ arr.astype(np.float64) + t_view
                    keypoints[i, p_idx, k_idx, :] = pt.astype(np.float32)
                else:
                    keypoints[i, p_idx, k_idx, :] = arr

    # Detection-pipeline parameters. NaN sentinel for unspecified, so
    # the comparator can tell "old npz that pre-dates this field" from
    # "explicitly recorded value of zero".
    _pc = float(person_confidence) if person_confidence is not None else float("nan")
    _mtd = float(max_track_distance) if max_track_distance is not None else float("nan")
    _tp = int(track_patience) if track_patience is not None else -1

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        timestamps=timestamps,
        keypoints_3d=keypoints,
        person_count=person_counts,
        primary_person_index=np.array(primary_indices, dtype=np.int32),
        view_transform_R=R_view.astype(np.float64),
        view_transform_t=t_view.astype(np.float64),
        # Detection provenance: which backend + model produced these
        # keypoints. Stored as 0-d unicode arrays — empty string when
        # the caller didn't pass it (legacy compatibility).
        model_backend=np.array(model_backend or "", dtype="<U32"),
        model_name=np.array(model_name or "", dtype="<U64"),
        person_confidence=np.float32(_pc),
        max_track_distance=np.float32(_mtd),
        track_patience=np.int32(_tp),
        **({} if extrinsic_session_id is None else
           {"extrinsic_session_id": np.array(extrinsic_session_id, dtype=np.int32)}),
        **({} if extrinsic_created_at is None else
           {"extrinsic_created_at": np.array(extrinsic_created_at, dtype="<U32")}),
    )


def load_keypoints_3d(path: Path) -> dict:
    """
    Load a keypoints .npz file.

    Returns
    -------
    dict with keys: "timestamps", "keypoints_3d", "person_count",
    and optionally "primary_person_index".
    """
    data = np.load(path)
    result = {
        "timestamps": data["timestamps"],
        "keypoints_3d": data["keypoints_3d"],
        "person_count": data["person_count"],
    }
    if "primary_person_index" in data:
        result["primary_person_index"] = data["primary_person_index"]
    return result


def extract_hip_z_series(
    keypoints_3d: np.ndarray,
    person_index: int = 0,
    body_rotation: np.ndarray | None = None,
    body_origin: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract hip midpoint Z (vertical) for a single person over time.

    Parameters
    ----------
    keypoints_3d : (N, P, K, 3) array
        Per-frame per-person keypoint array (NaN = missing).
    person_index : int
        Which person to track (default 0).
    body_rotation : (3, 3) array, optional
        World-to-body rotation. If given, hip coordinates are rotated and
        translated into the body frame (Z = up).
    body_origin : (3,) array, optional
        World-space origin of the body frame (typically L_ankle position).

    Returns
    -------
    valid_indices : (M,) array of frame indices where hip was available
    hip_z : (M,) array of hip Z values in the (optionally body-centred) frame
    """
    N = keypoints_3d.shape[0]
    # COCO-17 indices
    L_HIP, R_HIP = 11, 12

    valid_indices = []
    hip_z_values = []

    for i in range(N):
        if keypoints_3d.shape[1] <= person_index:
            continue
        l_hip = keypoints_3d[i, person_index, L_HIP]
        r_hip = keypoints_3d[i, person_index, R_HIP]

        l_ok = not np.isnan(l_hip).any()
        r_ok = not np.isnan(r_hip).any()

        if l_ok and r_ok:
            hip = (l_hip + r_hip) / 2
        elif l_ok:
            hip = l_hip
        elif r_ok:
            hip = r_hip
        else:
            continue

        if body_rotation is not None and body_origin is not None:
            hip = body_rotation @ (hip - body_origin)

        valid_indices.append(i)
        hip_z_values.append(float(hip[2]))

    return np.array(valid_indices, dtype=np.int64), np.array(hip_z_values, dtype=np.float64)
