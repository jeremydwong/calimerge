"""
Extrinsic camera calibration and bundle adjustment.

Pure functions - no classes, no state.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from ..types import (
    CameraExtrinsics,
    CameraIntrinsics,
    CalibratedCamera,
    PointPacket,
    SyncedPoints,
    extrinsics_from_vector,
    extrinsics_to_vector,
)


# ============================================================================
# Data Structures for Bundle Adjustment
# ============================================================================


@dataclass
class PointEstimates:
    """
    Data structure for bundle adjustment optimization.

    Holds 2D image observations and their corresponding 3D point estimates.
    """

    sync_indices: np.ndarray  # (n,) sync index for each 2D observation
    camera_indices: np.ndarray  # (n,) camera port for each 2D observation
    point_ids: np.ndarray  # (n,) point ID for each 2D observation
    img_points: np.ndarray  # (n, 2) 2D image coordinates
    obj_indices: np.ndarray  # (n,) index into obj_points for each observation
    obj_points: np.ndarray  # (m, 3) 3D point estimates

    @property
    def n_cameras(self) -> int:
        return len(np.unique(self.camera_indices))

    @property
    def n_obj_points(self) -> int:
        return self.obj_points.shape[0]

    @property
    def n_img_points(self) -> int:
        return self.img_points.shape[0]


# ============================================================================
# Stereo Calibration
# ============================================================================


def stereo_calibrate_pair(
    synced_points_list: list[SyncedPoints],
    intrinsics_a: CameraIntrinsics,
    intrinsics_b: CameraIntrinsics,
    port_a: int,
    port_b: int,
    min_corners: int = 6,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """
    Stereo calibrate a camera pair using shared ChArUco observations.

    Args:
        synced_points_list: List of SyncedPoints with ChArUco detections
        intrinsics_a: Intrinsics for camera A
        intrinsics_b: Intrinsics for camera B
        port_a: Port number for camera A
        port_b: Port number for camera B
        min_corners: Minimum shared corners required per frame

    Returns:
        (rotation_3x3, translation_3, rmse) of camera B relative to camera A,
        or None if insufficient shared data
    """
    # Collect matching observations
    obj_points_list = []
    img_points_a_list = []
    img_points_b_list = []

    for synced in synced_points_list:
        if port_a not in synced.frame_points or port_b not in synced.frame_points:
            continue

        fp_a = synced.frame_points.get(port_a)
        fp_b = synced.frame_points.get(port_b)

        if fp_a is None or fp_b is None:
            continue
        if fp_a.points is None or fp_b.points is None:
            continue
        if fp_a.points.obj_loc is None or fp_b.points.obj_loc is None:
            continue

        # Find common point IDs
        ids_a = set(fp_a.points.point_id.tolist())
        ids_b = set(fp_b.points.point_id.tolist())
        common_ids = ids_a & ids_b

        if len(common_ids) < min_corners:
            continue

        # Extract matching points
        obj_pts = []
        img_a = []
        img_b = []

        for pt_id in common_ids:
            idx_a = np.where(fp_a.points.point_id == pt_id)[0][0]
            idx_b = np.where(fp_b.points.point_id == pt_id)[0][0]

            obj_pts.append(fp_a.points.obj_loc[idx_a])
            img_a.append(fp_a.points.img_loc[idx_a])
            img_b.append(fp_b.points.img_loc[idx_b])

        obj_points_list.append(np.array(obj_pts, dtype=np.float32))
        img_points_a_list.append(np.array(img_a, dtype=np.float32))
        img_points_b_list.append(np.array(img_b, dtype=np.float32))

    if len(obj_points_list) < 3:
        return None

    # Run OpenCV stereo calibration
    flags = cv2.CALIB_FIX_INTRINSIC
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 1e-6)

    ret, _, _, _, _, R, T, _, _ = cv2.stereoCalibrate(
        obj_points_list,
        img_points_a_list,
        img_points_b_list,
        intrinsics_a.matrix,
        intrinsics_a.distortion,
        intrinsics_b.matrix,
        intrinsics_b.distortion,
        imageSize=None,
        criteria=criteria,
        flags=flags,
    )

    return R, T[:, 0], ret


def compute_initial_extrinsics(
    synced_points_list: list[SyncedPoints],
    cameras: dict[int, CalibratedCamera],
    reference_port: int | None = None,
) -> dict[int, CameraExtrinsics]:
    """
    Compute initial extrinsics for all cameras via pairwise stereo calibration.

    Uses the reference camera (or first camera) as the origin, then chains
    stereo calibrations to estimate other camera poses.

    Args:
        synced_points_list: List of SyncedPoints with ChArUco detections
        cameras: Dict of port -> CalibratedCamera (intrinsics only needed)
        reference_port: Port to use as origin (default: lowest port number)

    Returns:
        Dict of port -> CameraExtrinsics
    """
    ports = sorted(cameras.keys())

    if reference_port is None:
        reference_port = ports[0]

    # Initialize reference camera at origin
    extrinsics = {
        reference_port: CameraExtrinsics(
            rotation=np.eye(3, dtype=np.float64),
            translation=np.zeros(3, dtype=np.float64),
        )
    }

    # Chain stereo calibrations from reference
    calibrated = {reference_port}
    pairs = list(combinations(ports, 2))

    # Keep trying until we can't calibrate any more cameras
    changed = True
    while changed:
        changed = False
        for port_a, port_b in pairs:
            # Skip if both calibrated or neither calibrated
            if port_a in calibrated and port_b in calibrated:
                continue
            if port_a not in calibrated and port_b not in calibrated:
                continue

            # Determine which camera is the anchor
            if port_a in calibrated:
                anchor, target = port_a, port_b
            else:
                anchor, target = port_b, port_a

            # Stereo calibrate
            result = stereo_calibrate_pair(
                synced_points_list,
                cameras[anchor].intrinsics,
                cameras[target].intrinsics,
                anchor,
                target,
            )

            if result is None:
                continue

            R_rel, t_rel, _ = result

            # Compute absolute pose of target camera
            R_anchor = extrinsics[anchor].rotation
            t_anchor = extrinsics[anchor].translation

            R_target = R_rel @ R_anchor
            t_target = R_rel @ t_anchor + t_rel

            extrinsics[target] = CameraExtrinsics(
                rotation=R_target,
                translation=t_target,
            )
            calibrated.add(target)
            changed = True

    return extrinsics


# ============================================================================
# Bundle Adjustment
# ============================================================================


def build_point_estimates(
    synced_points_list: list[SyncedPoints],
    cameras: dict[int, CalibratedCamera],
) -> PointEstimates:
    """
    Build PointEstimates from synchronized point observations.

    Triangulates initial 3D point positions from the camera poses.

    Args:
        synced_points_list: List of SyncedPoints
        cameras: Dict of port -> CalibratedCamera (with extrinsics)

    Returns:
        PointEstimates for bundle adjustment
    """
    from ..triangulation import triangulate_points

    # Collect all observations
    sync_indices = []
    camera_indices = []
    point_ids = []
    img_points = []

    for synced in synced_points_list:
        for port, fp in synced.frame_points.items():
            if fp is None or fp.points is None:
                continue
            if port not in cameras:
                continue

            n = len(fp.points.point_id)
            sync_indices.extend([synced.sync_index] * n)
            camera_indices.extend([port] * n)
            point_ids.extend(fp.points.point_id.tolist())
            img_points.extend(fp.points.img_loc.tolist())

    sync_indices = np.array(sync_indices, dtype=np.int32)
    camera_indices = np.array(camera_indices, dtype=np.int32)
    point_ids = np.array(point_ids, dtype=np.int32)
    img_points = np.array(img_points, dtype=np.float64)

    # Create unique (sync_index, point_id) combinations for 3D points
    unique_combos = {}
    obj_indices = np.zeros(len(sync_indices), dtype=np.int32)

    for i, (sync_idx, pt_id) in enumerate(zip(sync_indices, point_ids)):
        key = (int(sync_idx), int(pt_id))
        if key not in unique_combos:
            unique_combos[key] = len(unique_combos)
        obj_indices[i] = unique_combos[key]

    # Triangulate 3D points
    obj_points = np.zeros((len(unique_combos), 3), dtype=np.float64)

    for (sync_idx, pt_id), obj_idx in unique_combos.items():
        # Collect observations for this point
        mask = (sync_indices == sync_idx) & (point_ids == pt_id)
        obs_cameras = camera_indices[mask]
        obs_img = img_points[mask]

        if len(obs_cameras) >= 2:
            # Simple triangulation using first two cameras
            xyz = triangulate_points(
                {port: cameras[port] for port in obs_cameras},
                obs_cameras,
                obs_img,
            )
            if xyz is not None:
                obj_points[obj_idx] = xyz

    return PointEstimates(
        sync_indices=sync_indices,
        camera_indices=camera_indices,
        point_ids=point_ids,
        img_points=img_points,
        obj_indices=obj_indices,
        obj_points=obj_points,
    )


def _get_sparsity_pattern(
    point_estimates: PointEstimates,
    n_cameras: int,
) -> lil_matrix:
    """
    Build sparse Jacobian pattern for least_squares.
    """
    CAMERA_PARAM_COUNT = 6

    m = point_estimates.n_img_points * 2  # 2 residuals per observation (x, y)
    n = n_cameras * CAMERA_PARAM_COUNT + point_estimates.n_obj_points * 3

    A = lil_matrix((m, n), dtype=int)

    i = np.arange(point_estimates.n_img_points)

    # Camera parameters affect their observations
    for s in range(CAMERA_PARAM_COUNT):
        A[2 * i, point_estimates.camera_indices * CAMERA_PARAM_COUNT + s] = 1
        A[2 * i + 1, point_estimates.camera_indices * CAMERA_PARAM_COUNT + s] = 1

    # 3D point parameters affect their observations
    offset = n_cameras * CAMERA_PARAM_COUNT
    for s in range(3):
        A[2 * i, offset + point_estimates.obj_indices * 3 + s] = 1
        A[2 * i + 1, offset + point_estimates.obj_indices * 3 + s] = 1

    return A


def _xy_reprojection_error(
    params: np.ndarray,
    point_estimates: PointEstimates,
    cameras: dict[int, CalibratedCamera],
    port_to_idx: dict[int, int],
) -> np.ndarray:
    """
    Compute reprojection error for bundle adjustment.
    """
    CAMERA_PARAM_COUNT = 6
    n_cameras = len(port_to_idx)

    # Unpack camera parameters
    camera_params = params[: n_cameras * CAMERA_PARAM_COUNT].reshape(
        n_cameras, CAMERA_PARAM_COUNT
    )

    # Unpack 3D points
    points_3d = params[n_cameras * CAMERA_PARAM_COUNT :].reshape(-1, 3)

    # Compute reprojections per camera
    projected = np.zeros((point_estimates.n_img_points, 2), dtype=np.float64)

    for port, cam in cameras.items():
        port_idx = port_to_idx[port]
        mask = point_estimates.camera_indices == port

        if not np.any(mask):
            continue

        obj_pts = points_3d[point_estimates.obj_indices[mask]]

        rvec = camera_params[port_idx, 0:3]
        tvec = camera_params[port_idx, 3:6]

        proj, _ = cv2.projectPoints(
            obj_pts,
            rvec,
            tvec,
            cam.intrinsics.matrix,
            cam.intrinsics.distortion,
        )

        projected[mask] = proj[:, 0, :]

    # Compute error
    error = (projected - point_estimates.img_points).ravel()
    return error


def run_bundle_adjustment(
    cameras: dict[int, CalibratedCamera],
    point_estimates: PointEstimates,
    fix_first_camera: bool = True,
) -> tuple[dict[int, CalibratedCamera], PointEstimates, float]:
    """
    Run bundle adjustment to refine camera extrinsics and 3D point estimates.

    Args:
        cameras: Dict of port -> CalibratedCamera
        point_estimates: Initial point estimates
        fix_first_camera: If True, first camera stays at origin

    Returns:
        (refined_cameras, refined_point_estimates, final_rmse)
    """
    CAMERA_PARAM_COUNT = 6

    # Build port index mapping
    ports = sorted(cameras.keys())
    port_to_idx = {port: idx for idx, port in enumerate(ports)}
    n_cameras = len(ports)

    # Build initial parameter vector
    camera_params = np.zeros((n_cameras, CAMERA_PARAM_COUNT), dtype=np.float64)
    for port, cam in cameras.items():
        idx = port_to_idx[port]
        camera_params[idx] = extrinsics_to_vector(cam.extrinsics)

    initial_params = np.hstack([
        camera_params.ravel(),
        point_estimates.obj_points.ravel(),
    ])

    # Build sparsity pattern
    sparsity = _get_sparsity_pattern(point_estimates, n_cameras)

    # Run optimization
    result = least_squares(
        _xy_reprojection_error,
        initial_params,
        jac_sparsity=sparsity,
        verbose=0,
        x_scale="jac",
        loss="linear",
        ftol=1e-8,
        method="trf",
        args=(point_estimates, cameras, port_to_idx),
    )

    # Unpack results
    optimized_camera_params = result.x[: n_cameras * CAMERA_PARAM_COUNT].reshape(
        n_cameras, CAMERA_PARAM_COUNT
    )
    optimized_points = result.x[n_cameras * CAMERA_PARAM_COUNT :].reshape(-1, 3)

    # Build new cameras
    refined_cameras = {}
    for port, cam in cameras.items():
        idx = port_to_idx[port]
        extrinsics = extrinsics_from_vector(optimized_camera_params[idx])
        refined_cameras[port] = CalibratedCamera(
            serial_number=cam.serial_number,
            port=port,
            intrinsics=cam.intrinsics,
            extrinsics=extrinsics,
        )

    # Build new point estimates
    refined_point_estimates = PointEstimates(
        sync_indices=point_estimates.sync_indices,
        camera_indices=point_estimates.camera_indices,
        point_ids=point_estimates.point_ids,
        img_points=point_estimates.img_points,
        obj_indices=point_estimates.obj_indices,
        obj_points=optimized_points,
    )

    # Compute final RMSE
    final_error = result.fun.reshape(-1, 2)
    rmse = float(np.sqrt(np.mean(np.sum(final_error**2, axis=1))))

    return refined_cameras, refined_point_estimates, rmse


def compute_reprojection_rmse(
    cameras: dict[int, CalibratedCamera],
    point_estimates: PointEstimates,
) -> dict[str, float]:
    """
    Compute RMSE of reprojection error.

    Args:
        cameras: Dict of port -> CalibratedCamera
        point_estimates: Point estimates

    Returns:
        Dict with 'overall' RMSE and per-camera RMSE
    """
    ports = sorted(cameras.keys())
    port_to_idx = {port: idx for idx, port in enumerate(ports)}

    # Build parameter vector from current state
    CAMERA_PARAM_COUNT = 6
    n_cameras = len(ports)

    camera_params = np.zeros((n_cameras, CAMERA_PARAM_COUNT), dtype=np.float64)
    for port, cam in cameras.items():
        idx = port_to_idx[port]
        camera_params[idx] = extrinsics_to_vector(cam.extrinsics)

    params = np.hstack([
        camera_params.ravel(),
        point_estimates.obj_points.ravel(),
    ])

    # Compute error
    error = _xy_reprojection_error(params, point_estimates, cameras, port_to_idx)
    error = error.reshape(-1, 2)
    euclidean = np.sqrt(np.sum(error**2, axis=1))

    rmse = {"overall": float(np.sqrt(np.mean(euclidean**2)))}

    # Per-camera RMSE
    for port in ports:
        mask = point_estimates.camera_indices == port
        if np.any(mask):
            cam_error = euclidean[mask]
            rmse[str(port)] = float(np.sqrt(np.mean(cam_error**2)))

    return rmse
