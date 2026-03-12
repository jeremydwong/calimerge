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
    scale_intrinsics,
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
        intrinsics_a.matrix.copy(),
        intrinsics_a.distortion.copy(),
        intrinsics_b.matrix.copy(),
        intrinsics_b.distortion.copy(),
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


# ============================================================================
# End-to-End Pipeline
# ============================================================================


def run_extrinsic_from_videos(
    video_paths: dict[int, "Path"],
    intrinsics: dict[int, CameraIntrinsics],
    charuco_config: "CharucoConfig",
    frame_time_csv: "Path | None" = None,
    sample_interval: int = 10,
    progress_callback: "Callable | None" = None,
    frame_callback: "Callable | None" = None,
) -> tuple[dict[int, CalibratedCamera], float]:
    """
    Run the full extrinsic calibration pipeline from video files.

    Steps:
    1. Detect charuco corners in all videos (every sample_interval frames)
    2. Build SyncedPoints list (frame-for-frame sync, or from CSV if provided)
    3. Compute initial extrinsics via pairwise stereo calibration
    4. Triangulate initial 3D points
    5. Run bundle adjustment for joint optimization

    Args:
        video_paths: Dict of port -> video file path
        intrinsics: Dict of port -> CameraIntrinsics
        charuco_config: ChArUco board configuration
        frame_time_csv: Optional CSV with sync timing (sync_index, port, frame_index, frame_time)
        sample_interval: Process every Nth frame
        progress_callback: Optional callback(fraction: float, message: str)
        frame_callback: Optional callback(port, frame_index, frame_bgr, packet) called per detection

    Returns:
        (calibrated_cameras, rmse)
    """
    from pathlib import Path
    from .charuco import create_charuco_board
    from .intrinsic import detect_charuco_points
    from ..types import FramePoints, SyncedPoints, CalibratedCamera, CameraExtrinsics

    def report(fraction: float, message: str):
        if progress_callback:
            progress_callback(fraction, message)

    ports = sorted(video_paths.keys())
    board = create_charuco_board(charuco_config)

    # Step 1: Detect charuco corners in all videos
    report(0.0, "Step 1: Detecting charuco corners in videos...")

    # per_port_detections[port] = list of (frame_index, PointPacket)
    per_port_detections: dict[int, list[tuple[int, "PointPacket"]]] = {}

    # Scale intrinsics to match video resolution where needed
    intrinsics = dict(intrinsics)  # shallow copy — don't mutate caller's dict
    for port in ports:
        if port not in intrinsics:
            continue
        intr = intrinsics[port]
        probe = cv2.VideoCapture(str(video_paths[port]))
        vid_w = int(probe.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(probe.get(cv2.CAP_PROP_FRAME_HEIGHT))
        probe.release()
        vid_res = (vid_w, vid_h)
        if intr.resolution != vid_res:
            intrinsics[port] = scale_intrinsics(intr, vid_res)
            report(
                0.0,
                f"  Port {port}: scaled intrinsics from {intr.resolution} to {vid_res}",
            )

    total_corners = 0
    for port_i, port in enumerate(ports):
        video_path = video_paths[port]
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        detections = []
        port_corners = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % sample_interval == 0:
                packet = detect_charuco_points(frame, charuco_config, board)
                if packet.point_id is not None and len(packet.point_id) >= 4:
                    detections.append((frame_idx, packet))
                    port_corners += len(packet.point_id)
                    if frame_callback:
                        frame_callback(port, frame_idx, frame, packet)

            frame_idx += 1

        cap.release()
        per_port_detections[port] = detections
        total_corners += port_corners

        fraction = 0.3 * (port_i + 1) / len(ports)
        report(fraction, f"  Port {port}: {len(detections)} frames, {port_corners} corners from {frame_idx} frames")

    report(0.3, f"  Total: {total_corners} corner detections across {len(ports)} cameras")

    # Step 2: Build SyncedPoints
    report(0.3, "Step 2: Building synchronized point correspondences...")

    synced_points_list = _build_synced_points(
        per_port_detections, ports, frame_time_csv, sample_interval
    )

    report(0.35, f"  {len(synced_points_list)} sync frames with shared observations")

    # Report per-camera corner counts in synced frames
    for port in ports:
        frames_with_det = 0
        corner_sum = 0
        for sp in synced_points_list:
            fp = sp.frame_points.get(port)
            if fp is not None and fp.points is not None and fp.points.point_id is not None:
                frames_with_det += 1
                corner_sum += len(fp.points.point_id)
        report(0.35, f"    Port {port}: {frames_with_det} synced frames, {corner_sum} corners")

    # Report shared corners per camera pair
    for i, pa in enumerate(ports):
        for pb in ports[i + 1:]:
            shared_frames = 0
            shared_corners = 0
            for sp in synced_points_list:
                fp_a = sp.frame_points.get(pa)
                fp_b = sp.frame_points.get(pb)
                if (fp_a and fp_a.points and fp_a.points.point_id is not None and
                    fp_b and fp_b.points and fp_b.points.point_id is not None):
                    common = len(set(fp_a.points.point_id.tolist()) & set(fp_b.points.point_id.tolist()))
                    if common >= 4:
                        shared_frames += 1
                        shared_corners += common
            report(0.35, f"    Pair ({pa}, {pb}): {shared_frames} shared frames, {shared_corners} shared corners")

    if len(synced_points_list) < 3:
        raise RuntimeError(
            f"Only {len(synced_points_list)} synced frames with shared charuco "
            "detections. Need at least 3. Check that the board is visible in all cameras."
        )

    # Step 3: Build initial CalibratedCamera dict and compute initial extrinsics
    report(0.4, "Step 3: Computing initial extrinsics via stereo pairs...")

    cameras = {}
    for port in ports:
        cameras[port] = CalibratedCamera(
            serial_number=intrinsics[port].serial_number,
            port=port,
            intrinsics=intrinsics[port],
            extrinsics=CameraExtrinsics(
                rotation=np.eye(3, dtype=np.float64),
                translation=np.zeros(3, dtype=np.float64),
            ),
        )

    initial_extrinsics = compute_initial_extrinsics(synced_points_list, cameras)

    calibrated_ports = set(initial_extrinsics.keys())
    if len(calibrated_ports) < len(ports):
        missing = set(ports) - calibrated_ports
        report(0.45, f"  Warning: Could not calibrate cameras {missing}")

    # Update cameras with initial extrinsics
    for port in ports:
        if port in initial_extrinsics:
            cameras[port] = CalibratedCamera(
                serial_number=intrinsics[port].serial_number,
                port=port,
                intrinsics=intrinsics[port],
                extrinsics=initial_extrinsics[port],
            )

    report(0.5, f"  Initial extrinsics computed for {len(calibrated_ports)} cameras")

    # Step 4: Build point estimates (triangulate initial 3D points)
    report(0.5, "Step 4: Triangulating initial 3D point estimates...")

    point_estimates = build_point_estimates(synced_points_list, cameras)

    report(0.6, f"  {point_estimates.n_obj_points} 3D points, {point_estimates.n_img_points} observations")

    # Step 5: Bundle adjustment
    report(0.6, "Step 5: Running bundle adjustment...")

    refined_cameras, refined_points, rmse = run_bundle_adjustment(
        cameras, point_estimates
    )

    report(1.0, f"  Bundle adjustment complete. RMSE: {rmse:.4f}")

    return refined_cameras, rmse


def _build_synced_points(
    per_port_detections: dict[int, list[tuple[int, "PointPacket"]]],
    ports: list[int],
    frame_time_csv: "Path | None",
    sample_interval: int,
) -> list["SyncedPoints"]:
    """
    Build SyncedPoints from per-port charuco detections.

    If frame_time_csv is provided, uses sync_index from CSV.
    Otherwise, uses frame-for-frame sync: detection N in all cameras = sync_index N
    (based on sampled frame index / sample_interval).
    """
    from ..types import FramePoints, SyncedPoints

    if frame_time_csv is not None:
        return _build_synced_from_csv(per_port_detections, ports, frame_time_csv)

    # Frame-for-frame sync: group by sampled frame index
    # Each detection was at frame_idx where frame_idx % sample_interval == 0
    # Use frame_idx // sample_interval as sync_index

    # Build lookup: port -> {sync_key -> PointPacket}
    port_sync_map: dict[int, dict[int, "PointPacket"]] = {}
    for port in ports:
        lookup = {}
        for frame_idx, packet in per_port_detections[port]:
            sync_key = frame_idx // sample_interval
            lookup[sync_key] = (frame_idx, packet)
        port_sync_map[port] = lookup

    # Find sync keys where at least 2 cameras have detections
    all_keys: set[int] = set()
    for lookup in port_sync_map.values():
        all_keys.update(lookup.keys())

    synced_list = []
    for sync_key in sorted(all_keys):
        frame_points = {}
        cam_count = 0
        for port in ports:
            if sync_key in port_sync_map[port]:
                frame_idx, packet = port_sync_map[port][sync_key]
                frame_points[port] = FramePoints(
                    port=port, frame_index=frame_idx, points=packet
                )
                cam_count += 1
            else:
                frame_points[port] = None

        if cam_count >= 2:
            synced_list.append(SyncedPoints(sync_index=sync_key, frame_points=frame_points))

    return synced_list


def _build_synced_from_csv(
    per_port_detections: dict[int, list[tuple[int, "PointPacket"]]],
    ports: list[int],
    frame_time_csv: "Path",
) -> list["SyncedPoints"]:
    """Build SyncedPoints using sync timing from a frame_time_history CSV."""
    import csv
    from ..types import FramePoints, SyncedPoints

    # Parse CSV: build mapping of (port, frame_index) -> sync_index
    frame_to_sync: dict[tuple[int, int], int] = {}
    with open(frame_time_csv, "r") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].startswith("#"):
                continue
            if row[0] == "sync_index":
                continue  # header
            sync_idx = int(row[0])
            port = int(row[1])
            frame_idx = int(row[2])
            frame_to_sync[(port, frame_idx)] = sync_idx

    # Build lookup: port -> {sync_index -> PointPacket}
    port_sync_map: dict[int, dict[int, tuple[int, "PointPacket"]]] = {}
    for port in ports:
        lookup = {}
        for frame_idx, packet in per_port_detections[port]:
            # Find the nearest sync_index for this frame
            key = (port, frame_idx)
            if key in frame_to_sync:
                sync_idx = frame_to_sync[key]
                lookup[sync_idx] = (frame_idx, packet)
        port_sync_map[port] = lookup

    # Find sync indices where at least 2 cameras have detections
    all_sync_indices: set[int] = set()
    for lookup in port_sync_map.values():
        all_sync_indices.update(lookup.keys())

    synced_list = []
    for sync_idx in sorted(all_sync_indices):
        frame_points = {}
        cam_count = 0
        for port in ports:
            if sync_idx in port_sync_map[port]:
                frame_idx, packet = port_sync_map[port][sync_idx]
                frame_points[port] = FramePoints(
                    port=port, frame_index=frame_idx, points=packet
                )
                cam_count += 1
            else:
                frame_points[port] = None

        if cam_count >= 2:
            synced_list.append(SyncedPoints(sync_index=sync_idx, frame_points=frame_points))

    return synced_list
