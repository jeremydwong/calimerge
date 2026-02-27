"""
Multi-view triangulation functions.

Pure math: projection matrices, SVD triangulation, reprojection.
Adapted from posetrack/cs_parse.py.
"""

from __future__ import annotations

import cv2
import numpy as np


def calculate_projection_matrices(camera_params: list[dict]) -> list[np.ndarray]:
    """
    Calculate 3x4 projection matrices from camera parameter dicts.

    Each dict must have keys: matrix (3x3), rotation (3, rodrigues), translation (3,).
    Returns list of P = K @ [R|t] matrices.
    """
    projection_matrices = []
    for params in camera_params:
        K = params["matrix"]
        rvec = params["rotation"]
        tvec = params["translation"].reshape(3, 1)

        R, _ = cv2.Rodrigues(rvec)
        extrinsic = np.hstack((R, tvec))
        P = K @ extrinsic
        projection_matrices.append(P)

    return projection_matrices


def triangulate_keypoints(
    person_kp_dict: dict[int, np.ndarray],
    port_to_cam_index: dict[int, int],
    camera_params: list[dict],
    projection_matrices: list[np.ndarray],
    confidence_threshold: float = 0.1,
) -> list[np.ndarray | None]:
    """
    Triangulate 3D points for one person from multi-view 2D keypoints.

    Args:
        person_kp_dict: port -> keypoints array (N, 2 or 3). Column 2 is confidence if present.
        port_to_cam_index: port -> index into camera_params/projection_matrices.
        camera_params: list of camera dicts with matrix, distortions.
        projection_matrices: list of 3x4 projection matrices.
        confidence_threshold: skip keypoints below this confidence.

    Returns:
        List of length N, each element is 3D point (3,) or None if not triangulable.
    """
    available_ports = list(person_kp_dict.keys())
    if len(available_ports) < 2:
        return []

    num_keypoints = len(person_kp_dict[available_ports[0]])
    points_3d: list[np.ndarray | None] = [None] * num_keypoints

    available_cam_indices = [
        port_to_cam_index[p] for p in available_ports if p in port_to_cam_index
    ]
    if len(available_cam_indices) < 2:
        return []

    for kp_idx in range(num_keypoints):
        points_2d_undistorted: dict[int, np.ndarray] = {}

        for port in available_ports:
            cam_idx = port_to_cam_index.get(port)
            if cam_idx is None:
                continue

            kp_data = person_kp_dict[port][kp_idx]

            if isinstance(kp_data, (list, np.ndarray)) and len(kp_data) >= 2:
                point_2d_raw = np.array(kp_data[:2], dtype=np.float32).reshape(1, 1, 2)
                confidence = kp_data[2] if len(kp_data) > 2 else 1.0
                if np.isnan(point_2d_raw).any() or confidence < confidence_threshold:
                    continue
            else:
                continue

            K = camera_params[cam_idx]["matrix"]
            dist = camera_params[cam_idx]["distortions"]
            point_2d_undistorted = cv2.undistortPoints(point_2d_raw, K, dist, P=K)
            points_2d_undistorted[cam_idx] = point_2d_undistorted.reshape(2, 1)

        if len(points_2d_undistorted) < 2:
            continue

        # SVD multi-view triangulation
        num_valid = len(points_2d_undistorted)
        A = np.zeros((2 * num_valid, 4))
        valid_cam_indices = list(points_2d_undistorted.keys())

        for i, cam_idx in enumerate(valid_cam_indices):
            P = projection_matrices[cam_idx]
            x, y = points_2d_undistorted[cam_idx].flatten()
            A[2 * i] = x * P[2, :] - P[0, :]
            A[2 * i + 1] = y * P[2, :] - P[1, :]

        try:
            _, _, vh = np.linalg.svd(A)
            point_4d = vh[-1, :]
            if abs(point_4d[3]) < 1e-10:
                continue
            points_3d[kp_idx] = point_4d[:3] / point_4d[3]
        except np.linalg.LinAlgError:
            continue

    return points_3d


def project_3d_to_2d(point_3d: np.ndarray, P: np.ndarray) -> np.ndarray | None:
    """Project a single 3D point to 2D using a 3x4 projection matrix."""
    if point_3d is None or np.isnan(point_3d).any():
        return None
    point_4d = np.append(point_3d, 1.0)
    point_2d_hom = P @ point_4d
    if abs(point_2d_hom[2]) < 1e-6:
        return None
    return (point_2d_hom[:2] / point_2d_hom[2]).flatten()


def project_keypoints_to_all_cameras(
    keypoints_3d: list[np.ndarray | None],
    projection_matrices: list[np.ndarray],
    common_ports: list[int],
    port_to_cam_index: dict[int, int],
) -> dict[int, list[list[float]]]:
    """
    Project 3D keypoints to 2D pixel coordinates for all cameras.
    Vectorized implementation.
    """
    n_keypoints = len(keypoints_3d)
    n_cameras = len(common_ports)

    valid_mask = np.array([kp is not None for kp in keypoints_3d], dtype=bool)
    valid_idx = np.where(valid_mask)[0]

    kp_homo = np.zeros((n_keypoints, 4))
    kp_homo[:, 3] = 1
    if len(valid_idx) > 0:
        kp_homo[valid_idx, :3] = np.array([keypoints_3d[i] for i in valid_idx])

    P_all = np.array([projection_matrices[port_to_cam_index[port]] for port in common_ports])

    # Project to all cameras: (n_cameras, 3, 4) @ (4, n_keypoints)
    all_projected = np.einsum("cij,jk->cik", P_all, kp_homo.T)

    all_pixels = np.full((n_cameras, n_keypoints, 2), np.nan)
    if len(valid_idx) > 0:
        z_vals = all_projected[:, 2, valid_idx]
        for c in range(n_cameras):
            z_valid = z_vals[c] > 1e-6
            final_idx = valid_idx[z_valid]
            if len(final_idx) > 0:
                all_pixels[c, final_idx, 0] = (
                    all_projected[c, 0, final_idx] / all_projected[c, 2, final_idx]
                )
                all_pixels[c, final_idx, 1] = (
                    all_projected[c, 1, final_idx] / all_projected[c, 2, final_idx]
                )

    return {port: all_pixels[i].tolist() for i, port in enumerate(common_ports)}


def calculate_fundamental_matrix(P1: np.ndarray, P2: np.ndarray) -> np.ndarray:
    """Calculate fundamental matrix F from two 3x4 projection matrices."""
    U, S, Vt = np.linalg.svd(P1)
    C1 = Vt[-1, :]
    C1 = C1 / C1[3]

    e2 = P2 @ C1
    e2 = e2 / e2[2]

    e2_cross = np.array([
        [0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0],
    ])

    P1_pinv = np.linalg.pinv(P1)
    F = e2_cross @ P2 @ P1_pinv

    # Enforce rank-2
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0
    F = U @ np.diag(S) @ Vt
    F = F / np.linalg.norm(F)

    return F


def point_to_epipolar_line_distance(
    point1: np.ndarray, point2: np.ndarray, F: np.ndarray
) -> float:
    """Distance from point2 to the epipolar line of point1."""
    p1_homo = np.array([point1[0], point1[1], 1.0])
    p2_homo = np.array([point2[0], point2[1], 1.0])

    l2 = F @ p1_homo
    numerator = abs(np.dot(l2, p2_homo))
    denominator = np.sqrt(l2[0] ** 2 + l2[1] ** 2)

    if denominator < 1e-10:
        return float("inf")
    return numerator / denominator
