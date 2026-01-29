"""
Triangulation functions for 3D reconstruction.

Preserves Numba-optimized code from caliscope/Anipose.
Pure functions operating on dataclasses.
"""

from __future__ import annotations

import numpy as np
from numba import jit
from numba.typed import Dict, List

from .types import (
    CalibratedCamera,
    CameraIntrinsics,
    SyncedPoints,
    XYZPoints,
    compute_projection_matrix,
)


# ============================================================================
# Numba Helpers
# ============================================================================


@jit(nopython=True, cache=True)
def _unique_with_counts(arr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Helper to avoid np.unique(return_counts=True) which doesn't work with jit.
    """
    sorted_arr = np.sort(arr)
    unique_values = [sorted_arr[0]]
    counts = [1]

    for i in range(1, len(sorted_arr)):
        if sorted_arr[i] != sorted_arr[i - 1]:
            unique_values.append(sorted_arr[i])
            counts.append(1)
        else:
            counts[-1] += 1

    return np.array(unique_values), np.array(counts)


# ============================================================================
# Core Triangulation (adapted from Anipose)
# ============================================================================

#####################################################################################
# The following code is adapted from the `Anipose` project,
# in particular the `triangulate_simple` function of `aniposelib`
# Original author:  Lili Karashchuk
# Project: https://github.com/lambdaloop/aniposelib/
# Original Source Code:
#   https://github.com/lambdaloop/aniposelib/blob/d03b485c4e178d7cff076e9fe1ac36837db49158/aniposelib/cameras.py#L21
# This code is licensed under the BSD 2-Clause License
#
# BSD 2-Clause License
#
# Copyright (c) 2019, Lili Karashchuk
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


@jit(nopython=True, parallel=True, cache=True)
def _triangulate_sync_index(
    projection_matrices: Dict,
    camera_indices: np.ndarray,
    point_ids: np.ndarray,
    img_xy: np.ndarray,
) -> tuple[List, List]:
    """
    Triangulate 3D points from 2D observations at a single sync index.

    Uses Direct Linear Transform (DLT) via SVD.

    Args:
        projection_matrices: Numba Dict of port -> 3x4 projection matrix
        camera_indices: (n,) array of camera ports for each observation
        point_ids: (n,) array of point IDs for each observation
        img_xy: (n, 2) array of 2D image coordinates

    Returns:
        (point_ids_3d, xyz_points) as Numba Lists
    """
    point_indices_xyz = List()
    obj_xyz = List()

    unique_points, point_counts = _unique_with_counts(point_ids)

    for index in range(len(point_counts)):
        if point_counts[index] > 1:
            # This point was seen by multiple cameras - triangulate it
            point = unique_points[index]
            mask = point_ids == point
            points_xy = img_xy[mask]
            camera_ids = camera_indices[mask]

            num_cams = len(camera_ids)
            A = np.zeros((num_cams * 2, 4))

            for i in range(num_cams):
                x, y = points_xy[i]
                P = projection_matrices[camera_ids[i]]
                A[(i * 2) : (i * 2 + 1)] = x * P[2] - P[0]
                A[(i * 2 + 1) : (i * 2 + 2)] = y * P[2] - P[1]

            # SVD to find null space
            u, s, vh = np.linalg.svd(A, full_matrices=True)
            point_xyzw = vh[-1]
            point_xyz = point_xyzw[:3] / point_xyzw[3]

            point_indices_xyz.append(point)
            obj_xyz.append(point_xyz)

    return point_indices_xyz, obj_xyz


# End of adapted code
#####################################################################################


# ============================================================================
# Undistortion
# ============================================================================


def undistort_points(
    points: np.ndarray,
    intrinsics: CameraIntrinsics,
    iterations: int = 3,
) -> np.ndarray:
    """
    Undistort 2D points using camera intrinsics.

    Uses iterative algorithm for better accuracy than cv2.undistortPoints.
    Based on: https://yangyushi.github.io/code/2020/03/04/opencv-undistort.html

    Args:
        points: (n, 2) array of distorted image coordinates
        intrinsics: Camera intrinsics with distortion coefficients
        iterations: Number of refinement iterations

    Returns:
        (n, 2) array of undistorted image coordinates
    """
    if points.size == 0:
        return points.copy()

    k1, k2, p1, p2, k3 = intrinsics.distortion[:5]
    fx, fy = intrinsics.matrix[0, 0], intrinsics.matrix[1, 1]
    cx, cy = intrinsics.matrix[0, 2], intrinsics.matrix[1, 2]

    x = (points[:, 0] - cx) / fx
    y = (points[:, 1] - cy) / fy
    x0, y0 = x.copy(), y.copy()

    for _ in range(iterations):
        r2 = x**2 + y**2
        k_inv = 1 / (1 + k1 * r2 + k2 * r2**2 + k3 * r2**3)
        delta_x = 2 * p1 * x * y + p2 * (r2 + 2 * x**2)
        delta_y = p1 * (r2 + 2 * y**2) + 2 * p2 * x * y
        x = (x0 - delta_x) * k_inv
        y = (y0 - delta_y) * k_inv

    undistorted = np.column_stack([x * fx + cx, y * fy + cy])
    return undistorted


# ============================================================================
# High-Level API
# ============================================================================


def triangulate_frame(
    cameras: dict[int, CalibratedCamera],
    synced_points: SyncedPoints,
) -> XYZPoints:
    """
    Triangulate all points from a single synchronized frame set.

    Args:
        cameras: Dict of port -> CalibratedCamera
        synced_points: 2D points from all cameras at one sync index

    Returns:
        XYZPoints with triangulated 3D positions
    """
    # Build projection matrix dict for Numba
    proj_matrices = Dict()
    for port, cam in cameras.items():
        proj_matrices[port] = compute_projection_matrix(cam)

    # Collect all observations with undistortion
    camera_indices = []
    point_ids = []
    img_xy = []

    for port, fp in synced_points.frame_points.items():
        if fp is None or fp.points is None:
            continue
        if fp.points.point_id is None or fp.points.point_id.size == 0:
            continue
        if port not in cameras:
            continue

        # Undistort points
        undist = undistort_points(fp.points.img_loc, cameras[port].intrinsics)

        camera_indices.extend([port] * len(fp.points.point_id))
        point_ids.extend(fp.points.point_id.tolist())
        img_xy.extend(undist.tolist())

    if len(point_ids) == 0:
        return XYZPoints(
            sync_index=synced_points.sync_index,
            point_ids=np.array([], dtype=np.int32),
            xyz=np.array([], dtype=np.float64).reshape(0, 3),
        )

    camera_indices = np.array(camera_indices, dtype=np.int32)
    point_ids = np.array(point_ids, dtype=np.int32)
    img_xy = np.array(img_xy, dtype=np.float64)

    # Triangulate
    result_ids, result_xyz = _triangulate_sync_index(
        proj_matrices, camera_indices, point_ids, img_xy
    )

    if len(result_ids) == 0:
        return XYZPoints(
            sync_index=synced_points.sync_index,
            point_ids=np.array([], dtype=np.int32),
            xyz=np.array([], dtype=np.float64).reshape(0, 3),
        )

    return XYZPoints(
        sync_index=synced_points.sync_index,
        point_ids=np.array(list(result_ids), dtype=np.int32),
        xyz=np.array(list(result_xyz), dtype=np.float64),
    )


def triangulate_points(
    cameras: dict[int, CalibratedCamera],
    camera_indices: np.ndarray,
    img_points: np.ndarray,
) -> np.ndarray | None:
    """
    Triangulate a single 3D point from multiple camera observations.

    This is a simpler interface for triangulating when you already
    have the observations collected.

    Args:
        cameras: Dict of port -> CalibratedCamera
        camera_indices: (n,) array of camera ports
        img_points: (n, 2) array of 2D image coordinates (already undistorted)

    Returns:
        (3,) array of 3D coordinates, or None if < 2 observations
    """
    if len(camera_indices) < 2:
        return None

    # Build projection matrices
    proj_matrices = {}
    for port in np.unique(camera_indices):
        if port in cameras:
            proj_matrices[port] = compute_projection_matrix(cameras[port])

    # Build DLT matrix
    num_cams = len(camera_indices)
    A = np.zeros((num_cams * 2, 4), dtype=np.float64)

    for i, (port, (x, y)) in enumerate(zip(camera_indices, img_points)):
        if port not in proj_matrices:
            continue
        P = proj_matrices[port]
        A[i * 2] = x * P[2] - P[0]
        A[i * 2 + 1] = y * P[2] - P[1]

    # SVD to find null space
    try:
        _, _, vh = np.linalg.svd(A, full_matrices=True)
        point_xyzw = vh[-1]
        point_xyz = point_xyzw[:3] / point_xyzw[3]
        return point_xyz
    except np.linalg.LinAlgError:
        return None


def triangulate_all(
    cameras: dict[int, CalibratedCamera],
    synced_points_list: list[SyncedPoints],
    progress_callback: callable | None = None,
) -> list[XYZPoints]:
    """
    Triangulate all points from a list of synchronized frames.

    Args:
        cameras: Dict of port -> CalibratedCamera
        synced_points_list: List of SyncedPoints to triangulate
        progress_callback: Optional callback(current, total) for progress

    Returns:
        List of XYZPoints, one per input SyncedPoints
    """
    results = []
    total = len(synced_points_list)

    for i, synced in enumerate(synced_points_list):
        xyz = triangulate_frame(cameras, synced)
        results.append(xyz)

        if progress_callback is not None:
            progress_callback(i + 1, total)

    return results
