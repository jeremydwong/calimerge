"""
Extrinsic perturbation sensitivity study.

Measures how errors in extrinsic calibration parameters propagate into
3D reconstruction error. All logic in pure functions following the
codebase's data-oriented style.

Pipeline:
    1. Parse legacy calibration (config.toml) -> dict[int, CalibratedCamera]
    2. Load reference 3D poses (CSV) -> (sync_indices, xyz, keypoint_names)
    3. Back-project 3D -> 2D observations per camera (using cv2.projectPoints)
    4. Perturb extrinsics -> re-triangulate -> measure error vs reference
    5. Sweep individual parameters and run Monte Carlo trials
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from calimerge.triangulation import undistort_points
from calimerge.types import (
    CalibratedCamera,
    CameraExtrinsics,
    CameraIntrinsics,
    compute_projection_matrix,
    extrinsics_from_vector,
    extrinsics_to_vector,
)


# ============================================================================
# Parsing
# ============================================================================


def parse_legacy_calibration(config_path: Path) -> dict[int, CalibratedCamera]:
    """
    Parse a legacy caliscope config.toml into CalibratedCamera instances.

    The legacy format stores per-camera sections as [cam_0], [cam_1], etc.
    Each section contains: port, size, matrix, distortions, translation,
    rotation (Rodrigues vector), error, grid_count.

    Args:
        config_path: Path to config.toml

    Returns:
        dict mapping port -> CalibratedCamera
    """
    with open(config_path, "rb") as f:
        raw = tomllib.load(f)

    cameras: dict[int, CalibratedCamera] = {}

    for key, section in raw.items():
        if not key.startswith("cam_"):
            continue

        port = section["port"]
        w, h = section["size"]
        resolution = (int(w), int(h))

        matrix = np.array(section["matrix"], dtype=np.float64)
        distortion = np.array(section["distortions"], dtype=np.float64)

        # Rodrigues vector (3 elements) -> 3x3 rotation matrix
        rvec = np.array(section["rotation"], dtype=np.float64)
        rotation_matrix = cv2.Rodrigues(rvec)[0]

        translation = np.array(section["translation"], dtype=np.float64)

        error = float(section["error"])
        grid_count = int(section["grid_count"])

        serial = f"cam_{port}"

        intrinsics = CameraIntrinsics(
            serial_number=serial,
            resolution=resolution,
            matrix=matrix,
            distortion=distortion,
            error=error,
            grid_count=grid_count,
        )

        extrinsics = CameraExtrinsics(
            rotation=rotation_matrix,
            translation=translation,
        )

        cameras[port] = CalibratedCamera(
            serial_number=serial,
            port=port,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )

    return cameras


# ============================================================================
# Reference Data Loading
# ============================================================================


def load_reference_3d(
    csv_path: Path,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Load reference 3D poses from a tracked output CSV.

    Format: sync_index, person_id, Nose_X, Nose_Y, Nose_Z, L_Eye_X, ...
    Columns are grouped in _X, _Y, _Z triples per keypoint.

    Returns only keypoints that have at least some valid (non-NaN) data.

    Args:
        csv_path: Path to the 3D poses CSV

    Returns:
        (sync_indices, xyz_array, keypoint_names)
        - sync_indices: (N,) int array
        - xyz_array: (N, K, 3) float array (NaN where missing)
        - keypoint_names: list of K keypoint names
    """
    df = pd.read_csv(csv_path)

    sync_indices = df["sync_index"].values.astype(np.int64)

    # Find all keypoint names by looking for _X suffix columns
    x_cols = [c for c in df.columns if c.endswith("_X")]
    all_keypoints = [c[:-2] for c in x_cols]  # strip "_X"

    # Build (N, K_all, 3) array
    n_frames = len(df)
    n_kp_all = len(all_keypoints)
    xyz_all = np.full((n_frames, n_kp_all, 3), np.nan, dtype=np.float64)

    for k, name in enumerate(all_keypoints):
        for dim, suffix in enumerate(["_X", "_Y", "_Z"]):
            col = name + suffix
            if col in df.columns:
                vals = pd.to_numeric(df[col], errors="coerce").values
                xyz_all[:, k, dim] = vals

    # Filter to keypoints with at least some valid data
    valid_mask = np.any(np.isfinite(xyz_all), axis=(0, 2))  # (K_all,)
    valid_indices = np.where(valid_mask)[0]

    xyz = xyz_all[:, valid_indices, :]
    keypoint_names = [all_keypoints[i] for i in valid_indices]

    return sync_indices, xyz, keypoint_names


# ============================================================================
# Back-Projection
# ============================================================================


def backproject_to_2d(
    xyz_array: np.ndarray,
    cameras: dict[int, CalibratedCamera],
) -> dict[int, np.ndarray]:
    """
    Project 3D points to 2D image coordinates for each camera.

    Uses cv2.projectPoints which applies the full projection model
    including lens distortion.

    Args:
        xyz_array: (N, K, 3) array of 3D points (may contain NaN)
        cameras: dict of port -> CalibratedCamera

    Returns:
        dict of port -> (N, K, 2) array of 2D image coordinates
        NaN where input 3D points were NaN.
    """
    n_frames, n_kp, _ = xyz_array.shape
    result: dict[int, np.ndarray] = {}

    for port, cam in cameras.items():
        obs_2d = np.full((n_frames, n_kp, 2), np.nan, dtype=np.float64)

        rvec = cv2.Rodrigues(cam.extrinsics.rotation)[0]  # (3, 1)
        tvec = cam.extrinsics.translation.reshape(3, 1)
        K = cam.intrinsics.matrix
        dist = cam.intrinsics.distortion

        for i in range(n_frames):
            # Find valid (non-NaN) keypoints in this frame
            valid = np.all(np.isfinite(xyz_array[i]), axis=1)  # (K,)
            if not np.any(valid):
                continue

            pts_3d = xyz_array[i, valid, :]  # (n_valid, 3)
            pts_2d, _ = cv2.projectPoints(pts_3d, rvec, tvec, K, dist)
            obs_2d[i, valid, :] = pts_2d.reshape(-1, 2)

        result[port] = obs_2d

    return result


# ============================================================================
# Perturbation
# ============================================================================

_PARAM_INDICES = {"rx": 0, "ry": 1, "rz": 2, "tx": 3, "ty": 4, "tz": 5}
_ROTATION_PARAMS = {"rx", "ry", "rz"}
_TRANSLATION_PARAMS = {"tx", "ty", "tz"}


def perturb_extrinsics(
    cameras: dict[int, CalibratedCamera],
    port: int,
    param_name: str,
    magnitude: float,
) -> dict[int, CalibratedCamera]:
    """
    Apply a perturbation to a single extrinsic parameter of one camera.

    Args:
        cameras: dict of port -> CalibratedCamera
        port: which camera to perturb
        param_name: "rx", "ry", "rz" (degrees), "tx", "ty", "tz" (mm)
        magnitude: perturbation size in degrees (rotation) or mm (translation)

    Returns:
        New cameras dict with the target camera's extrinsics perturbed.
    """
    if param_name not in _PARAM_INDICES:
        raise ValueError(f"Unknown parameter: {param_name}")

    cam = cameras[port]
    vec = extrinsics_to_vector(cam.extrinsics)

    idx = _PARAM_INDICES[param_name]

    if param_name in _ROTATION_PARAMS:
        # Convert degrees to radians
        delta = magnitude * math.pi / 180.0
    else:
        # Convert mm to meters
        delta = magnitude / 1000.0

    vec[idx] += delta
    new_extrinsics = extrinsics_from_vector(vec)

    new_cam = CalibratedCamera(
        serial_number=cam.serial_number,
        port=cam.port,
        intrinsics=cam.intrinsics,
        extrinsics=new_extrinsics,
    )

    new_cameras = dict(cameras)
    new_cameras[port] = new_cam
    return new_cameras


def _perturb_extrinsics_vector(
    cameras: dict[int, CalibratedCamera],
    port: int,
    delta_vec: np.ndarray,
) -> dict[int, CalibratedCamera]:
    """
    Apply a 6-element perturbation vector [drx, dry, drz, dtx, dty, dtz]
    to a single camera. Rotations in radians, translations in meters.

    Args:
        cameras: dict of port -> CalibratedCamera
        port: which camera to perturb
        delta_vec: (6,) perturbation in [radians, radians, radians, m, m, m]

    Returns:
        New cameras dict with the target camera's extrinsics perturbed.
    """
    cam = cameras[port]
    vec = extrinsics_to_vector(cam.extrinsics)
    vec = vec + delta_vec
    new_extrinsics = extrinsics_from_vector(vec)

    new_cam = CalibratedCamera(
        serial_number=cam.serial_number,
        port=cam.port,
        intrinsics=cam.intrinsics,
        extrinsics=new_extrinsics,
    )

    new_cameras = dict(cameras)
    new_cameras[port] = new_cam
    return new_cameras


# ============================================================================
# Triangulation
# ============================================================================


def triangulate_from_observations(
    obs_2d: dict[int, np.ndarray],
    cameras: dict[int, CalibratedCamera],
) -> np.ndarray:
    """
    Triangulate 3D points from 2D observations across cameras.

    Uses DLT (Direct Linear Transform) via SVD, with undistortion
    applied to the 2D points before triangulation.

    Args:
        obs_2d: dict of port -> (N, K, 2) observed 2D points (may contain NaN)
        cameras: dict of port -> CalibratedCamera

    Returns:
        (N, K, 3) array of triangulated 3D points (NaN where < 2 cameras)
    """
    ports = sorted(obs_2d.keys())
    sample = obs_2d[ports[0]]
    n_frames, n_kp, _ = sample.shape

    # Pre-compute projection matrices
    proj_matrices: dict[int, np.ndarray] = {}
    for port, cam in cameras.items():
        proj_matrices[port] = compute_projection_matrix(cam)

    result = np.full((n_frames, n_kp, 3), np.nan, dtype=np.float64)

    for i in range(n_frames):
        for k in range(n_kp):
            # Collect valid observations across cameras
            valid_ports = []
            valid_pts = []

            for port in ports:
                pt = obs_2d[port][i, k]
                if np.all(np.isfinite(pt)):
                    valid_ports.append(port)
                    valid_pts.append(pt)

            if len(valid_ports) < 2:
                continue

            # Undistort 2D points
            undistorted_pts = []
            for port, pt in zip(valid_ports, valid_pts):
                pts_arr = np.array([pt], dtype=np.float64)  # (1, 2)
                undist = undistort_points(pts_arr, cameras[port].intrinsics)
                undistorted_pts.append(undist[0])

            # Build DLT system: A is (2*n_cams, 4)
            n_cams = len(valid_ports)
            A = np.zeros((n_cams * 2, 4), dtype=np.float64)

            for j, (port, upt) in enumerate(
                zip(valid_ports, undistorted_pts)
            ):
                x, y = upt
                P = proj_matrices[port]
                A[j * 2] = x * P[2] - P[0]
                A[j * 2 + 1] = y * P[2] - P[1]

            # SVD solve
            try:
                _, _, vh = np.linalg.svd(A, full_matrices=True)
                point_xyzw = vh[-1]
                if abs(point_xyzw[3]) < 1e-12:
                    continue
                result[i, k] = point_xyzw[:3] / point_xyzw[3]
            except np.linalg.LinAlgError:
                continue

    return result


# ============================================================================
# Error Computation
# ============================================================================


def compute_errors(
    ref_xyz: np.ndarray,
    perturbed_xyz: np.ndarray,
) -> dict:
    """
    Compute Euclidean distance errors between reference and perturbed 3D points.

    Args:
        ref_xyz: (N, K, 3) reference 3D points
        perturbed_xyz: (N, K, 3) perturbed 3D points

    Returns:
        dict with: mean_cm, median_cm, p95_cm, max_cm,
                   per_keypoint_mean_cm (K,), valid_count
    """
    diff = perturbed_xyz - ref_xyz  # (N, K, 3)
    dist = np.sqrt(np.nansum(diff**2, axis=2))  # (N, K)

    # Mask out pairs where either ref or perturbed is NaN
    valid = np.isfinite(dist)
    valid_dists = dist[valid]

    if len(valid_dists) == 0:
        n_kp = ref_xyz.shape[1]
        return {
            "mean_cm": np.nan,
            "median_cm": np.nan,
            "p95_cm": np.nan,
            "max_cm": np.nan,
            "per_keypoint_mean_cm": np.full(n_kp, np.nan),
            "valid_count": 0,
        }

    # Outlier cap: clip errors > 10x median
    median_val = np.median(valid_dists)
    if median_val > 0:
        cap = 10.0 * median_val
        valid_dists = np.clip(valid_dists, 0, cap)

    # Convert meters to centimeters
    valid_dists_cm = valid_dists * 100.0

    # Per-keypoint mean error
    n_kp = ref_xyz.shape[1]
    per_kp_mean = np.full(n_kp, np.nan)
    for k in range(n_kp):
        kp_valid = np.isfinite(dist[:, k])
        kp_dists = dist[:, k][kp_valid]
        if len(kp_dists) > 0:
            if median_val > 0:
                kp_dists = np.clip(kp_dists, 0, cap)
            per_kp_mean[k] = np.mean(kp_dists) * 100.0

    return {
        "mean_cm": float(np.mean(valid_dists_cm)),
        "median_cm": float(np.median(valid_dists_cm)),
        "p95_cm": float(np.percentile(valid_dists_cm, 95)),
        "max_cm": float(np.max(valid_dists_cm)),
        "per_keypoint_mean_cm": per_kp_mean,
        "valid_count": int(len(valid_dists_cm)),
    }


# ============================================================================
# Single-Parameter Sweep
# ============================================================================

_ROTATION_MAGNITUDES = [0.05, 0.1, 0.5, 1.0, 2.0, 5.0]  # degrees
_TRANSLATION_MAGNITUDES = [1, 2, 5, 10, 20, 50]  # mm


def run_single_param_sweep(
    cameras: dict[int, CalibratedCamera],
    ref_xyz: np.ndarray,
    obs_2d: dict[int, np.ndarray],
    keypoint_names: list[str],
    port: int = 0,
) -> pd.DataFrame:
    """
    Sweep each extrinsic parameter individually across a range of magnitudes.

    For each of the 6 parameters, for each magnitude, applies both positive
    and negative perturbations, re-triangulates, and computes errors.

    Args:
        cameras: dict of port -> CalibratedCamera
        ref_xyz: (N, K, 3) reference 3D points
        obs_2d: dict of port -> (N, K, 2) back-projected 2D observations
        keypoint_names: list of K keypoint names
        port: which camera to perturb (default 0)

    Returns:
        DataFrame with columns: param, magnitude, unit, sign,
        mean_cm, median_cm, p95_cm, max_cm, + per-keypoint columns
    """
    rows = []

    for param_name in ["rx", "ry", "rz", "tx", "ty", "tz"]:
        if param_name in _ROTATION_PARAMS:
            magnitudes = _ROTATION_MAGNITUDES
            unit = "deg"
        else:
            magnitudes = _TRANSLATION_MAGNITUDES
            unit = "mm"

        for mag in magnitudes:
            for sign_label, sign_val in [("+", 1.0), ("-", -1.0)]:
                actual_mag = mag * sign_val

                perturbed_cams = perturb_extrinsics(
                    cameras, port, param_name, actual_mag
                )
                tri_xyz = triangulate_from_observations(obs_2d, perturbed_cams)
                errs = compute_errors(ref_xyz, tri_xyz)

                row = {
                    "param": param_name,
                    "magnitude": mag,
                    "unit": unit,
                    "sign": sign_label,
                    "mean_cm": errs["mean_cm"],
                    "median_cm": errs["median_cm"],
                    "p95_cm": errs["p95_cm"],
                    "max_cm": errs["max_cm"],
                }

                # Add per-keypoint columns
                for k, name in enumerate(keypoint_names):
                    row[f"{name}_mean_cm"] = errs["per_keypoint_mean_cm"][k]

                rows.append(row)
                print(
                    f"  Sweep: {param_name} {sign_label}{mag}{unit} "
                    f"-> mean {errs['mean_cm']:.2f}cm, "
                    f"p95 {errs['p95_cm']:.2f}cm"
                )

    return pd.DataFrame(rows)


# ============================================================================
# Monte Carlo
# ============================================================================

_SIGMA_LEVELS = [
    (0.1, 1),    # (degrees, mm)
    (0.5, 5),
    (1.0, 10),
    (2.0, 20),
]


def run_monte_carlo(
    cameras: dict[int, CalibratedCamera],
    ref_xyz: np.ndarray,
    obs_2d: dict[int, np.ndarray],
    keypoint_names: list[str],
    port: int = 0,
    n_trials: int = 50,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Monte Carlo perturbation study with jointly sampled extrinsic noise.

    For each sigma level, runs n_trials with all 6 extrinsic parameters
    perturbed simultaneously from independent Gaussian distributions.

    Args:
        cameras: dict of port -> CalibratedCamera
        ref_xyz: (N, K, 3) reference 3D points
        obs_2d: dict of port -> (N, K, 2) back-projected 2D observations
        keypoint_names: list of K keypoint names
        port: which camera to perturb (default 0)
        n_trials: number of Monte Carlo trials per sigma level
        seed: random seed for reproducibility

    Returns:
        DataFrame with columns: sigma_rot_deg, sigma_trans_mm, trial,
        mean_cm, median_cm, p95_cm, max_cm, + per-keypoint columns,
        + individual perturbation values drx..dtz
    """
    rng = np.random.default_rng(seed)
    rows = []

    for sigma_deg, sigma_mm in _SIGMA_LEVELS:
        sigma_rad = sigma_deg * math.pi / 180.0
        sigma_m = sigma_mm / 1000.0

        print(
            f"  Monte Carlo: sigma_rot={sigma_deg}deg, "
            f"sigma_trans={sigma_mm}mm, {n_trials} trials"
        )

        for trial in range(n_trials):
            # Sample perturbations
            drx = rng.normal(0, sigma_rad)
            dry = rng.normal(0, sigma_rad)
            drz = rng.normal(0, sigma_rad)
            dtx = rng.normal(0, sigma_m)
            dty = rng.normal(0, sigma_m)
            dtz = rng.normal(0, sigma_m)

            delta_vec = np.array(
                [drx, dry, drz, dtx, dty, dtz], dtype=np.float64
            )

            perturbed_cams = _perturb_extrinsics_vector(
                cameras, port, delta_vec
            )
            tri_xyz = triangulate_from_observations(obs_2d, perturbed_cams)
            errs = compute_errors(ref_xyz, tri_xyz)

            row = {
                "sigma_rot_deg": sigma_deg,
                "sigma_trans_mm": sigma_mm,
                "trial": trial,
                "mean_cm": errs["mean_cm"],
                "median_cm": errs["median_cm"],
                "p95_cm": errs["p95_cm"],
                "max_cm": errs["max_cm"],
                "drx_rad": drx,
                "dry_rad": dry,
                "drz_rad": drz,
                "dtx_m": dtx,
                "dty_m": dty,
                "dtz_m": dtz,
            }

            for k, name in enumerate(keypoint_names):
                row[f"{name}_mean_cm"] = errs["per_keypoint_mean_cm"][k]

            rows.append(row)

        # Print summary for this sigma level
        level_rows = [
            r for r in rows
            if r["sigma_rot_deg"] == sigma_deg
            and r["sigma_trans_mm"] == sigma_mm
        ]
        means = [r["mean_cm"] for r in level_rows]
        print(
            f"    -> mean error: {np.mean(means):.2f}cm "
            f"(std {np.std(means):.2f}cm)"
        )

    return pd.DataFrame(rows)


# ============================================================================
# Per-Keypoint Sensitivity
# ============================================================================


def compute_per_keypoint_sensitivity(
    cameras: dict[int, CalibratedCamera],
    ref_xyz: np.ndarray,
    obs_2d: dict[int, np.ndarray],
    keypoint_names: list[str],
    port: int = 0,
    rot_magnitude_deg: float = 1.0,
    trans_magnitude_mm: float = 10.0,
) -> pd.DataFrame:
    """
    Compute per-keypoint sensitivity at reference perturbation magnitudes.

    For each of the 6 parameters, applies the reference magnitude (positive)
    and reports per-keypoint error.

    Args:
        cameras: dict of port -> CalibratedCamera
        ref_xyz: (N, K, 3) reference 3D points
        obs_2d: dict of port -> (N, K, 2) back-projected 2D observations
        keypoint_names: list of K keypoint names
        port: which camera to perturb
        rot_magnitude_deg: reference rotation perturbation (degrees)
        trans_magnitude_mm: reference translation perturbation (mm)

    Returns:
        DataFrame with columns: keypoint, rx_cm, ry_cm, rz_cm, tx_cm, ty_cm, tz_cm
    """
    # Compute per-keypoint error for each parameter
    param_errors: dict[str, np.ndarray] = {}

    for param_name in ["rx", "ry", "rz", "tx", "ty", "tz"]:
        if param_name in _ROTATION_PARAMS:
            mag = rot_magnitude_deg
        else:
            mag = trans_magnitude_mm

        perturbed_cams = perturb_extrinsics(cameras, port, param_name, mag)
        tri_xyz = triangulate_from_observations(obs_2d, perturbed_cams)
        errs = compute_errors(ref_xyz, tri_xyz)
        param_errors[param_name] = errs["per_keypoint_mean_cm"]

    rows = []
    for k, name in enumerate(keypoint_names):
        row = {"keypoint": name}
        for param_name in ["rx", "ry", "rz", "tx", "ty", "tz"]:
            row[f"{param_name}_cm"] = param_errors[param_name][k]
        rows.append(row)

    return pd.DataFrame(rows)


# ============================================================================
# Top-Level Orchestration
# ============================================================================


def run_study(
    config_path: Path,
    ref_csv_path: Path,
    output_dir: Path,
    camera_port: int = 0,
    mc_trials: int = 50,
) -> None:
    """
    Run the full extrinsic perturbation sensitivity study.

    Steps:
        1. Parse legacy calibration
        2. Load reference 3D poses
        3. Back-project to 2D (once, reused for all trials)
        4. Run single-param sweep -> sweep_results.csv
        5. Run Monte Carlo -> monte_carlo_results.csv
        6. Compute per-keypoint sensitivity -> per_keypoint_sensitivity.csv
        7. Print summary

    Args:
        config_path: Path to legacy config.toml
        ref_csv_path: Path to reference 3D poses CSV
        output_dir: Directory for output CSVs
        camera_port: Which camera to perturb (default 0)
        mc_trials: Number of Monte Carlo trials per sigma level
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Parse calibration
    print(f"Loading calibration from {config_path}")
    cameras = parse_legacy_calibration(config_path)
    print(f"  Found {len(cameras)} cameras: ports {sorted(cameras.keys())}")
    for port, cam in sorted(cameras.items()):
        print(
            f"    cam_{port}: {cam.intrinsics.resolution}, "
            f"error={cam.intrinsics.error:.3f}, "
            f"grids={cam.intrinsics.grid_count}"
        )

    if camera_port not in cameras:
        raise ValueError(
            f"Camera port {camera_port} not found. "
            f"Available: {sorted(cameras.keys())}"
        )

    # 2. Load reference 3D
    print(f"\nLoading reference 3D from {ref_csv_path}")
    sync_indices, ref_xyz, keypoint_names = load_reference_3d(ref_csv_path)
    n_frames, n_kp, _ = ref_xyz.shape
    valid_count = np.sum(np.all(np.isfinite(ref_xyz), axis=2))
    print(
        f"  {n_frames} frames, {n_kp} keypoints "
        f"({', '.join(keypoint_names[:5])}, ...)"
    )
    print(f"  {valid_count} valid 3D observations total")

    # 3. Back-project to 2D
    print("\nBack-projecting 3D -> 2D for all cameras...")
    obs_2d = backproject_to_2d(ref_xyz, cameras)
    for port in sorted(obs_2d.keys()):
        valid_2d = np.sum(np.all(np.isfinite(obs_2d[port]), axis=2))
        print(f"  cam_{port}: {valid_2d} valid 2D observations")

    # Verify round-trip: triangulate from back-projected 2D with original cams
    print("\nVerifying round-trip (triangulate from back-projected 2D)...")
    roundtrip_xyz = triangulate_from_observations(obs_2d, cameras)
    roundtrip_errs = compute_errors(ref_xyz, roundtrip_xyz)
    print(
        f"  Round-trip error: mean={roundtrip_errs['mean_cm']:.4f}cm, "
        f"max={roundtrip_errs['max_cm']:.4f}cm"
    )

    # 4. Single-param sweep
    print(f"\nRunning single-parameter sweep (perturbing cam_{camera_port})...")
    sweep_df = run_single_param_sweep(
        cameras, ref_xyz, obs_2d, keypoint_names, port=camera_port
    )
    sweep_path = output_dir / "sweep_results.csv"
    sweep_df.to_csv(sweep_path, index=False)
    print(f"  Saved to {sweep_path}")

    # 5. Monte Carlo
    print(f"\nRunning Monte Carlo ({mc_trials} trials per sigma level)...")
    mc_df = run_monte_carlo(
        cameras,
        ref_xyz,
        obs_2d,
        keypoint_names,
        port=camera_port,
        n_trials=mc_trials,
    )
    mc_path = output_dir / "monte_carlo_results.csv"
    mc_df.to_csv(mc_path, index=False)
    print(f"  Saved to {mc_path}")

    # 6. Per-keypoint sensitivity
    print("\nComputing per-keypoint sensitivity (1deg rot, 10mm trans)...")
    kp_df = compute_per_keypoint_sensitivity(
        cameras, ref_xyz, obs_2d, keypoint_names, port=camera_port
    )
    kp_path = output_dir / "per_keypoint_sensitivity.csv"
    kp_df.to_csv(kp_path, index=False)
    print(f"  Saved to {kp_path}")

    # 7. Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    print(f"\nRound-trip baseline error: {roundtrip_errs['mean_cm']:.4f} cm")

    print("\nSingle-param sweep (mean error in cm, positive perturbation):")
    pos_sweep = sweep_df[sweep_df["sign"] == "+"]
    for param in ["rx", "ry", "rz", "tx", "ty", "tz"]:
        param_rows = pos_sweep[pos_sweep["param"] == param]
        if param in _ROTATION_PARAMS:
            unit = "deg"
        else:
            unit = "mm"
        summary_parts = []
        for _, row in param_rows.iterrows():
            summary_parts.append(f"{row['magnitude']}{unit}={row['mean_cm']:.2f}")
        print(f"  {param}: {', '.join(summary_parts)}")

    print("\nMonte Carlo summary (mean +/- std of mean error in cm):")
    for sigma_deg, sigma_mm in _SIGMA_LEVELS:
        level = mc_df[
            (mc_df["sigma_rot_deg"] == sigma_deg)
            & (mc_df["sigma_trans_mm"] == sigma_mm)
        ]
        mean_err = level["mean_cm"].mean()
        std_err = level["mean_cm"].std()
        print(
            f"  sigma=({sigma_deg}deg, {sigma_mm}mm): "
            f"{mean_err:.2f} +/- {std_err:.2f} cm"
        )

    print("\nPer-keypoint sensitivity (cm error at 1deg rotation):")
    for _, row in kp_df.iterrows():
        rot_mean = np.mean([row["rx_cm"], row["ry_cm"], row["rz_cm"]])
        print(f"  {row['keypoint']:15s}: {rot_mean:.2f} cm (avg across rx/ry/rz)")

    print(f"\nAll results saved to {output_dir}/")
