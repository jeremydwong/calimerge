"""
Main pose tracking pipeline.

Orchestrates: video loading -> person detection -> pose estimation ->
cross-view matching -> triangulation -> tracking -> export.

Replaces posetrack's process_synced_mwc_frames_multi_person_perf(),
adapted to accept calimerge data structures directly.
"""

from __future__ import annotations

import os
import time
import pickle
from pathlib import Path
from typing import Callable

import cv2
import numpy as np
import pandas as pd
from PIL import Image

from ..types import CalibratedCamera
from .markers import SYNTHPOSE_MARKERS, NUM_MARKERS, HIP_INDICES
from .triangulation import (
    calculate_projection_matrices,
    project_keypoints_to_all_cameras,
)
from .tracker import (
    PersonTrack,
    calculate_2d_com,
    group_detections_across_views_bipartite,
    generate_3d_candidates_from_groups,
    assign_3d_candidates_to_tracks,
)
from .pose_detector import (
    setup_device,
    load_models,
    detect_persons_batch,
    estimate_poses_batch,
)


def calibrated_cameras_to_params(
    cameras: dict[int, CalibratedCamera],
) -> tuple[list[dict], dict[int, int]]:
    """
    Convert calimerge CalibratedCamera objects to posetrack camera_params format.

    Returns:
        camera_params: list of dicts with keys:
            matrix, distortions, size, rotation, translation, port
        port_to_cam_index: dict mapping port -> index in camera_params
    """
    camera_params = []
    port_to_cam_index = {}

    for i, (port, cam) in enumerate(sorted(cameras.items())):
        # CameraExtrinsics.rotation is 3x3 -> convert to Rodrigues vector
        rvec, _ = cv2.Rodrigues(cam.extrinsics.rotation)
        camera_params.append({
            "matrix": cam.intrinsics.matrix,
            "distortions": cam.intrinsics.distortion,
            "size": np.array(cam.intrinsics.resolution),
            "rotation": rvec.flatten(),
            "translation": cam.extrinsics.translation,
            "port": port,
        })
        port_to_cam_index[port] = i

    return camera_params, port_to_cam_index


def run_pose_tracking(
    cameras: dict[int, CalibratedCamera],
    video_paths: dict[int, Path],
    frame_time_csv: Path | None,
    output_path: Path,
    device_name: str = "auto",
    skip_sync_indices: int = 1,
    person_confidence: float = 0.1,
    keypoint_confidence: float = 0.1,
    max_persons: int = 2,
    batch_size: int = 8,
    track_frames_til_lost_patience: int = 30,
    epipolar_threshold: float = 10.0,
    hip_indices: tuple[int, int] = HIP_INDICES,
    progress_callback: Callable[[str, float], None] | None = None,
    log_callback: Callable[[str], None] | None = None,
) -> Path | None:
    """
    Run the full multi-person 3D pose tracking pipeline.

    Args:
        cameras: port -> CalibratedCamera with intrinsics and extrinsics.
        video_paths: port -> Path to video file.
        frame_time_csv: path to frame_time_history.csv (for sync).
        output_path: directory to write results into.
        device_name: "auto", "mps", "cuda", or "cpu".
        skip_sync_indices: process every Nth sync index (1 = all).
        person_confidence: YOLO detection confidence threshold.
        keypoint_confidence: minimum keypoint confidence to include.
        max_persons: maximum concurrent tracked persons.
        batch_size: frames processed simultaneously.
        track_frames_til_lost_patience: frames before a lost track is deactivated.
        epipolar_threshold: pixel threshold for cross-view matching.
        hip_indices: keypoint indices for center-of-mass (default: L_Hip, R_Hip).
        progress_callback: called with (step_name, fraction) for GUI updates.
        log_callback: called with log messages.

    Returns:
        Path to output directory, or None on failure.
    """
    def log(msg: str):
        if log_callback:
            log_callback(msg)

    def progress(step: str, frac: float):
        if progress_callback:
            progress_callback(step, frac)

    # --- 1. Setup device ---
    device = setup_device(device_name)
    log(f"Using device: {device}")

    # --- 2. Load frame history ---
    progress("Loading frame history", 0.0)
    frame_history_df = _load_frame_history(frame_time_csv, log)
    if frame_history_df is None:
        log("ERROR: Could not load frame history")
        return None

    # --- 3. Convert calibration ---
    progress("Setting up calibration", 0.05)
    camera_params, port_to_cam_index = calibrated_cameras_to_params(cameras)

    # Filter to common ports between CSV, calibration, and videos
    csv_ports = set(frame_history_df["port"].unique())
    calib_ports = set(port_to_cam_index.keys())
    video_ports = set(video_paths.keys())
    common_ports = sorted(list(csv_ports & calib_ports & video_ports))

    log(f"Common ports: {common_ports}")

    if len(common_ports) < 2:
        log("ERROR: Need >= 2 common ports for triangulation")
        return None

    # Re-filter camera params to common ports only
    filtered_params = []
    filtered_port_map = {}
    for new_idx, port in enumerate(common_ports):
        orig_idx = port_to_cam_index[port]
        filtered_params.append(camera_params[orig_idx])
        filtered_port_map[port] = new_idx

    camera_params = filtered_params
    port_to_cam_index = filtered_port_map
    log(f"Using {len(camera_params)} cameras")

    # --- 4. Projection matrices ---
    projection_matrices = calculate_projection_matrices(camera_params)

    # --- 5. Load models ---
    progress("Loading models", 0.1)
    log("Loading detection and pose estimation models...")
    person_model, pose_processor, pose_model = load_models(device=device, log_fn=log)

    # --- 6. Open videos ---
    progress("Opening videos", 0.15)
    caps, video_lengths = _open_videos(common_ports, video_paths, log)
    if caps is None:
        return None

    # --- 7. Process frames ---
    all_sync_indices = sorted(frame_history_df["sync_index"].unique())
    filtered_indices = all_sync_indices[::skip_sync_indices]
    log(f"Processing {len(filtered_indices)} of {len(all_sync_indices)} sync indices (skip={skip_sync_indices})")

    output_path.mkdir(parents=True, exist_ok=True)

    results_by_person, pixel_coords_by_person, cameras_by_person = _process_all_frames(
        filtered_indices=filtered_indices,
        frame_history_df=frame_history_df,
        common_ports=common_ports,
        caps=caps,
        video_lengths=video_lengths,
        projection_matrices=projection_matrices,
        port_to_cam_index=port_to_cam_index,
        camera_params=camera_params,
        person_model=person_model,
        pose_processor=pose_processor,
        pose_model=pose_model,
        device=device,
        person_confidence=person_confidence,
        keypoint_confidence=keypoint_confidence,
        hip_indices=hip_indices,
        max_persons=max_persons,
        batch_size=batch_size,
        track_frames_til_lost_patience=track_frames_til_lost_patience,
        epipolar_threshold=epipolar_threshold,
        progress_fn=progress,
        log_fn=log,
    )

    # --- 8. Release videos ---
    for cap in caps.values():
        cap.release()

    # --- 9. Save results ---
    progress("Saving results", 0.95)
    _save_all_results(
        results_by_person,
        pixel_coords_by_person,
        cameras_by_person,
        common_ports,
        output_path,
        log,
    )

    progress("Complete", 1.0)
    log(f"Tracking complete. {len(results_by_person)} person(s) tracked.")
    return output_path


# ============================================================================
# Internal helpers
# ============================================================================


def _load_frame_history(csv_path: Path | None, log) -> pd.DataFrame | None:
    """Load and process frame_time_history.csv."""
    if csv_path is None or not csv_path.exists():
        log(f"Frame history CSV not found: {csv_path}")
        return None

    try:
        # Skip comment lines starting with #
        df = pd.read_csv(csv_path, comment="#")
        df["sync_index"] = df["sync_index"].astype(int)
        df["port"] = df["port"].astype(int)
        df["frame_time"] = df["frame_time"].astype(float)
        df = df.sort_values(by=["port", "frame_time"])
        df["derived_frame_index"] = (
            df.groupby("port")["frame_time"].rank(method="min").astype(int) - 1
        )
        log(f"Loaded {df['sync_index'].nunique()} sync indices, ports: {sorted(df['port'].unique())}")
        return df
    except Exception as e:
        log(f"Error reading frame history: {e}")
        return None


def _open_videos(
    common_ports: list[int], video_paths: dict[int, Path], log
) -> tuple[dict[int, cv2.VideoCapture] | None, dict[int, int] | None]:
    """Open video captures for all common ports."""
    caps = {}
    video_lengths = {}

    for port in common_ports:
        path = video_paths.get(port)
        if path is None or not path.exists():
            log(f"ERROR: Video not found for port {port}: {path}")
            for c in caps.values():
                c.release()
            return None, None

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            log(f"ERROR: Cannot open video: {path}")
            for c in caps.values():
                c.release()
            return None, None

        caps[port] = cap
        video_lengths[port] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        log(f"Opened port {port}: {path.name} ({video_lengths[port]} frames)")

    return caps, video_lengths


def _process_all_frames(
    filtered_indices,
    frame_history_df,
    common_ports,
    caps,
    video_lengths,
    projection_matrices,
    port_to_cam_index,
    camera_params,
    person_model,
    pose_processor,
    pose_model,
    device,
    person_confidence,
    keypoint_confidence,
    hip_indices,
    max_persons,
    batch_size,
    track_frames_til_lost_patience,
    epipolar_threshold,
    progress_fn,
    log_fn,
):
    """Main batch processing loop."""
    import torch

    active_tracks: list[PersonTrack] = []
    next_person_id = 0
    results_by_person: dict[int, list] = {}
    pixel_coords_by_person: dict[int, list] = {}
    cameras_by_person: dict[int, list] = {}
    previous_views_used = []

    total_batches = (len(filtered_indices) + batch_size - 1) // batch_size
    start_time = time.time()

    for batch_num, batch_start in enumerate(range(0, len(filtered_indices), batch_size)):
        batch_end = min(batch_start + batch_size, len(filtered_indices))
        current_batch = filtered_indices[batch_start:batch_end]

        frac = 0.2 + 0.75 * (batch_num / max(1, total_batches))
        progress_fn(f"Processing batch {batch_num + 1}/{total_batches}", frac)

        # Read frames for this batch
        batch_frames, valid_indices = _read_batch_frames(
            current_batch, frame_history_df, common_ports, caps, video_lengths
        )
        if not valid_indices:
            continue

        # Batch detection and pose estimation
        with torch.no_grad():
            batch_detections = _perform_batch_detection(
                batch_frames,
                valid_indices,
                common_ports,
                person_model,
                pose_processor,
                pose_model,
                device,
                person_confidence,
                keypoint_confidence,
                hip_indices,
                batch_size,
            )

        # Process each frame sequentially (tracking is stateful)
        for sync_index in valid_indices:
            if sync_index not in batch_detections:
                continue

            (
                active_tracks,
                next_person_id,
                results_by_person,
                pixel_coords_by_person,
                cameras_by_person,
                previous_views_used,
            ) = _process_single_frame(
                sync_index=sync_index,
                view_detections=batch_detections[sync_index],
                active_tracks=active_tracks,
                next_person_id=next_person_id,
                previous_views_used=previous_views_used,
                results_by_person=results_by_person,
                pixel_coords_by_person=pixel_coords_by_person,
                cameras_by_person=cameras_by_person,
                projection_matrices=projection_matrices,
                port_to_cam_index=port_to_cam_index,
                camera_params=camera_params,
                common_ports=common_ports,
                hip_indices=hip_indices,
                max_persons=max_persons,
                track_frames_til_lost_patience=track_frames_til_lost_patience,
                epipolar_threshold=epipolar_threshold,
            )

        # Log progress periodically
        if (batch_num + 1) % 10 == 0 or batch_num == total_batches - 1:
            elapsed = time.time() - start_time
            frames_done = min(batch_end, len(filtered_indices))
            fps = frames_done / max(elapsed, 0.001)
            log_fn(f"  {frames_done}/{len(filtered_indices)} frames ({elapsed:.1f}s, {fps:.1f} fps)")

    elapsed = time.time() - start_time
    log_fn(f"Processing finished in {elapsed:.1f}s")

    return results_by_person, pixel_coords_by_person, cameras_by_person


def _read_batch_frames(current_batch, frame_history_df, common_ports, caps, video_lengths):
    """Read frames for a batch of sync indices."""
    batch_frames = {}
    valid_indices = []

    for sync_index in current_batch:
        sync_data = frame_history_df[frame_history_df["sync_index"] == sync_index]
        if set(sync_data["port"]) != set(common_ports):
            continue

        current_frames = {}
        success = True

        for _, row in sync_data.iterrows():
            port = row["port"]
            frame_idx = int(row["derived_frame_index"])

            cap = caps.get(port)
            total = video_lengths.get(port, -1)

            if cap is None or total == -1 or not (0 <= frame_idx < total):
                success = False
                break

            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                success = False
                break

            current_frames[port] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

        if success:
            batch_frames[sync_index] = current_frames
            valid_indices.append(sync_index)

    return batch_frames, valid_indices


def _perform_batch_detection(
    batch_frames,
    valid_indices,
    common_ports,
    person_model,
    pose_processor,
    pose_model,
    device,
    person_confidence,
    keypoint_confidence,
    hip_indices,
    batch_size,
):
    """Batch person detection and pose estimation."""
    batch_detections = {}

    all_images = []
    image_metadata = []

    for sync_index in valid_indices:
        for port in common_ports:
            if port in batch_frames[sync_index]:
                all_images.append(batch_frames[sync_index][port])
                image_metadata.append((sync_index, port))

    if not all_images:
        return batch_detections

    # Batch person detection
    batch_person_results = detect_persons_batch(
        all_images, person_model, device, person_confidence, batch_size
    )

    # Prepare for batch pose estimation
    images_with_boxes = [
        (img, result[1])  # (image, boxes_coco)
        for img, result in zip(all_images, batch_person_results)
    ]

    # Batch pose estimation
    batch_pose_results = estimate_poses_batch(
        images_with_boxes, pose_processor, pose_model, device, batch_size
    )

    # Organize into sync_index/port structure
    for img_idx, (sync_index, port) in enumerate(image_metadata):
        if sync_index not in batch_detections:
            batch_detections[sync_index] = {}

        _, boxes_coco, scores = batch_person_results[img_idx]
        all_kps, all_scores = batch_pose_results[img_idx]

        port_detections = []

        if boxes_coco.size > 0:
            for person_idx in range(len(all_kps)):
                if person_idx >= len(all_scores):
                    break

                kps_2d = all_kps[person_idx]
                scores_2d = all_scores[person_idx]
                det_confidence = scores[person_idx] if person_idx < len(scores) else 0.0

                valid_kps = []
                for kp_idx in range(len(kps_2d)):
                    kp, score = kps_2d[kp_idx], scores_2d[kp_idx]
                    if score >= keypoint_confidence:
                        valid_kps.append([kp[0], kp[1], score])
                    else:
                        valid_kps.append([np.nan, np.nan, score])

                com_2d = calculate_2d_com(valid_kps, hip_indices)
                if com_2d is None:
                    continue

                port_detections.append({
                    "keypoints": np.array(valid_kps),
                    "com_2d": com_2d,
                    "confidence": det_confidence,
                })

        batch_detections[sync_index][port] = port_detections

    return batch_detections


def _process_single_frame(
    sync_index,
    view_detections,
    active_tracks,
    next_person_id,
    previous_views_used,
    results_by_person,
    pixel_coords_by_person,
    cameras_by_person,
    projection_matrices,
    port_to_cam_index,
    camera_params,
    common_ports,
    hip_indices,
    max_persons,
    track_frames_til_lost_patience,
    epipolar_threshold,
):
    """Process a single synced frame: group, triangulate, track."""
    groups = group_detections_across_views_bipartite(
        view_detections,
        projection_matrices,
        port_to_cam_index,
        camera_params,
        epipolar_threshold=epipolar_threshold,
    )

    candidate_groups = generate_3d_candidates_from_groups(
        groups, port_to_cam_index, camera_params, projection_matrices, hip_indices
    )

    track_assignments, _, new_track_assignments = assign_3d_candidates_to_tracks(
        active_tracks,
        candidate_groups,
        max_distance=0.15,
        default_views=None,
        max_tracks=max_persons,
        max_new_track_distance=4.0,
        min_new_track_distance=0.3,
    )

    # Bad frame handling
    if active_tracks and not track_assignments:
        for track in active_tracks:
            track.increment_lost_counter()
        active_tracks = [t for t in active_tracks if t.is_active]
        return (
            active_tracks,
            next_person_id,
            results_by_person,
            pixel_coords_by_person,
            cameras_by_person,
            previous_views_used,
        )

    # Update/create tracks
    for track_id, curtuple in track_assignments.items():
        if curtuple is None:
            continue

        grp_idx, cand_idx = curtuple
        if cand_idx is None:
            continue

        candidate = candidate_groups[grp_idx][cand_idx]

        if track_id < len(active_tracks):
            existing_track = active_tracks[track_id]
            existing_track.update(candidate["keypoints_3d"], sync_index, candidate["views"])
            track = existing_track
        else:
            if len(active_tracks) >= max_persons:
                continue
            new_track = PersonTrack(
                person_id=next_person_id,
                track_id=track_id,
                keypoints_3d=candidate["keypoints_3d"],
                sync_index=sync_index,
                hip_indices=hip_indices,
                views_used=candidate["views"],
                track_frames_til_lost_patience=track_frames_til_lost_patience,
            )
            active_tracks.append(new_track)
            next_person_id += 1
            track = new_track

        person_id = track.person_id

        if person_id not in results_by_person:
            results_by_person[person_id] = []
        if person_id not in pixel_coords_by_person:
            pixel_coords_by_person[person_id] = []
        if person_id not in cameras_by_person:
            cameras_by_person[person_id] = []

        # Store 3D keypoints
        kps_3d_list = [
            [np.nan] * 3 if kp is None else kp.tolist()
            for kp in candidate["keypoints_3d"]
        ]
        results_by_person[person_id].append({
            "sync_index": sync_index,
            "person_id": person_id,
            "keypoints_3d": kps_3d_list,
        })

        # Project to 2D for all cameras
        pixel_coords = project_keypoints_to_all_cameras(
            candidate["keypoints_3d"],
            projection_matrices,
            common_ports,
            port_to_cam_index,
        )
        pixel_coords_by_person[person_id].append({
            "sync_index": sync_index,
            "person_id": person_id,
            "pixel_coords": pixel_coords,
        })

        cameras_by_person[person_id].append({
            "sync_index": sync_index,
            "person_id": person_id,
            "cameras_used": candidate["views"],
        })

    # Update lost counters
    matched = {
        tid for tid, a in track_assignments.items() if a is not None and a[1] is not None
    }
    for track in active_tracks:
        if track.track_id not in matched:
            track.increment_lost_counter()

    # Update views used
    current_views = []
    for _, (g, c) in track_assignments.items():
        if c is not None:
            current_views.append(candidate_groups[g][c]["views"])
    previous_views_used = current_views

    active_tracks = [t for t in active_tracks if t.is_active]

    return (
        active_tracks,
        next_person_id,
        results_by_person,
        pixel_coords_by_person,
        cameras_by_person,
        previous_views_used,
    )


def _save_all_results(
    results_by_person,
    pixel_coords_by_person,
    cameras_by_person,
    common_ports,
    output_path,
    log,
):
    """Save per-person CSVs and pickle files."""
    for person_id, results in results_by_person.items():
        if not results:
            continue

        log(f"Saving person {person_id} ({len(results)} frames)...")

        # 3D keypoints CSV
        csv_path = output_path / f"person{person_id}_3d.csv"
        _save_person_csv(results, csv_path, NUM_MARKERS)

        # Pickle
        pkl_path = output_path / f"person{person_id}_3d.pkl"
        with open(pkl_path, "wb") as f:
            pickle.dump(results, f)

        # Pixel coordinates per port
        if person_id in pixel_coords_by_person:
            for port in common_ports:
                px_csv = output_path / f"person{person_id}_pixels_port{port}.csv"
                _save_pixel_coords_csv(
                    pixel_coords_by_person[person_id], px_csv, NUM_MARKERS, port
                )

        # Camera info
        if person_id in cameras_by_person:
            cam_csv = output_path / f"person{person_id}_cameras.csv"
            _save_cameras_csv(cameras_by_person[person_id], cam_csv)

    log(f"Results saved to {output_path}")


def _save_person_csv(results: list, csv_path: Path, num_kps: int):
    """Save 3D keypoints for one person to CSV."""
    rows = []
    for result in results:
        row = {
            "sync_index": result["sync_index"],
            "person_id": result["person_id"],
        }
        kps = result["keypoints_3d"]
        # Pad/truncate
        while len(kps) < num_kps:
            kps.append([np.nan, np.nan, np.nan])
        kps = kps[:num_kps]

        for kp_idx in range(num_kps):
            name = SYNTHPOSE_MARKERS.get(kp_idx, f"KP_{kp_idx}")
            coords = kps[kp_idx]
            if coords is None or not isinstance(coords, (list, tuple)) or len(coords) != 3:
                x, y, z = np.nan, np.nan, np.nan
            else:
                x, y, z = coords
            row[f"{name}_X"] = x
            row[f"{name}_Y"] = y
            row[f"{name}_Z"] = z
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False, float_format="%.4f")


def _save_pixel_coords_csv(
    pixel_results: list, csv_path: Path, num_kps: int, port: int
):
    """Save 2D pixel coordinates for one person and port."""
    rows = []
    for result in pixel_results:
        row = {
            "sync_index": result["sync_index"],
            "person_id": result["person_id"],
        }
        coords = result["pixel_coords"].get(port, [])
        while len(coords) < num_kps:
            coords.append([np.nan, np.nan])
        coords = coords[:num_kps]

        for kp_idx in range(num_kps):
            name = SYNTHPOSE_MARKERS.get(kp_idx, f"KP_{kp_idx}")
            pt = coords[kp_idx]
            if pt is None or not isinstance(pt, (list, tuple)) or len(pt) != 2:
                px, py = np.nan, np.nan
            else:
                px, py = pt
            row[f"{name}_px"] = px
            row[f"{name}_py"] = py
        rows.append(row)

    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False, float_format="%.4f")


def _save_cameras_csv(camera_results: list, csv_path: Path):
    """Save camera info per frame."""
    rows = []
    for result in camera_results:
        rows.append({
            "sync_index": result["sync_index"],
            "person_id": result["person_id"],
            "cameras_used": ",".join(str(c) for c in result["cameras_used"]),
        })
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(csv_path, index=False)
