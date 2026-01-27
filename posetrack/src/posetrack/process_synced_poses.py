# In process_synced_poses.py
import pandas as pd
import numpy as np
import cv2
from PIL import Image
import torch
import os
from tqdm import tqdm
import time
import argparse
import math # For distance calculation
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import glob
from re import match 

from .cs_parse import parse_calibration_mwc, parse_calibration_fmc, calculate_projection_matrices, triangulate_keypoints
from .pose_detector import load_models, detect_persons, detect_persons_batch, estimate_poses, estimate_poses_batch, SynthPoseMarkers, LOCAL_DET_DIR, LOCAL_SP_DIR
from .libwalk import quick_rotation_matrix


    # --- Multi-Person Tracking State ---
class PersonTrack:
    def __init__(self, person_id, track_id, keypoints_3d, sync_index, hip_indices, views_used=None, track_frames_til_lost_patience = 10, min_keypoints_for_com=2):
        self.track_id = track_id
        self.person_id = person_id
        self.keypoints_3d_history = [keypoints_3d]  # List of 3D keypoints over time
        self.views_used_history = [views_used] if views_used is not None else []  # List of views_used over time
        self.last_seen_sync = sync_index
        self.frames_since_seen = 0
        self.is_active = True
        self.hip_indices = hip_indices
        self.track_frames_til_lost_patience = track_frames_til_lost_patience
        self.min_keypoints_for_com = min_keypoints_for_com
        
    def update(self, keypoints_3d, sync_index, views_used=None):
        self.keypoints_3d_history.append(keypoints_3d)
        if views_used is not None:
            self.views_used_history.append(views_used)
        self.last_seen_sync = sync_index
        self.frames_since_seen = 0
        
    def get_last_views_used(self):
        """Get the views_used from the most recent frame."""
        if not self.views_used_history:
            return None
        return self.views_used_history[-1]
        
    def increment_lost_counter(self):
        self.frames_since_seen += 1
        if self.frames_since_seen > self.track_frames_til_lost_patience:
            self.is_active = False
            
    def get_com_3d(self):
        """Get 3D center of mass from last known keypoints."""
        if not self.keypoints_3d_history:
            return None
        last_kps = self.keypoints_3d_history[-1]
        if last_kps is None:
            return None
        
        # Get hip keypoints
        left_hip = last_kps[self.hip_indices[0]] if self.hip_indices[0] < len(last_kps) else None
        right_hip = last_kps[self.hip_indices[1]] if self.hip_indices[1] < len(last_kps) else None
        
        valid_hips = []
        if left_hip is not None and not np.isnan(left_hip).any():
            valid_hips.append(left_hip)
        if right_hip is not None and not np.isnan(right_hip).any():
            valid_hips.append(right_hip)
            
        if len(valid_hips) >= self.min_keypoints_for_com:
            return np.mean(valid_hips, axis=0)
        return None

# --- Add the project_3d_to_2d helper function here or import it ---
def project_3d_to_2d(point_3d, P):
    """Projects a 3D point to 2D using a projection matrix."""
    if point_3d is None or np.isnan(point_3d).any(): return None
    point_4d = np.append(point_3d, 1.0)
    point_2d_hom = P @ point_4d
    if abs(point_2d_hom[2]) < 1e-6 : return None # Check for near-zero depth
    point_2d = point_2d_hom[:2] / point_2d_hom[2]
    return point_2d.flatten()

def project_keypoints_to_all_cameras_ultrafast(keypoints_3d, projection_matrices, common_ports, port_to_cam_index):
    """Process all cameras simultaneously for maximum speed."""
    
    n_keypoints = len(keypoints_3d)
    n_cameras = len(common_ports)
    
    # Create mask and indices once
    valid_mask = np.array([kp is not None for kp in keypoints_3d], dtype=bool)
    valid_idx = np.where(valid_mask)[0]
    
    # Build homogeneous coordinates
    kp_homo = np.zeros((n_keypoints, 4))
    kp_homo[:, 3] = 1
    if len(valid_idx) > 0:
        kp_homo[valid_idx, :3] = np.array([keypoints_3d[i] for i in valid_idx])
    
    # Stack all projection matrices
    P_all = np.array([projection_matrices[port_to_cam_index[port]] for port in common_ports])
    
    # Project to all cameras at once: (n_cameras, 3, 4) @ (4, n_keypoints)
    all_projected = np.einsum('cij,jk->cik', P_all, kp_homo.T)  # (n_cameras, 3, n_keypoints)
    
    # Pre-allocate all results
    all_pixels = np.full((n_cameras, n_keypoints, 2), np.nan)
    
    # Process all cameras
    if len(valid_idx) > 0:
        z_vals = all_projected[:, 2, valid_idx]  # (n_cameras, n_valid)
        
        for c in range(n_cameras):
            z_valid = z_vals[c] > 1e-6
            final_idx = valid_idx[z_valid]
            if len(final_idx) > 0:
                all_pixels[c, final_idx, 0] = all_projected[c, 0, final_idx] / all_projected[c, 2, final_idx]
                all_pixels[c, final_idx, 1] = all_projected[c, 1, final_idx] / all_projected[c, 2, final_idx]
    
    # Convert to dictionary
    pixel_coords = {port: all_pixels[i].tolist() for i, port in enumerate(common_ports)}
    
    return pixel_coords

def project_keypoints_to_all_cameras(keypoints_3d, projection_matrices, common_ports, port_to_cam_index):
    """Project 3D keypoints to 2D pixel coordinates for all cameras."""
    pixel_coords = {}
    
    for port in common_ports:
        cam_idx = port_to_cam_index[port]
        P = projection_matrices[cam_idx]
        
        port_pixels = []
        for kp_3d in keypoints_3d:
            if kp_3d is None or np.isnan(kp_3d).any():
                port_pixels.append([np.nan, np.nan])
            else:
                pixel_2d = project_3d_to_2d(kp_3d, P)
                if pixel_2d is not None:
                    port_pixels.append(pixel_2d.tolist())
                else:
                    port_pixels.append([np.nan, np.nan])
        
        pixel_coords[port] = port_pixels
    
    return pixel_coords

def plot_debug_keypoints(debug_frames, all_pixel_coords_by_person, common_ports, first_sync_index):
    """Plot projected keypoints on the first frame for debugging."""
    if not debug_frames or not all_pixel_coords_by_person:
        return
    
    num_ports = len(common_ports)
    fig, axes = plt.subplots(1, num_ports, figsize=(5 * num_ports, 5))
    if num_ports == 1:
        axes = [axes]
    
    for i, port in enumerate(common_ports):
        ax = axes[i]
        
        # Convert PIL image to numpy array for plotting
        img_array = np.array(debug_frames[port])
        ax.imshow(img_array)
        ax.set_title(f'Port {port} - First Frame Projected Keypoints')
        ax.set_xlabel('X (pixels)')
        ax.set_ylabel('Y (pixels)')
        
        # Plot keypoints for each person
        colors = ['red', 'blue', 'green', 'yellow', 'purple', 'orange']
        for person_idx, (person_id, pixel_results) in enumerate(all_pixel_coords_by_person.items()):
            if not pixel_results:
                continue
                
            # Find the result for the first sync_index
            first_result = None
            for result in pixel_results:
                if result['sync_index'] == first_sync_index:
                    first_result = result
                    break
            
            if first_result is None:
                continue
                
            pixel_coords = first_result['pixel_coords'].get(port, [])
            color = colors[person_idx % len(colors)]
            
            # Plot keypoints
            for kp_idx, coords in enumerate(pixel_coords):
                if coords and len(coords) == 2 and not np.isnan(coords).any():
                    ax.plot(coords[0], coords[1], 'o', color=color, markersize=3, 
                           label=f'Person {person_id}' if kp_idx == 0 else "")
        
        ax.legend()
        ax.set_xlim(0, img_array.shape[1])
        ax.set_ylim(img_array.shape[0], 0)  # Invert y-axis for image coordinates
    
    plt.tight_layout()
    plt.show()
# --- Helper Functions ---

def calculate_2d_com(keypoints_2d, hip_indices):
    """Calculate 2D center of mass from hip keypoints."""
    valid_hips = []
    for hip_idx in hip_indices:
        if hip_idx < len(keypoints_2d):
            hip = keypoints_2d[hip_idx][:2]  # Just x,y
            if not np.isnan(hip).any():
                valid_hips.append(hip)
    
    if len(valid_hips) >= 1:  # Even one hip is enough for 2D
        return np.mean(valid_hips, axis=0)
    return None

def generate_3d_candidates_from_groups(groups, port_to_cam_index, camera_params, projection_matrices, hip_indices):
    """
    Generate 3D person candidates by triangulating detections across groups.
    
    Returns:
        List of dicts, one per group. Each dict contains:
        - 'candidates': list of 3D candidates for this person
        - 'best': the highest quality candidate (or None if no valid candidates)
    """
    import itertools
    
    results = []
    
    for group_idx, group in enumerate(groups):
        active_ports = list(group.keys())
        
        if len(active_ports) < 2:
            # Still track this group even if we can't triangulate
            results.append({'candidates': [], 'best': None})
            continue
        
        group_candidates = []
        
        # Generate all combinations of 2 or more views
        for num_views in range(2, len(active_ports) + 1):
            for view_combination in itertools.combinations(active_ports, num_views):
                kps_to_triangulate = {}
                total_confidence = 0
                valid_views = 0
                
                for port in view_combination:
                    detection = group[port]
                    
                    if detection['com_2d'] is not None:  # Using com_2d to match Hungarian grouping
                        kps_to_triangulate[port] = detection['keypoints']
                        total_confidence += detection['confidence']
                        valid_views += 1
                
                if valid_views >= 2:
                    keypoints_3d = triangulate_keypoints(
                        kps_to_triangulate, port_to_cam_index, 
                        camera_params, projection_matrices
                    )
                    
                    if keypoints_3d and any(kp is not None for kp in keypoints_3d):
                        com_3d = calculate_3d_com_from_keypoints(keypoints_3d, hip_indices)
                        
                        if com_3d is not None:
                            avg_confidence = total_confidence / valid_views
                            triangulation_quality = calculate_triangulation_quality(
                                kps_to_triangulate, keypoints_3d, projection_matrices, 
                                port_to_cam_index, hip_indices
                            )
                            
                            candidate = {
                                'keypoints_3d': keypoints_3d,
                                'com_3d': com_3d,
                                'triang_quality': triangulation_quality,
                                'views': view_combination,
                                'num_views': len(view_combination)
                            }
                            group_candidates.append(candidate)

        results.append(group_candidates)

    return results


def calculate_triangulation_quality(kps_to_triangulate, keypoints_3d, projection_matrices, port_to_cam_index, hip_indices):
    """
    Calculate quality of triangulation by measuring reprojection error of hip keypoints.
    Returns score between 0 and 1 (1 = perfect, 0 = terrible).
    """
    # Focus on hip keypoints for quality assessment
    hip_errors = []
    
    for hip_idx in hip_indices:
        if hip_idx < len(keypoints_3d) and keypoints_3d[hip_idx] is not None:
            com_3d = keypoints_3d[hip_idx]
            
            # Calculate reprojection error across all views
            for port, kps_2d in kps_to_triangulate.items():
                if hip_idx < len(kps_2d):
                    observed_2d = kps_2d[hip_idx][:2]
                    if not np.isnan(observed_2d).any():
                        # Project 3D back to 2D
                        cam_idx = port_to_cam_index[port]
                        P = projection_matrices[cam_idx]
                        projected_2d = project_3d_to_2d(com_3d, P)
                        
                        if projected_2d is not None:
                            error = np.linalg.norm(observed_2d - projected_2d)
                            hip_errors.append(error)
    
    if not hip_errors:
        return 0.0
    
    # Convert pixel error to quality score (lower error = higher quality)
    avg_error = np.mean(hip_errors)
    max_acceptable_error = 50  # pixels
    quality = max(0, 1 - (avg_error / max_acceptable_error))
    
    return quality

def assign_3d_candidates_to_tracks(active_tracks:list[PersonTrack], candidate_groups, max_distance=0.2, default_views=None, max_tracks=None, min_new_track_distance=0.3, max_new_track_distance=5.0):
    """
    Assign candidate that matches existing view for each track by taking the detected person closest to each of the tracks.
    If n_tracks < max_tracks, also add the best new candidate groups as new tracks.
    
    Args:
        active_tracks: List of active tracks
        candidate_groups: List returned by generate_3d_candidates_from_groups,
                         where each element represents one person with multiple view combinations
        max_distance: Maximum distance for any match to existing tracks
        default_views: Default views to use for selecting candidates when no tracks exist
        max_tracks: Maximum number of tracks allowed. If current tracks < max_tracks,
                   best unassigned groups will be added as new tracks
        min_new_track_distance: Minimum distance (meters) a new track must be from existing tracks
        max_new_track_distance: Maximum distance (meters) from reference track for a new track
        
    Returns:
        assignments (dict): which group, and which candidate in that group. 
            Keys are track indices for existing tracks
        unassigned_groups: List of group indices not assigned to any track
        new_track_assignments (dict): Assignments for new tracks to be created
            Keys are new track indices (starting from n_tracks), values are (group_idx, candidate_idx)
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    
    n_tracks = len(active_tracks)
    n_groups = len(candidate_groups)
    
    # Build assignments
    assignments = {}
    used_groups = set()

    def set_default_views(ingroups):
        default_views = []
        default_views.append(ingroups[0][0]['views'])
        return default_views

    if n_tracks == 0:
        if default_views is None:
            default_views = set_default_views(candidate_groups)
        
        for i,(curgroup) in enumerate(candidate_groups):
            for j,cur3dcandidate in enumerate(curgroup):
                if set(cur3dcandidate['views']) == set(default_views[0]):
                    assignments[i] = (i,j)
        
        # If max_tracks is specified and we have no tracks, fill the assignments but new_track_assignments must be the same. 
        if max_tracks is not None and max_tracks > 0:
            new_track_assignments = {}
            for idx, (group_idx, cand_idx) in assignments.items():
                if idx < max_tracks:
                    new_track_assignments[idx] = (group_idx, cand_idx)
                    used_groups.add(group_idx)
            unassigned_groups = [g for g in range(n_groups) if g not in used_groups]
            return assignments, unassigned_groups, new_track_assignments
        
        return assignments, None, {}
    
    # Build cost matrix: tracks x groups
    # which track should be paired with which group? 
    HI = 1000.0
    cost_matrix = np.full((n_tracks, n_groups), HI) # cost matrix assigns a group
    best_candidate_per_cell = {}  # (track_idx, group_idx) -> candidate_idx
    
    # populate the cost_matrix that maps tracks to groups
    for i_track, track in enumerate(active_tracks):
        track_com = track.get_com_3d()
        track_views = track.get_last_views_used()
        
        if track_com is None or track_views is None:
            continue
        
        track_views_set = set(track_views)
        
        for group_idx, candidate_list in enumerate(candidate_groups):
            
            best_distance = np.inf
            best_cand_idx = None
            
            for cand_idx, candidate in enumerate(candidate_list):
                cand_views = set(candidate.get('views', []))
                
                # Hard constraint: must have exact same views
                if cand_views != track_views_set:
                    continue
                
                cand_com = candidate.get('hip_3d') or candidate.get('com_3d')
                if cand_com is None:
                    continue
                
                distance = np.linalg.norm(track_com - cand_com)
                if distance < best_distance and distance < max_distance:
                    best_distance = distance
                    best_cand_idx = cand_idx
            
            if best_cand_idx is not None:
                cost_matrix[i_track, group_idx] = best_distance
                best_candidate_per_cell[(i_track, group_idx)] = best_cand_idx
    
    # Solve assignment problem between tracks and groups
    if np.all(cost_matrix == HI):
        # No valid candidates for existing tracks
        unassigned_groups = list(range(n_groups))
    else:
        track_opt_indices, group_opt_indices = linear_sum_assignment(cost_matrix)
        
        for i_track, group_idx in zip(track_opt_indices, group_opt_indices):
            if cost_matrix[i_track, group_idx] < max_distance:
                # Get the specific candidate that was best for this pairing
                cand_idx = best_candidate_per_cell[(i_track, group_idx)]
                assignments[i_track] = (group_idx, cand_idx)
                used_groups.add(group_idx)
        
        # Find unassigned groups
        unassigned_groups = [g for g in range(n_groups) if g not in used_groups]
    
    # Add new tracks from unassigned groups if we haven't reached max_tracks
    new_track_assignments = {}
    if max_tracks is not None and n_tracks < max_tracks and len(unassigned_groups) > 0:
        n_new_tracks_to_add = min(max_tracks - n_tracks, len(unassigned_groups))
        
        if n_new_tracks_to_add > 0:
            # Get reference position from first track's assignment if available
            reference_position = None
            if 0 in assignments:
                group_idx, cand_idx = assignments[0]
                ref_candidate = candidate_groups[group_idx][cand_idx]
                reference_position = ref_candidate.get('hip_3d') or ref_candidate.get('com_3d')
            elif len(active_tracks) > 0:
                # Fallback to first track's current position if no assignment
                reference_position = active_tracks[0].get_com_3d()
            
            # If we still don't have a reference position, skip adding new tracks
            if reference_position is None:
                return assignments, unassigned_groups, new_track_assignments
            
            # Score unassigned groups to find the best ones to add as new tracks
            group_scores = []
            
            for group_idx in unassigned_groups:
                candidate_list = candidate_groups[group_idx]
                
                # Find the best candidate in this group based on distance criteria
                best_score = -np.inf
                best_cand_idx = None
                
                for cand_idx, candidate in enumerate(candidate_list):
                    # Get candidate's 3D position
                    cand_com = candidate.get('hip_3d') or candidate.get('com_3d')
                    if cand_com is None:
                        continue
                    
                    # Check distance from reference position (first track)
                    distance_from_reference = np.linalg.norm(cand_com - reference_position)
                    if distance_from_reference > max_new_track_distance:
                        continue  # Too far away from reference
                    
                    # Check minimum distance from ALL existing track positions
                    too_close = False
                    min_dist_to_tracks = np.inf
                    
                    # Check distance from existing tracks' current positions
                    for track in active_tracks:
                        track_com = track.get_com_3d()
                        if track_com is not None:
                            dist_to_track = np.linalg.norm(cand_com - track_com)
                            min_dist_to_tracks = min(min_dist_to_tracks, dist_to_track)
                            if dist_to_track < min_new_track_distance:
                                too_close = True
                                break
                    
                    # Also check distance from already assigned candidates
                    if not too_close:
                        for track_idx, (g_idx, c_idx) in assignments.items():
                            assigned_candidate = candidate_groups[g_idx][c_idx]
                            assigned_com = assigned_candidate.get('hip_3d') or assigned_candidate.get('com_3d')
                            if assigned_com is not None:
                                dist_to_assigned = np.linalg.norm(cand_com - assigned_com)
                                min_dist_to_tracks = min(min_dist_to_tracks, dist_to_assigned)
                                if dist_to_assigned < min_new_track_distance:
                                    too_close = True
                                    break
                    
                    if too_close:
                        continue  # Too close to an existing or assigned track
                    
                    # Score based on distance criteria
                    score = 0.0
                    
                    # Prefer candidates that are not too close but not too far
                    if min_dist_to_tracks != np.inf:
                        # Score higher for appropriate separation from existing tracks
                        optimal_distance = min_new_track_distance * 2  # Sweet spot is 2x minimum
                        if min_dist_to_tracks >= min_new_track_distance and min_dist_to_tracks <= optimal_distance:
                            score += 10.0
                        elif min_dist_to_tracks > optimal_distance:
                            # Penalize being too far from existing tracks
                            score += 5.0 - (min_dist_to_tracks - optimal_distance) * 0.5
                    
                    # Prefer candidates closer to the reference within the allowed range
                    # Linear penalty for distance from reference
                    score += (max_new_track_distance - distance_from_reference) / max_new_track_distance * 5.0
                    
                    # Bonus for matching default views if specified
                    if default_views is not None:
                        cand_views = set(candidate.get('views', []))
                        if cand_views == set(default_views[0]):
                            score += 15.0  # High bonus for matching default views
                    
                    if score > best_score:
                        best_score = score
                        best_cand_idx = cand_idx
                
                if best_cand_idx is not None:
                    group_scores.append((best_score, group_idx, best_cand_idx))
            
            # Sort by score and take the best ones
            if len(group_scores) > 0:
                group_scores.sort(reverse=True)
                
                for i in range(min(n_new_tracks_to_add, len(group_scores))):
                    score, group_idx, cand_idx = group_scores[i]
                    new_track_idx = n_tracks + i
                    assignments[new_track_idx] = (group_idx, cand_idx)
                    new_track_assignments[new_track_idx] = (group_idx, cand_idx)
                    used_groups.add(group_idx)
            
            # Update unassigned groups after adding new tracks
            unassigned_groups = [g for g in range(n_groups) if g not in used_groups]
    
    return assignments, unassigned_groups, new_track_assignments

def process_synced_mwc_frames_multi_person_perf(
    frame_history_csv_path, calibration_path, video_dir, output_path, model_dir=LOCAL_SP_DIR,
    detector_dir=LOCAL_DET_DIR, calib_type='mwc', skip_sync_indices=1, person_confidence=0.1,
    keypoint_confidence=0.1, device_name="auto",
    # --- Tracking parameters ---
    max_persons=2,  # Maximum number of persons to track
    com_distance_threshold=0.3,  # meters - minimum distance between COMs to be different people
    track_frames_til_lost_patience=30,  # frames to wait before considering a track lost
    min_keypoints_for_com=2,  # minimum valid hip keypoints needed to compute COM
    hip_indices=(11, 12),  # COCO format: left hip, right hip
    epipolar_threshold=30,  # pixels - max distance from epipolar line
    reprojection_error_threshold=50,  # pixels - max reprojection error after triangulation
    min_views_for_detection=2,  # minimum camera views to confirm a person
    iou_threshold=0.3,  # for matching bounding boxes across views
    temporal_smoothing_window=5,  # frames for temporal consistency
    min_track_length=5,  # minimum frames before considering a track valid
    max_com_velocity=2.0,  # m/s - maximum reasonable COM velocity between frames
    verbose_debug=False,  # Enable debugging visualization for first frame
    override_views_used=None,  # Optional list of view combinations to prioritize instead of previous frame
    # --- NEW BATCH PROCESSING PARAMETERS ---
    batch_size=8,  # Number of frames to process simultaneously (configurable)
    ):
    """Processes synchronized frames with multi-person 3D tracking using COM-based matching with batch processing."""

    # --- 1. Setup Device ---
    def setup_device():
        """Setup and return the compute device (cuda/mps/cpu)."""
        if device_name == "auto":
            if torch.backends.mps.is_available() and torch.backends.mps.is_built(): 
                device = "mps"
            elif torch.cuda.is_available(): 
                device = "cuda"
            else: 
                device = "cpu"
        else:
            device = device_name
            if device == "mps" and (not torch.backends.mps.is_available() or not torch.backends.mps.is_built()): 
                print("Warn: MPS unavailable.")
                device = "cpu"
            elif device == "cuda" and not torch.cuda.is_available(): 
                print("Warn: CUDA unavailable.")
                device = "cpu"
        print(f"Using device: {device}")
        return device

    # --- 2. Load Frame History & Derive 0-Based Index ---
    def load_frame_history():
        """Load and process frame history CSV."""
        print(f"Loading frame history from: {frame_history_csv_path}")
        try:
            frame_history_df = pd.read_csv(frame_history_csv_path)
            frame_history_df['sync_index'] = frame_history_df['sync_index'].astype(int)
            frame_history_df['port'] = frame_history_df['port'].astype(int)
            frame_history_df['frame_time'] = frame_history_df['frame_time'].astype(float)
            frame_history_df = frame_history_df.sort_values(by=['port', 'frame_time'])
            frame_history_df['derived_frame_index'] = frame_history_df.groupby('port')['frame_time'].rank(method='min').astype(int) - 1
        except Exception as e: 
            print(f"Error reading/processing frame history CSV: {e}")
            return None
        
        print(f"Found {frame_history_df['sync_index'].nunique()} unique sync indices.")
        print(f"Ports found: {sorted(frame_history_df['port'].unique())}")
        return frame_history_df

    # --- 3. Load Calibration & Filter ---
    def load_and_filter_calibration(frame_history_df):
        """Load calibration and filter to common ports between CSV and calibration."""
        print(f"Loading calibration ({calib_type}) from: {calibration_path}")
        
        if calib_type.lower() == 'mwc': 
            camera_params = parse_calibration_mwc(calibration_path)
        elif calib_type.lower() == 'fmc': 
            camera_params = parse_calibration_fmc(calibration_path)
        else: 
            print(f"Error: Invalid calib type '{calib_type}'.")
            return None, None, None
        
        if not camera_params: 
            print("Error: Failed to load camera calibration.")
            return None, None, None
        
        print(f"Loaded {len(camera_params)} cameras initially.")
        
        # Map ports to camera indices
        port_to_cam_index = {}
        calibration_ports = set()
        for idx, params in enumerate(camera_params):
            port = params.get('port')
            if port is not None: 
                port_to_cam_index[port] = idx
                calibration_ports.add(port)
            else:
                try: 
                    inferred_port = int(params.get('name', f'cam_{idx}').split('_')[-1])
                    port_to_cam_index[inferred_port] = idx
                    calibration_ports.add(inferred_port)
                except ValueError: 
                    print(f"Warn: Cannot determine port for camera index {idx}.")
        
        # Find common ports
        csv_ports = set(frame_history_df['port'].unique())
        if not csv_ports.issubset(calibration_ports): 
            print(f"Warn: Ports mismatch CSV:{csv_ports} vs Calib:{calibration_ports}.")
        
        common_ports = sorted(list(csv_ports.intersection(calibration_ports)))
        print(f"Using common ports for processing: {common_ports}")
        
        if len(common_ports) < 2: 
            print("Error: Need >= 2 common ports.")
            return None, None, None
        
        # Filter camera params to common ports
        filtered_cam_params = []
        filtered_port_map = {}
        new_idx = 0
        for port in common_ports: 
            original_idx = port_to_cam_index[port]
            filtered_cam_params.append(camera_params[original_idx])
            filtered_port_map[port] = new_idx
            new_idx += 1
        
        camera_params = filtered_cam_params
        port_to_cam_index = filtered_port_map
        num_cameras = len(camera_params)
        print(f"Filtered calibration to {num_cameras} cameras.")
        
        return camera_params, port_to_cam_index, common_ports

    # --- 4. Calculate Projection Matrices ---
    def calculate_projections(camera_params, num_cameras):
        """Calculate projection matrices for all cameras."""
        print("Calculating projection matrices...")
        projection_matrices = calculate_projection_matrices(camera_params)
        if len(projection_matrices) != num_cameras: 
            print("Error: Proj matrix count mismatch.")
            return None
        return projection_matrices

    # --- 5. Load Models with Optimization ---
    def load_and_optimize_models(device):
        """Load detection and pose estimation models with optimization."""
        print("Loading detection and pose estimation models...")
        try:
            person_processor, person_model, pose_processor, pose_model = load_models(
                detect_path=detector_dir, pose_model_path=model_dir, device=device
            )
            
            # Optimize models for inference
            print("Optimizing models for batch inference...")
            person_model.eval()
            pose_model.eval()
            
            # Warm up models with dummy data
            with torch.no_grad():
                dummy_image = torch.randn(1, 3, 224, 224).to(device)
                try:
                    _ = person_model(dummy_image)
                    print("Person detection model warmed up")
                except:
                    print("Person model warmup skipped (different input format)")
                
                try:
                    dummy_pose_input = torch.randn(1, 3, 256, 192).to(device)
                    _ = pose_model(dummy_pose_input)
                    print("Pose estimation model warmed up")
                except:
                    print("Pose model warmup skipped (different input format)")
                    
        except Exception as e: 
            print(f"Error loading models: {e}")
            return None
        
        return person_processor, person_model, pose_processor, pose_model

    # --- 6. Open Video Files ---
    def open_video_files(common_ports, frame_history_df):
        """Open video capture for all required ports and validate sync ranges."""
        print("Opening video capture for required ports...")
        caps = {}
        video_lengths = {}
        
        for port in common_ports:
            video_path = os.path.join(video_dir, f"port_{port}.mp4")
            if not os.path.exists(video_path): 
                print(f"Error: Vid not found: {video_path}")
                [c.release() for c in caps.values()]
                return None, None
            
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened(): 
                print(f"Error: Cannot open vid: {video_path}")
                [c.release() for c in caps.values()]
                return None, None
            
            caps[port] = cap
            video_lengths[port] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            print(f"Opened video for port {port}: {video_path} ({video_lengths[port]} frames)")
        
        # Validate sync range vs video length
        all_sync_indices = sorted(frame_history_df['sync_index'].unique())
        sync_range = all_sync_indices[-1] - all_sync_indices[0] + 1
        
        print(f"\n--- Sync Validation ---")
        print(f"Sync index range: {all_sync_indices[0]} to {all_sync_indices[-1]} ({len(all_sync_indices)} unique indices)")
        print(f"Expected sync range: {sync_range} frames")
        
        sync_matches_video = True
        for port in common_ports:
            video_len = video_lengths[port]
            if abs(sync_range - video_len) > 1:
                print(f"WARNING: Port {port} video length ({video_len}) doesn't match sync range ({sync_range})")
                sync_matches_video = False
            else:
                print(f"Port {port}: sync range ({sync_range}) matches video length ({video_len}) ✓")
        
        if sync_matches_video:
            print("✓ Frame alignment validation PASSED")
        else:
            print("⚠ Frame alignment validation FAILED")
        
        return caps, video_lengths

    # --- 7. Process Synchronized Frames with Batch Processing ---
    def process_frames_in_batches(frame_history_df, common_ports, caps, video_lengths, 
                                   projection_matrices, port_to_cam_index, camera_params,
                                   person_processor, person_model, pose_processor, pose_model, device):
        """Main batch processing loop for all synchronized frames."""
        
        # Initialize tracking state
        active_tracks = []
        next_track_id = 0
        all_results_by_person = {}
        all_pixel_coords_by_person = {}
        all_cameras_by_person = {}
        previous_views_used = []
        debug_frames = {}
        next_person_id = 0
        
        all_sync_indices = sorted(frame_history_df['sync_index'].unique())
        start_time = time.time()
        
        print(f"Starting multi-person processing for {len(all_sync_indices)} sync indices with batch size {batch_size}...")
        
        # Filter sync indices based on skip_sync_indices
        sync_index_counter = 0
        filtered_sync_indices = []
        for sync_index in all_sync_indices:
            sync_index_counter += 1
            if (sync_index_counter - 1) % skip_sync_indices != 0: 
                continue
            filtered_sync_indices.append(sync_index)
        
        print(f"Processing {len(filtered_sync_indices)} filtered sync indices...")
        
        # Process in batches
        for batch_start_idx in tqdm(range(0, len(filtered_sync_indices), batch_size), desc="Processing Batches"):
            batch_end_idx = min(batch_start_idx + batch_size, len(filtered_sync_indices))
            current_batch_indices = filtered_sync_indices[batch_start_idx:batch_end_idx]
            
            # Read batch of frames
            batch_frames, valid_batch_indices = read_batch_frames(
                current_batch_indices, frame_history_df, common_ports, caps, video_lengths
            )
            
            if not valid_batch_indices:
                continue
            
            # Perform batch detection and pose estimation
            batch_detections = perform_batch_detection_and_pose(
                batch_frames, valid_batch_indices, common_ports,
                person_processor, person_model, pose_processor, pose_model,
                device, person_confidence, keypoint_confidence, hip_indices, batch_size
            )
            
            # Process each frame in the batch
            for sync_index in valid_batch_indices:
                active_tracks, next_person_id, all_results_by_person,\
                    all_pixel_coords_by_person, all_cameras_by_person, \
                        previous_views_used = process_single_syncedframe(
                    sync_index, batch_detections[sync_index], 
                    active_tracks, next_person_id, previous_views_used,
                    all_results_by_person, all_pixel_coords_by_person, all_cameras_by_person,
                    projection_matrices, port_to_cam_index, camera_params, common_ports,
                    hip_indices, max_persons, override_views_used, track_frames_til_lost_patience
                )
            
            # Store first frame for debugging if enabled
            if verbose_debug and all_sync_indices[0] in valid_batch_indices:
                debug_frames = batch_frames[all_sync_indices[0]].copy()
        
        end_time = time.time()
        print(f"Processing finished in {end_time - start_time:.2f} seconds.")
        
        return all_results_by_person, all_pixel_coords_by_person, all_cameras_by_person, debug_frames

    def read_batch_frames(current_batch_indices, frame_history_df, common_ports, caps, video_lengths):
        """Read frames for a batch of sync indices."""
        batch_frames = {}
        valid_batch_indices = []
        
        for sync_index in current_batch_indices:
            sync_data = frame_history_df[frame_history_df['sync_index'] == sync_index]
            if set(sync_data['port']) != set(common_ports): 
                continue
            
            current_frames_pil = {}
            frame_read_success = True
            
            for _, row in sync_data.iterrows():
                port = row['port']
                frame_idx_to_read = row['derived_frame_index']
                
                try: 
                    frame_idx_int = int(frame_idx_to_read)
                except (ValueError, TypeError): 
                    print(f"Warn: Invalid derived index '{frame_idx_to_read}' p{port} s{sync_index}")
                    frame_read_success = False
                    break
                
                cap = caps.get(port)
                total_frames = video_lengths.get(port, -1)
                
                if cap is None or total_frames == -1: 
                    print(f"CritErr: No capture/len p{port}")
                    frame_read_success = False
                    break
                
                if not (0 <= frame_idx_int < total_frames): 
                    print(f"Warn: Idx {frame_idx_int} out of bounds (0-{total_frames-1}) p{port} s{sync_index}")
                    frame_read_success = False
                    break
                
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_int)
                ret, frame = cap.read()
                
                if not ret: 
                    print(f"Warn: Failed read idx {frame_idx_int} p{port} s{sync_index}")
                    frame_read_success = False
                    break
                
                current_frames_pil[port] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
            if frame_read_success:
                batch_frames[sync_index] = current_frames_pil
                valid_batch_indices.append(sync_index)
        
        return batch_frames, valid_batch_indices

    def perform_batch_detection_and_pose(batch_frames, valid_batch_indices, common_ports,
                                          person_processor, person_model, pose_processor, pose_model,
                                          device, person_confidence, keypoint_confidence, hip_indices, batch_size):
        """Perform batch person detection and pose estimation."""
        batch_detections = {}
        
        # Collect all images for batch processing
        all_images = []
        image_metadata = []
        
        for sync_index in valid_batch_indices:
            for port in common_ports:
                if port in batch_frames[sync_index]:
                    all_images.append(batch_frames[sync_index][port])
                    image_metadata.append((sync_index, port))
        
        if not all_images:
            return batch_detections
        
        print(f"True batch processing {len(all_images)} images...")
        
        with torch.no_grad():
            # Batch person detection
            batch_person_results = detect_persons_batch(
                all_images, person_processor, person_model, device, person_confidence, batch_size
            )
            
            # Prepare data for batch pose estimation
            images_with_boxes = []
            for img_idx, (image, (person_boxes_voc, person_boxes_coco, person_scores)) in enumerate(zip(all_images, batch_person_results)):
                images_with_boxes.append((image, person_boxes_coco))
            
            # Batch pose estimation
            batch_pose_results = estimate_poses_batch(
                images_with_boxes, pose_processor, pose_model, device, batch_size
            )
        
        # Organize results back into sync_index/port structure
        for img_idx, (sync_index, port) in enumerate(image_metadata):
            if sync_index not in batch_detections:
                batch_detections[sync_index] = {}
            
            person_boxes_voc, person_boxes_coco, person_scores = batch_person_results[img_idx]
            all_keypoints_2d, all_keypoint_scores = batch_pose_results[img_idx]
            
            port_detections = []
            
            if person_boxes_coco.size > 0:
                for person_idx in range(len(all_keypoints_2d)):
                    if person_idx < len(all_keypoints_2d) and person_idx < len(all_keypoint_scores):
                        kps_2d, scores_2d = all_keypoints_2d[person_idx], all_keypoint_scores[person_idx]
                        detection_confidence = person_scores[person_idx] if person_idx < len(person_scores) else 0.0
                        
                        valid_kps_view = []
                        
                        # Process keypoints
                        for kp_idx in range(len(kps_2d)):
                            kp, score = kps_2d[kp_idx], scores_2d[kp_idx]
                            if score >= keypoint_confidence:
                                valid_kps_view.append([kp[0], kp[1], score])
                            else:
                                valid_kps_view.append([np.nan, np.nan, score])
                        
                        # Calculate 2D hip center for tracking
                        com_2d = calculate_2d_com(valid_kps_view, hip_indices)
                        if com_2d is None:
                            print(f"Warn: No valid hips for COM p{port} s{sync_index} person{person_idx}")
                            com_2d = calculate_2d_com(valid_kps_view, (hip_indices[0], hip_indices[0]))
                            if com_2d is None:
                                com_2d = [np.nan, np.nan]
                                print(f"Warn: No valid single hips for COM p{port} s{sync_index} person{person_idx}")
                                continue
                        
                        person_data = {
                            'keypoints': np.array(valid_kps_view),
                            'com_2d': com_2d,
                            'confidence': detection_confidence
                        }
                        
                        port_detections.append(person_data)
            
            batch_detections[sync_index][port] = port_detections
        
        return batch_detections

    def process_single_syncedframe(sync_index, view_detections, active_tracks, next_person_id, previous_views_used,
                              all_results_by_person, all_pixel_coords_by_person, all_cameras_by_person,
                              projection_matrices, port_to_cam_index, camera_params, common_ports,
                              hip_indices, max_persons, override_views_used, track_frames_til_lost_patience):
        """Process a single frame's detections and update tracking."""
        
        # Group detections across views
        groups = group_detections_across_views_bipartite_full(
            view_detections, None, projection_matrices, port_to_cam_index,
            camera_params, epipolar_threshold=10
        )
        
        # Generate 3D candidates
        candidate_3d_groups = generate_3d_candidates_from_groups(
            groups, port_to_cam_index, camera_params, projection_matrices, hip_indices
        )
        
        # Associate 3D candidates with existing tracks
        track_assignments, unused_3d_groups, newtrackassignments = assign_3d_candidates_to_tracks(
            active_tracks, candidate_3d_groups, 0.15, default_views=override_views_used, max_tracks = max_persons,max_new_track_distance=4,min_new_track_distance=.3
        )
        
        # Check if this is a bad frame and skip 
        if len(active_tracks) > 0 and not track_assignments:
            print(f"Bad frame detected at sync {sync_index}: no tracks matched")
            for track in active_tracks:
                track.increment_lost_counter()
                if not track.is_active:
                    print(f"Lost track of person {track.person_id} at sync {sync_index}")
            active_tracks = [t for t in active_tracks if t.is_active]
            return active_tracks, next_person_id, all_results_by_person, all_pixel_coords_by_person, all_cameras_by_person, previous_views_used
        
        #### for all track_assignments, update existing tracks or create new ones
        for track_id, curtuple in track_assignments.items():
            if curtuple is None:
                continue
            
            grp_idx, candidate_idx = curtuple
            if candidate_idx is None:
                continue
            
            candidate = candidate_3d_groups[grp_idx][candidate_idx]
            existing_track = None
            
            # Update existing track or create new one
            if track_id < len(active_tracks):
                existing_track = active_tracks[track_id]
                existing_track.update(
                    candidate['keypoints_3d'], 
                    sync_index, 
                    candidate['views']
                )
            else:
                # Create new track only if we haven't hit the max concurrent tracks
                if len(active_tracks) < max_persons:
                    new_track = PersonTrack(
                        person_id=next_person_id,
                        track_id=track_id,
                        keypoints_3d=candidate['keypoints_3d'],
                        sync_index=sync_index,
                        hip_indices=hip_indices,
                        views_used=candidate['views'],
                        track_frames_til_lost_patience=track_frames_til_lost_patience
                    )
                    active_tracks.append(new_track)
                    next_person_id += 1
                else:
                    print(f"Warning: Cannot create new track {track_id}, max persons ({max_persons}) reached")
                    continue
            
            # Get the person_id from the track
            track = existing_track if existing_track else new_track
            person_id = track.person_id
            
            # Initialize storage for this person if needed
            if person_id not in all_results_by_person:
                all_results_by_person[person_id] = []
            if person_id not in all_pixel_coords_by_person:
                all_pixel_coords_by_person[person_id] = []
            if person_id not in all_cameras_by_person:
                all_cameras_by_person[person_id] = []
            
            # Store 3D keypoints
            kps_3d_list = [[np.nan]*3 if kp is None else kp.tolist() 
                          for kp in candidate['keypoints_3d']]
            all_results_by_person[person_id].append({
                'sync_index': sync_index,
                'person_id': person_id,
                'keypoints_3d': kps_3d_list
            })
            
            # Project and store pixel coordinates
            pixel_coords = project_keypoints_to_all_cameras_ultrafast(
                candidate['keypoints_3d'], 
                projection_matrices, 
                common_ports, 
                port_to_cam_index
            )
            all_pixel_coords_by_person[person_id].append({
                'sync_index': sync_index,
                'person_id': person_id,
                'pixel_coords': pixel_coords
            })
            
            # Store camera information
            all_cameras_by_person[person_id].append({
                'sync_index': sync_index,
                'person_id': person_id,
                'cameras_used': candidate['views']
            })
        
        # Update lost counters for unmatched tracks
        matched_track_ids = {track_id for track_id, assignment in track_assignments.items() 
                           if assignment is not None and assignment[1] is not None}
        
        for track in active_tracks:
            if track.track_id not in matched_track_ids:
                track.increment_lost_counter()
                if not track.is_active:
                    print(f"Lost track of person {track.person_id} (track {track.track_id}) at sync {sync_index}")
        
        # Update previous_views_used
        current_views_used = []
        for track_idx, tuple_idxs in track_assignments.items():
            grp_idx = tuple_idxs[0]
            candidate_idx = tuple_idxs[1]
            if candidate_idx is not None:
                candidate = candidate_3d_groups[grp_idx][candidate_idx]
                current_views_used.append(candidate['views'])
        
        previous_views_used = current_views_used
        
        # Remove inactive tracks
        active_tracks = [t for t in active_tracks if t.is_active]
        
        return active_tracks, next_person_id, all_results_by_person, all_pixel_coords_by_person, all_cameras_by_person, previous_views_used

    # --- 8. Cleanup and Save Results ---
    def save_all_results(all_results_by_person, all_pixel_coords_by_person, all_cameras_by_person, 
                        common_ports, debug_frames, all_sync_indices):
        """Save results for each tracked person."""
        
        for person_id, results in all_results_by_person.items():
            if not results:
                continue
            
            print(f"\nSaving results for Person {person_id} ({len(results)} frames)...")
            
            # Define output filenames for this person
            person_csv_path = f"{output_path}_person{person_id}.csv"
            person_pickle_path = f"{output_path}_person{person_id}.pkl"
            
            # Save pickle
            try:
                with open(person_pickle_path, 'wb') as f_pkl:
                    pickle.dump(results, f_pkl)
                print(f"Saved pickle: {person_pickle_path}")
            except Exception as e:
                print(f"Error saving pickle for person {person_id}: {e}")
            
            # Save CSV
            save_person_csv(results, person_csv_path, expected_num_kps=52)
            
            # Save pixel coordinates CSV for each port
            if person_id in all_pixel_coords_by_person:
                pixel_results = all_pixel_coords_by_person[person_id]
                for port in common_ports:
                    pixel_csv_path = f"{output_path}_person{person_id}_pixelcoords_port_{port}.csv"
                    save_pixel_coords_csv(pixel_results, pixel_csv_path, expected_num_kps=52, port=port)
            
            # Save camera information CSV
            if person_id in all_cameras_by_person:
                camera_results = all_cameras_by_person[person_id]
                cameras_csv_path = f"{output_path}_person{person_id}_cameras.csv"
                save_cameras_csv(camera_results, cameras_csv_path)
        
        # Debug visualization if enabled
        if verbose_debug and debug_frames and all_pixel_coords_by_person:
            print("\nGenerating debug visualization...")
            plot_debug_keypoints(debug_frames, all_pixel_coords_by_person, common_ports, all_sync_indices[0])
        
        print(f"\nTotal persons tracked: {len(all_results_by_person)}")

    # ==================== MAIN EXECUTION FLOW ====================
    # Step 1: Setup device
    device = setup_device()
    
    # Step 2: Load frame history
    frame_history_df = load_frame_history()
    if frame_history_df is None:
        return
    
    # Step 3: Load and filter calibration
    camera_params, port_to_cam_index, common_ports = load_and_filter_calibration(frame_history_df)
    if camera_params is None:
        return
    
    num_cameras = len(camera_params)
    
    # Step 4: Calculate projection matrices
    projection_matrices = calculate_projections(camera_params, num_cameras)
    if projection_matrices is None:
        return
    
    # Step 5: Load and optimize models
    models = load_and_optimize_models(device)
    if models is None:
        return
    person_processor, person_model, pose_processor, pose_model = models
    
    # Step 6: Open video files and validate sync
    caps, video_lengths = open_video_files(common_ports, frame_history_df)
    if caps is None:
        return
    
    # Step 7: Process frames in batches
    all_sync_indices = sorted(frame_history_df['sync_index'].unique())
    all_results_by_person, all_pixel_coords_by_person, all_cameras_by_person, debug_frames = process_frames_in_batches(
        frame_history_df, common_ports, caps, video_lengths,
        projection_matrices, port_to_cam_index, camera_params,
        person_processor, person_model, pose_processor, pose_model, device
    )
    
    # Cleanup: Release video captures
    print("\nReleasing video captures...")
    [cap.release() for cap in caps.values()]
    
    # Step 8: Save all results
    save_all_results(
        all_results_by_person, all_pixel_coords_by_person, all_cameras_by_person,
        common_ports, debug_frames, all_sync_indices
    )
    
    return all_results_by_person

#-- helper functions

def calculate_fundamental_matrix(P1, P2):
    """
    Calculate the fundamental matrix F from two projection matrices.
    
    The fundamental matrix relates corresponding points x1 and x2 in two images:
    x2^T * F * x1 = 0
    
    Args:
        P1: 3x4 projection matrix for camera 1
        P2: 3x4 projection matrix for camera 2
    
    Returns:
        F: 3x3 fundamental matrix
    """
    # Method: F = [e2]_x * P2 * P1^+
    # where P1^+ is the pseudo-inverse of P1
    # and [e2]_x is the skew-symmetric matrix of the epipole e2
    
    # Calculate the camera center C1 (null space of P1)
    # P1 * C1 = 0, where C1 is in homogeneous coordinates
    U, S, Vt = np.linalg.svd(P1)
    C1 = Vt[-1, :]  # Last row of V^T (null space)
    C1 = C1 / C1[3]  # Normalize so last coordinate is 1
    
    # Project C1 into image 2 to get epipole e2
    e2 = P2 @ C1
    e2 = e2 / e2[2]  # Normalize
    
    # Create skew-symmetric matrix [e2]_x
    e2_cross = np.array([
        [0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0]
    ])
    
    # Calculate F = [e2]_x * P2 * P1^+
    P1_pinv = np.linalg.pinv(P1)
    F = e2_cross @ P2 @ P1_pinv
    
    # Enforce rank-2 constraint (fundamental matrix should have rank 2)
    U, S, Vt = np.linalg.svd(F)
    S[2] = 0  # Set smallest singular value to 0
    F = U @ np.diag(S) @ Vt
    
    # Normalize F
    F = F / np.linalg.norm(F)
    
    return F


def point_to_epipolar_line_distance(point1, point2, F):
    """
    Calculate the distance from point2 to the epipolar line of point1.
    
    The epipolar line l2 in image 2 corresponding to point1 in image 1 is:
    l2 = F * point1
    
    Args:
        point1: (x, y) coordinates in image 1
        point2: (x, y) coordinates in image 2
        F: 3x3 fundamental matrix
    
    Returns:
        distance: Perpendicular distance from point2 to epipolar line
    """
    # Convert points to homogeneous coordinates
    p1_homo = np.array([point1[0], point1[1], 1.0])
    p2_homo = np.array([point2[0], point2[1], 1.0])
    
    # Calculate epipolar line in image 2: l2 = F * p1
    l2 = F @ p1_homo
    
    # Distance from point to line ax + by + c = 0 is |ax + by + c| / sqrt(a² + b²)
    # For line l = [a, b, c] and point p = [x, y, 1]:
    # distance = |l^T * p| / sqrt(a² + b²)
    
    numerator = abs(np.dot(l2, p2_homo))
    denominator = np.sqrt(l2[0]**2 + l2[1]**2)
    
    if denominator < 1e-10:  # Avoid division by zero
        return float('inf')
    
    distance = numerator / denominator
    
    return distance

def group_detections_across_views_bipartite_full(detected_persons_2d, view_results,
                                           projection_matrices, port_to_cam_index,
                                           camera_params, epipolar_threshold=30):
    """
    Improved bipartite matching that handles missing detections and order independence.
    
    Strategy: Match all pairs of views independently, then merge matches that share detections.
    """
    import numpy as np
    from scipy.optimize import linear_sum_assignment
    
    ports = list(detected_persons_2d.keys())
    if len(ports) < 2:
        return []
    
    # Store all pairwise matches
    # Key: (port1, detection_idx1, port2, detection_idx2)
    # Value: epipolar distance
    pairwise_matches = {}
    
    # Match all pairs of views
    for i, port1 in enumerate(ports):
        detections1 = detected_persons_2d[port1]
        if not detections1:
            continue
            
        for port2 in ports[i+1:]:  # Only process each pair once
            detections2 = detected_persons_2d[port2]
            if not detections2:
                continue
            
            # Calculate fundamental matrix
            cam_idx1 = port_to_cam_index[port1]
            cam_idx2 = port_to_cam_index[port2]
            F = calculate_fundamental_matrix(
                projection_matrices[cam_idx1],
                projection_matrices[cam_idx2]
            )
            
            # Build cost matrix
            n1 = len(detections1)
            n2 = len(detections2)
            cost_matrix = np.full((n1, n2), 1000.0)
            
            for idx1, det1 in enumerate(detections1):
                for idx2, det2 in enumerate(detections2):
                    dist = point_to_epipolar_line_distance(
                        det1['com_2d'], det2['com_2d'], F
                    )
                    if dist < epipolar_threshold:
                        cost_matrix[idx1, idx2] = dist
            
            # Solve assignment problem
            row_indices, col_indices = linear_sum_assignment(cost_matrix)
            
            # Store valid matches
            for idx1, idx2 in zip(row_indices, col_indices):
                if cost_matrix[idx1, idx2] < epipolar_threshold:
                    # Store both directions for easy lookup
                    pairwise_matches[(port1, idx1, port2, idx2)] = cost_matrix[idx1, idx2]
                    pairwise_matches[(port2, idx2, port1, idx1)] = cost_matrix[idx1, idx2]
    
    # Build groups from pairwise matches using Union-Find
    parent = {}
    
    def find(x):
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]
    
    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py
    
    # Union all matched detections
    for (port1, idx1, port2, idx2), dist in pairwise_matches.items():
        if port1 < port2:  # Process each pair only once
            union((port1, idx1), (port2, idx2))
    
    # Collect groups
    groups_dict = {}
    for port in ports:
        for idx, detection in enumerate(detected_persons_2d[port]):
            root = find((port, idx))
            if root not in groups_dict:
                groups_dict[root] = {}
            groups_dict[root][port] = detection
    
    # Convert to list and filter
    final_groups = [group for group in groups_dict.values() if len(group) >= 2]
    
    return final_groups

def calculate_3d_com_from_keypoints(keypoints_3d, hip_indices):
    """Calculate 3D center of mass from keypoints."""
    if keypoints_3d is None:
        return None
        
    valid_hips = []
    for hip_idx in hip_indices:
        if hip_idx < len(keypoints_3d) and keypoints_3d[hip_idx] is not None:
            hip = keypoints_3d[hip_idx]
            if not np.isnan(hip).any():
                valid_hips.append(hip)
    
    if len(valid_hips) >= 1:
        return np.mean(valid_hips, axis=0)
    return None


def save_person_csv(results, csv_path, expected_num_kps):
    """Save results for one person to CSV."""
    csv_rows = []
    
    for result in results:
        row_data = {
            'sync_index': result['sync_index'],
            'person_id': result['person_id']
        }
        
        kps_list = result['keypoints_3d']
        
        # Pad/truncate to expected length
        if len(kps_list) != expected_num_kps:
            kps_list.extend([[np.nan, np.nan, np.nan]] * (expected_num_kps - len(kps_list)))
            kps_list = kps_list[:expected_num_kps]
        
        for kp_idx in range(expected_num_kps):
            marker_name = SynthPoseMarkers.markers.get(kp_idx, f"KP_{kp_idx}")
            coords = kps_list[kp_idx]
            
            if coords is None or not isinstance(coords, (list, tuple)) or len(coords) != 3:
                x, y, z = np.nan, np.nan, np.nan
            else:
                x, y, z = coords[0], coords[1], coords[2]
            
            row_data[f"{marker_name}_X"] = x
            row_data[f"{marker_name}_Y"] = y
            row_data[f"{marker_name}_Z"] = z
        
        csv_rows.append(row_data)
    
    if csv_rows:
        #  Create directory if it doesn't exist
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        results_df = pd.DataFrame(csv_rows)
        results_df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"Saved CSV: {csv_path}")

def save_cameras_csv(camera_results, csv_path):
    """Save camera information used for each 3D candidate to CSV for diagnostic purposes."""
    csv_rows = []
    
    for result in camera_results:
        row_data = {
            'sync_index': result['sync_index'],
            'person_id': result['person_id'],
            'cameras_used': ','.join(str(cam) for cam in result['cameras_used'])  # Convert list to comma-separated string
        }
        csv_rows.append(row_data)
    
    if csv_rows:
        cameras_df = pd.DataFrame(csv_rows)
        cameras_df.to_csv(csv_path, index=False)
        print(f"Saved cameras CSV: {csv_path}")


def save_pixel_coords_csv(pixel_results, csv_path, expected_num_kps, port):
    """Save pixel coordinates for one person and one port to CSV."""
    csv_rows = []
    
    for result in pixel_results:
        row_data = {
            'sync_index': result['sync_index'],
            'person_id': result['person_id']
        }
        
        # Get pixel coordinates for this port
        pixel_coords = result['pixel_coords'].get(port, [])
        
        # Pad/truncate to expected length
        if len(pixel_coords) != expected_num_kps:
            pixel_coords.extend([[np.nan, np.nan]] * (expected_num_kps - len(pixel_coords)))
            pixel_coords = pixel_coords[:expected_num_kps]
        
        for kp_idx in range(expected_num_kps):
            marker_name = SynthPoseMarkers.markers.get(kp_idx, f"KP_{kp_idx}")
            coords = pixel_coords[kp_idx]
            
            if coords is None or not isinstance(coords, (list, tuple)) or len(coords) != 2:
                px, py = np.nan, np.nan
            else:
                px, py = coords[0], coords[1]
            
            row_data[f"{marker_name}_px"] = px
            row_data[f"{marker_name}_py"] = py
        
        csv_rows.append(row_data)
    
    if csv_rows:
        results_df = pd.DataFrame(csv_rows)
        results_df.to_csv(csv_path, index=False, float_format='%.4f')
        print(f"Saved pixel coordinates CSV: {csv_path}")

def read_posetrack_csv(csv_path):
    """
    Read a posetrack CSV file and return a dictionary with nx3 numpy arrays 
    grouped by body part prefix (e.g., 'NOSE', 'LEFT_EYE', etc.).
    
    Args:
        csv_path (str): Path to the CSV file
        
    Returns:
        dict: Dictionary with body part names as keys and nx3 numpy arrays as values
              where n is the number of frames and columns are [x, y, z]
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Get sync_index and person_id columns
        sync_indices = df['sync_index'].values
        person_ids = df['person_id'].values if 'person_id' in df.columns else np.zeros(len(df))
        
        # Find all body part columns (those ending with _X, _Y, _Z)
        all_columns = df.columns.tolist()
        body_part_names = set()
        
        for col in all_columns:
            if col.endswith('_X') or col.endswith('_Y') or col.endswith('_Z'):
                body_part = col[:-2]  # Remove _X, _Y, or _Z suffix
                body_part_names.add(body_part)
        
        # Create dictionary with body part data
        body_parts_data = {}
        
        for body_part in body_part_names:
            x_col = f"{body_part}_X"
            y_col = f"{body_part}_Y"
            z_col = f"{body_part}_Z"
            
            if x_col in df.columns and y_col in df.columns and z_col in df.columns:
                x_data = df[x_col].values
                y_data = df[y_col].values
                z_data = df[z_col].values
                
                # Stack into nx3 array
                body_parts_data[body_part] = np.column_stack([x_data, y_data, z_data])
        
        # Also include metadata and direct access to sync_index/person_id
        body_parts_data['_metadata'] = {
            'sync_indices': sync_indices,
            'person_ids': person_ids,
            'num_frames': len(df)
        }
        
        # Add sync_index and person_id as direct arrays if they exist
        if 'sync_index' in df.columns:
            body_parts_data['sync_index'] = sync_indices
        if 'person_id' in df.columns:
            body_parts_data['person_id'] = person_ids
        
        return body_parts_data
        
    except Exception as e:
        print(f"Error reading CSV file {csv_path}: {e}")
        return None

def batch_process_subfolders(base_dir, overwrite_tracked_files = False, **kwargs):
    """
    Batch process all subfolders within a directory.
    Each subfolder should contain the necessary files for multi-person pose processing.
    
    Args:
        base_dir: Parent directory containing subfolders to process
        **kwargs: Additional arguments to pass to process_synced_mwc_frames_multi_person
    """
    import os
    
    if not os.path.isdir(base_dir):
        print(f"Error: Base directory does not exist: {base_dir}")
        return
    
    # Get all subdirectories
    subfolders = [f for f in os.listdir(base_dir) 
                  if os.path.isdir(os.path.join(base_dir, f)) and not f.startswith('.')]
    
    if not subfolders:
        print(f"No subfolders found in {base_dir}")
        return
    
    print(f"Found {len(subfolders)} subfolders to process: {subfolders}")
    
    for subfolder in subfolders:
        subfolder_path = os.path.join(base_dir, subfolder)
        print(f"\n{'='*60}")
        print(f"Processing subfolder: {subfolder}")
        print("hi!")
        # Expected file paths within each subfolder
        frame_time_history_csv = os.path.join(subfolder_path, "frame_time_history.csv")
        
        calibration_path = os.path.join(os.path.dirname(base_dir), "config.toml")
        
        # Create synthpose output directory within the subfolder
        output_dir = os.path.join(subfolder_path, "synthpose")
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, "output_3d_poses_tracked.csv")

        matching_files = []        
        if not overwrite_tracked_files:
            # check for existing files 
            print(f"looking for existing files of type {output_path}")
            pattern = r"output_3d_poses_tracked\.csv_person[0-9]\.csv"

            for filename in os.listdir(output_dir):
                if match(pattern, filename):
                    matching_files.append(filename)
            if not matching_files: 
                print("no matching files. No risk of overwrite, Proceeding.")
            else:
                print("Overwrite set to false and there are tracked.csv files. skipping.")
                continue
            
        # Check if required files exist
        if not os.path.exists(frame_time_history_csv):
            print(f"Warning: frame_time_history.csv not found in {subfolder_path}, skipping...")
            continue
        if not os.path.exists(calibration_path):
            print(f"Warning: config.toml not found in {subfolder_path}, skipping...")
            continue
        
        # try:
        # Process this subfolder
        process_synced_mwc_frames_multi_person_perf(
            frame_history_csv_path=frame_time_history_csv,
            calibration_path=calibration_path,
            video_dir=subfolder_path,
            output_path=output_path,
            **kwargs
        )
        print(f"Successfully processed {subfolder}")
        # except Exception as e:
        #     print(f"Error processing {subfolder}: {e}")
        #     continue
    
    print(f"\nBatch processing completed.")

# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process synchronized multi-camera video for 3D pose estimation with tracking.")
    # (Add the new arguments if needed, or keep defaults)
    testfolder = "recording_balance_stage1_v2"
    parser.add_argument("--csv_path", default=f"test/caliscope/{testfolder}/frame_time_history.csv",help="Path...")
    parser.add_argument("--calibration_path",default=f"test/caliscope/{testfolder}/config.toml", help="Path...")
    parser.add_argument("--video_dir", default=f"test/caliscope/{testfolder}", help="Dir...")
    parser.add_argument("--output_path", default=f"output/caliscope/{testfolder}/output_3d_poses_tracked.csv",help="Path...")
    parser.add_argument("--model_dir", default=LOCAL_SP_DIR, help="Path...")
    parser.add_argument("--detector_dir", default=LOCAL_DET_DIR, help="Path...")
    parser.add_argument("--calib_type", default="mwc", choices=['mwc', 'fmc'], help="Type...")
    parser.add_argument("--skip", type=int, default=1, help="Skip N...")
    parser.add_argument("--person_conf", type=float, default=0.8, help="Conf...")
    parser.add_argument("--keypoint_conf", type=float, default=0.1, help="Conf...")
    parser.add_argument("--device", default="mps", choices=['auto', 'cpu', 'mps', 'cuda'], help="Device...")
    parser.add_argument("--track_max_dist", type=float, default=100.0, help="Max 2D pixel distance for head tracking match (default: 100).")
    parser.add_argument("--head_idx", type=int, default=0, help="Index of keypoint used for tracking (default: 0, Nose).")

    args = parser.parse_args()

    process_synced_mwc_frames_multi_person_perf(
        frame_history_csv_path=args.csv_path, calibration_path=args.calibration_path, video_dir=args.video_dir,
        output_path=args.output_path, model_dir=args.model_dir, detector_dir=args.detector_dir,
        calib_type=args.calib_type, skip_sync_indices=args.skip, person_confidence=args.person_conf,
        keypoint_confidence=args.keypoint_conf, device_name=args.device)

