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
from .process_synced_poses import PersonTrack, group_detections_across_views_bipartite_full, generate_3d_candidates_from_groups, assign_3d_candidates_to_tracks, project_keypoints_to_all_cameras,project_keypoints_to_all_cameras_ultrafast, save_person_csv,save_pixel_coords_csv,save_cameras_csv, project_3d_to_2d, plot_debug_keypoints, calculate_2d_com

def process_synced_mwc_frames_multi_person_perf(
    frame_history_csv_path, calibration_path, video_dir, output_path, model_dir=LOCAL_SP_DIR,
    detector_dir=LOCAL_DET_DIR, calib_type='mwc', skip_sync_indices=1, person_confidence=0.3,
    keypoint_confidence=0.1, device_name="auto",
    # --- Tracking parameters ---
    max_persons=2,  # Maximum number of persons to track
    track_lost_patience=10,  # frames to wait before considering a track lost
    min_keypoints_for_com=2,  # minimum valid hip keypoints needed to compute COM
    hip_indices=(11, 12),  # COCO format: left hip, right hip
    verbose_debug=False,  # Enable debugging visualization for first frame
    # --- NEW BATCH PROCESSING PARAMETER ---
    batch_size=8,  # Number of frames to process simultaneously (configurable)
    ):
    """Processes synchronized frames with multi-person 3D tracking using COM-based matching with batch processing."""

    # --- 1. Setup Device (same as before) ---
    if device_name == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built(): device = "mps"
        elif torch.cuda.is_available(): device = "cuda"
        else: device = "cpu"
    else:
        device = device_name
        if device == "mps" and (not torch.backends.mps.is_available() or not torch.backends.mps.is_built()): 
            print("Warn: MPS unavailable."); device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available(): 
            print("Warn: CUDA unavailable."); device = "cpu"
    print(f"Using device: {device}")

    # --- 2. Load Frame History & Derive 0-Based Index ---
    # (Keep as is)
    print(f"Loading frame history from: {frame_history_csv_path}")
    try:
        frame_history_df = pd.read_csv(frame_history_csv_path)
        frame_history_df['sync_index'] = frame_history_df['sync_index'].astype(int)
        frame_history_df['port'] = frame_history_df['port'].astype(int)
        frame_history_df['frame_time'] = frame_history_df['frame_time'].astype(float)
        frame_history_df = frame_history_df.sort_values(by=['port', 'frame_time'])
        frame_history_df['derived_frame_index'] = frame_history_df.groupby('port')['frame_time'].rank(method='min').astype(int) - 1
    except Exception as e: print(f"Error reading/processing frame history CSV: {e}"); return
    print(f"Found {frame_history_df['sync_index'].nunique()} unique sync indices.")
    print(f"Ports found: {sorted(frame_history_df['port'].unique())}")


    # --- 3. Load Calibration & Filter ---
    # (Keep as is)
    print(f"Loading calibration ({calib_type}) from: {calibration_path}")
    if calib_type.lower() == 'mwc': camera_params = parse_calibration_mwc(calibration_path)
    elif calib_type.lower() == 'fmc': camera_params = parse_calibration_fmc(calibration_path)
    else: print(f"Error: Invalid calib type '{calib_type}'."); return
    if not camera_params: print("Error: Failed to load camera calibration."); return
    print(f"Loaded {len(camera_params)} cameras initially.")
    port_to_cam_index={}; calibration_ports=set()
    for idx, params in enumerate(camera_params):
        port = params.get('port')
        if port is not None: port_to_cam_index[port] = idx; calibration_ports.add(port)
        else:
            try: inferred_port = int(params.get('name', f'cam_{idx}').split('_')[-1]); port_to_cam_index[inferred_port] = idx; calibration_ports.add(inferred_port)
            except ValueError: print(f"Warn: Cannot determine port for camera index {idx}.")
    csv_ports = set(frame_history_df['port'].unique())
    if not csv_ports.issubset(calibration_ports): print(f"Warn: Ports mismatch CSV:{csv_ports} vs Calib:{calibration_ports}.")
    common_ports = sorted(list(csv_ports.intersection(calibration_ports)))
    print(f"Using common ports for processing: {common_ports}")
    if len(common_ports) < 2: print("Error: Need >= 2 common ports."); return
    
    filtered_cam_params=[]; filtered_port_map={}; new_idx=0
    for port in common_ports: 
        original_idx = port_to_cam_index[port]
        filtered_cam_params.append(camera_params[original_idx])
        filtered_port_map[port] = new_idx
        new_idx += 1
    
    camera_params = filtered_cam_params; port_to_cam_index = filtered_port_map; num_cameras = len(camera_params)
    print(f"Filtered calibration to {num_cameras} cameras.")

    # --- 4. Calculate Projection Matrices ---
    # (Keep as is)
    print("Calculating projection matrices...")
    projection_matrices = calculate_projection_matrices(camera_params)
    if len(projection_matrices) != num_cameras: print("Error: Proj matrix count mismatch."); return

    # --- 5. Load Models with Optimization ---
    print("Loading detection and pose estimation models...")
    try:
        person_processor, person_model, pose_processor, pose_model = load_models(detect_path=detector_dir, pose_model_path=model_dir, device=device)
        
        # Optimize models for inference
        print("Optimizing models for batch inference...")
        person_model.eval()  # Set to evaluation mode
        pose_model.eval()    # Set to evaluation mode
        
        # Enable inference optimizations
        with torch.no_grad():
            # Warm up models with dummy data to optimize
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
                
    except Exception as e: print(f"Error loading models: {e}"); return

    # --- 6. Open Video Files ---
    # (Keep as is, including getting video_lengths)
    print("Opening video capture for required ports...")
    caps = {}; video_lengths = {}
    for port in common_ports:
        video_path = os.path.join(video_dir, f"port_{port}.mp4")
        if not os.path.exists(video_path): print(f"Error: Vid not found: {video_path}"); [c.release() for c in caps.values()]; return
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): print(f"Error: Cannot open vid: {video_path}"); [c.release() for c in caps.values()]; return
        caps[port] = cap; video_lengths[port] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Opened video for port {port}: {video_path} ({video_lengths[port]} frames)")

    # --- 6.5. Validate Sync Range vs Video Length ---
    all_sync_indices = sorted(frame_history_df['sync_index'].unique())
    sync_range = all_sync_indices[-1] - all_sync_indices[0] + 1  # +1 because both endpoints inclusive
    
    print(f"\n--- Sync Validation ---")
    print(f"Sync index range: {all_sync_indices[0]} to {all_sync_indices[-1]} ({len(all_sync_indices)} unique indices)")
    print(f"Expected sync range: {sync_range} frames")
    
    # Check if sync range matches video lengths (should be exact or very close)
    sync_matches_video = True
    for port in common_ports:
        video_len = video_lengths[port]
        if abs(sync_range - video_len) > 1:  # Allow only 1 frame difference
            print(f"WARNING: Port {port} video length ({video_len}) doesn't match sync range ({sync_range})")
            sync_matches_video = False
        else:
            print(f"Port {port}: sync range ({sync_range}) matches video length ({video_len}) ✓")
    
    if sync_matches_video:
        print("✓ Frame alignment validation PASSED - videos appear to share first and last frames with sync data")
    else:
        print("⚠ Frame alignment validation FAILED - cannot assume sync_index corresponds to video frame positions!")
        print("  This may indicate missing frames at start/end or sync timing issues.")

    active_tracks = []  # List of PersonTrack objects
    next_track_id = 0
    all_results_by_person = {}  # {person_id: [results]}
    all_pixel_coords_by_person = {}  # {person_id: [pixel_coords]} for each port
    previous_views_used = []  # Track views_used from previous frame for continuity
    all_cameras_by_person = {}  # {person_id: [camera_info]} for diagnostic purposes
    debug_frames = {}  # Store first frame for debugging

    # --- 7. Process Synchronized Frames with Batch Processing ---
    all_sync_indices = sorted(frame_history_df['sync_index'].unique())
    start_time = time.time()

    print(f"Starting multi-person processing for {len(all_sync_indices)} sync indices with batch size {batch_size}...")
    sync_index_counter = 0
    
    # Filter sync indices based on skip_sync_indices
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
        
        # --- BATCH FRAME READING ---
        batch_frames = {}  # {sync_index: {port: PIL_image}}
        valid_batch_indices = []
        
        for sync_index in current_batch_indices:
            sync_data = frame_history_df[frame_history_df['sync_index'] == sync_index]
            if set(sync_data['port']) != set(common_ports): 
                continue

            # Read frames for this sync_index
            current_frames_pil = {}
            frame_read_success = True
            
            for _, row in sync_data.iterrows():
                port = row['port']
                frame_idx_to_read = row['derived_frame_index']
                try: frame_idx_int = int(frame_idx_to_read)
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
        
        if not valid_batch_indices:
            continue
            
        # --- BATCH PERSON DETECTION AND POSE ESTIMATION ---
        batch_detections = {}  # {sync_index: {port: [person_data, ...]}}
        
        # Collect all images for batch processing
        all_images = []
        image_metadata = []  # (sync_index, port)
        
        for sync_index in valid_batch_indices:
            for port in common_ports:
                if port in batch_frames[sync_index]:
                    all_images.append(batch_frames[sync_index][port])
                    image_metadata.append((sync_index, port))
        
        if not all_images:
            continue
            
        # TRUE BATCH PROCESSING - Process all images simultaneously
        print(f"True batch processing {len(all_images)} images...")
        with torch.no_grad():
            # Batch person detection - process all images at once
            batch_person_results = detect_persons_batch(
                all_images, person_processor, person_model, device, person_confidence, batch_size
            )
            
            # Prepare data for batch pose estimation
            images_with_boxes = []
            for img_idx, (image, (person_boxes_voc, person_boxes_coco, person_scores)) in enumerate(zip(all_images, batch_person_results)):
                images_with_boxes.append((image, person_boxes_coco))
            
            # Batch pose estimation on all images with their detected persons
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
        
        # --- PROCESS EACH FRAME IN THE BATCH (Individual Processing) ---
        for sync_index in valid_batch_indices:
            view_detections = batch_detections[sync_index]
            
            groups = group_detections_across_views_bipartite_full(view_detections, None, 
                                               projection_matrices, port_to_cam_index,
                                               camera_params, epipolar_threshold=10)
            
            # --- Generate 3D candidates using all possible combinations ---
            candidate_3d_groups = generate_3d_candidates_from_groups(
                groups, port_to_cam_index, camera_params, projection_matrices, hip_indices
            )
            
            # --- Associate 3D candidates with existing tracks ---
            track_assignments, unused_3d_groups = assign_3d_candidates_to_tracks(
                active_tracks, candidate_3d_groups, 0.5
            )
            
            # Check if this is a bad frame (no successful assignments)
            if len(active_tracks) > 0 and not track_assignments:
                print(f"Bad frame detected at sync {sync_index}: no tracks matched")
                
                # Just increment lost counters for all tracks and skip the rest of the updating. 
                for track in active_tracks:
                    track.increment_lost_counter()
                    if not track.is_active:
                        print(f"Lost track of person {track.id} at sync {sync_index}")
                
                # Remove inactive tracks
                active_tracks = [t for t in active_tracks if t.is_active]
                
                # Continue to next frame without updating any results
                continue

            # Update existing tracks
            for track_id,curtuple in track_assignments.items():
                if curtuple is not None:
                    (grp_idx, candidate_idx) = curtuple
                
                if candidate_idx is not None:
                    candidate = candidate_3d_groups[grp_idx][candidate_idx]
                    if len(active_tracks) < max_persons:
                        new_track = PersonTrack(track_id, candidate_3d_groups[grp_idx][candidate_idx]['keypoints_3d'], sync_index,hip_indices,candidate['views'])
                        active_tracks.append(new_track)
                    else:
                        # find which track to update
                        track_idx = None
                        for i, t in enumerate(active_tracks):
                            if t.id == track_id:
                                track_idx = i
                                break
                        if track_idx is not None:
                            active_tracks[track_idx].update(candidate['keypoints_3d'], sync_index, candidate['views'])
                        else:
                            print(f"Warn: Track ID {track_id} not found in active tracks.")
                            continue

                    # Store result
                    person_id = active_tracks[track_id].id
                    if person_id not in all_results_by_person:
                        all_results_by_person[person_id] = []
                    if person_id not in all_pixel_coords_by_person:
                        all_pixel_coords_by_person[person_id] = []
                    if person_id not in all_cameras_by_person:
                        all_cameras_by_person[person_id] = []
                        
                    kps_3d_list = [[np.nan]*3 if kp is None else kp.tolist() 
                                   for kp in candidate['keypoints_3d']]
                    all_results_by_person[person_id].append({
                        'sync_index': sync_index,
                        'person_id': person_id,
                        'keypoints_3d': kps_3d_list
                    })
                    
                    # Project 3D keypoints to 2D pixel coordinates for all cameras
                    pixel_coords = project_keypoints_to_all_cameras(
                        candidate['keypoints_3d'], projection_matrices, common_ports, port_to_cam_index
                    )
                    all_pixel_coords_by_person[person_id].append({
                        'sync_index': sync_index,
                        'person_id': person_id,
                        'pixel_coords': pixel_coords
                    })
                    
                    # Store camera information for diagnostics
                    all_cameras_by_person[person_id].append({
                        'sync_index': sync_index,
                        'person_id': person_id,
                        'cameras_used': candidate_3d_groups[grp_idx][candidate_idx]['views']
                    })
                    
            # Update lost counters for unmatched tracks
            for i, track in enumerate(active_tracks):
                if i not in track_assignments or track_assignments[i] is None:
                    track.increment_lost_counter()
                    if not track.is_active:
                        print(f"Lost track of person {track.id} at sync {sync_index}")
            
            # Update previous_views_used with views from candidates that were actually used
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
            
            # Store first frame for debugging if enabled
            if verbose_debug and sync_index == all_sync_indices[0]:
                debug_frames = batch_frames[sync_index].copy()

    # --- 8. Cleanup and Save Results ---
    print("\nReleasing video captures...")
    [cap.release() for cap in caps.values()]
    end_time = time.time()
    print(f"Processing finished in {end_time - start_time:.2f} seconds.")

    # Save results for each person
    base_output_path = output_path
    
    for person_id, results in all_results_by_person.items():
        if not results:
            continue
            
        print(f"\nSaving results for Person {person_id} ({len(results)} frames)...")
        
        # Define output filenames for this person
        person_csv_path = f"{base_output_path}_person{person_id}.csv"
        person_pickle_path = f"{base_output_path}_person{person_id}.pkl"
        
        # Save pickle
        try:
            with open(person_pickle_path, 'wb') as f_pkl:
                pickle.dump(results, f_pkl)
            print(f"Saved pickle: {person_pickle_path}")
        except Exception as e:
            print(f"Error saving pickle for person {person_id}: {e}")
        
        # Save CSV (same format as original)
        save_person_csv(results, person_csv_path, expected_num_kps=52)
        
        # Save pixel coordinates CSV for each port
        if person_id in all_pixel_coords_by_person:
            pixel_results = all_pixel_coords_by_person[person_id]
            for port in common_ports:
                pixel_csv_path = f"{base_output_path}_person{person_id}_pixelcoords_port_{port}.csv"
                save_pixel_coords_csv(pixel_results, pixel_csv_path, expected_num_kps=52, port=port)
        
        # Save camera information CSV for diagnostics
        if person_id in all_cameras_by_person:
            camera_results = all_cameras_by_person[person_id]
            cameras_csv_path = f"{base_output_path}_person{person_id}_cameras.csv"
            save_cameras_csv(camera_results, cameras_csv_path)
    
    # Debug visualization if enabled
    if verbose_debug and debug_frames and all_pixel_coords_by_person:
        print("\nGenerating debug visualization...")
        plot_debug_keypoints(debug_frames, all_pixel_coords_by_person, common_ports, all_sync_indices[0])
    
    print(f"\nTotal persons tracked: {len(all_results_by_person)}")
    return all_results_by_person
# --- Helper Functions ---


def process_synced_mwc_frames_multi_person(
    frame_history_csv_path, calibration_path, video_dir, output_path, model_dir=LOCAL_SP_DIR,
    detector_dir=LOCAL_DET_DIR, calib_type='mwc', skip_sync_indices=1, person_confidence=0.3,
    keypoint_confidence=0.1, device_name="auto",
    # --- Tracking parameters ---
    max_persons=2,  # Maximum number of persons to track
    com_distance_threshold=0.3,  # meters - minimum distance between COMs to be different people
    track_frames_til_lost_patience=10,  # frames to wait before considering a track lost
    min_keypoints_for_com=2,  # minimum valid hip keypoints needed to compute COM
    hip_indices=(11, 12),  # COCO format: left hip, right hip
    epipolar_threshold=10,  # pixels - max distance from epipolar line
    reprojection_error_threshold=50,  # pixels - max reprojection error after triangulation
    min_views_for_detection=2,  # minimum camera views to confirm a person
    iou_threshold=0.3,  # for matching bounding boxes across views
    temporal_smoothing_window=5,  # frames for temporal consistency
    min_track_length=5,  # minimum frames before considering a track valid
    max_com_velocity=2.0,  # m/s - maximum reasonable COM velocity between frames
    verbose_debug=False,  # Enable debugging visualization for first frame
    override_views_used=None,  # Optional list of view combinations to prioritize instead of previous frame
    ):
    """Processes synchronized frames with multi-person 3D tracking using COM-based matching."""

    # --- 1. Setup Device (same as before) ---
    if device_name == "auto":
        if torch.backends.mps.is_available() and torch.backends.mps.is_built(): device = "mps"
        elif torch.cuda.is_available(): device = "cuda"
        else: device = "cpu"
    else:
        device = device_name
        if device == "mps" and (not torch.backends.mps.is_available() or not torch.backends.mps.is_built()): 
            print("Warn: MPS unavailable."); device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available(): 
            print("Warn: CUDA unavailable."); device = "cpu"
    print(f"Using device: {device}")

    # --- 2. Load Frame History & Derive 0-Based Index ---
    # (Keep as is)
    print(f"Loading frame history from: {frame_history_csv_path}")
    try:
        frame_history_df = pd.read_csv(frame_history_csv_path)
        frame_history_df['sync_index'] = frame_history_df['sync_index'].astype(int)
        frame_history_df['port'] = frame_history_df['port'].astype(int)
        frame_history_df['frame_time'] = frame_history_df['frame_time'].astype(float)
        frame_history_df = frame_history_df.sort_values(by=['port', 'frame_time'])
        frame_history_df['derived_frame_index'] = frame_history_df.groupby('port')['frame_time'].rank(method='min').astype(int) - 1
    except Exception as e: print(f"Error reading/processing frame history CSV: {e}"); return
    print(f"Found {frame_history_df['sync_index'].nunique()} unique sync indices.")
    print(f"Ports found: {sorted(frame_history_df['port'].unique())}")


    # --- 3. Load Calibration & Filter ---
    # (Keep as is)
    print(f"Loading calibration ({calib_type}) from: {calibration_path}")
    if calib_type.lower() == 'mwc': camera_params = parse_calibration_mwc(calibration_path)
    elif calib_type.lower() == 'fmc': camera_params = parse_calibration_fmc(calibration_path)
    else: print(f"Error: Invalid calib type '{calib_type}'."); return
    if not camera_params: print("Error: Failed to load camera calibration."); return
    print(f"Loaded {len(camera_params)} cameras initially.")
    port_to_cam_index={}; calibration_ports=set()
    for idx, params in enumerate(camera_params):
        port = params.get('port')
        if port is not None: port_to_cam_index[port] = idx; calibration_ports.add(port)
        else:
            try: inferred_port = int(params.get('name', f'cam_{idx}').split('_')[-1]); port_to_cam_index[inferred_port] = idx; calibration_ports.add(inferred_port)
            except ValueError: print(f"Warn: Cannot determine port for camera index {idx}.")
    csv_ports = set(frame_history_df['port'].unique())
    if not csv_ports.issubset(calibration_ports): print(f"Warn: Ports mismatch CSV:{csv_ports} vs Calib:{calibration_ports}.")
    common_ports = sorted(list(csv_ports.intersection(calibration_ports)))
    print(f"Using common ports for processing: {common_ports}")
    if len(common_ports) < 2: print("Error: Need >= 2 common ports."); return
    
    filtered_cam_params=[]; filtered_port_map={}; new_idx=0
    for port in common_ports: 
        original_idx = port_to_cam_index[port]
        filtered_cam_params.append(camera_params[original_idx])
        filtered_port_map[port] = new_idx
        new_idx += 1
    
    camera_params = filtered_cam_params; port_to_cam_index = filtered_port_map; num_cameras = len(camera_params)
    print(f"Filtered calibration to {num_cameras} cameras.")

    # --- 4. Calculate Projection Matrices ---
    # (Keep as is)
    print("Calculating projection matrices...")
    projection_matrices = calculate_projection_matrices(camera_params)
    if len(projection_matrices) != num_cameras: print("Error: Proj matrix count mismatch."); return

    # --- 5. Load Models ---
    # (Keep as is)
    print("Loading detection and pose estimation models...")
    try:
        person_processor, person_model, pose_processor, pose_model = load_models(detect_path=detector_dir, pose_model_path=model_dir, device=device)
    except Exception as e: print(f"Error loading models: {e}"); return

    # --- 6. Open Video Files ---
    # (Keep as is, including getting video_lengths)
    print("Opening video capture for required ports...")
    caps = {}; video_lengths = {}
    for port in common_ports:
        video_path = os.path.join(video_dir, f"port_{port}.mp4")
        if not os.path.exists(video_path): print(f"Error: Vid not found: {video_path}"); [c.release() for c in caps.values()]; return
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): print(f"Error: Cannot open vid: {video_path}"); [c.release() for c in caps.values()]; return
        caps[port] = cap; video_lengths[port] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print(f"Opened video for port {port}: {video_path} ({video_lengths[port]} frames)")

    # --- 6.5. Validate Sync Range vs Video Length ---
    all_sync_indices = sorted(frame_history_df['sync_index'].unique())
    sync_range = all_sync_indices[-1] - all_sync_indices[0] + 1  # +1 because both endpoints inclusive
    
    print(f"\n--- Sync Validation ---")
    print(f"Sync index range: {all_sync_indices[0]} to {all_sync_indices[-1]} ({len(all_sync_indices)} unique indices)")
    print(f"Expected sync range: {sync_range} frames")
    
    # Check if sync range matches video lengths (should be exact or very close)
    sync_matches_video = True
    for port in common_ports:
        video_len = video_lengths[port]
        if abs(sync_range - video_len) > 1:  # Allow only 1 frame difference
            print(f"WARNING: Port {port} video length ({video_len}) doesn't match sync range ({sync_range})")
            sync_matches_video = False
        else:
            print(f"Port {port}: sync range ({sync_range}) matches video length ({video_len}) ✓")
    
    if sync_matches_video:
        print("✓ Frame alignment validation PASSED - videos appear to share first and last frames with sync data")
    else:
        print("⚠ Frame alignment validation FAILED - cannot assume sync_index corresponds to video frame positions!")
        print("  This may indicate missing frames at start/end or sync timing issues.")

    active_tracks = []  # List of PersonTrack objects
    next_track_id = 0
    all_results_by_person = {}  # {person_id: [results]}
    all_pixel_coords_by_person = {}  # {person_id: [pixel_coords]} for each port
    previous_views_used = []  # Track views_used from previous frame for continuity
    all_cameras_by_person = {}  # {person_id: [camera_info]} for diagnostic purposes
    debug_frames = {}  # Store first frame for debugging

    # --- 7. Process Synchronized Frames ---
    all_sync_indices = sorted(frame_history_df['sync_index'].unique())
    start_time = time.time()

    print(f"Starting multi-person processing for {len(all_sync_indices)} sync indices...")
    sync_index_counter = 0
    
    for sync_index in tqdm(all_sync_indices, desc="Processing Sync Indices"):
        sync_index_counter += 1
        if (sync_index_counter - 1) % skip_sync_indices != 0: 
            continue

        sync_data = frame_history_df[frame_history_df['sync_index'] == sync_index]
        if set(sync_data['port']) != set(common_ports): continue

        # --- Read frames ---
        # (Keep frame reading logic as is)
        current_frames_pil = {}
        frame_read_success = True
        for _, row in sync_data.iterrows():
            port = row['port']
            frame_idx_to_read = row['derived_frame_index']
            try: frame_idx_int = int(frame_idx_to_read)
            except (ValueError, TypeError): print(f"Warn: Invalid derived index '{frame_idx_to_read}' p{port} s{sync_index}"); frame_read_success = False; break
            cap = caps.get(port); total_frames = video_lengths.get(port, -1)
            if cap is None or total_frames == -1: print(f"CritErr: No capture/len p{port}"); frame_read_success = False; break
            if not (0 <= frame_idx_int < total_frames): print(f"Warn: Idx {frame_idx_int} out of bounds (0-{total_frames-1}) p{port} s{sync_index}"); frame_read_success = False; break
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx_int)
            ret, frame = cap.read()
            if not ret: print(f"Warn: Failed read idx {frame_idx_int} p{port} s{sync_index}"); frame_read_success = False; break
            current_frames_pil[port] = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not frame_read_success: continue

        # --- Detect and estimate poses in all views ---
        view_detections = {}  # {port: [(confidence, person_data), ...]} sorted by confidence
        
        for port in common_ports:
            image = current_frames_pil[port]
            person_boxes_voc, person_boxes_coco, person_scores = detect_persons(
                image, person_processor, person_model, device, person_confidence
            )
            
            if person_boxes_coco.size == 0: 
                view_detections[port] = []
                continue

            all_keypoints_2d, all_keypoint_scores = estimate_poses(
                image, person_boxes_coco, pose_processor, pose_model, device
            )

            port_detections = []
            
            for person_idx in range(len(all_keypoints_2d)):
                kps_2d, scores_2d = all_keypoints_2d[person_idx], all_keypoint_scores[person_idx]
                detection_confidence = person_scores[person_idx]  # Use detection confidence
                
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
                    #print text warning to screen
                    print(f"Warn: No valid hips for COM p:{port} s {sync_index} person{person_idx}")
                    # try to use only 1 hip
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
                
                
                # port_detections.append((detection_confidence, person_data))
                port_detections.append(person_data)

            # Sort by confidence (highest first)
            view_detections[port] = port_detections

        groups = group_detections_across_views_bipartite_full(view_detections, None, 
                                           projection_matrices, port_to_cam_index,
                                           camera_params, epipolar_threshold=epipolar_threshold)
        # or epipolar:
        # group_detections_epipolar(view_detections, None,
        #                          projection_matrices, port_to_cam_index,
        #                          camera_params, epipolar_threshold=30)
        
        # --- Generate 3D candidates using all possible combinations ---
        candidate_3d_groups = generate_3d_candidates_from_groups(
            groups, port_to_cam_index, camera_params, projection_matrices, hip_indices
        )
        
        # --- Associate 3D candidates with existing tracks ---
        track_assignments, unused_3d_groups = assign_3d_candidates_to_tracks(
            active_tracks, candidate_3d_groups, 0.15, default_views=override_views_used
        )
        
        # new code to skip bad frames:
        # Check if this is a bad frame (no successful assignments)
        if len(active_tracks) > 0 and not track_assignments:
            print(f"Bad frame detected at sync {sync_index}: no tracks matched")
            
            # Just increment lost counters for all tracks and skip the rest of the updating. 
            for track in active_tracks:
                track.increment_lost_counter()
                if not track.is_active:
                    print(f"Lost track of person {track.id} at sync {sync_index}")
            
            # Remove inactive tracks
            active_tracks = [t for t in active_tracks if t.is_active]
            
            # Continue to next frame without updating any results
            continue

        # Update existing tracks.
        # since nothing in the above prevents us from tracking more than the number of desired active tracks, 
        # we need to do so here. 
        # here track_id is an actual id. 
        for track_id,curtuple in track_assignments.items():
            if curtuple is not None:
                (grp_idx, candidate_idx) = curtuple
    

            if candidate_idx is not None:
                candidate = candidate_3d_groups[grp_idx][candidate_idx]
                if len(active_tracks) < max_persons:
                    new_track = PersonTrack(track_id, candidate_3d_groups[grp_idx][candidate_idx]['keypoints_3d'], sync_index,hip_indices,candidate['views'])
                    active_tracks.append(new_track) 
                    print("added new track")
                    #track_idx = len(active_tracks)
                else:
                    # find which track to update
                    track_idx = None
                    for i, t in enumerate(active_tracks):
                        if t.id == track_id:
                            track_idx = i
                            break
                    if track_idx is not None:
                        active_tracks[track_idx].update(candidate['keypoints_3d'], sync_index, candidate['views'])
                    else:
                        print(f"Warn: Track ID {track_id} not found in active tracks, and we're full! Continue-ing.")
                        continue

                #
                person_id = active_tracks[track_id].id
                if person_id not in all_results_by_person:
                    all_results_by_person[person_id] = []
                if person_id not in all_pixel_coords_by_person:
                    all_pixel_coords_by_person[person_id] = []
                if person_id not in all_cameras_by_person:
                    all_cameras_by_person[person_id] = []
                    
                kps_3d_list = [[np.nan]*3 if kp is None else kp.tolist() 
                               for kp in candidate['keypoints_3d']]
                all_results_by_person[person_id].append({
                    'sync_index': sync_index,
                    'person_id': person_id,
                    'keypoints_3d': kps_3d_list
                })
                
                # Project 3D keypoints to 2D pixel coordinates for all cameras; we will output these later. 
                # note: this is somewhat misleading because we do not necessarily use all the cameras when we make estimates. 
                # TODO: update these outputs to add an additional column for cameras used.
                pixel_coords = project_keypoints_to_all_cameras(
                    candidate['keypoints_3d'], projection_matrices, common_ports, port_to_cam_index
                )
                all_pixel_coords_by_person[person_id].append({
                    'sync_index': sync_index,
                    'person_id': person_id,
                    'pixel_coords': pixel_coords
                })
                
                # Store camera information for diagnostics
                all_cameras_by_person[person_id].append({
                    'sync_index': sync_index,
                    'person_id': person_id,
                    'cameras_used': candidate_3d_groups[grp_idx][candidate_idx]['views']
                })
                
        # Update lost counters for unmatched tracks
        for i, track in enumerate(active_tracks):
            if i not in track_assignments or track_assignments[i] is None:
                track.increment_lost_counter()
                if not track.is_active:
                    print(f"Lost track of person {track.id} at sync {sync_index}")
        
        # Update previous_views_used with views from candidates that were actually used
        current_views_used = []
        for track_idx, tuple_idxs in track_assignments.items():
            grp_idx = tuple_idxs[0]
            candidate_idx = tuple_idxs[1]
            if candidate_idx is not None:
                candidate = candidate_3d_groups[grp_idx][candidate_idx]
                current_views_used.append(candidate['views'])
        
        # # Also add views from new tracks that were created
        # for candidate_idx in used_candidates:
        #     if candidate_idx < len(candidate_3d_groups):
        #         candidate = candidate_3d_groups[candidate_idx]
        #         # Only add if not already added from track assignments
        #         if candidate['views_used'] not in current_views_used:
        #             current_views_used.append(candidate['views_used'])
        
        previous_views_used = current_views_used
        
        # Remove inactive tracks
        active_tracks = [t for t in active_tracks if t.is_active]
        
        # Store first frame for debugging if enabled
        if verbose_debug and sync_index == all_sync_indices[0]:
            debug_frames = current_frames_pil.copy()

    # --- 8. Cleanup and Save Results ---
    print("\nReleasing video captures...")
    [cap.release() for cap in caps.values()]
    end_time = time.time()
    print(f"Processing finished in {end_time - start_time:.2f} seconds.")

    # Save results for each person
    base_output_path = output_path
    
    for person_id, results in all_results_by_person.items():
        if not results:
            continue
            
        print(f"\nSaving results for Person {person_id} ({len(results)} frames)...")
        
        # Define output filenames for this person
        person_csv_path = f"{base_output_path}_person{person_id}.csv"
        person_pickle_path = f"{base_output_path}_person{person_id}.pkl"
        
        # Save pickle
        try:
            with open(person_pickle_path, 'wb') as f_pkl:
                pickle.dump(results, f_pkl)
            print(f"Saved pickle: {person_pickle_path}")
        except Exception as e:
            print(f"Error saving pickle for person {person_id}: {e}")
        
        # Save CSV (same format as original)
        save_person_csv(results, person_csv_path, expected_num_kps=52)
        
        # Save pixel coordinates CSV for each port
        if person_id in all_pixel_coords_by_person:
            pixel_results = all_pixel_coords_by_person[person_id]
            for port in common_ports:
                pixel_csv_path = f"{base_output_path}_person{person_id}_pixelcoords_port_{port}.csv"
                save_pixel_coords_csv(pixel_results, pixel_csv_path, expected_num_kps=52, port=port)
        
        # Save camera information CSV for diagnostics
        if person_id in all_cameras_by_person:
            camera_results = all_cameras_by_person[person_id]
            cameras_csv_path = f"{base_output_path}_person{person_id}_cameras.csv"
            save_cameras_csv(camera_results, cameras_csv_path)
    
    # Debug visualization if enabled
    if verbose_debug and debug_frames and all_pixel_coords_by_person:
        print("\nGenerating debug visualization...")
        plot_debug_keypoints(debug_frames, all_pixel_coords_by_person, common_ports, all_sync_indices[0])
    
    print(f"\nTotal persons tracked: {len(all_results_by_person)}")
    return all_results_by_person