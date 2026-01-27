# Visualization functions for processed pose data and analysis results
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Slider, Button
import glob
import os
import colorsys

# Import required functions from other modules
from .cs_parse import parse_calibration_mwc, calculate_projection_matrices
from .pose_detector import SynthPoseMarkers
from .libwalk import quick_rotation_matrix

def batch_project_poses_to_video(base_dir):
    subfolders = [f.path for f in os.scandir(base_dir) if f.is_dir()]
    print(subfolders)


    for subfolder in subfolders:
        print(f"Processing subfolder: {subfolder}")
        results_directory = os.path.join(subfolder, 'synthpose')
        if not os.path.exists(results_directory):
            print(f"Skipping {results_directory} as it does not exist.")
            continue

        # project the poses to video
        project_poses_to_video(results_directory, verbose = False, port_number=0, output_video_name="detected_people.mp4")

def project_3d_to_2d(point_3d, P):
    """Projects a 3D point to 2D using a projection matrix."""
    if point_3d is None or np.isnan(point_3d).any(): 
        return None
    point_4d = np.append(point_3d, 1.0)
    point_2d_hom = P @ point_4d
    if abs(point_2d_hom[2]) < 1e-6: 
        return None  # Check for near-zero depth
    point_2d = point_2d_hom[:2] / point_2d_hom[2]
    return point_2d.flatten()

def create_clothing_debug_plot(video_file_path, elbow_pixels, knee_pixels, 
                              elbow_sample_frames, knee_sample_frames,
                              shirt_median_hsl, pants_median_hsl,
                              movement_threshold, sampling_radius, max_samples,
                              output_base_name):
    """
    Create comprehensive debug visualization for clothing color detection.
    
    Args:
        video_file_path (str): Path to video file
        elbow_pixels (list): List of elbow pixel coordinates
        knee_pixels (list): List of knee pixel coordinates
        elbow_sample_frames (list): Frame indices used for elbow sampling
        knee_sample_frames (list): Frame indices used for knee sampling
        shirt_median_hsl (list): HSL values for shirt color
        pants_median_hsl (list): HSL values for pants color
        movement_threshold (float): Movement threshold used
        sampling_radius (int): Sampling radius used
        max_samples (int): Maximum samples parameter
        output_base_name (str): Base name for output files
    
    Returns:
        str: Path to saved debug plot
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Top left: Show first frame with detected points
    if elbow_sample_frames or knee_sample_frames:
        # Use the first available frame for visualization
        viz_frame_idx = None
        if elbow_sample_frames:
            viz_frame_idx = elbow_sample_frames[0]
        elif knee_sample_frames:
            viz_frame_idx = knee_sample_frames[0]
        
        if viz_frame_idx is not None:
            # Load the frame
            cap = cv2.VideoCapture(video_file_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, viz_frame_idx)
            ret, frame = cap.read()
            cap.release()
            
            if ret:
                # Convert BGR to RGB for matplotlib
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                axes[0,0].imshow(frame_rgb)
                axes[0,0].set_title(f'Detection Frame {viz_frame_idx}')
                
                # Draw circles around detected points
                if viz_frame_idx < len(elbow_pixels) and elbow_pixels[viz_frame_idx] is not None:
                    px, py = elbow_pixels[viz_frame_idx]
                    circle = plt.Circle((px, py), sampling_radius, color='red', fill=False, linewidth=2)
                    axes[0,0].add_patch(circle)
                    axes[0,0].text(px + sampling_radius + 5, py, 'Elbow', color='red', fontsize=10, weight='bold')
                
                if viz_frame_idx < len(knee_pixels) and knee_pixels[viz_frame_idx] is not None:
                    px, py = knee_pixels[viz_frame_idx]
                    circle = plt.Circle((px, py), sampling_radius, color='blue', fill=False, linewidth=2)
                    axes[0,0].add_patch(circle)
                    axes[0,0].text(px + sampling_radius + 5, py, 'Knee', color='blue', fontsize=10, weight='bold')
                
                axes[0,0].axis('off')
            else:
                axes[0,0].text(0.5, 0.5, 'Could not load frame', ha='center', va='center', transform=axes[0,0].transAxes)
                axes[0,0].set_title('Frame Not Available')
        else:
            axes[0,0].text(0.5, 0.5, 'No sample frames available', ha='center', va='center', transform=axes[0,0].transAxes)
            axes[0,0].set_title('No Detection Frames')
    else:
        axes[0,0].text(0.5, 0.5, 'No detections found', ha='center', va='center', transform=axes[0,0].transAxes)
        axes[0,0].set_title('No Detections')
    
    # Top middle: Elbow pixel trajectory
    elbow_x_coords = [p[0] if p is not None else np.nan for p in elbow_pixels]
    elbow_y_coords = [p[1] if p is not None else np.nan for p in elbow_pixels]
    frame_indices = list(range(len(elbow_pixels)))
    
    axes[0,1].plot(frame_indices, elbow_x_coords, 'r-', alpha=0.7, label='X coordinate')
    axes[0,1].plot(frame_indices, elbow_y_coords, 'b-', alpha=0.7, label='Y coordinate')
    
    # Highlight frames used for elbow color sampling
    for frame_idx in elbow_sample_frames:
        if frame_idx < len(elbow_pixels) and elbow_pixels[frame_idx] is not None:
            axes[0,1].axvline(x=frame_idx, color='red', linestyle='--', alpha=0.8, linewidth=2)
            axes[0,1].plot(frame_idx, elbow_x_coords[frame_idx], 'ro', markersize=8)
            axes[0,1].plot(frame_idx, elbow_y_coords[frame_idx], 'bo', markersize=8)
    
    axes[0,1].set_title(f'Elbow Pixel Trajectory\n({len(elbow_sample_frames)} frames used)')
    axes[0,1].set_xlabel('Frame Index')
    axes[0,1].set_ylabel('Pixel Coordinate')
    axes[0,1].legend()
    axes[0,1].grid(True, alpha=0.3)
    
    # Top right: Knee pixel trajectory
    knee_x_coords = [p[0] if p is not None else np.nan for p in knee_pixels]
    knee_y_coords = [p[1] if p is not None else np.nan for p in knee_pixels]
    
    axes[0,2].plot(frame_indices, knee_x_coords, 'r-', alpha=0.7, label='X coordinate')
    axes[0,2].plot(frame_indices, knee_y_coords, 'b-', alpha=0.7, label='Y coordinate')
    
    # Highlight frames used for knee color sampling
    for frame_idx in knee_sample_frames:
        if frame_idx < len(knee_pixels) and knee_pixels[frame_idx] is not None:
            axes[0,2].axvline(x=frame_idx, color='blue', linestyle='--', alpha=0.8, linewidth=2)
            axes[0,2].plot(frame_idx, knee_x_coords[frame_idx], 'ro', markersize=8)
            axes[0,2].plot(frame_idx, knee_y_coords[frame_idx], 'bo', markersize=8)
    
    axes[0,2].set_title(f'Knee Pixel Trajectory\n({len(knee_sample_frames)} frames used)')
    axes[0,2].set_xlabel('Frame Index')
    axes[0,2].set_ylabel('Pixel Coordinate')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # Convert HSL back to RGB for display
    def hsl_to_rgb(h, s, l):
        """Convert HSL to RGB for display"""
        h = h / 360.0
        s = s / 100.0
        l = l / 100.0
        r, g, b = colorsys.hls_to_rgb(h, l, s)
        return [r, g, b]
    
    # Bottom left: Shirt color
    shirt_rgb = hsl_to_rgb(shirt_median_hsl[0], shirt_median_hsl[1], shirt_median_hsl[2])
    axes[1,0].imshow([[shirt_rgb]], aspect='auto')
    axes[1,0].set_title(f'Shirt Color (from Elbow)\nH:{shirt_median_hsl[0]:.1f} S:{shirt_median_hsl[1]:.1f} L:{shirt_median_hsl[2]:.1f}')
    axes[1,0].axis('off')
    
    # Bottom middle: Pants color
    pants_rgb = hsl_to_rgb(pants_median_hsl[0], pants_median_hsl[1], pants_median_hsl[2])
    axes[1,1].imshow([[pants_rgb]], aspect='auto')
    axes[1,1].set_title(f'Pants Color (from Knee)\nH:{pants_median_hsl[0]:.1f} S:{pants_median_hsl[1]:.1f} L:{pants_median_hsl[2]:.1f}')
    axes[1,1].axis('off')
    
    # Bottom right: Summary statistics
    axes[1,2].text(0.1, 0.9, f'Elbow Frames Used: {elbow_sample_frames}', transform=axes[1,2].transAxes, fontsize=10)
    axes[1,2].text(0.1, 0.8, f'Knee Frames Used: {knee_sample_frames}', transform=axes[1,2].transAxes, fontsize=10)
    axes[1,2].text(0.1, 0.7, f'Movement Threshold: {movement_threshold:.1f} pixels', transform=axes[1,2].transAxes, fontsize=10)
    axes[1,2].text(0.1, 0.6, f'Sampling Radius: {sampling_radius} pixels', transform=axes[1,2].transAxes, fontsize=10)
    axes[1,2].text(0.1, 0.5, f'Max Samples: {max_samples}', transform=axes[1,2].transAxes, fontsize=10)
    axes[1,2].text(0.1, 0.4, f'Total Frames: {len(elbow_pixels)}', transform=axes[1,2].transAxes, fontsize=10)
    axes[1,2].text(0.1, 0.3, f'Valid Elbow Frames: {sum(1 for p in elbow_pixels if p is not None)}', transform=axes[1,2].transAxes, fontsize=10)
    axes[1,2].text(0.1, 0.2, f'Valid Knee Frames: {sum(1 for p in knee_pixels if p is not None)}', transform=axes[1,2].transAxes, fontsize=10)
    axes[1,2].set_title('Summary Statistics')
    axes[1,2].axis('off')
    
    plt.tight_layout()
    
    # Save debug plot
    debug_plot_path = f"{output_base_name}_clothes_debug.png"
    plt.savefig(debug_plot_path, dpi=150, bbox_inches='tight')
    plt.show()
    
    return debug_plot_path

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

def show_multi_person_results(results_directory, output_plot_path=None, alpha = 0.2, verbose = True):
    """
    Read all person CSV files from a directory and create a timeline plot 
    showing when each detected person was active.
    
    Args:
        results_directory (str): Directory containing *_person*.csv files
        output_plot_path (str, optional): Path to save the plot. If None, displays plot.
    """
    # Find all person CSV files
    person_files = glob.glob(os.path.join(results_directory, "*_person[0-9].csv"))
    
    if not person_files:
        print(f"No person CSV files found in {results_directory}")
        return
    
    print(f"Found {len(person_files)} person files:")
    for f in person_files:
        print(f"  - {os.path.basename(f)}")
    
    # Read data from each person file
    person_data = {}
    person_files_data = {}
    
    for person_file in person_files:
        # Extract person ID from filename
        filename = os.path.basename(person_file)
        if '_person' in filename:
            try:
                person_id = int(filename.split('_person')[1].split('.')[0])
            except (ValueError, IndexError):
                person_id = filename  # fallback to filename if parsing fails
        else:
            person_id = filename
        
        # Read the CSV data
        data = read_posetrack_csv(person_file)
        if data is not None and '_metadata' in data:
            person_data[person_id] = data['_metadata']
            person_files_data[person_id] = data  # Store full data for animation
    
    if not person_data:
        print("No valid person data found")
        return
    
    if verbose==False:
        return person_data, person_files_data

    # Create the timeline plot
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(person_data)))
    
    # Plot each person's activity
    for i, (person_id, metadata) in enumerate(person_data.items()):
        sync_indices = metadata['sync_indices']
        num_frames = metadata['num_frames']
        
        # Create y-position for this person
        y_pos = i + 1
        
        # Plot the active frames
        plt.scatter(sync_indices, [y_pos] * len(sync_indices), 
                   color=colors[i], alpha=alpha, s=20, label=f'Person {person_id}')
        
        # Add text annotation
        if len(sync_indices) > 0:
            plt.text(sync_indices[0], y_pos + 0.1, f'Person {person_id} ({num_frames} frames)', 
                    fontsize=10, ha='left')
    
    # Customize the plot
    plt.xlabel('Sync Index')
    plt.ylabel('Person ID')
    plt.title(f'Multi-Person Detection Timeline\nDirectory: {os.path.basename(results_directory)}')
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Set y-axis limits and labels
    if person_data:
        plt.ylim(0.5, len(person_data) + 0.5)
        plt.yticks(range(1, len(person_data) + 1), [f'Person {pid}' for pid in person_data.keys()])
    
    # Show statistics
    total_detections = sum(metadata['num_frames'] for metadata in person_data.values())
    print(f"\nDetection Summary:")
    print(f"  Total persons detected: {len(person_data)}")
    print(f"  Total frame detections: {total_detections}")
    
    for person_id, metadata in person_data.items():
        sync_indices = metadata['sync_indices']
        if len(sync_indices) > 0:
            sync_range = f"{sync_indices.min()}-{sync_indices.max()}"
            print(f"  Person {person_id}: {metadata['num_frames']} frames, sync range {sync_range}")
    
    plt.tight_layout()
    
    if output_plot_path:
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {output_plot_path}")
        
    if verbose:
        plt.show()
    else:
        plt.close()
    
    return person_data, person_files_data

def animate_multi_person_results(results_directory, output_plot_path=None, alpha=0.7):
    """
    Create an interactive animation of multi-person 3D pose data with timeline and 3D visualization.

    Args:
        results_directory (str): Directory containing *_person*.csv files
        output_plot_path (str, optional): Path to save the plot. If None, displays plot.
        alpha (float, optional): Transparency value for plotted points and lines (0.0 to 1.0). Default is 0.7.
    """
    # Get person data and loaded CSV data
    person_data, person_files_data = show_multi_person_results(results_directory, output_plot_path, verbose=False)
    
    if not person_data or not person_files_data:
        print("No person data available for animation")
        return
    
    # Define keypoints to animate (subset for clarity) - using actual SynthPoseMarkers names
    keypoints_to_animate = [
        'L_Ankle', 'R_Ankle',  # feet
        'L_Knee', 'R_Knee',    # knees  
        'L_Hip', 'R_Hip',      # hips
        'L_Shoulder', 'R_Shoulder',  # shoulders
        'L_Elbow', 'R_Elbow',  # elbows
        'L_Wrist', 'R_Wrist',  # wrists
        'L_Eye', 'R_Eye',      # eyes
        'Nose'                 # head
    ]
    
    # Get all unique sync indices across all persons
    all_sync_indices = set()
    for data in person_files_data.values():
        if 'sync_index' in data:
            all_sync_indices.update(data['sync_index'])
    
    if not all_sync_indices:
        print("No sync indices found in data")
        return
    
    all_sync_indices = sorted(list(all_sync_indices))
    min_sync = min(all_sync_indices)
    max_sync = max(all_sync_indices)
    
    print(f"Animation range: sync_index {min_sync} to {max_sync} ({len(all_sync_indices)} frames)")
    
    # Calculate rotation matrix and offset for the first person only
    rotation_matrix = np.eye(3)
    x0 = np.zeros(3)
    if person_files_data:
        # Find the person whose first sync_index is the smallest (i.e., detected first)
        min_first_sync = float('inf')
        first_person_id = None
        for pid, pdata in person_files_data.items():
            sync_indices = pdata.get('sync_index', [])
            if len(sync_indices) > 0 and sync_indices[0] < min_first_sync:
                min_first_sync = sync_indices[0]
                first_person_id = pid
        if first_person_id is None:
            # Fallback: just pick the first key if none found
            first_person_id = list(person_files_data.keys())[0]
        first_person_data = person_files_data[first_person_id]
        print(f"\nCalculating rotation matrix using Person {first_person_id}...")
        rotation_matrix, x0 = quick_rotation_matrix(first_person_data)
    
    # Create figure with subplots
    fig, (ax_timeline, ax_3d) = plt.subplots(2, 1, figsize=(16, 10), 
                                            gridspec_kw={'height_ratios': [1, 2]})
    
    # Make 3D subplot
    ax_3d.remove()
    ax_3d = fig.add_subplot(2, 1, 2, projection='3d')
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(person_data)))
    
    # Plot timeline (same as show_multi_person_results)
    for i, (person_id, metadata) in enumerate(person_data.items()):
        sync_indices = metadata['sync_indices']
        y_pos = i + 1
        ax_timeline.scatter(sync_indices, [y_pos] * len(sync_indices),
                          color=colors[i], alpha=alpha, s=20, label=f'Person {person_id}')
    
    ax_timeline.set_xlabel('Sync Index')
    ax_timeline.set_ylabel('Person ID')
    ax_timeline.set_title('Multi-Person Detection Timeline')
    ax_timeline.grid(True, alpha=0.3)
    ax_timeline.legend()
    
    # Add current frame indicator
    current_frame_line = ax_timeline.axvline(x=min_sync, color='red', linewidth=2, label='Current Frame')
    
    # 3D pose subplot
    ax_3d.set_xlabel('X (m)')
    ax_3d.set_ylabel('Y (m)')
    ax_3d.set_zlabel('Z (m)')
    ax_3d.set_title('3D Pose Visualization')
    
    # Set fixed axis limits
    ax_3d.set_xlim([-2.5, 2.5])
    ax_3d.set_ylim([-2.5, 2.5])
    ax_3d.set_zlim([-2.5, 2.5])
    
    # Initialize 3D scatter plots for each person with dummy data
    person_scatters = {}
    for i, person_id in enumerate(person_data.keys()):
        person_scatters[person_id] = ax_3d.scatter([0], [0], [0],
                                                  color=colors[i], s=50, alpha=alpha,
                                                  label=f'Person {person_id}')
    ax_3d.legend()
    
    # Animation state
    current_sync_idx_index = 0  # Index into all_sync_indices array
    is_playing = False
    animation_timer = None
    
    def update_frame(current_sync_index):
        """Update the 3D visualization for a given sync index."""
        # Update timeline indicator
        current_frame_line.set_xdata([current_sync_index, current_sync_index])
        
        # Count people displayed
        people_displayed = 0
        
        print(f"\n=== FRAME UPDATE: sync_index = {current_sync_index} ===")
        
        # Update 3D poses
        for i, (person_id, data) in enumerate(person_files_data.items()):
            if 'sync_index' not in data:
                continue
                
            # Find frame with matching sync index
            sync_mask = data['sync_index'] == current_sync_index
            
            if np.any(sync_mask):
                # Get keypoints for this frame
                x_coords, y_coords, z_coords = [], [], []
                
                # Track feet coordinates for debugging
                left_ankle_coord = None
                right_ankle_coord = None
                
                for keypoint_name in keypoints_to_animate:
                    if keypoint_name in data:
                        kp_data = data[keypoint_name][sync_mask]
                        if len(kp_data) > 0:
                            x, y, z = kp_data[0]  # First (and should be only) matching frame
                            if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                                # Apply rotation matrix and translation
                                original_point = np.array([x, y, z])
                                rotated_point = rotation_matrix.T @ (original_point - x0)
                                
                                x_coords.append(rotated_point[0])
                                y_coords.append(rotated_point[1])
                                z_coords.append(rotated_point[2])
                                
                                # Capture feet coordinates for debugging (rotated coordinates)
                                if keypoint_name == 'L_Ankle':
                                    left_ankle_coord = tuple(rotated_point)
                                elif keypoint_name == 'R_Ankle':
                                    right_ankle_coord = tuple(rotated_point)
                
                # Update scatter plot data
                if x_coords:
                    people_displayed += 1
                    person_scatters[person_id]._offsets3d = (x_coords, y_coords, z_coords)
                    
                    # Debug info for this person
                    print(f"Person {person_id}: {len(x_coords)} keypoints")
                    if left_ankle_coord:
                        print(f"  Left Ankle:  ({left_ankle_coord[0]:.3f}, {left_ankle_coord[1]:.3f}, {left_ankle_coord[2]:.3f})")
                    if right_ankle_coord:
                        print(f"  Right Ankle: ({right_ankle_coord[0]:.3f}, {right_ankle_coord[1]:.3f}, {right_ankle_coord[2]:.3f})")
                else:
                    print(f"Person {person_id}: no valid keypoints")
                    person_scatters[person_id]._offsets3d = ([], [], [])
            else:
                # No data for this sync index
                person_scatters[person_id]._offsets3d = ([], [], [])
                print(f"Person {person_id}: not present at this sync_index")
        
        print(f"Total people displayed: {people_displayed}")
        print("=" * 50)
        
        # Redraw the plot
        fig.canvas.draw()
    
    def on_slider_change(val):
        """Handle slider changes."""
        nonlocal current_sync_idx_index, is_playing, animation_timer
        current_sync_value = int(val)
        # Find index in all_sync_indices that corresponds to this sync_index value
        try:
            current_sync_idx_index = all_sync_indices.index(current_sync_value)
        except ValueError:
            # Find closest sync_index
            current_sync_idx_index = min(range(len(all_sync_indices)), 
                                        key=lambda i: abs(all_sync_indices[i] - current_sync_value))
        
        # Stop animation when user interacts with slider
        if is_playing and animation_timer:
            animation_timer.stop()
            is_playing = False
            play_button.label.set_text('Play')
        
        update_frame(current_sync_value)
    
    def on_play_button(event):
        """Handle play button clicks."""
        nonlocal is_playing, current_sync_idx_index, animation_timer
        is_playing = not is_playing
        
        if is_playing:
            play_button.label.set_text('Pause')
            # Start timer-based animation
            animation_timer = fig.canvas.new_timer(interval=100)  # 100ms = 10 FPS
            animation_timer.add_callback(animate_step)
            animation_timer.start()
        else:
            play_button.label.set_text('Play')
            # Stop timer
            if animation_timer:
                animation_timer.stop()
    
    def animate_step():
        """Single animation step called by timer."""
        nonlocal current_sync_idx_index, is_playing
        
        if not is_playing:
            return
        
        # Move to next frame
        current_sync_idx_index += 1
        if current_sync_idx_index >= len(all_sync_indices):
            current_sync_idx_index = 0  # Loop back to start
        
        # Get current sync_index value and update display
        current_sync_value = all_sync_indices[current_sync_idx_index]
        slider.set_val(current_sync_value)
        update_frame(current_sync_value)
    
    # Create slider - use actual sync_index values, not frame indices
    slider_ax = plt.axes([0.1, 0.02, 0.6, 0.03])
    slider = Slider(slider_ax, 'Sync Index', min_sync, max_sync, 
                   valinit=min_sync, valfmt='%d', valstep=1)
    slider.on_changed(on_slider_change)
    
    # Create play button
    play_button_ax = plt.axes([0.75, 0.02, 0.1, 0.03])
    play_button = Button(play_button_ax, 'Play')
    play_button.on_clicked(on_play_button)
    
    # Initial frame
    update_frame(min_sync)
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.1)  # Make room for controls
    
    if output_plot_path:
        # For static save, just save the first frame
        plt.savefig(output_plot_path, dpi=300, bbox_inches='tight')
        print(f"Static plot saved to: {output_plot_path}")
    else:
        plt.show()
    
    return fig

def project_poses_to_video(synthpose_directory_path, port_number, output_video_name="detected_people.mp4", verbose=True):
    """
    Project 3D pose data back onto a video from a specific camera port and save as new video.
    
    Args:
        results_directory (str): Directory containing *_person*.csv files and config.toml
        port_number (int): Camera port number (e.g., 0, 1, 2)
        output_video_name (str): Name of output video file
    """
    import cv2
    import os
    
    # Load person data
    print(f"Loading person data from: {synthpose_directory_path}")
    bothret = show_multi_person_results(synthpose_directory_path, verbose=verbose, output_plot_path=None)

    if bothret is None:
        print("No person data available for video projection")
        return
    else: 
        (person_data, person_files_data) = bothret

    # Load calibration. typically the config.toml is uh 3 levels up from the specific synthpose_directory_path
    # so use os.path.basename to get the parent directory
    calibration_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(synthpose_directory_path))), "config.toml")
    print(calibration_path)
    video_path = os.path.join(os.path.dirname(synthpose_directory_path), f"port_{port_number}.mp4")
    
    if not os.path.exists(calibration_path):
        print(f"Calibration file not found: {calibration_path}")
        print("f trying the results directory instead")
        calibration_path = os.path.join(synthpose_directory_path, "config.toml")
        if not os.path.exists(calibration_path):
            print(f"Calibration file still not found: {calibration_path}")  
        return
    
    if not os.path.exists(video_path):
        print(f"Video file not found: {video_path}")
        print('trying this directory instead')
        video_path = os.path.join(synthpose_directory_path, f"port_{port_number}.mp4")
        if not os.path.exists(video_path):
            print(f"Video file still not found: {video_path}")
            return

    print(f"Loading calibration from: {calibration_path}")
    camera_params = parse_calibration_mwc(calibration_path)
    if not camera_params:
        print("Failed to load camera calibration")
        return
    
    # Find camera parameters for this port using the parsed data
    target_cam_params = None
    target_cam_name = f"cam_{port_number}"
    
    for params in camera_params:
        # Check if this camera has the matching port number
        if params.get('port') == port_number:
            target_cam_params = params
            print(f"Found camera with explicit port {port_number}")
            break
        # Check if camera name matches expected pattern (cam_0, cam_1, etc.)
        elif params.get('id') == target_cam_name:
            target_cam_params = params
            print(f"Assuming {target_cam_name} == port_{port_number}")
            break
    
    if target_cam_params is None:
        print(f"Camera parameters for port {port_number} not found")
        print(f"Available cameras: {[p.get('id', 'unknown') for p in camera_params]}")
        print(f"Expected camera name: {target_cam_name}")
        return
    
    # Calculate projection matrix for this camera
    projection_matrices = calculate_projection_matrices([target_cam_params])
    if not projection_matrices:
        print("Failed to calculate projection matrix")
        return
    
    P = projection_matrices[0]
    
    # Load frame time history to map sync_index to frame numbers. again, typically one level up.
    frame_history_path = os.path.join(os.path.dirname(synthpose_directory_path), "frame_time_history.csv")
    if not os.path.exists(frame_history_path):
        print(f"Frame history not found: {frame_history_path}")
        print('trying this directory instead')
        frame_history_path = os.path.join(synthpose_directory_path, "frame_time_history.csv")
        if not os.path.exists(frame_history_path):
            print(f"Frame history still not found: {frame_history_path}")
        return      
    
    import pandas as pd
    frame_history_df = pd.read_csv(frame_history_path)
    
    print(f"Frame history loaded: {len(frame_history_df)} total rows")
    print(f"Ports in frame history: {sorted(frame_history_df['port'].unique())}")
    print(f"Sync index range: {frame_history_df['sync_index'].min()} to {frame_history_df['sync_index'].max()}")
    
    # Filter for this port
    frame_history_df = frame_history_df[frame_history_df['port'] == port_number]
    print(f"Rows for port {port_number}: {len(frame_history_df)}")
    
    if len(frame_history_df) == 0:
        print(f"No frame history data found for port {port_number}")
        return
    
    frame_history_df = frame_history_df.sort_values(by='frame_time')
    frame_history_df['derived_frame_index'] = frame_history_df.groupby('port')['frame_time'].rank(method='min').astype(int) - 1
    
    print(f"Port {port_number} sync index range: {frame_history_df['sync_index'].min()} to {frame_history_df['sync_index'].max()}")
    print(f"Port {port_number} derived frame range: {frame_history_df['derived_frame_index'].min()} to {frame_history_df['derived_frame_index'].max()}")
    print(f"First few sync->frame mappings:")
    for i, (_, row) in enumerate(frame_history_df.head(10).iterrows()):
        print(f"  sync_index {row['sync_index']} -> frame {row['derived_frame_index']}")
        
    # Also show person data sync ranges
    print(f"\nPerson data sync index ranges:")
    for person_id, data in person_files_data.items():
        if 'sync_index' in data:
            sync_range = f"{data['sync_index'].min()} to {data['sync_index'].max()}"
            print(f"  Person {person_id}: {sync_range} ({len(data['sync_index'])} frames)")
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video: {video_path}")
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties: {frame_width}x{frame_height}, {fps} FPS, {total_frames} frames")
    
    # Setup output video
    output_path = os.path.join(synthpose_directory_path, output_video_name)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
    
    # Color map for different people
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    
    print(f"Processing {total_frames} frames...")
    
    # Track projection statistics
    frames_with_poses = 0
    frames_processed = 0
    
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        frames_processed += 1
        
        # Find corresponding sync_index for this frame
        matching_rows = frame_history_df[frame_history_df['derived_frame_index'] == frame_idx]
        if matching_rows.empty:
            # No sync data for this frame, just write original frame
            if frame_idx % 200 == 0:
                print(f"Frame {frame_idx}: No sync data available")
            out.write(frame)
            continue
        
        current_sync_index = matching_rows.iloc[0]['sync_index']
        
        # Debug sync mapping every 200 frames
        if frame_idx % 200 == 0:
            print(f"Frame {frame_idx}: sync_index = {current_sync_index}")
        
        # 1. Count unique people present at this sync_index
        people_in_frame = 0
        for person_id, data in person_files_data.items():
            if 'sync_index' not in data:
                continue
            # 2. Check if this person has data at current_sync_index
            sync_mask = data['sync_index'] == current_sync_index
            if np.any(sync_mask):
                # 3. Count them once (not multiple times for multiple camera views)
                people_in_frame += 1
        
        # Create fresh overlay for this frame
        overlay = frame.copy()

        # Draw poses for each person
        for person_idx, (person_id, data) in enumerate(person_files_data.items()):
            if 'sync_index' not in data:
                continue

            # Find frame with matching sync index
            sync_mask = data['sync_index'] == current_sync_index
            if not np.any(sync_mask):
                continue

            # Found a person at this sync index!
            if frame_idx % 200 == 0:  # Log every 200 frames
                print(f"  -> Found Person {person_id} at sync_index {current_sync_index} (frame {frame_idx})")

            # Get color for this person
            color = colors[person_idx % len(colors)]

            # Project and draw keypoints
            keypoints_2d = []

            for keypoint_name in ['L_Ankle','R_Ankle', 'L_Knee', 'R_Knee', 'L_Hip', 'R_Hip',
                                'L_Shoulder', 'R_Shoulder', 'L_Elbow', 'R_Elbow', 'L_Wrist', 'R_Wrist',
                                'L_Eye', 'R_Eye', 'Nose','l_calc','l_big_toe','r_calc',
                                'r_big_toe',]: #6 per column
                if keypoint_name in data:
                    kp_data = data[keypoint_name][sync_mask]
                    if len(kp_data) > 0:
                        x, y, z = kp_data[0]
                        if not (np.isnan(x) or np.isnan(y) or np.isnan(z)):
                            # Project 3D point to 2D
                            point_3d = np.array([x, y, z])
                            point_2d = project_3d_to_2d(point_3d, P)
                            if point_2d is not None:
                                keypoints_2d.append((int(point_2d[0]), int(point_2d[1])))
                            else:
                                keypoints_2d.append(None)
                        else:
                            keypoints_2d.append(None)
                    else:
                        keypoints_2d.append(None)
                else:
                    keypoints_2d.append(None)

            # Draw skeleton connections on the overlay with anti-aliasing
            connections = [
                # Legs
                (0, 2), (1, 3),  # ankles to knees
                (2, 4), (3, 5),  # knees to hips
                (4, 5),          # hips together
                # Torso
                (4, 6), (5, 7),  # hips to shoulders
                (6, 7),          # shoulders together
                # Arms
                (6, 8), (7, 9),  # shoulders to elbows
                (8, 10), (9, 11), # elbows to wrists
                # Head
                (12, 14), (13, 14), # eyes to nose
                (6, 14), (7, 14),    # shoulders to nose (approximate neck)


                (0, 15), (15,16),
                (1, 17), (17,18)   # right ankle to calc and then toe

            ]

            for start_idx, end_idx in connections:
                if (start_idx < len(keypoints_2d) and end_idx < len(keypoints_2d) and
                    keypoints_2d[start_idx] is not None and keypoints_2d[end_idx] is not None):
                    cv2.line(overlay, keypoints_2d[start_idx], keypoints_2d[end_idx], color, 2, cv2.LINE_AA)

            # Draw keypoints on the overlay
            for kp in keypoints_2d:
                if kp is not None:
                    cv2.circle(overlay, kp, 3, color, -1, cv2.LINE_AA)

            # Add person ID label on the overlay
            if keypoints_2d[14] is not None:  # Use nose position for label
                label_pos = (keypoints_2d[14][0], keypoints_2d[14][1] - 20)
                cv2.putText(overlay, f'Person {person_id}', label_pos,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)

        # Blend overlay with original frame once (50% opacity)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)
        
        # Track frames with poses
        if people_in_frame > 0:
            frames_with_poses += 1
        
        # Add frame info
        info_text = f'Frame: {frame_idx}, Sync: {current_sync_index}, People: {people_in_frame}'
        cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        out.write(frame)
        
        if frame_idx % 200 == 0:
            print(f"Processed {frame_idx}/{total_frames} frames, {frames_with_poses} frames with poses so far")
    
    # Cleanup
    cap.release()
    out.release()
    
    print(f"\n=== VIDEO PROJECTION SUMMARY ===")
    print(f"Total frames processed: {frames_processed}")
    print(f"Frames with pose data: {frames_with_poses}")
    print(f"Percentage with poses: {100*frames_with_poses/frames_processed:.1f}%")
    print(f"Output video saved: {output_path}")
    return output_path

def report_clothing(video_file_path, config_toml_path, csv_3d_coordinates_path, 
                    movement_threshold=10.0, sampling_radius=3, max_samples=5):
    """
    Extract clothing colors from pose data by projecting 3D coordinates to 2D pixel coordinates.
    
    Args:
        video_file_path (str): Path to the video file
        config_toml_path (str): Path to config.toml calibration file
        csv_3d_coordinates_path (str): Path to CSV file with 3D pose coordinates
        movement_threshold (float): Minimum pixel movement to consider stable frames
        sampling_radius (int): Radius in pixels around keypoint to sample colors
        max_samples (int): Maximum number of samples to collect
    
    Returns:
        dict: Dictionary with 'shirt' and 'pants' color data in HSL format
    """
    import pandas as pd
    import cv2
    import numpy as np
    import colorsys
    import os
    
    # Load calibration data
    camera_params = parse_calibration_mwc(config_toml_path)
    if not camera_params:
        raise ValueError(f"Failed to load calibration from {config_toml_path}")
    
    # Use first camera for projection (assuming single camera analysis)
    target_cam_params = camera_params[0]
    projection_matrices = calculate_projection_matrices([target_cam_params])
    P = projection_matrices[0]
    
    # Load 3D coordinate data
    df = pd.read_csv(csv_3d_coordinates_path)
    
    # Find elbow and knee keypoints columns
    elbow_cols = [col for col in df.columns if 'L_Elbow' in col or 'R_Elbow' in col]
    knee_cols = [col for col in df.columns if 'L_Knee' in col or 'R_Knee' in col]
    
    if not elbow_cols or not knee_cols:
        raise ValueError("Could not find elbow or knee keypoint columns in CSV")
    
    # Extract X,Y,Z coordinates for elbows and knees
    elbow_coords = []
    knee_coords = []
    
    for _, row in df.iterrows():
        # Get elbow coordinates (average of left and right)
        elbow_x_cols = [col for col in elbow_cols if col.endswith('_X')]
        elbow_y_cols = [col for col in elbow_cols if col.endswith('_Y')]
        elbow_z_cols = [col for col in elbow_cols if col.endswith('_Z')]
        
        if elbow_x_cols and elbow_y_cols and elbow_z_cols:
            elbow_x = np.nanmean([row[col] for col in elbow_x_cols])
            elbow_y = np.nanmean([row[col] for col in elbow_y_cols])
            elbow_z = np.nanmean([row[col] for col in elbow_z_cols])
            elbow_coords.append([elbow_x, elbow_y, elbow_z])
        else:
            elbow_coords.append([np.nan, np.nan, np.nan])
        
        # Get knee coordinates (average of left and right)
        knee_x_cols = [col for col in knee_cols if col.endswith('_X')]
        knee_y_cols = [col for col in knee_cols if col.endswith('_Y')]
        knee_z_cols = [col for col in knee_cols if col.endswith('_Z')]
        
        if knee_x_cols and knee_y_cols and knee_z_cols:
            knee_x = np.nanmean([row[col] for col in knee_x_cols])
            knee_y = np.nanmean([row[col] for col in knee_y_cols])
            knee_z = np.nanmean([row[col] for col in knee_z_cols])
            knee_coords.append([knee_x, knee_y, knee_z])
        else:
            knee_coords.append([np.nan, np.nan, np.nan])
    
    # Convert to numpy arrays
    elbow_coords = np.array(elbow_coords)
    knee_coords = np.array(knee_coords)
    
    # Project 3D coordinates to 2D pixel coordinates
    elbow_pixels = []
    knee_pixels = []
    
    for i in range(len(elbow_coords)):
        # Project elbow
        if not np.isnan(elbow_coords[i]).any():
            elbow_2d = project_3d_to_2d(elbow_coords[i], P)
            elbow_pixels.append(elbow_2d)
        else:
            elbow_pixels.append(None)
        
        # Project knee
        if not np.isnan(knee_coords[i]).any():
            knee_2d = project_3d_to_2d(knee_coords[i], P)
            knee_pixels.append(knee_2d)
        else:
            knee_pixels.append(None)
    
    # Find frames with small movement for stable sampling
    def find_stable_frames(pixel_coords, threshold=movement_threshold):
        stable_frames = []
        for i in range(1, len(pixel_coords)):
            if pixel_coords[i] is not None and pixel_coords[i-1] is not None:
                diff = np.linalg.norm(np.array(pixel_coords[i]) - np.array(pixel_coords[i-1]))
                if diff < threshold:
                    stable_frames.append(i)
        return stable_frames
    
    elbow_stable_frames = find_stable_frames(elbow_pixels)
    knee_stable_frames = find_stable_frames(knee_pixels)
    
    # Select frames for color sampling (up to max_samples)
    elbow_sample_frames = elbow_stable_frames[:max_samples] if len(elbow_stable_frames) >= 1 else elbow_stable_frames
    knee_sample_frames = knee_stable_frames[:max_samples] if len(knee_stable_frames) >= 1 else knee_stable_frames
    
    # Ensure we have at least 1 sample
    if not elbow_sample_frames and len(elbow_pixels) > 0:
        valid_elbow_frames = [i for i, p in enumerate(elbow_pixels) if p is not None]
        if valid_elbow_frames:
            elbow_sample_frames = [valid_elbow_frames[0]]
    
    if not knee_sample_frames and len(knee_pixels) > 0:
        valid_knee_frames = [i for i, p in enumerate(knee_pixels) if p is not None]
        if valid_knee_frames:
            knee_sample_frames = [valid_knee_frames[0]]
    
    # Extract colors from video
    def extract_colors_from_frames(video_path, pixel_coords, frame_indices, radius=sampling_radius):
        cap = cv2.VideoCapture(video_path)
        colors = []
        
        for frame_idx in frame_indices:
            if frame_idx >= len(pixel_coords) or pixel_coords[frame_idx] is None:
                continue
            
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get pixel coordinates
            px, py = pixel_coords[frame_idx]
            px, py = int(px), int(py)
            
            # Extract color patch around the keypoint
            h, w = frame_rgb.shape[:2]
            y_min = max(0, py - radius)
            y_max = min(h, py + radius + 1)
            x_min = max(0, px - radius)
            x_max = min(w, px + radius + 1)
            
            color_patch = frame_rgb[y_min:y_max, x_min:x_max]
            
            # Calculate median color
            if color_patch.size > 0:
                median_color = np.median(color_patch.reshape(-1, 3), axis=0)
                colors.append(median_color)
        
        cap.release()
        return colors
    
    # Extract shirt colors (from elbows)
    shirt_colors = extract_colors_from_frames(video_file_path, elbow_pixels, elbow_sample_frames)
    
    # Extract pants colors (from knees)
    pants_colors = extract_colors_from_frames(video_file_path, knee_pixels, knee_sample_frames)
    
    # Convert RGB to HSL
    def rgb_to_hsl(rgb_colors):
        hsl_colors = []
        for rgb in rgb_colors:
            r, g, b = rgb / 255.0
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            # Convert to standard HSL format (H: 0-360, S: 0-100, L: 0-100)
            h = h * 360
            s = s * 100
            l = l * 100
            hsl_colors.append([h, s, l])
        return hsl_colors
    
    shirt_hsl = rgb_to_hsl(shirt_colors) if shirt_colors else []
    pants_hsl = rgb_to_hsl(pants_colors) if pants_colors else []
    
    # Calculate median HSL values
    def median_hsl(hsl_colors):
        if not hsl_colors:
            return [0, 0, 0]
        hsl_array = np.array(hsl_colors)
        return np.median(hsl_array, axis=0).tolist()
    
    shirt_median_hsl = median_hsl(shirt_hsl)
    pants_median_hsl = median_hsl(pants_hsl)
    
    # Save results to CSV
    base_name = os.path.splitext(csv_3d_coordinates_path)[0]
    output_path = f"{base_name}_clothes.csv"
    
    # Create DataFrame with shirt on top row, pants on bottom row
    clothes_df = pd.DataFrame({
        'item': ['shirt', 'pants'],
        'H': [shirt_median_hsl[0], pants_median_hsl[0]],
        'S': [shirt_median_hsl[1], pants_median_hsl[1]],
        'L': [shirt_median_hsl[2], pants_median_hsl[2]]
    })
    
    clothes_df.to_csv(output_path, index=False)
    
    # Create debugging visualization using extracted function
    from .processed_visualizations import create_clothing_debug_plot
    
    debug_plot_path = create_clothing_debug_plot(
        video_file_path=video_file_path,
        elbow_pixels=elbow_pixels,
        knee_pixels=knee_pixels,
        elbow_sample_frames=elbow_sample_frames,
        knee_sample_frames=knee_sample_frames,
        shirt_median_hsl=shirt_median_hsl,
        pants_median_hsl=pants_median_hsl,
        movement_threshold=movement_threshold,
        sampling_radius=sampling_radius,
        max_samples=max_samples,
        output_base_name=base_name
    )
    
    return {
        'shirt': shirt_median_hsl,
        'pants': pants_median_hsl,
        'output_file': output_path,
        'debug_plot': debug_plot_path
    }