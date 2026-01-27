# indev: untested at the moment. 
# this includes per-session tracking ideas. 
# developing, including clothing. 
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