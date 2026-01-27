# code to read in MWC or FMC files and process keyframes using synpose
import configparser
import numpy as np
import ast # For safely evaluating literal expressions (like lists) from strings
import os # To check file existence
import io # For StringIO to simulate file reading

# sp-> superclass pose parsing configuration files. 
# parse calibration data from MWC or FMC files.
# parse_camera_section_data

# --- Re-using the helper function from the previous example ---
def parse_camera_section_data(config, section_name):
    """
    Helper function to parse data from a single camera section
    using configparser. Handles common keys and conversions.
    """
    cam_data = {}
    try:
        # --- Intrinsics ---
        matrix_str = config.get(section_name, 'matrix')
        cam_data['matrix'] = np.array(ast.literal_eval(matrix_str), dtype=float)
        if cam_data['matrix'].shape != (3, 3):
             raise ValueError(f"Matrix shape is not (3, 3) for {section_name}")

        dist_str = config.get(section_name, 'distortions')
        cam_data['distortions'] = np.array(ast.literal_eval(dist_str), dtype=float)

        size_str = config.get(section_name, 'size')
        cam_data['size'] = np.array(ast.literal_eval(size_str), dtype=int)
        if cam_data['size'].shape != (2,):
             raise ValueError(f"Size shape is not (2,) for {section_name}")

        # --- Extrinsics ---
        rot_str = config.get(section_name, 'rotation')
        rotation_val = ast.literal_eval(rot_str)
        if isinstance(rotation_val[0], list):
             print(f"Warning: Rotation format in {section_name} looks like a matrix, expected Rodrigues vector.")
             cam_data['rotation'] = np.array(rotation_val, dtype=float) # Store as is for now
             # Decide later if conversion is needed
             if cam_data['rotation'].shape != (3,3): # Basic check if it looked like a matrix
                 print(f"Warning: Rotation matrix in {section_name} is not 3x3.")

        else: # Assume it's a Rodrigues vector
            cam_data['rotation'] = np.array(rotation_val, dtype=float)
            if cam_data['rotation'].shape != (3,):
                 raise ValueError(f"Rotation vector shape is not (3,) for {section_name}")

        trans_str = config.get(section_name, 'translation')
        translation_val = ast.literal_eval(trans_str)
        if isinstance(translation_val[0], list) and len(translation_val) == 3:
             cam_data['translation'] = np.array(translation_val, dtype=float).flatten()
        else:
            cam_data['translation'] = np.array(translation_val, dtype=float)

        if cam_data['translation'].shape != (3,):
             raise ValueError(f"Translation vector shape is not (3,) for {section_name}")

        # --- Optional/Metadata ---
        cam_data['id'] = section_name # Use the section name as identifier (e.g., 'cam_0')
        if config.has_option(section_name, 'port'):
            cam_data['port'] = config.getint(section_name, 'port')
        if config.has_option(section_name, 'error'):
            cam_data['error'] = config.getfloat(section_name, 'error')
        if config.has_option(section_name, 'name'): # Check if 'name' exists (like in FMC)
             cam_data['name'] = config.get(section_name, 'name').strip('"')
        else:
             cam_data['name'] = section_name # Fallback to section name

        return cam_data

    except (configparser.NoOptionError, configparser.NoSectionError) as e:
        print(f"Error parsing section {section_name}: Missing expected key - {e}")
        return None
    except (SyntaxError, ValueError) as e:
        print(f"Error parsing data in section {section_name}: Invalid format - {e}")
        return None
    except Exception as e:
        print(f"An unexpected error occurred parsing section {section_name}: {e}")
        return None


def parse_calibration_mwc(file_path):
    """
    Parses camera calibration data from an MWC-style config.txt file,
    handling lines before the first section header.

    Args:
        file_path (str): The path to the config.txt file.

    Returns:
        list: A list of dictionaries, where each dictionary contains
              the parsed parameters for a single camera. Returns an
              empty list if the file cannot be read or no valid camera
              sections are found.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []

    valid_lines = []
    found_first_section = False
    try:
        with open(file_path, 'r') as f:
            for line in f:
                stripped_line = line.strip()
                # Start collecting lines once we hit the first section header
                if stripped_line.startswith('['):
                    found_first_section = True
                # Ignore blank lines and comments if desired (optional)
                # if stripped_line and not stripped_line.startswith('#'):
                if found_first_section:
                     # Keep the original line ending for configparser
                    valid_lines.append(line)

    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

    if not valid_lines:
        print(f"Error: No content found starting from the first section header in {file_path}")
        return []

    # Join the valid lines back into a single string
    config_string = "".join(valid_lines)

    # Use StringIO to treat the string as a file for configparser
    config_io = io.StringIO(config_string)

    config = configparser.ConfigParser()
    try:
        config.read_file(config_io)
    except configparser.ParsingError as e:
         print(f"Error parsing the INI structure from file {file_path}: {e}")
         return []
    except Exception as e:
        print(f"An unexpected error occurred during config parsing for {file_path}: {e}")
        return []


    camera_params = []
    # Identify camera sections (e.g., [cam_0], [cam_1], ...)
    camera_sections = [s for s in config.sections() if s.startswith('cam_')]

    print(f"Found potential camera sections in MWC file: {camera_sections}")

    for section_name in camera_sections:
        print(f"Parsing section: {section_name}")
        cam_data = parse_camera_section_data(config, section_name)
        if cam_data:
            # Check MWC units - translation seems small (meters?)
            # print(f"Note: MWC translation for {section_name}: {cam_data['translation']}. Units might be meters.")
            camera_params.append(cam_data)
        else:
            print(f"Skipping section {section_name} due to parsing errors.")

    print(f"Successfully parsed {len(camera_params)} cameras from MWC file.")
    return camera_params



def parse_calibration_fmc(file_path):
    """
    Parses camera calibration data from an FMC-style
    recording...camera_calibration.txt file.

    Args:
        file_path (str): The path to the camera_calibration.txt file.

    Returns:
        list: A list of dictionaries, where each dictionary contains
              the parsed parameters ('name', 'size', 'matrix', 'distortions',
              'rotation', 'translation') for a single camera. Returns an
              empty list if the file cannot be read or no valid camera
              sections are found.
    """
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return []

    config = configparser.ConfigParser()
    # Read file content
    try:
         with open(file_path, 'r') as f:
             config.read_string(f.read())
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

    camera_params = []
    # Identify camera sections (e.g., [cam_0], [cam_1], ...)
    camera_sections = [s for s in config.sections() if s.startswith('cam_')]

    print(f"Found potential camera sections in FMC file: {camera_sections}")

    for section_name in camera_sections:
        print(f"Parsing section: {section_name}")
        # Use the same helper function, but adjust for specific FMC keys if needed
        cam_data = {}
        try:
            # Use the helper for common fields
            common_data = parse_camera_section_data(config, section_name)
            if not common_data:
                raise ValueError("Core parsing failed in helper function.") # Trigger skip

            cam_data.update(common_data)

            # FMC specific: 'name' key inside the section
            if config.has_option(section_name, 'name'):
                 cam_data['name'] = config.get(section_name, 'name').strip('"') # Override section name if specific name exists
            else:
                 cam_data['name'] = section_name # Fallback to section name

            # Check units if available (e.g., from metadata)
            # The large translation values suggest mm, consistent with charuco_square_size=157.0
            # (which is likely mm)
            camera_params.append(cam_data)

        except Exception as e: # Catch errors specific to this loop or re-raised from helper
            print(f"Skipping section {section_name} due to parsing errors: {e}")
            continue # Skip to the next section

    # --- Optionally parse FMC metadata ---
    metadata = {}
    if config.has_section('metadata'):
         try:
             metadata['metadata'] = dict(config.items('metadata'))
             print("Parsed FMC metadata.")
             # Potentially parse specific metadata fields like charuco_square_size
             if config.has_option('metadata','charuco_square_size'):
                  metadata['charuco_square_size'] = config.getfloat('metadata','charuco_square_size')
                  print(f"Charuco square size (likely unit for translation): {metadata['charuco_square_size']}")

         except Exception as e:
            print(f"Warning: Could not parse [metadata] section fully: {e}")

    # return camera_params, metadata
    print(f"Successfully parsed {len(camera_params)} cameras from FMC file.")
    return camera_params

import cv2
import numpy as np
import re # For parsing

def calculate_projection_matrices(camera_params):
    """
    Calculate the projection matrices for each camera based on intrinsic
    and extrinsic parameters.

    Args:
        camera_params (list): List of dictionaries containing camera parameters.

    Returns:
        list: List of projection matrices for each camera.
    """
    projection_matrices = []
    for params in camera_params:
        K = params['matrix']
        rvec = params['rotation']
        tvec = params['translation'].reshape(3, 1) # Reshape for matrix math

        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)

        # Create extrinsic matrix [R | t]
        extrinsic_matrix = np.hstack((R, tvec))

        # Calculate projection matrix P = K @ [R | t]
        P = K @ extrinsic_matrix
        projection_matrices.append(P)
        # print("\nProjection Matrices:")
        # for i, P in enumerate(projection_matrices):
        #     print(f"Camera {i}:\n{P}\n")

    return projection_matrices

# --- 3 & 4 & 5: Undistort and Triangulate Keypoints ---

# --- Add/Modify this within sp_parse.py ---
import cv2
import numpy as np

# ... (keep existing parse functions and calculate_projection_matrices) ...

def triangulate_keypoints(
    person_kp_dict,    # Dict: {port: keypoints_array (17, 2 or 3)}
    port_to_cam_index, # Dict: mapping port number to index in camera_params/proj_matrices
    camera_params,     # List of camera parameter dicts
    projection_matrices, # List of 3x4 projection matrices
    confidence_threshold=0.1
    ):
    """
    Triangulates 3D points for a single person from 2D keypoints detected
    in multiple cameras. Handles missing views for the person.

    Args:
        person_kp_dict (dict): Dictionary mapping port number to the detected
                               2D keypoints array (e.g., shape (17, 2) or (17,3))
                               for this specific person.
        port_to_cam_index (dict): Maps port numbers (keys in person_kp_dict) to
                                  the corresponding index in camera_params and
                                  projection_matrices lists.
        camera_params (list): List of dictionaries containing camera parameters.
        projection_matrices (list): List of 3x4 projection matrices.

    Returns:
        A list of NumPy arrays (or None), where each array is the triangulated
        3D point [x, y, z] for a keypoint. Returns None for keypoints that
        could not be triangulated (e.g., seen in < 2 views).
        Returns an empty list if the person was seen in < 2 cameras overall.
    """
    available_ports = list(person_kp_dict.keys())
    num_available_views = len(available_ports)

    if num_available_views < 2:
        # print(f"Warning: Person seen in only {num_available_views} views. Triangulation requires at least 2. Skipping person.")
        return [] # Cannot triangulate

    # Determine the number of keypoints from the first available view
    first_port = available_ports[0]
    num_keypoints = len(person_kp_dict[first_port]) # Assumes all views have same number of keypoints
    points_3d = [None] * num_keypoints # Initialize list for 3D points

    # Get the indices corresponding to the available ports
    available_cam_indices = [port_to_cam_index[p] for p in available_ports if p in port_to_cam_index]
    if len(available_cam_indices) < 2:
        print("Warning: Could not map available ports to camera indices. Skipping person.")
        return []

    # Triangulate each keypoint individually
    for kp_idx in range(num_keypoints):
        points_2d_undistorted_this_kp = {} # Store undistorted points for this keypoint: {cam_idx: point}
        valid_views_for_kp = 0

        for port in available_ports:
            cam_idx = port_to_cam_index.get(port)
            if cam_idx is None:
                # print(f"Warning: Port {port} not found in camera mapping. Skipping view for KP {kp_idx}.")
                continue

            kp_data = person_kp_dict[port][kp_idx]

            # Check if keypoint data is valid (handle different formats)
            if isinstance(kp_data, (list, np.ndarray)) and len(kp_data) >= 2:
                 point_2d_raw = np.array(kp_data[:2], dtype=np.float32).reshape(1, 1, 2)
                 # Optional: Check confidence if available (index 2)
                 confidence = kp_data[2] if len(kp_data) > 2 else 1.0
                 if np.isnan(point_2d_raw).any() or confidence < confidence_threshold:
                     # print(f"Debug: Invalid/low confidence KP {kp_idx} in port {port}. Skipping view.")
                     continue # Skip this view for this specific keypoint
            else:
                 # print(f"Debug: Invalid keypoint data format for KP {kp_idx} in port {port}. Skipping view.")
                 continue # Skip this view

            # Undistort
            K = camera_params[cam_idx]['matrix']
            dist = camera_params[cam_idx]['distortions']
            point_2d_undistorted = cv2.undistortPoints(point_2d_raw, K, dist, P=K)
            points_2d_undistorted_this_kp[cam_idx] = point_2d_undistorted.reshape(2, 1)
            valid_views_for_kp += 1

        # --- Triangulation for this keypoint ---
        if valid_views_for_kp < 2:
            # print(f"Warning: Keypoint {kp_idx} visible in only {valid_views_for_kp} valid views. Cannot triangulate.")
            continue # Keep points_3d[kp_idx] as None

        # Use SVD for multi-view triangulation
        num_valid_cams = len(points_2d_undistorted_this_kp)
        A = np.zeros((2 * num_valid_cams, 4))
        i = 0
        valid_cam_indices_kp = list(points_2d_undistorted_this_kp.keys())

        for cam_idx in valid_cam_indices_kp:
            P = projection_matrices[cam_idx]
            x, y = points_2d_undistorted_this_kp[cam_idx].flatten()
            A[2 * i    ] = x * P[2, :] - P[0, :]
            A[2 * i + 1] = y * P[2, :] - P[1, :]
            i += 1

        # Solve AX=0 using SVD
        try:
            u, s, vh = np.linalg.svd(A)
            point_4d_hom_svd = vh[-1, :] # Solution is the last row of Vh
            # Convert from homogeneous to Cartesian coordinates
            point_3d = point_4d_hom_svd[:3] / point_4d_hom_svd[3]
            points_3d[kp_idx] = point_3d # Store the result
        except np.linalg.LinAlgError:
            print(f"Warning: SVD computation failed for keypoint {kp_idx}. Cannot triangulate.")
        except ZeroDivisionError:
             print(f"Warning: Homogeneous coordinate W is zero for keypoint {kp_idx}. Cannot triangulate.")


    return points_3d

if __name__ == "__main__":
    # set default filenames to be in test directory
    test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(test_dir, 'test','calibration')
    mwc_file = os.path.join(test_dir, 'test_mwc_config.toml')
    fmc_file = os.path.join(test_dir, 'test_fmc_recording_calibration.toml')
    # parse the MWC and FMC files
    camera_params = parse_calibration_mwc(mwc_file)
    fmc_camera_params = parse_calibration_fmc(fmc_file)
    # print the parsed camera parameters
    print("\nParsed Camera Parameters from MWC:")
    for i, params in enumerate(camera_params):
        print(f"Camera {i}: {params}")
    print("\nParsed Camera Parameters from FMC:")
    for i, params in enumerate(fmc_camera_params):
        print(f"Camera {i}: {params}")

    # Load camera_params from snapshot_*.pkl files and compare with current ones
    import pickle

    mwc_snapshot_path = os.path.join(test_dir, 'snapshot_camera_params_mwc.pkl')
    fmc_snapshot_path = os.path.join(test_dir, 'snapshot_camera_params_fmc.pkl')

    try:
        with open(mwc_snapshot_path, 'rb') as f:
            camera_params_snapshot = pickle.load(f)
        with open(fmc_snapshot_path, 'rb') as f:
            fmc_camera_params_snapshot = pickle.load(f)
    except FileNotFoundError as e:
        print(f"Snapshot file not found: {e}")
        camera_params_snapshot = None
        fmc_camera_params_snapshot = None

    def compare_camera_params(a, b):
        if type(a) != type(b):
            return False
        if isinstance(a, dict):
            if set(a.keys()) != set(b.keys()):
                return False
            for k in a:
                if not compare_camera_params(a[k], b[k]):
                    return False
            return True
        if isinstance(a, list):
            if len(a) != len(b):
                return False
            for x, y in zip(a, b):
                if not compare_camera_params(x, y):
                    return False
            return True
        if isinstance(a, np.ndarray):
            return np.allclose(a, b)
        return a == b

    if camera_params_snapshot is not None:
        mwc_equal = compare_camera_params(camera_params, camera_params_snapshot)
        print(f"MWC camera_params equal to snapshot: {mwc_equal}")
    else:
        print("MWC snapshot not loaded; skipping comparison.")

    if fmc_camera_params_snapshot is not None:
        fmc_equal = compare_camera_params(fmc_camera_params, fmc_camera_params_snapshot)
        print(f"FMC camera_params equal to snapshot: {fmc_equal}")
    else:
        print("FMC snapshot not loaded; skipping comparison.")

    # --- Add assert checks at the end ---
    if camera_params_snapshot is not None:
        assert mwc_equal, "Parsed MWC camera_params do not match the snapshot!"
    if fmc_camera_params_snapshot is not None:
        assert fmc_equal, "Parsed FMC camera_params do not match the snapshot!"
        
    # Example: Calculating projection matrices for the FMC data
    if fmc_camera_params:
        projection_matrices_fmc = []
        for params in fmc_camera_params:
            K = params['matrix']
            rvec = params['rotation']
            tvec = params['translation'].reshape(3, 1)
            R, _ = cv2.Rodrigues(rvec)
            extrinsic_matrix = np.hstack((R, tvec))
            P = K @ extrinsic_matrix
            projection_matrices_fmc.append(P)
        print(f"\nCalculated {len(projection_matrices_fmc)} projection matrices from FMC data.")
        # print(projection_matrices_fmc[0]) # Print first one as example
    else:
        print("\nNo FMC camera parameters were parsed to calculate projection matrices.")
