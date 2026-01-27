# for now, we have left test suites within the actual cs_parse.py file.

# run cs_parse as main to test the parsing
import os
import sys
path_to_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path_to_project_root)
import cv2
import numpy as np
import posetrack as pt

# call cs_parse.py __main__ function
# to run the test suite

if __name__ == "__main__":
    # set default filenames to be in test directory
    test_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    test_dir = os.path.join(test_dir, 'tests','calibration')
    mwc_file = os.path.join(test_dir, 'test_mwc_config.toml')
    fmc_file = os.path.join(test_dir, 'test_fmc_recording_calibration.toml')
    # parse the MWC and FMC files
    camera_params = pt.parse_calibration_mwc(mwc_file)
    fmc_camera_params = pt.parse_calibration_fmc(fmc_file)
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
        # print(projection_matrices_fmc[0]) # Print first one as example
    else:
        print("\nNo FMC camera parameters were parsed to calculate projection matrices.")
