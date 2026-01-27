# test_estimate_poses.py: tests pose_detector.estimate_poses().
# this depends on load_models() and detect_persons().
import torch
import numpy as np
import cv2 # Needed for visualization if using cv2 directly, or by supervision
from PIL import Image
import os
import argparse
import time
import supervision as sv # For visualization consistency with original test/pose_detector

# Ensure the src directory is in the Python path for imports
import sys
import os
path_to_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path_to_project_root)


# Import the refactored functions from pose_detector.py
from posetrack import (
    load_models,
    detect_persons,
    estimate_poses,
    LOCAL_SP_DIR, 
    LOCAL_DET_DIR)

def run_verification(image_path, output_dir=os.path.join(path_to_project_root,"output/output_verify"), device_name="auto",
                     detector_dir=LOCAL_DET_DIR, pose_model_dir=LOCAL_SP_DIR,
                     person_conf=0.3, show_result=True, save_result=True):
    """
    Run the pose detection pipeline on a single image and compare with snapshot of expected output.
    Args:
        image_path (str): Path to the input image.
        output_dir (str): Directory to save the annotated output image.
        device_name (str): Device to use ('auto', 'cpu', 'mps', 'cuda').
        detector_dir (str, optional): Path to local detector model. Defaults to None (Hugging Face).
        pose_model_dir (str): Path to the local pose estimation model.
        person_conf (float): Confidence threshold for person detection.
        show_result (bool): Whether to display the annotated image.
        save_result (bool): Whether to save the annotated image.
    """
    # there are 5 parts:
    # 1. Setup device for cpu, mps or cuda. 
    # 2. Load models
    # 3. Load image
    # 4. Detect persons
    # 5. Estimate poses
    print("Starting verification script...")

    # --- 1. Setup Device ---
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
            print("Warning: MPS explicitly requested but unavailable/not built. Falling back to CPU.")
            device = "cpu"
        elif device == "cuda" and not torch.cuda.is_available():
            print("Warning: CUDA explicitly requested but unavailable. Falling back to CPU.")
            device = "cpu"
    print(f"Using device: {device}")

    # --- 2. Load Models ---
    print("Loading models using pose_detector.load_models...")
    try:
        person_processor, person_model, pose_processor, pose_model = load_models(
            detect_path=detector_dir,
            pose_model_path=pose_model_dir,
            device=device
        )
    except Exception as e:
        print(f"Error loading models: {e}")
        return

    # --- 3. Load Image ---
    print(f"Loading image from: {image_path}")
    try:
        image_pil = Image.open(image_path).convert("RGB")
        print(f"Image loaded successfully. Size: {image_pil.size}")
    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return
    except Exception as e:
        print(f"Error loading image: {e}")
        return

    # --- 4. Detect Persons ---
    print("Detecting persons using pose_detector.detect_persons...")
    try:
        # Use the detect_persons function
        start_time = time.perf_counter()
        person_boxes_voc, person_boxes_coco, person_scores = detect_persons(
            image_pil, person_processor, person_model, device, confidence_threshold=person_conf
        )
        end_time = time.perf_counter()
        print(f"Human Detection took {end_time-start_time:.2f} s.")
        print(f"Detected {len(person_boxes_voc)} persons.")
        if len(person_boxes_voc) > 0:
            print(f"  Example VOC box (xyxy): {person_boxes_voc[0]}")
            print(f"  Example COCO box (xywh): {person_boxes_coco[0]}")
            print(f"  Example score: {person_scores[0]}")

    except Exception as e:
        print(f"Error during person detection: {e}")
        # It might be useful to continue to see if pose estimation crashes too
        person_boxes_voc = np.empty((0, 4), dtype=np.float32)
        person_boxes_coco = np.empty((0, 4), dtype=np.float32)
        person_scores = np.empty((0,), dtype=np.float32)


    os.makedirs(output_dir, exist_ok=True)
    # check if output_dir is not empty
    if not os.listdir(output_dir):
        output_filename = os.path.join(output_dir, f"verify_{os.path.basename(image_path)}")

    # --- 5. Estimate Poses ---
    all_keypoints = []
    all_keypoint_scores = []
    if len(person_boxes_coco) > 0: # Only proceed if persons were detected
        print("Estimating poses using pose_detector.estimate_poses...")
        try:
            # Make sure to pass the COCO boxes
            start_time = time.perf_counter()
            all_keypoints, all_keypoint_scores = estimate_poses(
                image_pil, person_boxes_coco, pose_processor, pose_model, device,
                debug_plot=show_result, 
                debug_save_prefix = output_dir)
            end_time = time.perf_counter()
            print(f"Keypoint pose inference took {end_time-start_time:.2f} seconds.")
            print(f"Estimated poses for {len(all_keypoints)} persons.") 
            #save all_keypoints using pickle, to be compared with during testing with assertions
            import pickle
            with open(os.path.join(output_dir, 'all_keypoints.pkl'), 'wb') as f:
                pickle.dump(all_keypoints, f)
            
        except Exception as e:
            print(f"Error during pose estimation: {e}")
            all_keypoints = [] # Ensure lists are empty on error
            all_keypoint_scores = []

    # compare all_keypoints with the expected output
    # parse the image filename as the filenam and the directory
    image_filename = os.path.basename(image_path).split('.')[0]  # Get filename without extension
    image_dir = os.path.dirname(image_path)
    all_keypoints_reference = os.path.join(image_dir, f'keypoints_{image_filename}.pkl')
    # load the pkl file
    all_keypoints_ref = pickle.load(open(all_keypoints_reference, 'rb')) if os.path.exists(all_keypoints_reference) else None

    # compare the all_keypoints_ref with the all_keypoints with assert
    try:
        assert len(all_keypoints) == len(all_keypoints_ref), "Number of detected poses does not match reference."
        for i in range(len(all_keypoints)):
            assert np.allclose(all_keypoints[i], all_keypoints_ref[i], atol=1e-5), f"Pose {i} does not match reference."
        print("TEST KEYPOINTS MATCH: PASSED. All expected keypoints within 1e-5 of the 2025-05-27 run.")
    except FileNotFoundError:
        print(f"Reference keypoints file not found: {all_keypoints_reference}")
    except AssertionError as e:
        print(f"Assertion error: {e}")


# --- Command Line Argument Parsing ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verification script for refactored pose detection.")

    # Use paths relative to the project structure assumed
    # Adjust these defaults if your structure is different
    parser.add_argument("--image", default="tests/image/1.png", help="Path to the input image.")
    parser.add_argument("--output_dir", default=os.path.join(path_to_project_root,"output/output_verify"), help="Directory to save the output annotated image.")
    parser.add_argument("--pose_model", default=LOCAL_SP_DIR, help="Path to the local SynthPose model directory.")
    parser.add_argument("--detector_model", default=LOCAL_DET_DIR, help="Path to the local RT-DETR model directory (optional, uses HF if None).")
    parser.add_argument("--device", default="mps", choices=['auto', 'cpu', 'mps', 'cuda'], help="Computation device.")
    parser.add_argument("--person_conf", type=float, default=0.05, help="Confidence threshold for person detection.")
    parser.add_argument("--hide", action="store_true", help="Do not display the annotated image.")
    parser.add_argument("--no_save", action="store_true", help="Do not save the annotated image.")
    parser.add_argument("--show_result", type=bool, default=False, help="show the resulting figures (which will affect performance estimates).")

    args = parser.parse_args()

    print("Starting first run_verification...")
    start_time = time.perf_counter()
    run_verification(
        image_path=args.image,
        output_dir=args.output_dir,
        device_name=args.device,
        detector_dir=args.detector_model,
        pose_model_dir=args.pose_model,
        person_conf=args.person_conf,
        show_result=False,
        save_result=not args.no_save
    )
    end_time = time.perf_counter()
    print(f"First run_verification completed in {end_time - start_time:.2f} seconds")
    
    print("Starting second run_verification...")
    start_time = time.perf_counter()
    run_verification(
        image_path="tests/image/many.png",
        output_dir=args.output_dir,
        device_name=args.device,
        detector_dir=args.detector_model,
        pose_model_dir=args.pose_model,
        person_conf=args.person_conf,
        show_result=True,
        save_result=not args.no_save
    )
    end_time = time.perf_counter()
    print(f"Second run_verification completed in {end_time - start_time:.2f} seconds")