
import torch
import numpy as np
import cv2
from PIL import Image
from transformers import (
    AutoProcessor,
    RTDetrForObjectDetection,
    VitPoseForPoseEstimation, 
    AutoModelForObjectDetection,
    AutoImageProcessor
)
import time
from tqdm import tqdm
import supervision as sv # Added for visualization types if needed later
import pickle # for saving/loading models / results
      
LOCAL_SP_DIR    = "/Users/jeremy/Git/KeypointInference/models/synthpose/checkpoints"
LOCAL_DET_DIR   = "/Users/jeremy/Git/KeypointInference/models/yolov10s/"

# Define the markers directly here. we could rewrite these names, since they become the keys in the dictionary.
class SynthPoseMarkers:
    markers = {
        0: "Nose", 1: "L_Eye", 2: "R_Eye", 3: "L_Ear", 4: "R_Ear",
        5: "L_Shoulder", 6: "R_Shoulder", 7: "L_Elbow", 8: "R_Elbow",
        9: "L_Wrist", 10: "R_Wrist", 11: "L_Hip", 12: "R_Hip",
        13: "L_Knee", 14: "R_Knee", 15: "L_Ankle", 16: "R_Ankle",
        17: "sternum", 18: "rshoulder", 19: "lshoulder", 20: "r_lelbow",
        21: "l_lelbow", 22: "r_melbow", 23: "l_melbow", 24: "r_lwrist",
        25: "l_lwrist", 26: "r_mwrist", 27: "l_mwrist", 28: "r_ASIS",
        29: "l_ASIS", 30: "r_PSIS", 31: "l_PSIS", 32: "r_knee",
        33: "l_knee", 34: "r_mknee", 35: "l_mknee", 36: "r_ankle",
        37: "l_ankle", 38: "r_mankle", 39: "l_mankle", 40: "r_5meta",
        41: "l_5meta", 42: "r_toe", 43: "l_toe", 44: "r_big_toe",
        45: "l_big_toe", 46: "l_calc", 47: "r_calc", 48: "C7",
        49: "L2", 50: "T11", 51: "T6"
    }
    num_markers = len(markers) # Should be 52

def load_models(detect_path=LOCAL_DET_DIR, pose_model_path=LOCAL_SP_DIR, device="cpu"):
    """Loads the detection and pose estimation models."""
    print(f"Loading models to device: {device}")

    # --- Person Detection Model ---
    # print("Loading detection model...")
    # if detect_path and detect_path != "PekingU/rtdetr_r50vd_coco_o365":
    #     print(f"Loading local detector from: {detect_path}")
    #     person_image_processor = AutoProcessor.from_pretrained(detect_path, local_files_only=True)
    #     person_model = RTDetrForObjectDetection.from_pretrained(detect_path, local_files_only=True, device_map=device)
    # else:
    #     print("Loading detector from Hugging Face: PekingU/rtdetr_r50vd_coco_o365")
    #     # Ensure use_fast=True is compatible or remove if causing issues
    #     person_image_processor = AutoProcessor.from_pretrained("PekingU/rtdetr_r50vd_coco_o365")
    #     person_model = RTDetrForObjectDetection.from_pretrained("PekingU/rtdetr_r50vd_coco_o365", device_map=device)

    from ultralytics import YOLO
    from huggingface_hub import hf_hub_download

    # Define where you want to store the model
    
    os.makedirs(detect_path,exist_ok=True)
    model_id = "jameslahm/yolov10s.pt"
    filename = model_id.split('/')[-1]
    model_path = os.path.join(detect_path,filename)

    # Check if model already exists locally.
    # note: ultralytics packages processor and model together. 
    if not os.path.exists(model_path):
        # Download only if it doesn't exist
        print("Downloading model...")
        model_path = hf_hub_download(
        repo_id=model_id,
        filename=filename,
        cache_dir=detect_path,
        local_dir_use_symlinks=False)
        
    person_model = YOLO(model_path)    
    person_model.to(device)

    # --- Pose Estimation Model ---
    print(f"Loading pose estimation model from: {pose_model_path}")
    # Ensure use_fast=True is compatible or remove if causing issues
    pose_image_processor = AutoProcessor.from_pretrained(pose_model_path, local_files_only=True)
    pose_model = VitPoseForPoseEstimation.from_pretrained(pose_model_path, local_files_only=True, device_map=device)

    print("Models loaded successfully.")
    return None, person_model, pose_image_processor, pose_model

def detect_persons(image, person_image_processor, person_model, device, confidence_threshold=0.3):
    """Drop-in replacement for RT-DETR using YOLO - same function signature"""
    
    results = person_model(image, conf=confidence_threshold, half=True, classes=[0], verbose = False)  # class 0 = person
        # imgsz=640,        # Input size (preprocessing)
    # conf=0.25,        # Confidence threshold (postprocessing)
    # iou=0.45,         # NMS IoU threshold (postprocessing)
    # max_det=300,      # Maximum detections (postprocessing)
    # augment=False,    # Test-time augmentation
    # half=False,       # FP16 inference
    # device='cpu'      # Device selection
    
    if len(results[0].boxes) == 0:
        # Return empty arrays - same as your original
        return np.empty((0, 4), dtype=np.float32), \
               np.empty((0, 4), dtype=np.float32), \
               np.empty((0,), dtype=np.float32)
    
    boxes = results[0].boxes
    
    person_boxes_voc = boxes.xyxy.cpu().numpy()      # (x1, y1, x2, y2) - VOC format
    person_scores = boxes.conf.cpu().numpy()
    
    # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h) - same as your original
    person_boxes_coco = person_boxes_voc.copy()
    person_boxes_coco[:, 2] = person_boxes_coco[:, 2] - person_boxes_coco[:, 0] # w = x2 - x1
    person_boxes_coco[:, 3] = person_boxes_coco[:, 3] - person_boxes_coco[:, 1] # h = y2 - y1
    
    return person_boxes_voc, person_boxes_coco, person_scores

def detect_persons_rtdetr(image, person_image_processor, person_model, device, confidence_threshold=0.3):
    """Detects persons in a PIL Image."""
    inputs = person_image_processor(images=image, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = person_model(**inputs)

    results = person_image_processor.post_process_object_detection(
        outputs, target_sizes=torch.tensor([image.size[::-1]]), threshold=confidence_threshold
    )
    result = results[0]

    person_labels_mask = (result["labels"] == 0) # COCO index for person is 0
    person_boxes_voc = result["boxes"][person_labels_mask]
    person_scores = result["scores"][person_labels_mask]

    # --- FIX: Ensure NumPy array return, even if empty ---
    if len(person_boxes_voc) == 0:
        # Return empty NumPy arrays with appropriate shape hint if possible
        return np.empty((0, 4), dtype=np.float32), \
               np.empty((0, 4), dtype=np.float32), \
               np.empty((0,), dtype=np.float32)

    person_boxes_voc = person_boxes_voc.cpu().numpy()
    person_scores = person_scores.cpu().numpy()

    # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h)
    person_boxes_coco = person_boxes_voc.copy()
    person_boxes_coco[:, 2] = person_boxes_coco[:, 2] - person_boxes_coco[:, 0] # w = x2 - x1
    person_boxes_coco[:, 3] = person_boxes_coco[:, 3] - person_boxes_coco[:, 1] # h = y2 - y1
    # --- END FIX ---

    return person_boxes_voc, person_boxes_coco, person_scores

import cv2 # Make sure cv2 is imported
import numpy as np # Make sure numpy is imported
import os # For creating directories

def estimate_poses(
    image,
    person_boxes_coco,
    pose_image_processor,
    pose_model,
    device,
    debug_plot=False,
    debug_save_prefix=None
    ):
    """
    Estimates poses for given person bounding boxes.
    Includes optional debugging visualization.

    returns:
        all_keypoints: List of keypoints for each detected person.
        all_keypoint_scores: List of scores for each keypoint.
    """
    if person_boxes_coco.size == 0:
         return [], []

    person_boxes_coco_list = person_boxes_coco.astype(np.float32).tolist()

    # Process boxes for pose estimation
    inputs = pose_image_processor(image, boxes=[person_boxes_coco_list], return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = pose_model(**inputs)

    
    num_persons = len(person_boxes_coco_list)
    # # Create a list with the original image size repeated for each person
    
    pose_results = pose_image_processor.post_process_pose_estimation(
        outputs,
        boxes=[person_boxes_coco_list]
    )

    # post_process_pose_estimation should still return a list per original image.
    # Since we passed one image, pose_results should be a list of length 1.
    if not pose_results:
        print("Warning estimate_poses: post_process_pose_estimation returned empty list.")
        return [], []

    # Access the results for the first (and only) image in the batch fed to the processor
    image_pose_results = pose_results[0]

    # This list should contain pose results for each person box processed
    if len(image_pose_results) != num_persons:
         print(f"Warning estimate_poses: Number of pose results ({len(image_pose_results)}) doesn't match number of input boxes ({num_persons}).")
         # Decide how to handle this - maybe return empty or try to process what's available

    all_keypoints = []
    all_keypoint_scores = []
    if not isinstance(image_pose_results, list):
        print(f"Warning estimate_poses: Unexpected format for image_pose_results: {type(image_pose_results)}")
        return [], []

    # Loop through the results for each person
    for person_idx, person_result in enumerate(image_pose_results):
        if not isinstance(person_result, dict):
             print(f"Warning estimate_poses: Unexpected format for person_result {person_idx}: {type(person_result)}")
             # Append placeholders or skip
             all_keypoints.append(np.full((17, 2), np.nan)) # Assuming 17 keypoints
             all_keypoint_scores.append(np.zeros(17))
             continue

        keypoints = person_result.get('keypoints', None)
        scores = person_result.get('scores', None)

        if keypoints is None or scores is None:
             print(f"Warning estimate_poses: Missing 'keypoints' or 'scores' in person_result {person_idx}.")
             all_keypoints.append(np.full((17, 2), np.nan))
             all_keypoint_scores.append(np.zeros(17))
             continue

        if isinstance(keypoints, torch.Tensor):
            keypoints = keypoints.cpu().numpy()
        if isinstance(scores, torch.Tensor):
             scores = scores.cpu().numpy()

        if not (isinstance(keypoints, np.ndarray) and keypoints.ndim == 2 and keypoints.shape[1] >= 2):
             print(f"Warning estimate_poses: Unexpected keypoints format/shape for person {person_idx}: {type(keypoints)}, shape {getattr(keypoints, 'shape', 'N/A')}")
             all_keypoints.append(np.full((17, 2), np.nan))
             all_keypoint_scores.append(np.zeros(17))
             continue

        all_keypoints.append(keypoints)
        all_keypoint_scores.append(scores)

    # --- Debugging Visualization ---
    # (Keep the plotting code as is)
    if debug_plot and len(all_keypoints) > 0:
        print("Debugging visualization enabled. Plotting keypoints and boxes...")
        try:
            frame_bgr = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
            box_color = (0, 255, 0); kp_color_high_conf = (255, 0, 0); kp_color_low_conf = (0, 0, 255); text_color = (255, 255, 255)

            # Check if number of keypoint sets matches number of boxes
            if len(all_keypoints) != len(person_boxes_coco):
                 print(f"Warning estimate_poses plot: Mismatch between keypoint sets ({len(all_keypoints)}) and boxes ({len(person_boxes_coco)}). Plotting may be incorrect.")
                 # Attempt to plot anyway, using the minimum of the two counts
                 num_to_plot = min(len(all_keypoints), len(person_boxes_coco))
            else:
                 num_to_plot = len(all_keypoints)

            
            for i in range(num_to_plot):
                box_coco = person_boxes_coco[i]
                x1, y1, w, h = map(int, box_coco)
                x2, y2 = x1 + w, y1 + h
                cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), box_color, 2)
                cv2.putText(frame_bgr, f"Person {i}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1)

                keypoints = all_keypoints[i]
                scores = all_keypoint_scores[i]

                if keypoints.shape[0] != scores.shape[0]: continue # Already warned above

                for kp_idx in range(keypoints.shape[0]):
                    kx, ky = map(int, keypoints[kp_idx, :2])
                    conf = scores[kp_idx]
                    kp_color = kp_color_high_conf if conf >= 0.3 else kp_color_low_conf
                    cv2.circle(frame_bgr, (kx, ky), 3, kp_color, -1)

            # show it!
            cv2.imshow("Pose Estimation Debug", frame_bgr)
            cv2.waitKey(0)  # Wait for a key press to close the window

        except Exception as e:
            print(f"Error during debug plotting in estimate_poses: {e}")

    if debug_plot and debug_save_prefix is not None:
            output_filename = os.path.join(debug_save_prefix,"pose.png")
            os.makedirs(os.path.dirname(debug_save_prefix), exist_ok=True)
            cv2.imwrite(output_filename, frame_bgr)
            cv2.destroyAllWindows()


    return all_keypoints, all_keypoint_scores

def detect_persons_rdetr(images, person_image_processor, person_model, device, confidence_threshold=0.3, batch_size=8):
    """Batch detects persons in multiple PIL Images with debugging."""

    import time
    
    batch_results = []
    
    # DEBUG: Time single image processing
    if len(images) > 0:
        start_time = time.time()
        single_inputs = person_image_processor(images=images[0], return_tensors="pt").to(device)
        with torch.no_grad():
            _ = person_model(**single_inputs)
        single_time = time.time() - start_time
        print(f"DEBUG: Single image processing time: {single_time:.4f}s")
    
    # Process images in batches
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        actual_batch_size = len(batch_images)
        
        # DEBUG: Time batch processing
        start_time = time.time()
        
        # Prepare batch inputs
        inputs = person_image_processor(images=batch_images, return_tensors="pt").to(device)
        target_sizes = torch.tensor([img.size[::-1] for img in batch_images])
        
        # DEBUG: Check if we're actually batching
        # print(f"DEBUG: Input tensor shape: {inputs['pixel_values'].shape}")
        # print(f"DEBUG: Expected batch dimension: {actual_batch_size}")
        
        with torch.no_grad():
            outputs = person_model(**inputs)
        
        batch_time = time.time() - start_time
        # print(f"DEBUG: Batch of {actual_batch_size} images time: {batch_time:.4f}s")
        # print(f"DEBUG: Time per image in batch: {batch_time/actual_batch_size:.4f}s")
        
        # Post-process batch results
        results = person_image_processor.post_process_object_detection(
            outputs, target_sizes=target_sizes, threshold=confidence_threshold
        )
        
        # Process each image result in the batch
        for result in results:
            person_labels_mask = (result["labels"] == 0)  # COCO index for person is 0
            person_boxes_voc = result["boxes"][person_labels_mask]
            person_scores = result["scores"][person_labels_mask]
            
            # --- FIX: Ensure NumPy array return, even if empty ---
            if len(person_boxes_voc) == 0:
                # Return empty NumPy arrays with appropriate shape hint if possible
                batch_results.append((
                    np.empty((0, 4), dtype=np.float32),
                    np.empty((0, 4), dtype=np.float32),
                    np.empty((0,), dtype=np.float32)
                ))
                continue
                
            person_boxes_voc = person_boxes_voc.cpu().numpy()
            person_scores = person_scores.cpu().numpy()
            
            # Convert boxes from VOC (x1, y1, x2, y2) to COCO (x1, y1, w, h)
            person_boxes_coco = person_boxes_voc.copy()
            person_boxes_coco[:, 2] = person_boxes_coco[:, 2] - person_boxes_coco[:, 0]  # w = x2 - x1
            person_boxes_coco[:, 3] = person_boxes_coco[:, 3] - person_boxes_coco[:, 1]  # h = y2 - y1
            # --- END FIX ---
            
            batch_results.append((person_boxes_voc, person_boxes_coco, person_scores))
    
    return batch_results
def detect_persons_batch_rtdetr(images, person_image_processor, person_model, device, confidence_threshold=0.3, batch_size=8):
    """Batch detects persons in multiple PIL Images with debugging."""
    
    import time
    
    batch_results = []
    
    # DEBUG: Time single image processing
    if len(images) > 0:
        start_time = time.perf_counter()
        single_inputs = person_model(images[0],conf=confidence_threshold, half=True, classes=[0])
        single_time = time.time() - start_time
        print(f"DEBUG: Single image processing time: {single_time:.4f}s")
    
    # Process images in batches
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        actual_batch_size = len(batch_images)
        
        # DEBUG: Time batch processing
        start_time = time.time()
        
        # do the batch processing.
        results = person_model(batch_images, conf=confidence_threshold, half=True,classes=[0],batch=len(batch_images))
        
        batch_time = time.time() - start_time
        # print(f"DEBUG: Batch of {actual_batch_size} images time: {batch_time:.4f}s")
        # print(f"DEBUG: Time per image in batch: {batch_time/actual_batch_size:.4f}s")
        
        # Process each image result in the batch
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
            # All boxes are persons since we filtered with classes=[0]
                person_boxes_voc = result.boxes.xyxy.cpu().numpy()  # VOC format
                person_boxes_coco = result.boxes.xywh.cpu().numpy() # COCO format
                person_scores = result.boxes.conf.cpu().numpy()
            else:
                # Empty arrays matching your format
                person_boxes_voc = np.empty((0, 4), dtype=np.float32)
                person_boxes_coco = np.empty((0, 4), dtype=np.float32)
                person_scores = np.empty((0,), dtype=np.float32)
            
            batch_results.append((person_boxes_voc, person_boxes_coco, person_scores))
    
    return batch_results

def detect_persons_batch(images, person_image_processor, person_model, device, confidence_threshold=0.3, batch_size=8):
    """Batch detects persons in multiple PIL Images with debugging."""
    import time
    
    batch_results = []
    
    # DEBUG: Time single image processing
    # if len(images) > 0:
    #     start_time = time.perf_counter()
    #     single_results = person_model(images[0], conf=confidence_threshold, half=True, classes=[0])
    #     single_time = time.perf_counter() - start_time
    #     # print(f"DEBUG: Single image processing time: {single_time:.4f}s")
    
    # Process images in batches
    for i in range(0, len(images), batch_size):
        batch_images = images[i:i + batch_size]
        actual_batch_size = len(batch_images)
        
        # DEBUG: Time batch processing
        start_time = time.perf_counter()
        results = person_model(batch_images, conf=confidence_threshold, half=True, classes=[0], batch=len(batch_images), verbose=False)
        batch_time = time.perf_counter() - start_time
        # print(f"DEBUG: Batch of {actual_batch_size} images time: {batch_time:.4f}s")
        # print(f"DEBUG: Time per image in batch: {batch_time/actual_batch_size:.4f}s")
        
        # Process each image result in the batch
        for result in results:
            if result.boxes is not None and len(result.boxes) > 0:
                person_boxes_voc = result.boxes.xyxy.cpu().numpy()  # VOC format
                
                # Convert VOC to COCO (x1, y1, w, h) - matching single function
                person_boxes_coco = person_boxes_voc.copy()
                person_boxes_coco[:, 2] = person_boxes_coco[:, 2] - person_boxes_coco[:, 0]  # w = x2 - x1
                person_boxes_coco[:, 3] = person_boxes_coco[:, 3] - person_boxes_coco[:, 1]  # h = y2 - y1
                
                person_scores = result.boxes.conf.cpu().numpy()
            else:
                # Empty arrays matching your format
                person_boxes_voc = np.empty((0, 4), dtype=np.float32)
                person_boxes_coco = np.empty((0, 4), dtype=np.float32)
                person_scores = np.empty((0,), dtype=np.float32)
            
            batch_results.append((person_boxes_voc, person_boxes_coco, person_scores))
    
    return batch_results

def estimate_poses_batch(images_with_boxes, pose_image_processor, pose_model, device, batch_size=8):
    """
    Batch estimates poses for given images and their person bounding boxes.
    
    Args:
        images_with_boxes: List of tuples (image, person_boxes_coco)
        pose_image_processor: The pose processor
        pose_model: The pose model
        device: Device to run on
        batch_size: Number of images to process simultaneously
    
    Returns:
        List of (all_keypoints, all_keypoint_scores) for each image
    """
    batch_results = []
    
    # Process images in batches
    for i in range(0, len(images_with_boxes), batch_size):
        batch_data = images_with_boxes[i:i + batch_size]
        
        # Prepare batch inputs
        batch_images = []
        batch_boxes_lists = []
        
        for image, person_boxes_coco in batch_data:
            if person_boxes_coco.size == 0:
                # Handle empty boxes - add dummy entry to maintain batch structure
                batch_images.append(image)
                batch_boxes_lists.append([[0, 0, 1, 1]])  # Dummy box
            else:
                batch_images.append(image)
                person_boxes_coco_list = person_boxes_coco.astype(np.float32).tolist()
                batch_boxes_lists.append(person_boxes_coco_list)
        
        # Process batch
        inputs = pose_image_processor(batch_images, boxes=batch_boxes_lists, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = pose_model(**inputs)
        
        # Post-process batch results
        pose_results = pose_image_processor.post_process_pose_estimation(
            outputs,
            boxes=batch_boxes_lists
        )
        
        # Process results for each image in the batch
        for j, (image, person_boxes_coco) in enumerate(batch_data):
            if person_boxes_coco.size == 0:
                batch_results.append(([], []))
                continue
                
            # Access the results for this image
            if j >= len(pose_results):
                print(f"Warning: No pose results for image {j}")
                batch_results.append(([], []))
                continue
                
            image_pose_results = pose_results[j]
            num_persons = len(batch_boxes_lists[j])
            
            # Check if results match input boxes
            if len(image_pose_results) != num_persons:
                print(f"Warning estimate_poses_batch: Number of pose results ({len(image_pose_results)}) doesn't match number of input boxes ({num_persons}).")
            
            all_keypoints = []
            all_keypoint_scores = []
            
            if not isinstance(image_pose_results, list):
                print(f"Warning estimate_poses_batch: Unexpected format for image_pose_results: {type(image_pose_results)}")
                batch_results.append(([], []))
                continue
            
            # Loop through the results for each person
            for person_idx, person_result in enumerate(image_pose_results):
                if not isinstance(person_result, dict):
                    print(f"Warning estimate_poses_batch: Unexpected format for person_result {person_idx}: {type(person_result)}")
                    all_keypoints.append(np.full((17, 2), np.nan))  # Assuming 17 keypoints
                    all_keypoint_scores.append(np.zeros(17))
                    continue
                
                keypoints = person_result.get('keypoints', None)
                scores = person_result.get('scores', None)
                
                if keypoints is None or scores is None:
                    print(f"Warning estimate_poses_batch: Missing 'keypoints' or 'scores' in person_result {person_idx}.")
                    all_keypoints.append(np.full((17, 2), np.nan))
                    all_keypoint_scores.append(np.zeros(17))
                    continue
                
                if isinstance(keypoints, torch.Tensor):
                    keypoints = keypoints.cpu().numpy()
                if isinstance(scores, torch.Tensor):
                    scores = scores.cpu().numpy()
                
                if not (isinstance(keypoints, np.ndarray) and keypoints.ndim == 2 and keypoints.shape[1] >= 2):
                    print(f"Warning estimate_poses_batch: Unexpected keypoints format/shape for person {person_idx}: {type(keypoints)}, shape {getattr(keypoints, 'shape', 'N/A')}")
                    all_keypoints.append(np.full((17, 2), np.nan))
                    all_keypoint_scores.append(np.zeros(17))
                    continue
                
                all_keypoints.append(keypoints)
                all_keypoint_scores.append(scores)
            
            batch_results.append((all_keypoints, all_keypoint_scores))
    
    return batch_results

# Test script to verify the pose detection functionality
if __name__ == "__main__":
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if device == "mps" and not torch.backends.mps.is_built():
         print("MPS not available, falling back to CPU")
         device = "cpu"

    # Load models
    person_processor, person_model, pose_processor, pose_model = load_models(device=device)

    # Load image
    try:
        image = Image.open("test/chanel.png").convert("RGB")
    except FileNotFoundError:
        print("Error: test/chanel.png not found. Please provide a valid image.")
        exit()

    # Detect persons
    person_boxes_voc, person_boxes_coco, person_scores = detect_persons(
        image, person_processor, person_model, device
    )
    # print(f"Detected {len(person_boxes_voc)} persons.")

    # Estimate poses
    all_keypoints, all_keypoint_scores = estimate_poses(
        image, person_boxes_coco, pose_processor, pose_model, device
    )
    print(f"Estimated poses for {len(all_keypoints)} persons.")

    # --- Optional: Visualization using supervision (as in original test.py) ---
    if all_keypoints:
        import supervision as sv

        # Prepare KeyPoints object for supervision
        # Stack keypoints and scores from all detected persons
        xy = np.array([kp[:, :2] for kp in all_keypoints]) # Take only x, y
        confidence = np.array(all_keypoint_scores)

        if xy.ndim == 3 and confidence.ndim == 2 and xy.shape[0] == confidence.shape[0] and xy.shape[1] == confidence.shape[1]:
             keypoints_sv = sv.KeyPoints(xy=xy, confidence=confidence)

             # Annotate
             frame_np = np.array(image) # Convert PIL to numpy for annotation
             vertex_annotator = sv.VertexAnnotator(color=sv.Color.RED, radius=4)
             box_annotator = sv.BoxAnnotator(
                color=sv.Color.GREEN,
                color_lookup=sv.ColorLookup.INDEX) #,text_color=sv.Color.WHITE, Add this line 

             annotated_frame = vertex_annotator.annotate(scene=frame_np.copy(), key_points=keypoints_sv)

             # Create Detections object for bounding boxes
             detections = sv.Detections(
                  xyxy=person_boxes_voc, # Supervision uses xyxy (voc)
                  confidence=person_scores,
                  # class_id=np.zeros(len(person_boxes_voc), dtype=int) # Assign class ID if needed
             )
             # Annotate boxes
             annotated_frame = box_annotator.annotate(scene=annotated_frame, detections=detections)


             # Display or save
             annotated_image = Image.fromarray(annotated_frame)
             annotated_image.show() # Display the image
             print("Displaying annotated image.")
             # annotated_image.save("test/output_pose.png")
             print("Saved annotated image to test/output_pose.png")
        else:
            print("Warning: Could not format keypoints for supervision visualization.")
            print(f"xy shape: {xy.shape}, confidence shape: {confidence.shape}")
