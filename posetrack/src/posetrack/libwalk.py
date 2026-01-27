#%libwalk
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import os
import pandas as pd
from .pose_detector import SynthPoseMarkers
from tkinter import Tk, filedialog

import cv2
   
def extract_frames(input_file, output_file, n_frames):
   cap = cv2.VideoCapture(input_file)
   fourcc = cv2.VideoWriter.fourcc(*'mp4v')  # Note the dot instead of underscore
   fps = cap.get(cv2.CAP_PROP_FPS)
   width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
   height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
   
   out = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
   
   for i in range(n_frames):
       ret, frame = cap.read()
       if not ret:
           break
       out.write(frame)
   
   cap.release()
   out.release()

def map_synthpose_to_mediapipe(df):
    """
    Maps synthpose marker data to Mediapipe format.
    
    Args:
        df: pandas DataFrame with synthpose markers (columns like 'Nose_x', 'Nose_y', 'Nose_z', etc.)
    
    Returns:
        pandas DataFrame with Mediapipe format (columns with nx3 arrays for x,y,z coordinates)
    """
    
    # Mapping from synthpose markers to mediapipe names
    marker_mapping = {
        # Face markers
        'L_Eye': 'left_eye',
        'R_Eye': 'right_eye', 
        'L_Ear': 'left_ear',
        'R_Ear': 'right_ear',
        'Nose': 'nose',
        
        # Body markers
        'L_Shoulder': 'left_shoulder',
        'R_Shoulder': 'right_shoulder',
        'L_Elbow': 'left_elbow',
        'R_Elbow': 'right_elbow',
        'L_Wrist': 'left_wrist',
        'R_Wrist': 'right_wrist',
        'L_Hip': 'left_hip',
        'R_Hip': 'right_hip',
        'L_Knee': 'left_knee',
        'R_Knee': 'right_knee',
        'L_Ankle': 'left_ankle',
        'R_Ankle': 'right_ankle',
        
        # Alternative naming patterns in synthpose
        'lshoulder': 'left_shoulder',
        'rshoulder': 'right_shoulder',
        'l_knee': 'left_knee',
        'r_knee': 'right_knee',
        'l_ankle': 'left_ankle',
        'r_ankle': 'right_ankle',
        
        # Foot markers - using available toe markers for foot_index
        'l_big_toe': 'left_foot_index',
        'r_big_toe': 'right_foot_index',
        'l_calc': 'left_heel',  # calcaneus = heel
        'r_calc': 'right_heel',
        
        # Fallback foot markers if big toe not available
        'l_toe': 'left_foot_index',
        'r_toe': 'right_foot_index',
        'l_5meta': 'left_foot_index',  # 5th metatarsal as fallback
        'r_5meta': 'right_foot_index'
    }

    # Create output dataframe
    output_df = pd.DataFrame(index=df.index)
    
    # Process each mapping
    for synthpose_marker, mediapipe_name in marker_mapping.items():
        # Check if this marker exists in the input dataframe
        x_col = f'{synthpose_marker}_X'
        y_col = f'{synthpose_marker}_Y'
        z_col = f'{synthpose_marker}_Z'

        #create the analogous mediapipe columns, ONLY if the mediapipe_x columns do not already exist
        if f'{mediapipe_name}_x' not in output_df.columns:
          mediapipe_x_col = f'{mediapipe_name}_x'
          mediapipe_y_col = f'{mediapipe_name}_y'
          mediapipe_z_col = f'{mediapipe_name}_z'
          # # Check if all three columns exist
          if all(col in df.columns for col in [x_col, y_col, z_col]):            
            output_df[mediapipe_x_col] = df[x_col]
            output_df[mediapipe_y_col] = df[y_col]
            output_df[mediapipe_z_col] = df[z_col]

    
    output_dict = {}
    # loop across marker_mapping to create a dictionary of nx3 arrays
    for mediapipe_name in marker_mapping.values():
        cols = [col for col in output_df.columns if col.startswith(mediapipe_name)]
        if len(cols) == 3:
            output_dict[mediapipe_name] = output_df[cols].to_numpy()
        else:
            print(f"Warning: Expected 3 columns for {mediapipe_name}, found {len(cols)}. Skipping this marker.")

    return output_df, output_dict

def quick_rotation_matrix(person_data):
    """
    Calculate a rotation matrix to orient pose data correctly (feet on ground, person upright).
    
    Args:
        person_data (dict): Dictionary containing body part data with 'L_Ankle', 'R_Ankle', 'L_Hip', 'R_Hip'
        
    Returns:
        tuple: (rotation_matrix, x0) where rotation_matrix is 3x3 and x0 is the initial right foot position
    """
    # Get foot and hip data
    left_ankle = person_data.get('L_Ankle')
    right_ankle = person_data.get('R_Ankle') 
    left_hip = person_data.get('L_Hip')
    right_hip = person_data.get('R_Hip')
    
    if any(data is None for data in [left_ankle, right_ankle, left_hip, right_hip]):
        print("Missing required keypoint data for rotation matrix calculation")
        return np.eye(3), np.zeros(3)
    
    # Find stable frame where feet aren't moving much
    stable_frame_idx = None
    
    # Try 5 frames of stability first, then 3
    for window_size in [5, 3]:
        for i in range(window_size, len(left_ankle)):
            # Check if both feet are stable over the window
            left_stable = True
            right_stable = True
            
            for j in range(1, window_size):
                # Calculate velocity (change in position)
                if i-j >= 0:
                    left_vel = np.linalg.norm(left_ankle[i] - left_ankle[i-j])
                    right_vel = np.linalg.norm(right_ankle[i] - right_ankle[i-j])
                    
                    if not (np.isnan(left_vel) or np.isnan(right_vel)):
                        if left_vel > 0.1 * j or right_vel > 0.1 * j:  # 0.1 m/s threshold
                            left_stable = False
                            right_stable = False
                            break
            
            if left_stable and right_stable:
                # Check that both feet are visible (not NaN)
                if not (np.isnan(left_ankle[i]).any() or np.isnan(right_ankle[i]).any() or
                       np.isnan(left_hip[i]).any() or np.isnan(right_hip[i]).any()):
                    stable_frame_idx = i
                    break
        
        if stable_frame_idx is not None:
            break
    
    # If no stable frame found, use first frame where both feet and hips are visible
    if stable_frame_idx is None:
        for i in range(len(left_ankle)):
            if not (np.isnan(left_ankle[i]).any() or np.isnan(right_ankle[i]).any() or
                   np.isnan(left_hip[i]).any() or np.isnan(right_hip[i]).any()):
                stable_frame_idx = i
                print(f"No stable frame found, using first visible frame {i}")
                break
    else:
        print(f"Found stable frame at index {stable_frame_idx}")
    
    # Get positions at stable frame
    left_foot = left_ankle[stable_frame_idx]
    right_foot = right_ankle[stable_frame_idx]
    left_hip_pos = left_hip[stable_frame_idx]
    right_hip_pos = right_hip[stable_frame_idx]
    
    # Calculate hip center
    hip_center = (left_hip_pos + right_hip_pos) / 2
    
    # Calculate ankle center  
    ankle_center = (left_foot + right_foot) / 2
    
    # X-axis: vector between feet (left to right)
    x_axis = right_foot - left_foot
    x_axis = x_axis / np.linalg.norm(x_axis)
    
    # Z-axis: orthogonal vector from ankle center to hip center (upward)
    z_axis_raw = hip_center - ankle_center
    z_axis_raw = z_axis_raw / np.linalg.norm(z_axis_raw)
    
    # Make z_axis orthogonal to x_axis
    z_axis = z_axis_raw - np.dot(z_axis_raw, x_axis) * x_axis
    z_axis = z_axis / np.linalg.norm(z_axis)
    
    # Y-axis: cross product (negative to make right-handed coordinate system)
    y_axis = -np.cross(x_axis, z_axis)
    y_axis = y_axis / np.linalg.norm(y_axis)
    
    # Create rotation matrix [x, y, z] as columns
    rotation_matrix = np.column_stack([x_axis, y_axis, z_axis])
    
    # Initial position (right foot)
    x0 = right_foot.copy()
    
    print(f"Rotation matrix calculated:")
    print(f"X-axis (between feet): {x_axis}")
    print(f"Y-axis (forward): {y_axis}")  
    print(f"Z-axis (upward): {z_axis}")
    print(f"Initial right foot position: {x0}")
    
    return rotation_matrix, x0

def load_synthpose_csv_as_dict():
  root = Tk()
  root.withdraw()
  file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])
  if not file_path:
    return None
  df = pd.read_csv(file_path)
  marker_dict = {}
  for marker in SynthPoseMarkers.markers:
    cols = [f"{marker}_x", f"{marker}_y", f"{marker}_z"]
    if all(col in df.columns for col in cols):
      marker_dict[marker] = df[cols].values
  return marker_dict

def get_rotmat_x0(data_nx3,pm = None, xdir=1):
  
  # plot the data, with a title
  f,ax = plt.subplots()
  plt.plot(data_nx3)
  titletxt = f"Click 3 times to back-left, front-left,front-right:"
  plt.title(titletxt)
  
  # initialize the list of x-clicks we will collect
  ind_xclicks = []
  # Function to capture mouse clicks
  def onclick(event):
    ind_xclicks.append((event.xdata))
    # break if coordinates are 3
    if len(ind_xclicks) == 3:
      plt.gcf().canvas.mpl_disconnect(cid)
      plt.close()

  # Connect the click event to the function
  cid = plt.gcf().canvas.mpl_connect('button_press_event', onclick)
  plt.draw()
  plt.show(block=True)

  # round the indices of the clicks
  ind_xclicks = [int(np.round(ind_xclicks[i])) for i in range(3)]
  
  # define vectors in 3D
  pt1 = data_nx3[ind_xclicks[0],:]
  pt2 = data_nx3[ind_xclicks[1],:]
  pt3 = data_nx3[ind_xclicks[2],:]
  
  # define which order. assume standard:
  if xdir == 1:
    v1 = pt3 - pt2 # positive right
  else:
    v1 = pt2 - pt3

  v2 = pt2 - pt1 # positive forward
  # normalize v1 and v2 to be unit length
  v1 = v1/np.linalg.norm(v1)
  v2 = v2/np.linalg.norm(v2)
  # remove any component of v2 that is in the direction of v1
  v2 = v2 - np.dot(v2,v1)*v1
  v2 = v2/np.linalg.norm(v2)
  # now take the cross product
  v3 = np.cross(v1,v2)

  # compute the rotation matrix to go from the data frame to that defined by v1,v2,v3
  R = np.array([v1,v2,v3])

  # test by first rotating data_nx3, then subtracting out pt1, then plotting
  data_nx3_rot = R @ data_nx3.T
  data_nx3_rot = data_nx3_rot.T
  x0           = data_nx3_rot[ind_xclicks[0],:]
  data_nx3_rot = data_nx3_rot - x0
  
  f,ax = plt.subplots()
  # make it a 3D figure
  ax = f.add_subplot(111, projection='3d')
  ax.plot(data_nx3_rot[ind_xclicks[0]:-1,0],data_nx3_rot[ind_xclicks[0]:-1,1],data_nx3_rot[ind_xclicks[0]:-1,2],'k.')
  
  # plot the rotated points with circles
  pt2_R = R @ (pt2 - pt1)
  pt3_R = R @ (pt3 - pt1)
  ax.plot(0,0,0,'bo')
  ax.plot(pt2_R[0],pt2_R[1],pt2_R[2],'ro')
  ax.plot(pt3_R[0],pt3_R[1],pt3_R[2],'go')
  
  ax.set_xlabel('X (m)')
  ax.set_ylabel('Y (m)')
  ax.set_zlabel('Z (m)')

  if pm is not None:
    ax.set_xlim3d([-pm,pm])
    ax.set_ylim3d([-pm,pm])
    ax.set_zlim3d([-pm,pm])

  #set el az to be 70, -90, a nice overhead view of the movement
  ax.view_init(elev=70,azim=-90)

  plt.show(block = True)
  # set g to be lengths of pt2-pt1 and pt3-pt1
  gy = np.linalg.norm(pt2 - pt1)
  gx = np.linalg.norm(pt3 - pt1)
  gs = np.array([gx,gy])
  return R,x0,gs

  if __name__ == "__main__":
    marker_dict = load_synthpose_csv_as_dict()
    if marker_dict is not None:
      print("Loaded markers:", list(marker_dict.keys()))
    else:
      print("No file selected or failed to load.")