#%%
import posetrack as pt
import os
import pandas as pd
import matplotlib.pyplot as plt
# example on how to loop across, load synthpose, and determine peake COM speed. 
list_dirs = ['/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-07-17/recordings_gerrt/recording_gerrt_peakspeed_v1' ,
'/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-07-17/recordings_gerrt/recording_gerrt_peakspeed_v2' ,
'/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-07-17/recordings_gerrt/recording_gerrt_peakspeed_v3' ,
'/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-07-17/recordings_gerrt/recording_juneb_peakspeed_v1' ,
'/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-07-17/recordings_gerrt/recording_juneb_peakspeed_v2' ,
'/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-07-17/recordings_gerrt/recording_juneb_peakspeed_v3' ,
'/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-07-17/recordings_gerrt/recording_tonyv_peakspeed_v1' ,
'/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-07-17/recordings_gerrt/recording_tonyv_peakspeed_v2',
'/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-07-17/recordings_gerrt/recording_colleeng_peakspeed_v1', 
'/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-07-17/recordings_gerrt/recording_colleeng_peakspeed_v2',
'/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-07-17/recordings_gerrt/recording_shirleyc_peakspeed_v1',
'/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-07-17/recordings_gerrt/recording_shirleyc_peakspeed_v2',
'/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-07-17/recordings_gerrt/recording_shirleyc_peakspeed_v3']

rotfile = '/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-07-17/recordings_colleeng/recording_3by1_v1/synthpose/output_3d_poses_tracked.csv_person0.csv'
data1 = pd.read_csv(rotfile)
(data, dat_dict) = pt.map_synthpose_to_mediapipe(data1)

# rotate matrix
R, X0, g = pt.get_rotmat_x0(dat_dict['left_heel'])

# set mode to be interactive matplotlib
import matplotlib.pyplot as plt

lheel = dat_dict['left_heel']
%matplotlib qtagg


#%% 
import numpy as np
which_person = [0,0,2,1,0,0,1,5,1,1,0,2,0]
f,ax = plt.subplots(2,1)
velocities = []
for i, dir in enumerate(list_dirs):
    print(f"Processing directory: {dir}")
    results_directory = os.path.join(dir, 'synthpose')
    if not os.path.exists(results_directory):
        print(f"Skipping {results_directory} as it does not exist.")
        continue
    
    # Load the synthpose data
    specific_file = 'output_3d_poses_tracked.csv_person'+str(which_person[i])+'.csv'
    csv_3d_coordinates_path = os.path.join(results_directory, specific_file)
    data = pd.read_csv(csv_3d_coordinates_path)
    (dat,dadict) = pt.map_synthpose_to_mediapipe(data)
    # Calculate the peak speed of the center of mass (COM), the mean of the left_shoulder, left_hip, right_shoulder, and right_hip
    # clean the columns left_shoulder, left_hip, right_shoulder, right_hip, by first interpolating any missing vals. 
    com = (dadict['left_shoulder'] + dadict['left_hip'] +
             dadict['right_shoulder'] + dadict['right_hip']) / 4
    # now rotate and center
    com_rotated = (R @ (com.T)).T - np.reshape(X0,(1,3)) # Rotate and center the COM
    ax[0].plot(com_rotated[:, 0],com_rotated[:, 1])  # Plot the x-coordinate of the COM
    si = data['sync_index'].to_numpy()
    ax[1].plot(si-si[0],com_rotated[:, 1])  # Plot the x-coordinate of the COM
    # compute average velocity from the beginning, up to when the y-coord passes 3 m
    idx_3m = np.where(com_rotated[:, 1] > 3)[0]
    if len(idx_3m) > 0:
        n_samples = idx_3m[0]
        idx_3m = idx_3m[0]  # Use the first index where y > 3 m
    else:
        n_samples = data.shape[0]
    time_per_sample = 1/30

    dt = (data['sync_index'][idx_3m] - data['sync_index'][0])*time_per_sample
    dl = com_rotated[idx_3m,1] - com_rotated[0,1]
    avg_velocity = dl / dt
    velocities.append(avg_velocity)
    print(f"Average velocity for {os.path.basename(dir)}: {avg_velocity:.2f} m/s")
# %%
ax[0].set_xlabel('x pos (m)')
ax[0].set_ylabel('y pos (m)')
#set axis equal
ax[0].set_aspect('equal', adjustable='box')
ax[1].set_xlabel('sync_index')
ax[1].set_ylabel('y pos (m)')
# %%
v_per_sub = np.array([np.mean(velocities[0:3]),np.mean(velocities[3:6]),np.mean(velocities[6:8]),np.mean(velocities[8:10]),np.mean(velocities[10:13])])
# %%
