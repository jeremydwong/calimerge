# note: this file is meant to r un as a script to test out new features. 
# 2025-07-17: batch-processing a synthpose run. 
import posetrack as pt 
import os
import time
#%%
base_dir = '/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-07-25/recordings_carylh_newsynth'
overwrite = True #If True: we will redo existing analyses.
#%%
start_time = time.perf_counter()
pt.batch_process_subfolders(base_dir = base_dir, overwrite_tracked_files = overwrite, 
                            person_confidence=0.1, keypoint_confidence=0.1, device_name="mps",verbose_debug=False, 
                            override_views_used=[(0,2)], batch_size=32, track_frames_til_lost_patience=60)
single_time = time.time() - start_time
print(f"processed whole folder in {single_time:.2f} s.")
pt.batch_project_poses_to_video(base_dir)

# base_dir = '/Users/jeremy/Library/]CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-07-10/'
# which_trial = 'recording_linda_balance4_v3'

# base_dir = '/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-01-07/'
# which_trial = 'recording_ing_4stage1v1'                                                               

# video_file_path = os.path.join(base_dir, 'recordings', which_trial,'port_0.mp4')
# config_toml_path = os.path.join(base_dir,'config.toml')
# specific_file = 'output_3d_poses_tracked.csv_person0.csv'
# csv_3d_coordinates_path = os.path.join(base_dir, 'recordings', which_trial,'synthpose',specific_file)
# returnvals = pt.report_clothing(video_file_path, config_toml_path, csv_3d_coordinates_path, 
#                     movement_threshold=10.0, sampling_radius=3, max_samples=5)
