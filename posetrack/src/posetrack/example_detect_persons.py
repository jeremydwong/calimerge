#%%
# note: this file is meant to run as a script to test out 'detect_people' batch processing, handy when we want to find the people. 
# 2025-07-17: batch-processing a synthpose run. 

# base_dir = '/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-07-17/recordings_presvideo'
base_dir = '/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-07-25/recordings_garnetm/'
base_dir = '/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-07-25/recordings_carylh_newsynth/'
trial_dir = 'recording_carylh_peakspeed_v1'
#%%
# note: this file is meant to run as a script to test out new features. 
# 2025-07-17: batch-processing a synthpose run. 
import posetrack as pt 
import os
import pandas as pd
import matplotlib
# matplotlib.use('qtagg')#tqagg
# %matplotlib qtagg % interactive mode flag

#%% now loop through each subfolder, and project into a video using
ports = [0]
for theport in ports:
    pt.project_poses_to_video(os.path.join(base_dir,trial_dir,"synthpose"),theport,"vid"+str(theport)+".mp4")# project_poses_to_video(results_directory, port_number, output_video_name="detected_people.mp4"):

    
    # optionally, you can also animate the results
    

# %%

# %%
