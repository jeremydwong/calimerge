# play with some parameters for the multiperson pose tracking
import posetrack as pt
import os

# base_dir = '/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-07-25/recordings_charlesn' 
# test_dir = 'recording_garnetm_b2_v3'
# test_dir = 'recording_charlesn_b1_v3'
base_dir = '/Users/jeremy/Library/CloudStorage/OneDrive-UniversityofCalgary/Project 2025 Older Adult distributed movement assessments/data/2025-08-07/recordings_speedtest' 
test_dir = 'recording_ront_b1_speed'



frame_history_csv_path = os.path.join(base_dir, test_dir, 'frame_time_history.csv')
calibration_path = os.path.join(os.path.dirname(base_dir), 'config.toml')
video_dir = os.path.join(base_dir, test_dir)
output_path = os.path.join(base_dir, test_dir, 'synthpose', 'output_3d_poses_tracked.csv')
model_dir = pt.LOCAL_SP_DIR
detector_dir = pt.LOCAL_DET_DIR
calib_type = 'mwc'

pt.process_synced_mwc_frames_multi_person_perf(frame_history_csv_path, calibration_path, video_dir, output_path, model_dir=model_dir,
    detector_dir=detector_dir, calib_type=calib_type, skip_sync_indices=1, person_confidence=0.1,track_frames_til_lost_patience=60,
    keypoint_confidence=0.1, device_name="mps",verbose_debug=False, override_views_used=[(0,2)],batch_size=16)

pt.project_poses_to_video(os.path.dirname(output_path), port_number=0, output_video_name="detected_people.mp4")