import os,sys,argparse
path_to_project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, path_to_project_root)

from posetrack import LOCAL_DET_DIR, LOCAL_SP_DIR
from posetrack import process_synced_mwc_frames, process_synced_mwc_frames_multi_person, process_synced_mwc_frames_multi_person_perf

# simply call process_synced_mwc_frames() 
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process synchronized multi-camera video for 3D pose estimation with tracking.")
    # (Add the new arguments if needed, or keep defaults)
    testfolder = "coord_3x1_3"
    parser.add_argument("--csv_path", default=f"tests/caliscope/{testfolder}/frame_time_history.csv",help="Path...")
    parser.add_argument("--calibration_path",default=f"tests/caliscope/{testfolder}/config.toml", help="Path...")
    parser.add_argument("--video_dir", default=f"tests/caliscope/{testfolder}", help="Dir...")
    parser.add_argument("--output_path", default=f"output/caliscope/{testfolder}/output_3d_poses_tracked.csv",help="Path...")
    parser.add_argument("--model_dir", default=LOCAL_SP_DIR, help="Path...")
    parser.add_argument("--detector_dir", default=LOCAL_DET_DIR, help="Path...")
    parser.add_argument("--calib_type", default="mwc", choices=['mwc', 'fmc'], help="Type...")
    parser.add_argument("--skip", type=int, default=1, help="Skip N...")
    parser.add_argument("--person_conf", type=float, default=0.8, help="Conf...")
    parser.add_argument("--keypoint_conf", type=float, default=0.1, help="Conf...")
    parser.add_argument("--device", default="mps", choices=['auto', 'cpu', 'mps', 'cuda'], help="Device...")
    parser.add_argument("--track_max_dist", type=float, default=100.0, help="Max 2D pixel distance for head tracking match (default: 100).")
    parser.add_argument("--head_idx", type=int, default=0, help="Index of keypoint used for tracking (default: 0, Nose).")

    args = parser.parse_args()
    
    process_synced_mwc_frames_multi_person_perf(frame_history_csv_path=args.csv_path, calibration_path=args.calibration_path, video_dir=args.video_dir,
        output_path=args.output_path, model_dir=args.model_dir, detector_dir=args.detector_dir,
        calib_type=args.calib_type, skip_sync_indices=args.skip, person_confidence=args.person_conf,
        keypoint_confidence=args.keypoint_conf, device_name=args.device)