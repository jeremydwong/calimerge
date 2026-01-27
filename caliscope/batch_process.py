# batch_process.py
import argparse
from pathlib import Path
from typing import Optional

from caliscope.controller import Controller
from caliscope.post_processing.post_processor import PostProcessor
from caliscope.trackers.tracker_enum import TrackerEnum
import caliscope.logger

def batch_process(
    workspace_dir: Path,
    tracker_enum: TrackerEnum,
    xy_gap_fill: int = 3,
    xyz_gap_fill: int = 3,
    cutoff_freq: int = 6,
    include_trc: bool = True,
):
    """
    Process all recordings in a workspace using a specified tracker.
    
    Args:
        workspace_dir: Path to Caliscope workspace directory
        tracker_enum: TrackerEnum member specifying which tracker to use
        xy_gap_fill: Max gap size for XY interpolation
        xyz_gap_fill: Max gap size for XYZ interpolation
        cutoff_freq: Butterworth filter cutoff frequency
        include_trc: Whether to generate .trc motion files
    """

    logger = caliscope.logger.get(__name__)

    # then use:
    logger.info("Beginning batch process.")
    
    logger.info(f"Processing workspace: {workspace_dir}")
    controller = Controller(workspace_dir)
    
    controller.load_camera_array()
    # controller.load_workspace()
    #print the camera count
    logger.info(f"camera count: {controller.camera_count}")    
    
    recordings_dir = workspace_dir / "recordings"
    
    # Get all trial directories directly under recordings
    trial_dirs = [d for d in recordings_dir.iterdir() if d.is_dir()]
    
    #print all trial_dirs
    logger.info(f"Found {len(trial_dirs)} trial directories:")

    #track success and failure
    successes = []
    failures = []
    for (i,trial_dir) in enumerate(trial_dirs):
        logger.info(f"Processing trial {i+1}/{len(trial_dirs)}: {trial_dir}")
        
        tracker_dir = trial_dir / tracker_enum.name
        
        # Skip if already processed
        if tracker_dir.exists():
            logger.info(f"Skipping {trial_dir} - already processed with {tracker_enum.name}")
            continue
            
        logger.info(f"\nProcessing {trial_dir} with {tracker_enum.name} tracker...")
        
        try:
            # Initialize post processor
            camera_array = controller.camera_array
            post_processor = PostProcessor(
                camera_array=camera_array,
                recording_path=trial_dir,
                tracker_enum=tracker_enum,
            )
            
            # Process data
            post_processor.create_xy()
            post_processor.create_xyz(
                xy_gap_fill=xy_gap_fill,
                xyz_gap_fill=xyz_gap_fill,
                cutoff_freq=cutoff_freq,
                include_trc=include_trc,
            )
            
            logger.info(f"Successfully processed {trial_dir}")
            successes.append(trial_dir)
            
        except Exception as e:
            logger.info(f"Failed to process {trial_dir}: {str(e)}")
            failures.append(trial_dir)
            continue

    logger.info("Batch process complete.")
    logger.info(f"Successes: {len(successes)/len(successes+failures)}")    
    for success in successes:
      logger.info(f"  {success}")
    logger.info(f"Failures: {len(failures)/len(successes+failures)}")
    for failure in failures:
      logger.info(f"  {failure}")

    return successes, failures

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Batch process Caliscope recordings")
    parser.add_argument("workspace", type=str, help="Path to Caliscope workspace directory")
    parser.add_argument(
        "--tracker",
        type=str,
        required=True,
        choices=[t.name for t in TrackerEnum],
        help="Name of tracker to use",
    )
    
    logger = caliscope.logger.get(__name__)
    args = parser.parse_args()
    
    workspace_path = Path(args.workspace)
    print("done setting workspace path")
    tracker_enum = TrackerEnum[args.tracker]
    
    successes,failures = batch_process(
        workspace_dir=workspace_path,
        tracker_enum=tracker_enum,
    )


