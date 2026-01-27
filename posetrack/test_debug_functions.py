#!/usr/bin/env python3
"""
Test script for the new debugging functions.
"""

import os
import sys
sys.path.insert(0, 'src')

from posetrack.process_synced_poses import read_posetrack_csv, show_multi_person_results, animate_multiperson_results, project_poses_to_video

def test_debug_functions():
    """Test the debugging functions with sample data."""
    
    # Test directory - adjust this path as needed
    test_dir = "output/caliscope/recording_balance_stage1_v2"
    
    if not os.path.exists(test_dir):
        print(f"Test directory {test_dir} does not exist.")
        print("Please run the multi-person processing first, or adjust the test directory path.")
        return
    
    print(f"Testing debug functions with directory: {test_dir}")
    
    # Test show_multi_person_results
    print("\n=== Testing show_multi_person_results ===")
    person_data, person_files_data = show_multi_person_results(test_dir, output_plot_path=f"{test_dir}/debug_timeline.png")
    
    if person_data:
        print(f"Successfully processed {len(person_data)} persons")
        
        # Test read_posetrack_csv on individual files
        print("\n=== Testing read_posetrack_csv ===")
        for person_id in person_data.keys():
            csv_file = f"{test_dir}/output_3d_poses_tracked_person{person_id}.csv"
            if os.path.exists(csv_file):
                print(f"Testing read_posetrack_csv on: {csv_file}")
                data = read_posetrack_csv(csv_file)
                if data:
                    print(f"  - Successfully read {data['_metadata']['num_frames']} frames")
                    print(f"  - Body parts found: {len([k for k in data.keys() if k != '_metadata'])}")
                    print(f"  - First few body parts: {list(data.keys())[:5]}")
                else:
                    print(f"  - Failed to read CSV")
            else:
                print(f"  - CSV file not found: {csv_file}")
        
        # Test animation function
        print("\n=== Testing animate_multiperson_results ===")
        print("Opening interactive animation window...")
        try:
            fig = animate_multiperson_results(test_dir)
            print("Animation window opened successfully!")
            print("Use the slider to navigate frames and the Play button to start/stop playback.")
        except Exception as e:
            print(f"Animation failed: {e}")
        
        # Test video projection function
        print("\n=== Testing project_poses_to_video ===")
        print("Creating pose projection video...")
        try:
            output_path = project_poses_to_video(test_dir, port_number=0)
            if output_path:
                print(f"Video projection successful! Output: {output_path}")
            else:
                print("Video projection failed")
        except Exception as e:
            print(f"Video projection failed: {e}")
    
    print("\nTesting complete!")

if __name__ == "__main__":
    test_debug_functions()