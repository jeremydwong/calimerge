# Posetrack
Hugging face keypoint tracking algorithms: testing conversion of caliscope to different backends

# Getting started
You will need to make reference to two offline algorithms for 
<li>
- detecting humans
- detecting the pose of those humans
</li>

We are currently using a DETR (detection transformer) model [here](https://huggingface.co/PekingU/rtdetr_r50vd_coco_o365), and a synthpose model [here](https://huggingface.co/stanfordmimi/synthpose-vitpose-base-hf). 


1. Clone the DETR and synthpose model repositories above from Hugging Face.
2. Update `src/pose_detector.py` variables 'LOCAL_SP_DIR' and 'LOCAL_DET_DIR' to reference the downloaded model directories; the directory you specify MUST have the config.json, model.safetensors, and preprocessor_config.json files.
3. Using conda, install dependencies in a new conda environment called posetrack: ```conda env create --f environment.yaml```
4. tests test_cs_parse.py, test_estimate_poses, and test_mwc_video. It will do snapshot checks for the camera parameter file parsing, identified keypoints, and video (video check is WIP).

## frame_time_history.csv and non-zero frame indices

The `frame_time_history.csv` produced by the multiwebcam recorder uses the camera's internal frame counter as `frame_index`. This counter starts from whenever the camera session was initialized, not from the start of recording. For example, a typical CSV might show frame indices starting at 73338 even though the video file only contains 796 frames (0-795).

**The `frame_index` column is NOT a 0-based position in the video file.** To get the actual video frame position, the Python pipeline computes a per-camera rank based on `frame_time`:

```python
df = df.sort_values(by=['port', 'frame_time'])
df['derived_frame_index'] = df.groupby('port')['frame_time'].rank(method='min').astype(int) - 1
```

This assigns each row a sequential 0-based index within its camera, ordered chronologically. The `derived_frame_index` is then used with `cv2.VideoCapture.set(CAP_PROP_POS_FRAMES, ...)` to seek to the correct position in the video file.

The C++ CUDA pipeline (`pt_pipeline.cpp:load_sync_table_csv`) implements the same logic: it sorts rows by (port, frame_time), assigns per-camera ranks, and stores those as the video frame indices in the sync table.

This is legacy behavior from the original multiwebcam package and could be simplified in future recordings to use 0-based frame indices directly.

