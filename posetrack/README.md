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

