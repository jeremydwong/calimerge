"""Debug script: test CUDA YOLO detection on recorded workout frames."""
import cv2
from pathlib import Path
from calimerge.config import load_app_settings, load_calibration_from_toml, write_cuda_calibration_toml
from calimerge.tracking.cuda_stream_binding import CudaStreamPipeline
import tempfile

repo = Path("pyproject.toml").resolve().parent
rec_dir = repo / "recordings" / "workouts" / "20260413_145752_pushup"

# Load calibration
cal_path = sorted((repo / "recordings").glob("*/calibration.toml"))[-1]
cameras = load_calibration_from_toml(cal_path)
cuda_cal = Path(tempfile.gettempdir()) / "test_cuda_cal.toml"
write_cuda_calibration_toml(cameras, cuda_cal)

ports = sorted(cameras.keys())
w, h = cameras[ports[0]].intrinsics.resolution
print(f"Calibrated ports: {ports}, resolution: {w}x{h}")

# Load one frame per camera from the pushup recording
frames = {}
for port in ports:
    vids = sorted(rec_dir.glob(f"port_{port}*.mp4"))
    if not vids:
        print(f"WARNING: no video for port {port}")
        continue
    cap = cv2.VideoCapture(str(vids[0]))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, nframes // 2)
    ret, f = cap.read()
    cap.release()
    if ret:
        frames[port] = cv2.resize(f, (w, h))
        print(f"Port {port}: {frames[port].shape}, mean={frames[port].mean():.1f}")

pipeline = CudaStreamPipeline(
    num_cameras=len(ports), frame_width=w, frame_height=h,
    calibration_toml_path=str(cuda_cal),
    yolo_onnx_path=str(repo / "models" / "onnx" / "yolo_v10s.onnx"),
    vitpose_onnx_path=str(repo / "models" / "onnx" / "vitpose_synthpose.onnx"),
    engine_cache_dir=str(repo / "engine_cache"),
)

frame_list = [(frames[p], p) for p in ports if p in frames]
for i in range(5):
    result = pipeline.process_frame(frame_list, sync_index=i)
    print(f"frame {i}: {result.num_persons} persons, {result.processing_time_ms:.1f}ms")

pipeline.close()
