# Task 2 — VRU detection and tracking

Task 1 is paused at the CSP checkpoint download. This task uses the separate
[YOLO26n-BDD100K](https://github.com/lochuan/yolo26n-bdd100k) repository and
does not modify the model architecture.

## Requirement 1 — do this part locally

The YOLO26n-BDD100K repository already provides an FP32 checkpoint, so this
path avoids the Baidu download used by Task 1:

```powershell
cd D:\GIT\VRU_DET
New-Item -ItemType Directory -Force .\session | Out-Null
git clone https://github.com/lochuan/yolo26n-bdd100k.git .\session\yolo26n-bdd100k

# Install the PyTorch build appropriate for your machine first if needed.
cd D:\GIT\VRU_DET
.\.venv\Scripts\python.exe -m pip install -r requirements-task2.txt
```

For the first trial use:

```text
D:\GIT\VRU_DET\session\yolo26n-bdd100k\weights\yolo26n-bdd100k-fp32.pt
```

The repository also contains fixed-shape FP32/INT8 ONNX exports. Do not start
with INT8 while validating tracking; switch to ONNX after the FP32 pipeline is
working.

## Requirements 2–7 — run the pipeline

The NavSim mini dataset is image-only. The runner processes one camera folder
as a frame sequence ordered by file timestamp, writes an annotated MP4, and emits one CSV row per
detection:

For image folders, `--fps 0` (the default) infers the output-video FPS from
the file timestamps. Use an explicit positive `--fps` only when the desired
playback speed is different from the recorded timeline.

For NavSim, prefer the metadata order because file mtimes have only one-second
resolution and hash filenames are not temporal:

```powershell
  --navsim-metadata D:\navsim_workspace\dataset\navsim_logs\mini\2021.06.09.14.58.55_veh-35_01894_02311.pkl `
  --camera-name CAM_F0
```

```powershell
cd D:\GIT\VRU_DET
python task2_vru_tracking.py `
  --model D:\GIT\VRU_DET\session\yolo26n-bdd100k\weights\yolo26n-bdd100k-fp32.pt `
  --input D:\navsim_workspace\dataset\sensor_blobs\mini\2021.06.09.14.58.55_veh-35_01894_02311\CAM_F0 `
  --image-sort mtime `
  --device 0 `
  --imgsz 640
```

Outputs:

- `output/task2_vru_tracking.mp4`: bounding box, class, confidence, track ID.
- `output/task2_tracks.csv`: `frame_id, track_id, class, confidence, bbox`.
- `output/task2_metrics.json`: mean/p50/p95 inference latency, end-to-end
  latency, and effective FPS.

## Kiểm tra từng frame

Video mặc định phát liên tục nên khó kiểm tra ByteTrack. Dùng `--step` để
hiển thị một frame rồi chờ phím bất kỳ trước khi chuyển sang frame tiếp theo;
nhấn `q` để dừng. Nên giới hạn số frame lúc debug:

```powershell
.\.venv\Scripts\python.exe .\task2_vru_tracking.py `
  --model D:\GIT\VRU_DET\session\yolo26n-bdd100k\weights\yolo26n-bdd100k-fp32.pt `
  --input D:\navsim_workspace\dataset\sensor_blobs\mini\2021.06.09.14.58.55_veh-35_01894_02311\CAM_F0 `
  --device 0 `
  --imgsz 640 `
  --show `
  --step `
  --max-frames 20 `
  --save-frames-dir output\debug_frames
```

Nếu chỉ muốn phát chậm, dùng `--show --delay-ms 500` thay cho `--step`.
Thời gian chờ của chế độ debug không được tính vào end-to-end latency.

The default tracker is ByteTrack. To compare BoT-SORT, run the same command
with `--tracker botsort`. The tracker is selected by configuration; no model
code changes are needed. An ONNX model can be tested by changing only
`--model`, for example:

```powershell
python task2_vru_tracking.py `
  --model D:\GIT\VRU_DET\session\yolo26n-bdd100k\models\yolo26n-bdd7-fp32-384x640.onnx `
  --input D:\navsim_workspace\dataset\sensor_blobs\mini\2021.06.09.14.58.55_veh-35_01894_02311\CAM_F0 `
  --imgsz 384 640 `
  --device cpu
```

Because the NavSim filenames are hashes rather than timestamps, a camera
folder must be ordered by file modification timestamp; this is now the default
(`--image-sort mtime`). `--image-sort name` is available only for comparison
and should not be used for ByteTrack on this shard. The timestamp is preserved
at one-second resolution, so frames sharing the same second use the hash name
as a deterministic tie-breaker. For highest-confidence temporal track IDs,
provide a video recorded from the camera.

## GT-based VRU evaluation

After running with the NavSim metadata order, evaluate only GT
`pedestrian`/`bicycle` (the local GT has no separate `rider`/`motorcycle` label):

```powershell
.\.venv\Scripts\python.exe .\evaluate_vru_metrics.py `
  --pred-csv output\task2_cam_f0_canonical.csv `
  --metadata D:\navsim_workspace\dataset\navsim_logs\mini\2021.06.09.14.58.55_veh-35_01894_02311.pkl `
  --sensor-root D:\navsim_workspace\dataset\sensor_blobs\mini `
  --camera-name CAM_F0 `
  --iou-threshold 0.5
```

The evaluator projects 3D GT boxes into the camera, matches predictions by
class and IoU, and writes detection metrics plus MOTA, IDF1, and ID switches
to `output/task2_vru_evaluation.json` and `.md`.

## Scope

- Required classes are resolved by name: `person`, `rider`, `bicycle`,
  `motorcycle`.
- The code uses Ultralytics' public tracking interface, keeping Python/PyTorch
  setup separate from the model and leaving a clean path to ONNX/TensorRT.
- No training is performed.
