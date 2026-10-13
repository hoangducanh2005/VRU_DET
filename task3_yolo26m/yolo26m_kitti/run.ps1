$ErrorActionPreference = "Stop"
$TaskRoot = Split-Path -Parent $PSScriptRoot
$RepoRoot = Split-Path -Parent $TaskRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Model = Join-Path $RepoRoot "session\yolo26m-kitti\best.pt"
$Input = "D:\navsim_workspace\dataset\sensor_blobs\mini\2021.06.09.14.58.55_veh-35_01894_02311\CAM_F0"
$Metadata = "D:\navsim_workspace\dataset\navsim_logs\mini\2021.06.09.14.58.55_veh-35_01894_02311.pkl"
$Results = Join-Path $PSScriptRoot "results"
$Frames = Join-Path $Results "frames"
if (-not (Test-Path -LiteralPath $Model -PathType Leaf)) { throw "Missing YOLO26m KITTI checkpoint: $Model" }
& $Python (Join-Path $RepoRoot "task2_vru_tracking.py") --model $Model --input $Input --navsim-metadata $Metadata --camera-name CAM_F0 --classes "Pedestrian,Cyclist" --csv-format xyxy --no-video --save-frames-dir $Frames --device 0 --imgsz 640 --output-csv (Join-Path $Results "tracks.csv") --metrics-json (Join-Path $Results "metrics.json")
if ($LASTEXITCODE -ne 0) { throw "YOLO26m-KITTI inference failed with exit code $LASTEXITCODE" }
& $Python (Join-Path $RepoRoot "evaluate_vru_metrics.py") --pred-csv (Join-Path $Results "tracks.csv") --metadata $Metadata --sensor-root "D:\navsim_workspace\dataset\sensor_blobs\mini" --camera-name CAM_F0 --class-map "Pedestrian:pedestrian,Cyclist:bicycle" --output-json (Join-Path $Results "gt_evaluation.json") --output-report (Join-Path $Results "evidence.md")
if ($LASTEXITCODE -ne 0) { throw "YOLO26m-KITTI GT evaluation failed with exit code $LASTEXITCODE" }
& $Python (Join-Path $RepoRoot "evaluate_tracking_ids.py") --pred-csv (Join-Path $Results "tracks.csv") --metadata $Metadata --sensor-root "D:\navsim_workspace\dataset\sensor_blobs\mini" --camera-name CAM_F0 --class-map "Pedestrian:pedestrian,Cyclist:bicycle" --output-json (Join-Path $Results "tracking_id_metrics.json") --output-report (Join-Path $Results "tracking_id_metrics.md")
if ($LASTEXITCODE -ne 0) { throw "YOLO26m-KITTI identity evaluation failed with exit code $LASTEXITCODE" }
