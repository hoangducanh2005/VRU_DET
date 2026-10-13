$ErrorActionPreference = "Stop"
$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"
$Model = Join-Path $RepoRoot "session\yolo26n-bdd100k\weights\yolo26n-bdd100k-fp32.pt"
$Input = "D:\navsim_workspace\dataset\sensor_blobs\mini\2021.06.09.14.58.55_veh-35_01894_02311\CAM_F0"
$Metadata = "D:\navsim_workspace\dataset\navsim_logs\mini\2021.06.09.14.58.55_veh-35_01894_02311.pkl"
$Results = Join-Path $PSScriptRoot "results"

& $Python (Join-Path $RepoRoot "task2_vru_tracking.py") `
  --model $Model `
  --input $Input `
  --navsim-metadata $Metadata `
  --camera-name CAM_F0 `
  --classes "person,rider,bicycle,motorcycle" `
  --device 0 `
  --imgsz 640 `
  --output-video (Join-Path $Results "annotated.mp4") `
  --output-csv (Join-Path $Results "tracks.csv") `
  --metrics-json (Join-Path $Results "metrics.json")
