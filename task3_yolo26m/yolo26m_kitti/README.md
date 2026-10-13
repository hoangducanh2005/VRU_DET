# YOLO26m-KITTI — Task 3.1

Model run code and artifacts are isolated in this folder. The shared tracking pipeline and evaluators remain at the repository root.

Launch inference and both evaluations from repository root:

```powershell
.\task3_yolo26m\yolo26m_kitti\run.ps1
```

The launcher uses `session/yolo26m-kitti/best.pt`, processes NavSim CAM_F0 in metadata sequence, and saves the annotated video, tracking CSV, latency/FPS, projected-GT evidence, and identity metrics. To download the KITTI model, run `download_weight.ps1`. Results are under `results/`; individual JPEG frames are local-only and excluded from Git. `annotated.mp4` contains 834 frames at 2 FPS and 1280×720.
