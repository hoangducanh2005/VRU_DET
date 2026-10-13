# YOLO26m nuImages, 5 epochs — Task 3.2

Checkpoint copied from `C:\Users\Admin\Downloads\runs\detect\runs\pedestrian\yolo26m_nuimages\weights\best.pt` to `session/yolo26m-nuimages-5ep/best.pt`. The archived `args.yaml` confirms `epochs: 5`, and checkpoint metadata confirms the only class is `pedestrian`.

Run from repository root:

```powershell
.\task3_yolo26m\yolo26m_nuimages_5ep\run.ps1
```

The launcher uses the same NavSim shard, CAM_F0, 640px inference, GPU 0, ByteTrack, and projected-LiDAR GT evaluation as Task 3.1. Outputs are stored in this directory's `results/`: `annotated.mp4`, tracks CSV, latency/FPS JSON, detection evidence, and ID-only metrics comparing ByteTrack `track_id` to `anns.track_tokens`. The video contains 834 frames at 2 FPS and 1280×720; individual JPEG frames are local-only and excluded from Git.
