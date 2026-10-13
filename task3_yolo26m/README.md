# Task 3 — YOLO26m checkpoints

Each checkpoint has its own launcher and results directory:

- [YOLO26m-KITTI, Task 3.1](yolo26m_kitti/README.md)
- [YOLO26m-nuImages, 5 epochs, Task 3.2](yolo26m_nuimages_5ep/README.md)

Both use the shared ByteTrack pipeline at the repository root and evaluate against NavSim GT. Each model folder stores its inference outputs and metrics separately. The committed visual output is `results/annotated.mp4` (834 frames in metadata order, 2 FPS, 1280×720); individual JPEG frames are kept locally and excluded from Git.
