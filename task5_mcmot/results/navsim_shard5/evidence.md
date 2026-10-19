# Phase 5: YOLO11s BDD100K multi-camera tracking evidence

## Run configuration

- Checkpoint: `D:\GIT\VRU_Det\task4_yolo11s_bdd100k\weights\best.pt`
- Checkpoint SHA-256: `27275d0b9c7fb2698e6642e96344aba8afb1c9efbd0915c89c170accd7725ef8`
- Checkpoint training epochs from metadata: `25`
- NavSim metadata: `D:\navsim_workspace\dataset\navsim_logs\mini\2021.06.09.14.58.55_veh-35_01894_02311.pkl`
- Sequence: `834` synchronized timestamps × `8` cameras = `6672` images
- Cameras: `CAM_F0, CAM_L0, CAM_R0, CAM_L1, CAM_R1, CAM_L2, CAM_R2, CAM_B0`
- Model input: `640`; batch size: `8`; CUDA FP16: `True`
- Geometry parameters: ground z `0.000 m`, same-time cluster radius `2.50 m`, max temporal speed `8.00 m/s`

## Detection evaluation

Predictions and NavSim GT are matched one-to-one within the same camera and timestamp. Classes must agree and 2D IoU must be at least 0.50. NavSim 3D pedestrian/bicycle boxes are projected into each camera using its calibration.

| Metric | Result |
|---|---:|
| Scored GT boxes | 14925 |
| Scored tracked predictions | 4194 |
| TP / FP / FN | 987 / 3207 / 13938 |
| Precision | 0.2353 |
| Recall | 0.0661 |

## Tracking and cross-camera association

Identity GT is NavSim `track_tokens`. IDF1 uses the maximum-weight global-ID/GT-token assignment over matched camera-frame observations. ID switches count changes in assigned global ID for a GT token within each camera sequence.

| Metric | Result |
|---|---:|
| IDF1 | 0.0588 |
| IDTP / IDFP / IDFN | 562 / 3632 / 14363 |
| ID switches | 108 |
| Detection Accuracy (DetA) @ IoU 0.50 | 0.0544 |
| Association Accuracy (AssA) @ IoU 0.50 | 0.1360 |
| HOTA @ IoU 0.50 | 0.0860 |
| Cross-camera matched observation pairs | 996 |
| Cross-camera association TP / FP / FN / TN | 11 / 23 / 7 / 955 |
| Cross-camera association precision / recall / F1 | 0.3235 / 0.6111 / 0.4231 |
| Cross-camera association accuracy | 0.9699 |

Cross-camera pair labels compare detections matched to GT at the same timestamp across different cameras. Pairwise F1 is reported alongside accuracy because true-negative pairs can dominate accuracy.

## Runtime measurements

| Metric | Result |
|---|---:|
| GPU | NVIDIA GeForce RTX 4050 Laptop GPU |
| Mean detector batch latency | 81.542 ms |
| P95 detector batch latency | 184.294 ms |
| Mean latency per camera image | 10.193 ms |
| Detector throughput | 98.11 images/s |
| End-to-end runtime | 243.96 s |
| End-to-end throughput | 27.35 images/s |

Detector batch latency includes Ultralytics preprocessing, model execution, and postprocessing with CUDA synchronization. End-to-end runtime also includes per-camera tracking, projection/evaluation, CSV writing, and visualization output.

## Interpretation and outputs

HOTA is computed at IoU 0.50 over synchronized camera-frame observations; it is not an official nuScenes benchmark score. Appearance ReID is disabled. Global positions come from bottom-center image rays intersecting the configured LiDAR ground plane, then transformed by `lidar2global`; errors in calibration, ground height, or box bottoms can cause wrong merges/splits.
NavSim has no separate `rider` or `motorcycle` GT labels. Those model classes are retained in tracking outputs but are excluded from scored detection and identity metrics.

- Machine-readable metrics: `metrics.json`
- Per-camera tracks and global IDs: `tracks.csv`
- Global XY trajectories: `trajectories.csv`
- Synchronized annotated frames: `visualized_frames/`
- Annotated montage video: `mcmot_montage.mp4`
