# YOLO26m NavSim 8-camera evaluation

- Metadata: `/home/anhhd/navsim_workspace/dataset/downloads/openscene-v1.1/openscene-v1.1/meta_datas/mini/2021.06.28.16.29.11_veh-38_01415_01821.pkl`
- Sensor root: `/home/anhhd/navsim_workspace/dataset/downloads/openscene-v1.1/openscene_sensor_mini_camera/openscene-v1.1/sensor_blobs/mini`
- Checkpoint: `/home/anhhd/Downloads/Datasets/nuimages/runs/detect/runs/detect/runs/pedestrian/yolo26m_nuimages/weights/best.pt`
- Cameras: `CAM_F0, CAM_L0, CAM_R0, CAM_L1, CAM_R1, CAM_L2, CAM_R2, CAM_B0`
- Timestamps: `812`; camera images: `6496`
- Image size / confidence threshold: `640` / `0.25`
- IoU threshold: `0.5`
- Projected pedestrian GT boxes across camera views: `26121`
- Training lineage: initial 5-epoch nuImages checkpoint + 15 additional fine-tuning epochs = 20 cumulative epochs (the checkpoint metadata records the final 15-epoch stage).

## Overall detection metrics

| Metric | Value |
|---|---:|
| Predictions | 19852 |
| TP / FP / FN | 2641 / 17211 / 23480 |
| Precision | 0.1330 |
| Recall | 0.1011 |
| F1 | 0.1149 |
| AP at IoU 0.5 | 0.0201 |
| Mean IoU of TP | 0.5822 |

Metrics pool confidence-ranked detections across all camera-view/timestamp pairs. GT 3D pedestrian boxes are projected into each camera using NavSim calibration. Other GT classes are ignored; predictions are pedestrian-only. This run reports detector metrics without a temporal tracker.

## Per-camera detection metrics

| Camera | GT | Pred | TP | FP | FN | Precision | Recall | F1 | AP@0.5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| CAM_F0 | 5273 | 4154 | 765 | 3389 | 4508 | 0.1842 | 0.1451 | 0.1623 | 0.0402 |
| CAM_L0 | 7277 | 5121 | 617 | 4504 | 6660 | 0.1205 | 0.0848 | 0.0995 | 0.0186 |
| CAM_R0 | 2575 | 2913 | 525 | 2388 | 2050 | 0.1802 | 0.2039 | 0.1913 | 0.0429 |
| CAM_L1 | 2733 | 1867 | 109 | 1758 | 2624 | 0.0584 | 0.0399 | 0.0474 | 0.0070 |
| CAM_R1 | 1034 | 1485 | 161 | 1324 | 873 | 0.1084 | 0.1557 | 0.1278 | 0.0255 |
| CAM_L2 | 3105 | 1614 | 201 | 1413 | 2904 | 0.1245 | 0.0647 | 0.0852 | 0.0119 |
| CAM_R2 | 1518 | 1465 | 161 | 1304 | 1357 | 0.1099 | 0.1061 | 0.1079 | 0.0201 |
| CAM_B0 | 2606 | 1233 | 102 | 1131 | 2504 | 0.0827 | 0.0391 | 0.0531 | 0.0039 |

## Runtime and visual outputs

- Inference mean per image: `7.11 ms`
- Total elapsed time including visualization and metrics: `120.60 s`
- Effective camera-image throughput: `53.86 images/s`
- Synchronized montage frames: `812`; each contains all 8 cameras at the same metadata timestamp in a 4×2 layout.
- Montage JPEGs: `montage_frames/` (local full-resolution frame sequence; not committed separately)
- Montage video: `all_cameras_montage.mp4` (H.264, contains all 812 timestamps)
- Per-detection CSV with camera: `predictions.csv`
- Full metrics JSON: `metrics.json`

The checkpoint metadata records the final fine-tuning stage (`train_args.epochs=15`). It was initialized from the earlier 5-epoch nuImages checkpoint, for 20 cumulative training epochs.
