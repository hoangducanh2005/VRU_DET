# Task 4 — YOLO11s BDD100K checkpoint on NavSim (8 cameras)

- Checkpoint: `/home/anhhd/Downloads/archive/runs/detect/runs/detect/bdd100k_yolo11s/weights/best.pt` (SHA-256 `27275d0b9c7fb2698e6642e96344aba8afb1c9efbd0915c89c170accd7725ef8`)
- Device: `NVIDIA GeForce RTX 5060 Laptop GPU`; CUDA `12.8`
- Ultralytics / PyTorch: `8.4.175` / `2.11.0+cu128`
- Dataset: NavSim shard 1; `812` synchronized timestamps × 8 cameras = `6496` images
- VRU classes: `pedestrian, rider, bicycle, motorcycle`; input size `640`, confidence `0.25`, batch size `8`
- Checkpoint lineage: `yolo11s.pt` COCO-pretrained initialization, fine-tuned on BDD100K for `25` epochs; inference only in this run.

## Ground-truth evaluation

NavSim 3D pedestrian and bicycle boxes are projected into each camera. NavSim has no separate rider or motorcycle GT labels; those predictions are visualized but excluded from mAP.
Projected GT counts: pedestrian `26121`, bicycle `88`, rider `0`, motorcycle `0`.
Other NavSim GT instances ignored (3D annotation counts, before camera projection): `{"barrier": 3805, "czone_sign": 135, "generic_object": 54875, "traffic_cone": 9035, "vehicle": 17943}`.
Precision and recall are measured at confidence 0.25 and IoU 0.50. AP uses confidence-ranked predictions retained from score floor 0.001, with 101-point interpolation; mAP50-95 averages IoU thresholds 0.50 through 0.95 in 0.05 increments. The aggregate mAP averages classes with projected GT support; unsupported classes are excluded rather than treated as zero.

| Class | GT boxes | Predictions | Precision @ conf | Recall @ conf | AP50 | AP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| pedestrian | 26121 | 300491 | 0.1705 | 0.0745 | 0.0414 | 0.0080 |
| rider | 0 | 802 | N/A | N/A | N/A | N/A |
| bicycle | 88 | 32067 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| motorcycle | 0 | 1997 | N/A | N/A | N/A | N/A |

### Aggregate

| Metric | Value |
|---|---:|
| TP / FP / FN at confidence 0.25, IoU 0.50 | 1945 / 9570 / 24264 |
| Precision | 0.1689 |
| Recall | 0.0742 |
| F1 | 0.1031 |
| mAP50 | 0.0207 |
| mAP50-95 | 0.0040 |
| Classes included in mAP | pedestrian, bicycle |

## Throughput and resource benchmark

Each measured batch consists of the eight camera images from one NavSim timestamp. Batch latency is end-to-end Ultralytics predict latency for that batch (preprocess + GPU inference + postprocess); one unmeasured warm-up batch is excluded. GPU timing is synchronized before and after each measured batch.

| Metric | Measurement |
|---|---:|
| Mean batch latency | 27.26 ms |
| P95 batch latency | 29.27 ms |
| Mean latency per camera image | 3.41 ms |
| Inference throughput | 293.45 images/s |
| Inference FPS per-camera equivalent | 36.68 timestamp batches/s |
| Peak GPU memory allocated / reserved | 270.6 / 452.0 MiB |
| End-to-end runtime (including GT projection and visualization writes) | 75.49 s |
| End-to-end throughput | 86.05 images/s |
| End-to-end FPS per-camera equivalent | 10.76 timestamp batches/s |

## Deliverables

- `run_bdd_yolo11s_navsim.py`: reproducible runner; defaults target the local NavSim shard and CUDA device 0.
- `metrics.json`: configuration, per-class and aggregate quality metrics, and hardware benchmark.
- `batch_benchmark.csv`: one measured row per timestamp batch.
- `predictions.csv`: frame/timestamp/camera, BDD100K class, confidence, and image-space xyxy boxes.
- `visualized_frames/`: `812` synchronized 4×2 montages; every tile shows camera ID and every box shows class and confidence.
- `preview_montage.jpg`: first timestamp for quick review.

## Reproducibility notes

No full BDD100K download is required. Evaluation uses only the available NavSim metadata and camera images. Since NavSim stores 3D annotations rather than native 2D camera labels, metrics are projection-based and are not directly comparable to benchmark results computed from native 2D annotations.
