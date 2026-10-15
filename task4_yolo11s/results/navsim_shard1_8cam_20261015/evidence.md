# Task 4 — YOLO11s COCO baseline on NavSim (8 cameras)

- Checkpoint: `/home/anhhd/GIT/VRU_DET/yolo11s.pt` (SHA-256 `85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5`)
- Device: `NVIDIA GeForce RTX 5060 Laptop GPU`; CUDA `12.8`
- Ultralytics / PyTorch: `8.4.175` / `2.11.0+cu128`
- Dataset: NavSim shard 1; `812` synchronized timestamps × 8 cameras = `6496` images
- Classes: COCO `person`, `bicycle`, and `motorcycle`; input size `640`, confidence `0.25`, batch size `8`
- Training: none; official COCO-pretrained `yolo11s.pt` checkpoint.

## Ground-truth evaluation

NavSim 3D boxes projected into each camera; pedestrian maps to COCO person, bicycle maps to bicycle. The shard has no motorcycle GT; motorcycle is reported as unsupported and excluded from mAP.
Projected GT counts: pedestrian/person `26121`, bicycle `88`, motorcycle `0`.
Other NavSim GT instances ignored (3D annotation counts, before camera projection): `{"barrier": 3805, "czone_sign": 135, "generic_object": 54875, "traffic_cone": 9035, "vehicle": 17943}`.
Precision and recall are measured at confidence 0.25 and IoU 0.50. AP uses confidence-ranked predictions retained from score floor 0.001, with 101-point interpolation; mAP50-95 averages IoU thresholds 0.50 through 0.95 in 0.05 increments. The aggregate mAP averages classes with projected GT support; unsupported classes are excluded rather than treated as zero.

| Class | GT boxes | Predictions | Precision @ conf | Recall @ conf | AP50 | AP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| person | 26121 | 311293 | 0.1297 | 0.0521 | 0.0262 | 0.0073 |
| bicycle | 88 | 8191 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| motorcycle | 0 | 866 | N/A | N/A | N/A | N/A |

### Aggregate

| Metric | Value |
|---|---:|
| TP / FP / FN at confidence 0.25, IoU 0.50 | 1362 / 9190 / 24847 |
| Precision | 0.1291 |
| Recall | 0.0520 |
| F1 | 0.0741 |
| mAP50 | 0.0131 |
| mAP50-95 | 0.0037 |
| Classes included in mAP | person, bicycle |

## Throughput and resource benchmark

Each measured batch consists of the eight camera images from one NavSim timestamp. Batch latency is end-to-end Ultralytics predict latency for that batch (preprocess + GPU inference + postprocess); one unmeasured warm-up batch is excluded. GPU timing is synchronized before and after each measured batch.

| Metric | Measurement |
|---|---:|
| Mean batch latency | 27.46 ms |
| P95 batch latency | 29.01 ms |
| Mean latency per camera image | 3.43 ms |
| Inference throughput | 291.38 images/s |
| Inference FPS per-camera equivalent | 36.42 timestamp batches/s |
| Peak GPU memory allocated / reserved | 270.7 / 452.0 MiB |
| End-to-end runtime (including GT projection and visualization writes) | 73.43 s |
| End-to-end throughput | 88.47 images/s |
| End-to-end FPS per-camera equivalent | 11.06 timestamp batches/s |

## Deliverables

- `run_yolo11s_navsim.py`: reproducible runner; defaults target the local NavSim shard and CUDA device 0.
- `metrics.json`: configuration, per-class and aggregate quality metrics, and hardware benchmark.
- `batch_benchmark.csv`: one measured row per timestamp batch.
- `predictions.csv`: frame/timestamp/camera, COCO class, confidence, and image-space xyxy boxes.
- `visualized_frames/`: `812` synchronized 4×2 montages; every tile shows camera ID and every box shows class and confidence.
- `preview_montage.jpg`: first timestamp for quick review.

## Reproducibility notes

No full BDD100K download is required. Evaluation uses only the available NavSim metadata and camera images. Since NavSim stores 3D annotations rather than native 2D camera labels, metrics are projection-based and are not directly comparable to benchmark results computed from native 2D annotations.
