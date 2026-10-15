# VRU detection and tracking evaluation

- Metadata: `/home/anhhd/navsim_workspace/dataset/downloads/openscene-v1.1/openscene-v1.1/meta_datas/mini/2021.06.28.16.29.11_veh-38_01415_01821.pkl`
- Prediction CSV: `task3_yolo26m/yolo26m_nuimages_20ep/results/predictions.csv`
- Camera: `CAM_F0`
- IoU threshold: `0.5`
- Frames: `812` at `2.000003 FPS`
- Image size / confidence threshold: `640` / `0.25`
- Tracker: `ByteTrack`
- Training lineage: initial 5-epoch nuImages checkpoint + 15 additional fine-tuning epochs = 20 cumulative epochs.

## Detection metrics

| Metric | Value |
|---|---:|
| GT boxes scored | 5273 |
| Predictions scored | 2024 |
| TP / FP / FN | 459 / 1565 / 4814 |
| Precision | 0.2268 |
| Recall | 0.0870 |
| F1 | 0.1258 |
| AP at IoU 0.5 | 0.0273 |
| Mean IoU of TP | 0.5758 |

| GT class | GT | Pred | TP | FP | FN | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pedestrian | 5273 | 2024 | 459 | 1565 | 4814 | 0.2268 | 0.0870 | 0.1258 | 0.0273 |

## Tracking metrics

- MOTA: `-0.2359`
- IDF1: `0.0672`
- IDTP / IDFP / IDFN: `245 / 1779 / 5028`
- ID switches: `138`

Only GT `pedestrian` is scored because this checkpoint has a single pedestrian
class. GT objects from every other category are ignored in detection metrics.

Ignored GT boxes from other classes: `85863`.
Projected GT boxes: `5273`.

## Inference performance

- Detections: `2024`
- Inference latency (mean / p50 / p95): `9.54 / 8.21 / 8.88 ms`
- End-to-end latency (mean / p50 / p95): `17.55 / 12.82 / 13.83 ms`
- Effective processing speed: `46.27 FPS`
- Annotated frames: `visualized_frames/` (`pedestrian` label, confidence, and track ID)

## Comparison with the 5-epoch checkpoint

Both runs use this same shard, camera, image size, confidence threshold, tracker, and evaluator.

| Metric | 5ep checkpoint | Checkpoint from supplied path |
|---|---:|---:|
| Predictions | 2022 | 2024 |
| Precision | 0.1795 | 0.2268 |
| Recall | 0.0688 | 0.0870 |
| F1 | 0.0995 | 0.1258 |
| AP at IoU 0.5 | 0.0179 | 0.0273 |
| Mean IoU of TP | 0.5822 | 0.5758 |
| IDF1 | 0.0444 | 0.0672 |
| MOTA | -0.2699 | -0.2359 |

The 20-epoch model improves precision, recall, F1, AP@0.5, IDF1, and MOTA on this shard; mean IoU of matched detections is slightly lower. This is a single-shard comparison. Training used the earlier 5-epoch nuImages checkpoint followed by 15 additional fine-tuning epochs; the saved configuration records the latter stage.
