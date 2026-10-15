# VRU detection and tracking evaluation

- Metadata: `/home/anhhd/navsim_workspace/dataset/downloads/openscene-v1.1/openscene-v1.1/meta_datas/mini/2021.06.28.16.29.11_veh-38_01415_01821.pkl`
- Prediction CSV: `task3_yolo26m/runs/yolo26m_nuimages_5ep_2021.06.28.16.29.11_veh-38_01415_01821/predictions.csv`
- Camera: `CAM_F0`
- IoU threshold: `0.5`
- Frames: `812` at `2.000003 FPS`

## Detection metrics

| Metric | Value |
|---|---:|
| GT boxes scored | 5273 |
| Predictions scored | 2022 |
| TP / FP / FN | 363 / 1659 / 4910 |
| Precision | 0.1795 |
| Recall | 0.0688 |
| F1 | 0.0995 |
| AP at IoU 0.5 | 0.0179 |
| Mean IoU of TP | 0.5822 |

| GT class | GT | Pred | TP | FP | FN | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pedestrian | 5273 | 2022 | 363 | 1659 | 4910 | 0.1795 | 0.0688 | 0.0995 | 0.0179 |

## Tracking metrics

- MOTA: `-0.2699`
- IDF1: `0.0444`
- IDTP / IDFP / IDFN: `162 / 1860 / 5111`
- ID switches: `127`

Only GT `pedestrian` is scored because this checkpoint has a single pedestrian
class. GT objects from every other category are ignored in detection metrics.

Ignored GT boxes from other classes: `85863`.
Projected GT boxes: `5273`.

## Inference configuration and performance

- Checkpoint: `/home/anhhd/Downloads/Datasets/nuimages/runs/detect/runs/pedestrian/yolo26m_nuimages/weights/best.pt`
- Input: `CAM_F0` only; NavSim metadata order; image size `640`; confidence threshold `0.25`
- Tracker: `ByteTrack`
- Predictions: `2022` across `812` frames
- Inference latency (mean / p50 / p95): `11.67 / 8.58 / 9.71 ms`
- End-to-end latency (mean / p50 / p95): `19.84 / 13.34 / 14.89 ms`
- Effective processing speed: `41.45 FPS`
- Annotated frames: `visualized_frames/` (`pedestrian` label, confidence, and track ID)
