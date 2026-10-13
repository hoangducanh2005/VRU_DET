# VRU detection and tracking evaluation

- Metadata: `D:\navsim_workspace\dataset\navsim_logs\mini\2021.06.09.14.58.55_veh-35_01894_02311.pkl`
- Prediction CSV: `task3_yolo26m\yolo26m_kitti\results\tracks.csv`
- Camera: `CAM_F0`
- IoU threshold: `0.5`
- Frames: `834` at `1.999996 FPS`

## Detection metrics

| Metric | Value |
|---|---:|
| GT boxes scored | 2995 |
| Predictions scored | 253 |
| TP / FP / FN | 65 / 188 / 2930 |
| Precision | 0.2569 |
| Recall | 0.0217 |
| F1 | 0.0400 |
| AP at IoU 0.5 | 0.0083 |
| Mean IoU of TP | 0.5882 |

| GT class | GT | Pred | TP | FP | FN | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pedestrian | 2995 | 253 | 65 | 188 | 2930 | 0.2569 | 0.0217 | 0.0400 | 0.0083 |

## Tracking metrics

- MOTA: `-0.0414`
- IDF1: `0.0142`
- IDTP / IDFP / IDFN: `23 / 230 / 2972`
- ID switches: `1`

Only GT `pedestrian` and `bicycle` are scored. The shard has no separate
`rider` or `motorcycle` GT label; unsupported prediction classes are ignored
and reported in the JSON rather than counted as false positives.

Ignored GT boxes from other classes: `80525`.
Projected GT boxes: `2995`.
