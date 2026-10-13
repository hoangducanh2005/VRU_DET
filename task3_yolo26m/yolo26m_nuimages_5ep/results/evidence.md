# VRU detection and tracking evaluation

- Metadata: `D:\navsim_workspace\dataset\navsim_logs\mini\2021.06.09.14.58.55_veh-35_01894_02311.pkl`
- Prediction CSV: `D:\GIT\VRU_Det\task3_yolo26m\yolo26m_nuimages_5ep\results\tracks.csv`
- Camera: `CAM_F0`
- IoU threshold: `0.5`
- Frames: `834` at `1.999996 FPS`

## Detection metrics

| Metric | Value |
|---|---:|
| GT boxes scored | 2995 |
| Predictions scored | 3928 |
| TP / FP / FN | 315 / 3613 / 2680 |
| Precision | 0.0802 |
| Recall | 0.1052 |
| F1 | 0.0910 |
| AP at IoU 0.5 | 0.0171 |
| Mean IoU of TP | 0.6097 |

| GT class | GT | Pred | TP | FP | FN | Precision | Recall | F1 | AP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| pedestrian | 2995 | 3928 | 315 | 3613 | 2680 | 0.0802 | 0.1052 | 0.0910 | 0.0171 |

## Tracking metrics

- MOTA: `-1.1265`
- IDF1: `0.0399`
- IDTP / IDFP / IDFN: `138 / 3790 / 2857`
- ID switches: `76`

Only GT `pedestrian` and `bicycle` are scored. The shard has no separate
`rider` or `motorcycle` GT label; unsupported prediction classes are ignored
and reported in the JSON rather than counted as false positives.

Ignored GT boxes from other classes: `80525`.
Projected GT boxes: `2995`.
