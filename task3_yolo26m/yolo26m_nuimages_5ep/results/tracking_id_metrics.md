# ByteTrack identity-only evaluation

- Prediction CSV: `D:\GIT\VRU_Det\task3_yolo26m\yolo26m_nuimages_5ep\results\tracks.csv`
- NavSim metadata: `D:\navsim_workspace\dataset\navsim_logs\mini\2021.06.09.14.58.55_veh-35_01894_02311.pkl`
- Camera: `CAM_F0`
- Association IoU threshold: `0.5`
- GT identity source: `anns.track_tokens`
- Prediction identity source: ByteTrack `track_id`

This report evaluates track identity only. Detection AP/precision/recall are intentionally excluded.
A prediction with `track_id=-1` is treated as untracked and cannot contribute IDTP.

| Metric | Value |
|---|---:|
| ID Precision | 0.0351 |
| ID Recall | 0.0461 |
| IDF1 | 0.0399 |
| MOTA | -1.1265 |
| IDTP / IDFP / IDFN | 138 / 3790 / 2857 |
| ID switches | 76 |
| Untracked prediction detections | 502 |
| Untracked GT matches | 45 |
| Track-ID coverage | 0.8722 |
| Unique GT track_tokens | 312 |
| Unique predicted track IDs | 139 |
| Matched GT track_tokens | 88 |
| Matched predicted track IDs | 60 |
