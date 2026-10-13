# ByteTrack identity-only evaluation

- Prediction CSV: `D:\GIT\VRU_Det\task3_yolo26m\yolo26m_kitti\results\tracks.csv`
- NavSim metadata: `D:\navsim_workspace\dataset\navsim_logs\mini\2021.06.09.14.58.55_veh-35_01894_02311.pkl`
- Camera: `CAM_F0`
- Association IoU threshold: `0.5`
- GT identity source: `anns.track_tokens`
- Prediction identity source: ByteTrack `track_id`

This report evaluates track identity only. Detection AP/precision/recall are intentionally excluded.
A prediction with `track_id=-1` is treated as untracked and cannot contribute IDTP.

| Metric | Value |
|---|---:|
| ID Precision | 0.0898 |
| ID Recall | 0.0077 |
| IDF1 | 0.0141 |
| MOTA | -0.0423 |
| IDTP / IDFP / IDFN | 23 / 233 / 2981 |
| ID switches | 1 |
| Untracked prediction detections | 146 |
| Untracked GT matches | 39 |
| Track-ID coverage | 0.4297 |
| Unique GT track_tokens | 318 |
| Unique predicted track IDs | 17 |
| Matched GT track_tokens | 12 |
| Matched predicted track IDs | 11 |
