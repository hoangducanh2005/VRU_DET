# Full Model Evaluation Comparison

This table consolidates the available evidence for the COCO-pretrained YOLO11s baseline, the YOLO11s checkpoint fine-tuned on BDD100K, and the YOLO26m checkpoint trained for 20 cumulative epochs on nuImages. All runs used an NVIDIA GeForce RTX 5060 Laptop GPU at image size 640.

| Model / training lineage | Evaluation split and protocol | Precision | Recall | F1 | AP/mAP@0.5 | mAP@0.5:0.95 | Latency per image | Evidence |
|---|---|---:|---:|---:|---:|---:|---:|---|
| YOLO11s — COCO pretrained, no fine-tuning | NavSim shard 1, 8 cameras; projected 3D GT; person + bicycle supported | 0.1291 | 0.0520 | 0.0741 | 0.0131 | 0.0037 | 3.43 ms | [NavSim evidence](task4_yolo11s/results/navsim_shard1_8cam_20261015/evidence.md) |
| YOLO11s — COCO pretrained, fine-tuned on BDD100K for 25 epochs | NavSim shard 1, 8 cameras; projected 3D GT; pedestrian + bicycle supported | 0.1689 | 0.0742 | 0.1031 | 0.0207 | 0.0040 | 3.41 ms | [NavSim evidence](task4_yolo11s_bdd100k/results/run_navsim/evidence.md) |
| YOLO26m — COCO pretrained, fine-tuned on nuImages for 20 cumulative epochs | NavSim shard 1, 8 cameras; projected 3D GT; pedestrian only | 0.1330 | 0.1011 | 0.1149 | 0.0201 | — | 7.11 ms | [NavSim evidence](task3_yolo26m/yolo26m_nuimages_20ep/results/navsim_shard1_8cam/evidence.md) |
| YOLO11s — COCO pretrained, fine-tuned on BDD100K for 25 epochs | nuImages validation; 16,445 images, 32,185 pedestrian GT instances | 0.6710 | 0.3845 | — | 0.4300 | 0.2159 | 5.42 ms | [nuImages evidence](task4_yolo11s_bdd100k/results/run_nuimage_val/evidence.md) |
| YOLO26m — COCO pretrained, fine-tuned on nuImages for 20 cumulative epochs | nuImages validation; 16,445 images, 32,185 pedestrian GT instances | 0.8211 | 0.6085 | — | 0.7098 | 0.4078 | 7.69 ms | [nuImages evidence](task3_yolo26m/yolo26m_nuimages_20ep/results/nuimages_val/evidence.md) |

## How to read the comparison

- The two nuImages rows use the same validation split, pedestrian class, image size, and batch size, so their detection metrics are the most directly comparable. YOLO26m has higher validation precision, recall, and mAP; YOLO11s has lower measured inference-stage latency.
- The NavSim rows use the same shard, timestamps, and eight cameras, but class coverage differs: the YOLO11s evaluations include pedestrian/person and bicycle, while YOLO26m is pedestrian-only. NavSim metrics are based on camera projections of 3D annotations, not native 2D labels. Treat these as run-level comparisons with that protocol difference in mind.
- NavSim latency is the measured per-image prediction latency from synchronized batches of eight camera images (preprocessing + inference + postprocessing). nuImages latency is the Ultralytics validation inference stage; it excludes preprocessing and postprocessing. Latency values across these two evaluation protocols are therefore not directly equivalent.
- Precision/recall use each evidence report's stated operating point. NavSim uses confidence 0.25 and IoU 0.50; nuImages reports Ultralytics' validation operating point. AP/mAP is confidence-ranked. A dash means that metric was not reported in that run's evidence.
