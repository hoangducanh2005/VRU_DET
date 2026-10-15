# YOLO26m nuImages — 20 cumulative epochs

The final checkpoint was initialized from the earlier nuImages checkpoint trained for 5 epochs, then fine-tuned for 15 additional epochs: **20 cumulative training epochs**. The saved `train_args.epochs=15` describes that final fine-tuning stage. Checkpoint SHA-256: `431250c268cc93266c22b97f9f5fdb26fb06c1295f8f87f07bbec03d9784e2d2`.

- Final checkpoint: [`weights/best.pt`](weights/best.pt)
- NavSim shard 1, all 8 cameras: [`results/navsim_shard1_8cam/evidence.md`](results/navsim_shard1_8cam/evidence.md)
- Synchronized 8-camera video: [`results/navsim_shard1_8cam/all_cameras_montage.mp4`](results/navsim_shard1_8cam/all_cameras_montage.mp4)
- Predictions and full metrics: `results/navsim_shard1_8cam/predictions.csv`, `results/navsim_shard1_8cam/metrics.json`
- nuImages validation evidence and metrics: [`results/nuimages_val/evidence.md`](results/nuimages_val/evidence.md)

Generated annotated frames and montage previews are kept locally and excluded from Git.

To reproduce the 8-camera NavSim run from the repository root with the `ai_env` environment:

```bash
python task3_yolo26m/yolo26m_nuimages_20ep/run_navsim_8cam.py \
  --weights task3_yolo26m/yolo26m_nuimages_20ep/weights/best.pt
```

The script expects the NavSim shard and metadata at their configured local paths. The 812 individual JPEG montages are generated under `results/navsim_shard1_8cam/montage_frames/`; the MP4 contains the full sequence.
