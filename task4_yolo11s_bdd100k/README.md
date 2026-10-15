# Task 4 — YOLO11s BDD100K checkpoint on NavSim

This is a separate 8-camera inference run using the provided YOLO11s checkpoint, initialized from COCO-pretrained YOLO11s and fine-tuned on BDD100K for 25 epochs. It uses the same NavSim shard, timestamp batches, 640-pixel inference size, CUDA device, confidence policy, projection-based evaluator, benchmark fields, and synchronized montage deliverables as the COCO-pretrained Task 4 baseline. This run performs inference only.

## Checkpoint

Default checkpoint: `/home/anhhd/Downloads/archive/runs/detect/runs/detect/bdd100k_yolo11s/weights/best.pt`. The checkpoint exposes these BDD100K classes:

```text
pedestrian, rider, car, bus, truck, bicycle, motorcycle, traffic light, traffic sign, train
```

The runner filters and visualizes the four VRU classes `pedestrian`, `rider`, `bicycle`, and `motorcycle`. NavSim shard 1 has 3D ground truth for `pedestrian` and `bicycle`, but no separate `rider` or `motorcycle` labels. Predictions for unsupported GT classes remain in the visualizations and CSV, while mAP excludes those classes.

## Run

From the repository root, using the CUDA environment with Ultralytics installed:

```bash
python task4_yolo11s_bdd100k/run_bdd_yolo11s_navsim.py \
  --weights /home/anhhd/Downloads/archive/runs/detect/runs/detect/bdd100k_yolo11s/weights/best.pt \
  --device 0 \
  --imgsz 640 \
  --batch-size 8
```

The checkpoint's saved training configuration reports `model=yolo11s.pt`, `pretrained=true`, 25 BDD100K fine-tuning epochs, and training `imgsz=768`. This Task 4 run uses the requested inference `imgsz=640` for a consistent comparison with the COCO baseline.

## NavSim run

NavSim output is kept separately under `results/run_navsim/`:

- One 4×2 synchronized annotated montage for each of the 812 timestamps.
- A preview montage, per-detection predictions CSV, per-batch benchmark CSV, metrics JSON, and evidence Markdown report.
- Precision, recall, mAP50, and mAP50-95 using confidence-ranked detections and projected NavSim 3D ground truth.
- Mean and p95 latency per synchronized batch, throughput, per-camera-equivalent FPS, peak GPU memory, and end-to-end runtime.

Precision/recall use confidence 0.25 at IoU 0.50. AP retains predictions from confidence floor 0.001. See the generated evidence report for exact measurements and the class-level breakdown.

See the [NavSim evidence report](results/run_navsim/evidence.md). Generated inference montages are kept locally and excluded from Git.

## NuImages validation run

Run the same checkpoint against the pedestrian-labeled NuImages validation split:

```bash
python task4_yolo11s_bdd100k/run_bdd_yolo11s_nuimage_val.py \
  --weights /home/anhhd/Downloads/archive/runs/detect/runs/detect/bdd100k_yolo11s/weights/best.pt \
  --dataset /home/anhhd/Downloads/Datasets/nuimages_yolo \
  --device 0 \
  --imgsz 640 \
  --batch-size 16
```

This evaluates the complete val split with only the `pedestrian` output class, then writes a separate confidence-0.25 prediction/visualization pass. Outputs are under `results/run_nuimage_val/`; see its generated [evidence report](results/run_nuimage_val/evidence.md).
