# Task 4 — YOLO11s COCO baseline on NavSim

This folder contains a no-training YOLO11s baseline for the eight-camera NavSim shard. A single Ultralytics model processes one synchronized image from each camera as a batch of eight at every timestamp. The runner saves a 4×2 visualization for every timestamp, predictions, measured batch timings, GPU memory usage, and projection-based detection metrics.

## Environment

Use Python with an NVIDIA CUDA-enabled PyTorch build compatible with the installed driver. Install the pinned Ultralytics version into that environment:

```bash
python -m pip install -r task4_yolo11s/requirements-task4.txt
```

The official COCO checkpoint `yolo11s.pt` downloads automatically on the first run if it is not present. No BDD100K download or fine-tuning is needed.

## Run

From the repository root:

```bash
python task4_yolo11s/run_yolo11s_navsim.py \
  --weights yolo11s.pt \
  --device 0 \
  --imgsz 640 \
  --batch-size 8
```

The default metadata and sensor paths point to NavSim shard 1 on the workstation used for this run. Override `--metadata` and `--sensor-root` on another machine. The output directory can be changed with `--output-dir`.

## Classes and evaluation

The detector filters COCO classes `person`, `bicycle`, and `motorcycle`. For GT scoring, NavSim's 3D `pedestrian` annotations are mapped to COCO `person`, and its `bicycle` annotations are mapped to COCO `bicycle`. GT boxes are projected into each calibrated camera using the existing NavSim projection utility. This shard has no `motorcycle` GT label, so motorcycle predictions are visualized and recorded but excluded from mAP. Metrics are projection-based and are not native 2D benchmark annotations.

Precision/recall are reported at confidence 0.25 and IoU 0.50. Confidence-ranked AP retains predictions from a 0.001 score floor and uses 101-point interpolation. mAP50-95 averages IoU thresholds 0.50 through 0.95 in 0.05 increments, across GT-supported classes only. Qualitative detections and labeled evaluation are reported separately in `results/.../evidence.md`.

The completed run measured **0.0131 mAP50**, **0.0037 mAP50-95**, **291.38 images/s** inference throughput, **27.46 ms** mean and **29.01 ms** p95 batch latency, and **270.7 MiB** peak allocated GPU memory. These are projection-based NavSim metrics; the per-class breakdown and limitations are in the evidence report.

## Benchmark definition

One measured batch is the eight images for one timestamp. The runner warms up once (excluded), synchronizes CUDA around each batch, and records mean and p95 batch latency, mean per-image latency, images/second, per-camera-equivalent timestamp throughput, and peak allocated/reserved GPU memory. End-to-end timings additionally include GT projection, CSV writing, visualization, and JPEG output.

## Outputs

The default run writes to `results/navsim_shard1_8cam_20261015/`:

- `visualized_frames/`: one 4×2 synchronized montage per timestamp, labeled with camera ID, class, and confidence.
- `preview_montage.jpg`: the first synchronized timestamp.
- `predictions.csv`: timestamp, camera, COCO class, confidence, and xyxy image coordinates.
- `batch_benchmark.csv`: one measured timing row per timestamp batch.
- `metrics.json` and `evidence.md`: configuration, class metrics, benchmark measurements, and evaluation caveats.

The completed run's [evidence report](results/navsim_shard1_8cam_20261015/evidence.md) is available in the results folder. Generated inference montages are kept locally and excluded from Git.

The same evaluation setup is also available with the [BDD100K-fine-tuned YOLO11s checkpoint](../task4_yolo11s_bdd100k/README.md).

Run the same command to reproduce the outputs. The script deliberately requires CUDA and a batch size of eight to match Task 4's benchmark setup.
