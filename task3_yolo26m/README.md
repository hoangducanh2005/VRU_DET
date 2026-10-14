# Task 3 — YOLO26m experiments

Task 3 contains isolated YOLO26m experiments. Shared inference and evaluation utilities remain at the repository root; each model run keeps its own launcher, evidence, and outputs.

## Experiments

| Experiment | Description | Results |
|---|---|---|
| [YOLO26m-KITTI](yolo26m_kitti/README.md) | KITTI pedestrian/cyclist checkpoint evaluated on NavSim CAM_F0. | [Evidence](yolo26m_kitti/results/evidence.md) |
| [YOLO26m nuImages, 5 epochs](yolo26m_nuimages_5ep/README.md) | Earlier single-class pedestrian checkpoint with NavSim CAM_F0 inference and tracking artifacts. | [Evidence](yolo26m_nuimages_5ep/results/evidence.md) |
| [YOLO26m nuImages, 20 cumulative epochs](yolo26m_nuimages_20ep/README.md) | A 5-epoch checkpoint fine-tuned for 15 additional epochs; includes NuImages validation and all-camera NavSim inference. | [NavSim 8-camera evidence](yolo26m_nuimages_20ep/results/navsim_shard1_8cam/evidence.md) · [NuImages validation evidence](yolo26m_nuimages_20ep/results/nuimages_val/evidence.md) |

The 20-epoch checkpoint run evaluates all eight NavSim cameras at 812 synchronized timestamps. Its deliverables include a video montage, per-camera and aggregate detection metrics, and CSV predictions. This run uses a direct detector without temporal tracking. The separate KITTI and 5-epoch experiments have their own tracking artifacts; metrics across different checkpoints and protocols should be compared with care.

The NuImages checkpoint detects only `pedestrian`. Its training history is 5 initial epochs plus 15 fine-tuning epochs; the checkpoint's `train_args.epochs=15` records the latter stage.
