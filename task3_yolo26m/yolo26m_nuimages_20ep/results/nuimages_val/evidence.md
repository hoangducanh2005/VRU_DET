# YOLO26m nuImages validation inference evidence

- Checkpoint: `/home/anhhd/Downloads/Datasets/nuimages/runs/detect/runs/detect/runs/pedestrian/yolo26m_nuimages/weights/best.pt`
- Dataset split: `/home/anhhd/Downloads/Datasets/nuimages_yolo/val` (`images` and `labels` link to the val split)
- Images evaluated: `16,445`
- Ground-truth instances: `32,185`
- Class: `pedestrian`
- Image size / batch: `640` / `16`
- Device: `NVIDIA GeForce RTX 5060 Laptop GPU (CUDA:0)`
- Training lineage: initial 5-epoch nuImages checkpoint + 15 additional fine-tuning epochs = 20 cumulative epochs (the checkpoint metadata records the final 15-epoch stage).

## Detection metrics

| Metric | Value |
|---|---:|
| Precision (Ultralytics validation operating point) | 0.8211 |
| Recall (Ultralytics validation operating point) | 0.6085 |
| mAP at IoU 0.5 | 0.7098 |
| mAP at IoU 0.5:0.95 | 0.4078 |
| Validation inference latency per image | 7.69 ms |
| Validation elapsed time | 156.40 s |

Precision and recall use Ultralytics' validation operating point. mAP is computed over the validation confidence sweep. The separate annotated prediction pass used confidence threshold `0.25`.

## Annotated prediction output

- Images processed and saved: `16,445 / 16,445`
- Prediction elapsed time: `185.55 s`
- Annotated frames with bounding boxes, `pedestrian` labels, and confidence scores are generated locally and excluded from Git.
- Validation metrics JSON: [`validation_metrics.json`](validation_metrics.json)
- Inference configuration and timing JSON: [`inference_metrics.json`](inference_metrics.json)
- Ultralytics validation artifact files are kept locally; aggregate metrics are recorded in `validation_metrics.json`.

## Comparison with the 5-epoch checkpoint

The earlier 5-epoch checkpoint was evaluated on the same nuImages validation split at `imgsz=640`.

| Metric | 5ep checkpoint | Checkpoint from supplied path |
|---|---:|---:|
| Precision | 0.7712 | 0.8211 |
| Recall | 0.5357 | 0.6085 |
| mAP at IoU 0.5 | 0.6354 | 0.7098 |
| mAP at IoU 0.5:0.95 | 0.3456 | 0.4078 |

The supplied checkpoint scores higher on these validation metrics. Its saved training configuration records the final 15-epoch fine-tuning stage; that stage started from the 5-epoch nuImages checkpoint, for 20 cumulative training epochs.
