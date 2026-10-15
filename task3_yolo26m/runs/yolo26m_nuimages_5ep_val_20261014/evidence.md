# YOLO26m nuImages validation inference evidence

- Weights: `/home/anhhd/Downloads/Datasets/nuimages/runs/detect/runs/pedestrian/yolo26m_nuimages/weights/best.pt`
- Dataset: `/home/anhhd/Downloads/Datasets/nuimages_yolo`
- Split: `val`
- Class: `pedestrian`
- Images evaluated: `16,445`
- Ground-truth instances: `32,185`
- Image size: `640`
- Device: `NVIDIA GeForce RTX 5060 Laptop GPU (CUDA:0)`

## Detection metrics

| Metric | Value |
|---|---:|
| Precision (Ultralytics validation operating point) | 0.7712 |
| Recall (Ultralytics validation operating point) | 0.5357 |
| mAP at IoU 0.5 | 0.6354 |
| mAP at IoU 0.5:0.95 | 0.3456 |
| Validation inference latency per image | 7.65 ms |
| Validation elapsed time | 159.57 s |

Precision and recall are reported at Ultralytics' validation operating point. mAP is computed from the validation confidence sweep. The separate annotated prediction pass used confidence threshold `0.25`.

## Annotated prediction output

- Images processed and saved: `16,445 / 16,445`
- Prediction threshold: `0.25`
- Prediction pass elapsed time: `188.21 s`
- Annotated images with bounding boxes, `pedestrian` labels, and confidence scores: [`visualized_frames/`](visualized_frames/)
- Full validation metrics JSON: [`validation_metrics.json`](validation_metrics.json)
- Inference configuration and timing JSON: [`inference_metrics.json`](inference_metrics.json)
- Ultralytics validation artifacts and predictions: [`validation/`](validation/)
