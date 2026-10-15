# YOLO11s BDD100K checkpoint — NuImages validation

- Checkpoint: `/home/anhhd/Downloads/archive/runs/detect/runs/detect/bdd100k_yolo11s/weights/best.pt` (SHA-256 `27275d0b9c7fb2698e6642e96344aba8afb1c9efbd0915c89c170accd7725ef8`)
- Training lineage: COCO-pretrained `yolo11s.pt` fine-tuned on BDD100K for `25` epochs.
- Validation set: `/home/anhhd/Downloads/Datasets/nuimages_yolo/images/val`
- Images / ground-truth instances: `16445` / `32185`
- Evaluated class: `pedestrian` (dataset label 0 mapped to checkpoint class `pedestrian`)
- Device / image size / batch: `NVIDIA GeForce RTX 5060 Laptop GPU` / `640` / `16`
- This is validation inference only; the checkpoint was not trained or fine-tuned in this run.

## Labeled validation metrics

Ultralytics detection validation was run over the complete val split with predictions restricted to the checkpoint's `pedestrian` class. Precision and recall are reported at Ultralytics' validation operating point; mAP is computed over its confidence sweep.

| Metric | Value |
|---|---:|
| Precision | 0.6710 |
| Recall | 0.3845 |
| mAP50 | 0.4300 |
| mAP50-95 | 0.2159 |
| Validation fitness | 0.2159 |

## Validation speed

Inference latency is **5.420 ms/image**. Including preprocessing and postprocessing, the measured pipeline stages total approximately **6.255 ms/image** (0.237 + 5.420 + 0.597 ms; loss computation is negligible).

| Stage | ms/image |
|---|---:|
| Preprocess | 0.237 |
| Inference | 5.420 |
| Loss | 0.000 |
| Postprocess | 0.597 |
| Validation wall time | 123.61 s |

## Qualitative inference outputs

- Confidence threshold for the saved prediction pass: `0.25`.
- Annotated images saved: `16445`.
- Predictions saved: `13325`.
- Prediction pass wall time (including JPEG writing): `333.03 s`.
- All visualized predictions are `pedestrian` boxes with confidence labels. This pass is separate from the full confidence-sweep evaluation above.

## Files

- `predictions.csv`: image path, class, confidence, and xyxy coordinates.
- `visualized_frames/`: annotated output for every validation image.
- `preview.jpg`: first annotated validation image.
- `validation_metrics.json`: machine-readable metrics and checkpoint provenance.
- `ultralytics_validation/`: Ultralytics plots and validation artifacts.
