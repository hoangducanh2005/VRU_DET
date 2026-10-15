#!/usr/bin/env python3
"""Validate the BDD100K YOLO11s checkpoint on the NuImages pedestrian val split."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import cv2
import torch
from ultralytics import YOLO

DEFAULT_WEIGHTS = Path("/home/anhhd/Downloads/archive/runs/detect/runs/detect/bdd100k_yolo11s/weights/best.pt")
DEFAULT_DATA = Path("/home/anhhd/Downloads/Datasets/nuimages_yolo")
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "run_nuimage_val"


def args_parser() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold for saved visualizations and CSV")
    return parser.parse_args()


def label_count(label_dir: Path) -> int:
    count = 0
    for label_file in label_dir.glob("*.txt"):
        with label_file.open("r", encoding="utf-8") as handle:
            count += sum(1 for line in handle if line.strip())
    return count


def draw_result(result: Any, destination: Path) -> int:
    image = result.orig_img.copy()
    count = 0
    boxes = result.boxes
    if boxes is not None and len(boxes):
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        for coords, confidence in zip(xyxy, confidences):
            x1, y1, x2, y2 = [int(round(float(value))) for value in coords]
            label = f"pedestrian {float(confidence):.2f}"
            color = (40, 220, 40)
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            label_top = max(0, y1 - text_h - baseline - 4)
            cv2.rectangle(image, (x1, label_top), (x1 + text_w + 6, max(y1, label_top + text_h + baseline + 4)), color, -1)
            cv2.putText(image, label, (x1 + 3, max(text_h + 1, y1 - 3)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 0, 0), 1, cv2.LINE_AA)
            count += 1
    if not cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 90]):
        raise RuntimeError(f"Failed to save visualization: {destination}")
    return count


def write_evidence(metrics: dict[str, Any], output: Path) -> None:
    val = metrics["validation"]
    speed = val["speed_ms_per_image"]
    lines = [
        "# YOLO11s BDD100K checkpoint — NuImages validation",
        "",
        f"- Checkpoint: `{metrics['weights']}` (SHA-256 `{metrics['weights_sha256']}`)",
        f"- Training lineage: COCO-pretrained `{metrics['checkpoint_initial_model']}` fine-tuned on BDD100K for `{metrics['checkpoint_training_epochs']}` epochs.",
        f"- Validation set: `{metrics['dataset']}/images/val`",
        f"- Images / ground-truth instances: `{val['images']}` / `{val['ground_truth_instances']}`",
        f"- Evaluated class: `pedestrian` (dataset label 0 mapped to checkpoint class `{metrics['class_names']['0']}`)",
        f"- Device / image size / batch: `{metrics['device']}` / `{metrics['imgsz']}` / `{metrics['batch_size']}`",
        "- This is validation inference only; the checkpoint was not trained or fine-tuned in this run.",
        "",
        "## Labeled validation metrics",
        "",
        "Ultralytics detection validation was run over the complete val split with predictions restricted to the checkpoint's `pedestrian` class. Precision and recall are reported at Ultralytics' validation operating point; mAP is computed over its confidence sweep.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Precision | {val['precision']:.4f} |",
        f"| Recall | {val['recall']:.4f} |",
        f"| mAP50 | {val['mAP50']:.4f} |",
        f"| mAP50-95 | {val['mAP50_95']:.4f} |",
        f"| Validation fitness | {val['fitness']:.4f} |",
        "",
        "## Validation speed",
        "",
        "| Stage | ms/image |",
        "|---|---:|",
        f"| Preprocess | {speed['preprocess']:.3f} |",
        f"| Inference | {speed['inference']:.3f} |",
        f"| Loss | {speed['loss']:.3f} |",
        f"| Postprocess | {speed['postprocess']:.3f} |",
        f"| Validation wall time | {val['elapsed_seconds']:.2f} s |",
        "",
        "## Qualitative inference outputs",
        "",
        f"- Confidence threshold for the saved prediction pass: `{metrics['prediction_confidence_threshold']}`.",
        f"- Annotated images saved: `{metrics['prediction_pass']['images']}`.",
        f"- Predictions saved: `{metrics['prediction_pass']['detections']}`.",
        f"- Prediction pass wall time (including JPEG writing): `{metrics['prediction_pass']['elapsed_seconds']:.2f} s`.",
        "- All visualized predictions are `pedestrian` boxes with confidence labels. This pass is separate from the full confidence-sweep evaluation above.",
        "",
        "## Files",
        "",
        "- `predictions.csv`: image path, class, confidence, and xyxy coordinates.",
        "- `visualized_frames/`: annotated output for every validation image.",
        "- `preview.jpg`: first annotated validation image.",
        "- `validation_metrics.json`: machine-readable metrics and checkpoint provenance.",
        "- `ultralytics_validation/`: Ultralytics plots and validation artifacts.",
        "",
    ]
    (output / "evidence.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = args_parser()
    weights = args.weights.expanduser().resolve()
    dataset = args.dataset.expanduser().resolve()
    output = args.output_dir.expanduser().resolve()
    images_dir = dataset / "images" / "val"
    labels_dir = dataset / "labels" / "val"
    if not weights.is_file():
        raise FileNotFoundError(weights)
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise FileNotFoundError(f"Expected NuImages val image/label folders under {dataset}")
    image_paths = sorted(path for path in images_dir.iterdir() if path.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if not image_paths:
        raise ValueError(f"No validation images in {images_dir}")
    output.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; this run is configured for NVIDIA GPU inference.")
    device: int | str = int(args.device) if str(args.device).isdigit() else args.device
    torch.cuda.set_device(device)

    model = YOLO(str(weights))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    pedestrian_ids = [int(class_id) for class_id, name in names.items() if str(name).lower() == "pedestrian"]
    if len(pedestrian_ids) != 1:
        raise ValueError(f"Expected exactly one checkpoint class named pedestrian; found {names}")
    pedestrian_id = pedestrian_ids[0]
    train_args = model.ckpt.get("train_args", {}) if model.ckpt else {}

    # Use the checkpoint taxonomy so validation labels (class 0 = pedestrian) align with model class 0.
    val_yaml = output / "nuimages_bdd100k_val.yaml"
    names_yaml = "\n".join(f"  {int(class_id)}: {json.dumps(str(name))}" for class_id, name in names.items())
    val_yaml.write_text(
        f"path: {json.dumps(str(dataset))}\ntrain: images/train\nval: images/val\nnames:\n{names_yaml}\n",
        encoding="utf-8",
    )

    torch.cuda.synchronize(device)
    val_start = time.perf_counter()
    val_results = model.val(data=str(val_yaml), imgsz=args.imgsz, batch=args.batch_size, device=device,
                            classes=[pedestrian_id], conf=0.001, iou=0.7, plots=True,
                            project=str(output), name="ultralytics_validation", exist_ok=True, verbose=False)
    torch.cuda.synchronize(device)
    val_elapsed = time.perf_counter() - val_start
    results_dict = {str(key): float(value) for key, value in val_results.results_dict.items()}
    speed = {key: float(value) for key, value in val_results.speed.items()}
    validation = {
        "images": len(image_paths),
        "ground_truth_instances": label_count(labels_dir),
        "operating_confidence": "Ultralytics best-F1 validation operating point",
        "prediction_confidence_floor": 0.001,
        "precision": results_dict["metrics/precision(B)"],
        "recall": results_dict["metrics/recall(B)"],
        "mAP50": results_dict["metrics/mAP50(B)"],
        "mAP50_95": results_dict["metrics/mAP50-95(B)"],
        "fitness": results_dict["fitness"],
        "speed_ms_per_image": speed,
        "elapsed_seconds": val_elapsed,
        "ultralytics_results": results_dict,
        "validation_artifacts_dir": str(getattr(val_results, "save_dir", output / "ultralytics_validation")),
    }

    visual_dir = output / "visualized_frames"
    visual_dir.mkdir(parents=True, exist_ok=True)
    predictions_csv = output / "predictions.csv"
    prediction_start = time.perf_counter()
    prediction_count = 0
    with predictions_csv.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["image", "class", "confidence", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        first_preview: Path | None = None
        image_count = 0
        # Keep only one small prediction stream alive at a time. A single stream
        # over all 16k paths can retain excessive loader state on memory-limited hosts.
        for offset in range(0, len(image_paths), args.batch_size * 8):
            chunk = image_paths[offset : offset + args.batch_size * 8]
            predictor = model.predict(source=[str(path) for path in chunk], stream=True, imgsz=args.imgsz,
                                      batch=args.batch_size, device=device, classes=[pedestrian_id],
                                      conf=args.conf, iou=0.7, max_det=300, verbose=False)
            for result in predictor:
                image_name = Path(result.path).name
                image_count += 1
                boxes = result.boxes
                if boxes is not None and len(boxes):
                    xyxy = boxes.xyxy.detach().cpu().numpy()
                    confidences = boxes.conf.detach().cpu().numpy()
                    for coords, confidence in zip(xyxy, confidences):
                        writer.writerow({"image": image_name, "class": "pedestrian", "confidence": f"{float(confidence):.7f}",
                                         "x1": f"{float(coords[0]):.2f}", "y1": f"{float(coords[1]):.2f}",
                                         "x2": f"{float(coords[2]):.2f}", "y2": f"{float(coords[3]):.2f}"})
                        prediction_count += 1
                visual_path = visual_dir / image_name
                draw_result(result, visual_path)
                if first_preview is None:
                    first_preview = visual_path
            del predictor
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    torch.cuda.synchronize(device)
    prediction_elapsed = time.perf_counter() - prediction_start
    if first_preview:
        shutil.copy2(first_preview, output / "preview.jpg")

    metrics = {
        "task": "BDD100K-trained YOLO11s on NuImages pedestrian validation",
        "weights": str(weights),
        "weights_sha256": hashlib.sha256(weights.read_bytes()).hexdigest(),
        "checkpoint_initial_model": train_args.get("model"),
        "checkpoint_pretrained": train_args.get("pretrained"),
        "checkpoint_training_epochs": train_args.get("epochs"),
        "checkpoint_training_data": train_args.get("data"),
        "dataset": str(dataset), "device": torch.cuda.get_device_name(device),
        "cuda_version": torch.version.cuda, "torch_version": torch.__version__,
        "ultralytics_version": __import__("ultralytics").__version__,
        "imgsz": args.imgsz, "batch_size": args.batch_size,
        "class_names": {str(key): str(name) for key, name in names.items()},
        "evaluated_class": "pedestrian", "evaluated_class_id": pedestrian_id,
        "prediction_confidence_threshold": args.conf,
        "validation": validation,
        "prediction_pass": {
            "images": image_count, "detections": prediction_count,
            "elapsed_seconds": prediction_elapsed,
            "images_per_second_including_visualization": image_count / prediction_elapsed if prediction_elapsed else 0.0,
            "output_dir": str(visual_dir),
        },
    }
    (output / "validation_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    write_evidence(metrics, output)
    print(json.dumps({"output": str(output), "validation": validation,
                      "prediction_pass": metrics["prediction_pass"]}, indent=2))


if __name__ == "__main__":
    main()
