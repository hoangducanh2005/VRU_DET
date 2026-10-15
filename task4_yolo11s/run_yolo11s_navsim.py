#!/usr/bin/env python3
"""Batch-infer COCO-pretrained YOLO11s on all eight cameras in a NavSim shard."""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from evaluate_vru_metrics import iou, project_box  # noqa: E402

CAMERA_ORDER = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0")
PRED_TO_GT = {"person": "pedestrian", "bicycle": "bicycle", "motorcycle": "motorcycle"}
GT_CLASSES = ("pedestrian", "bicycle", "motorcycle")
DEFAULT_METADATA = Path(
    "/home/anhhd/navsim_workspace/dataset/downloads/openscene-v1.1/openscene-v1.1/"
    "meta_datas/mini/2021.06.28.16.29.11_veh-38_01415_01821.pkl"
)
DEFAULT_SENSOR_ROOT = Path(
    "/home/anhhd/navsim_workspace/dataset/downloads/openscene-v1.1/"
    "openscene_sensor_mini_camera/openscene-v1.1/sensor_blobs/mini"
)
DEFAULT_OUTPUT = Path(__file__).resolve().parent / "results" / "navsim_shard1_8cam_20261015"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=Path("yolo11s.pt"), help="COCO pretrained Ultralytics checkpoint")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--sensor-root", type=Path, default=DEFAULT_SENSOR_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0", help="CUDA device; Task 4 requires NVIDIA CUDA")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.001, help="Minimum score retained for confidence-ranked AP")
    parser.add_argument("--metric-conf", type=float, default=0.25, help="Operating confidence threshold for precision/recall")
    parser.add_argument("--iou", type=float, default=0.7, help="NMS IoU threshold")
    parser.add_argument("--batch-size", type=int, default=8, help="One synchronized batch contains one image from each camera")
    parser.add_argument("--max-det", type=int, default=300)
    return parser.parse_args()


def quantile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else 0.0


def average_precision(scores: list[float], true_positives: list[bool], positives: int) -> float:
    if positives <= 0 or not scores:
        return 0.0
    order = np.argsort(-np.asarray(scores, dtype=np.float64))
    tp = np.asarray([true_positives[i] for i in order], dtype=np.float64)
    fp = 1.0 - tp
    recall = np.cumsum(tp) / positives
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-12)
    # COCO-style 101-point interpolated AP.
    return float(np.mean([np.max(precision[recall >= point], initial=0.0) for point in np.linspace(0, 1, 101)]))


def match_detections(
    pred_by_image: dict[tuple[int, str], list[dict[str, Any]]],
    gt_by_image: dict[tuple[int, str], list[np.ndarray]],
    gt_class: str,
    iou_threshold: float,
    conf_threshold: float | None = None,
) -> tuple[list[float], list[bool], int, int, int]:
    gt_count = sum(len(boxes) for boxes in gt_by_image.values())
    scores: list[float] = []
    true_positives: list[bool] = []
    for key, detections in pred_by_image.items():
        gt_boxes = gt_by_image.get(key, [])
        unmatched = set(range(len(gt_boxes)))
        ordered = sorted(detections, key=lambda item: -item["confidence"])
        for detection in ordered:
            score = float(detection["confidence"])
            if conf_threshold is not None and score < conf_threshold:
                continue
            best_gt, best_iou = None, 0.0
            for gt_index in unmatched:
                candidate = iou(np.asarray(detection["bbox"], dtype=np.float64), gt_boxes[gt_index])
                if candidate > best_iou:
                    best_gt, best_iou = gt_index, candidate
            is_tp = best_gt is not None and best_iou >= iou_threshold
            scores.append(score)
            true_positives.append(is_tp)
            if is_tp:
                unmatched.remove(best_gt)
    tp_count = int(sum(true_positives))
    fp_count = len(true_positives) - tp_count
    fn_count = gt_count - tp_count
    return scores, true_positives, tp_count, fp_count, fn_count


def draw_tile(image: np.ndarray, detections: list[dict[str, Any]], camera: str, timestamp: int) -> np.ndarray:
    height, width = image.shape[:2]
    tile_w = 960
    tile_h = max(1, int(round(height * tile_w / width)))
    resized = cv2.resize(image, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    sx, sy = tile_w / width, tile_h / height
    colors = {"person": (40, 220, 40), "bicycle": (255, 180, 0), "motorcycle": (255, 80, 200)}
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        left, top = int(round(x1 * sx)), int(round(y1 * sy))
        right, bottom = int(round(x2 * sx)), int(round(y2 * sy))
        color = colors[detection["class"]]
        cv2.rectangle(resized, (left, top), (right, bottom), color, 2)
        label = f"{detection['class']} {detection['confidence']:.2f}"
        (text_w, text_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        label_top = max(0, top - text_h - baseline - 3)
        cv2.rectangle(resized, (left, label_top), (left + text_w + 6, max(top, label_top + text_h + baseline + 3)), color, -1)
        cv2.putText(resized, label, (left + 3, max(text_h + 1, top - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    header = np.full((34, tile_w, 3), (35, 35, 35), dtype=np.uint8)
    cv2.putText(header, f"{camera} | timestamp {timestamp} | VRU detections {len(detections)}", (12, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack((header, resized))


def main() -> None:
    args = parse_args()
    if args.batch_size != 8:
        raise ValueError("Task 4 requires batch-size=8 (one synchronized image per camera).")
    if not torch.cuda.is_available():
        raise RuntimeError("Task 4 requires an available NVIDIA CUDA device; CUDA is unavailable.")
    args.weights = args.weights.expanduser().resolve()
    args.metadata = args.metadata.expanduser().resolve()
    args.sensor_root = args.sensor_root.expanduser().resolve()
    out = args.output_dir.expanduser().resolve()
    frames_dir = out / "visualized_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    if not args.metadata.is_file():
        raise FileNotFoundError(args.metadata)

    with args.metadata.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected non-empty NavSim metadata: {args.metadata}")
    cameras = [camera for camera in CAMERA_ORDER if camera in records[0].get("cams", {})]
    if cameras != list(CAMERA_ORDER):
        raise ValueError(f"Expected all eight cameras, found {cameras}")
    for record in records:
        if any(camera not in record.get("cams", {}) for camera in cameras):
            raise ValueError(f"A metadata timestamp is missing one or more cameras: {record.get('timestamp')}")

    device = int(args.device) if str(args.device).isdigit() else args.device
    torch.cuda.set_device(device)
    model = YOLO(str(args.weights))  # Downloads the official yolo11s.pt COCO checkpoint if absent.
    model_names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    class_ids = {str(name).lower(): int(class_id) for class_id, name in model_names.items()}
    required = {"person", "bicycle", "motorcycle"}
    if not required.issubset(class_ids):
        raise ValueError(f"Checkpoint does not expose required COCO classes {sorted(required)}: {model_names}")
    selected_ids = [class_ids[name] for name in ("person", "bicycle", "motorcycle")]
    weights_sha256 = __import__("hashlib").sha256(args.weights.read_bytes()).hexdigest()

    # Warm up on one synchronized camera batch; warm-up is excluded from measured timings.
    warm_images = []
    for camera in cameras:
        image_path = args.sensor_root / records[0]["cams"][camera]["data_path"]
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise RuntimeError(f"Could not decode image: {image_path}")
        warm_images.append(image)
    model.predict(warm_images, imgsz=args.imgsz, conf=args.conf, iou=args.iou, classes=selected_ids,
                  device=device, batch=args.batch_size, max_det=args.max_det, verbose=False)
    torch.cuda.synchronize(device)
    torch.cuda.reset_peak_memory_stats(device)

    gt_by_class: dict[str, dict[tuple[int, str], list[np.ndarray]]] = {
        name: defaultdict(list) for name in GT_CLASSES
    }
    pred_by_class: dict[str, dict[tuple[int, str], list[dict[str, Any]]]] = {
        name: defaultdict(list) for name in PRED_TO_GT
    }
    pred_writer_fields = ["frame_id", "timestamp", "camera", "class", "confidence", "x1", "y1", "x2", "y2"]
    batch_latencies_ms: list[float] = []
    inference_batch_ms: list[float] = []
    batch_csv = out / "batch_benchmark.csv"
    predictions_csv = out / "predictions.csv"
    start_total = time.perf_counter()
    total_predictions = 0
    ignored_gt_counts: dict[str, int] = defaultdict(int)

    with predictions_csv.open("w", newline="", encoding="utf-8") as pred_file, batch_csv.open("w", newline="", encoding="utf-8") as batch_file:
        pred_writer = csv.DictWriter(pred_file, fieldnames=pred_writer_fields)
        pred_writer.writeheader()
        batch_writer = csv.DictWriter(batch_file, fieldnames=["frame_id", "timestamp", "batch_latency_ms", "per_image_latency_ms", "detections"])
        batch_writer.writeheader()

        for frame_id, record in enumerate(records):
            timestamp = int(record["timestamp"])
            record_gt_names = np.asarray(record.get("anns", {}).get("gt_names", np.empty((0,), dtype=str))).astype(str).tolist()
            for gt_name in record_gt_names:
                if gt_name not in GT_CLASSES:
                    ignored_gt_counts[gt_name] += 1
            images: list[np.ndarray] = []
            for camera in cameras:
                path = args.sensor_root / record["cams"][camera]["data_path"]
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"Could not decode image: {path}")
                images.append(image)

            torch.cuda.synchronize(device)
            batch_start = time.perf_counter()
            results = model.predict(images, imgsz=args.imgsz, conf=args.conf, iou=args.iou,
                                    classes=selected_ids, device=device, batch=args.batch_size,
                                    max_det=args.max_det, verbose=False)
            torch.cuda.synchronize(device)
            batch_ms = (time.perf_counter() - batch_start) * 1000.0
            batch_latencies_ms.append(batch_ms)
            inference_batch_ms.append(batch_ms)

            tiles: list[np.ndarray] = []
            frame_detections = 0
            for camera, image, result in zip(cameras, images, results):
                key = (frame_id, camera)
                detections: list[dict[str, Any]] = []
                boxes = result.boxes
                if boxes is not None and len(boxes):
                    xyxy = boxes.xyxy.detach().cpu().numpy()
                    confs = boxes.conf.detach().cpu().numpy()
                    cls_ids = boxes.cls.detach().cpu().numpy().astype(int)
                    for box, score, cls_id in zip(xyxy, confs, cls_ids):
                        class_name = str(model_names[int(cls_id)]).lower()
                        if class_name not in PRED_TO_GT:
                            continue
                        bbox = [float(value) for value in box]
                        detection = {"class": class_name, "gt_class": PRED_TO_GT[class_name], "confidence": float(score), "bbox": bbox}
                        detections.append(detection)
                        pred_by_class[class_name][key].append(detection)
                        pred_writer.writerow({"frame_id": frame_id, "timestamp": timestamp, "camera": camera,
                                              "class": class_name, "confidence": f"{float(score):.7f}",
                                              "x1": f"{bbox[0]:.2f}", "y1": f"{bbox[1]:.2f}",
                                              "x2": f"{bbox[2]:.2f}", "y2": f"{bbox[3]:.2f}"})
                frame_detections += len(detections)
                total_predictions += len(detections)

                h, w = image.shape[:2]
                anns = record.get("anns", {})
                gt_boxes_3d = np.asarray(anns.get("gt_boxes", np.empty((0, 7))), dtype=np.float64).reshape(-1, 7)
                gt_names = np.asarray(anns.get("gt_names", np.empty((0,), dtype=str))).astype(str).tolist()
                for box3d, gt_name in zip(gt_boxes_3d, gt_names):
                    if gt_name not in GT_CLASSES:
                        continue
                    projected = project_box(box3d, record["cams"][camera], (h, w))
                    if projected is not None:
                        gt_by_class[gt_name][key].append(projected)
                display_detections = [item for item in detections if item["confidence"] >= args.metric_conf]
                tiles.append(draw_tile(image, display_detections, camera, timestamp))

            batch_writer.writerow({"frame_id": frame_id, "timestamp": timestamp,
                                   "batch_latency_ms": f"{batch_ms:.4f}",
                                   "per_image_latency_ms": f"{batch_ms / len(cameras):.4f}",
                                   "detections": frame_detections})
            montage = np.vstack((np.hstack(tiles[:4]), np.hstack(tiles[4:])))
            frame_path = frames_dir / f"frame_{frame_id:06d}_t{timestamp}.jpg"
            if not cv2.imwrite(str(frame_path), montage, [cv2.IMWRITE_JPEG_QUALITY, 90]):
                raise RuntimeError(f"Could not write visualization: {frame_path}")

    total_elapsed = time.perf_counter() - start_total
    gpu_peak_allocated_mb = torch.cuda.max_memory_allocated(device) / (1024**2)
    gpu_peak_reserved_mb = torch.cuda.max_memory_reserved(device) / (1024**2)
    total_images = len(records) * len(cameras)
    mean_batch_ms = statistics.fmean(batch_latencies_ms) if batch_latencies_ms else 0.0
    batch_p95_ms = quantile(batch_latencies_ms, 95)
    inference_seconds = sum(batch_latencies_ms) / 1000.0
    images_per_second = total_images / inference_seconds if inference_seconds else 0.0
    timestamps_per_second = len(records) / inference_seconds if inference_seconds else 0.0
    e2e_images_per_second = total_images / total_elapsed if total_elapsed else 0.0

    supported_gt_classes = [name for name in GT_CLASSES if sum(map(len, gt_by_class[name].values())) > 0]
    per_class: dict[str, Any] = {}
    all_tp = all_fp = all_fn = 0
    for pred_name, gt_name in PRED_TO_GT.items():
        gt_boxes = gt_by_class[gt_name]
        pred_boxes = pred_by_class[pred_name]
        gt_count = sum(map(len, gt_boxes.values()))
        if gt_count == 0:
            per_class[pred_name] = {"ground_truth_boxes": 0, "predictions": sum(map(len, pred_boxes.values())),
                                    "precision_at_conf": None, "recall_at_conf": None, "AP50": None,
                                    "AP50_95": None, "note": "No projected NavSim ground-truth boxes for this class; excluded from mAP."}
            continue
        _, _, tp, fp, fn = match_detections(pred_boxes, gt_boxes, gt_name, 0.5, args.metric_conf)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        aps = []
        for threshold in np.arange(0.5, 0.951, 0.05):
            scores, score_tp, _, _, _ = match_detections(pred_boxes, gt_boxes, gt_name, float(threshold))
            aps.append(average_precision(scores, score_tp, gt_count))
        ap50 = aps[0]
        ap50_95 = float(np.mean(aps))
        per_class[pred_name] = {"ground_truth_boxes": gt_count, "predictions": sum(map(len, pred_boxes.values())),
                                "TP_at_conf_iou50": tp, "FP_at_conf_iou50": fp, "FN_at_conf_iou50": fn,
                                "precision_at_conf": precision, "recall_at_conf": recall,
                                "AP50": ap50, "AP50_95": ap50_95}
        all_tp += tp
        all_fp += fp
        all_fn += fn

    total_gt_scored = all_tp + all_fn
    overall_precision = all_tp / (all_tp + all_fp) if all_tp + all_fp else 0.0
    overall_recall = all_tp / total_gt_scored if total_gt_scored else 0.0
    supported_ap50 = [per_class[name]["AP50"] for name in per_class if per_class[name].get("AP50") is not None]
    supported_ap95 = [per_class[name]["AP50_95"] for name in per_class if per_class[name].get("AP50_95") is not None]
    metrics = {
        "task": "Task 4 YOLO11s COCO-pretrained 8-camera NavSim baseline",
        "weights": str(args.weights), "weights_sha256": weights_sha256,
        "metadata": str(args.metadata), "sensor_root": str(args.sensor_root), "output_dir": str(out),
        "device": torch.cuda.get_device_name(device), "cuda_version": torch.version.cuda,
        "ultralytics_version": __import__("ultralytics").__version__, "torch_version": torch.__version__,
        "imgsz": args.imgsz, "confidence_threshold": args.metric_conf,
        "inference_confidence_floor": args.conf, "nms_iou_threshold": args.iou,
        "batch_size": args.batch_size, "cameras": list(cameras), "timestamps": len(records), "images": total_images,
        "classes": {"person": 0, "bicycle": 1, "motorcycle": 3},
        "ground_truth_note": "NavSim 3D boxes projected into each camera; pedestrian maps to COCO person, bicycle maps to bicycle. The shard has no motorcycle GT; motorcycle is reported as unsupported and excluded from mAP.",
        "ground_truth_boxes_projected": {name: sum(map(len, gt_by_class[name].values())) for name in GT_CLASSES},
        "ignored_navsim_gt_counts": dict(ignored_gt_counts),
        "evaluation_iou_thresholds": [round(float(x), 2) for x in np.arange(0.5, 0.951, 0.05)],
        "metrics_at_conf_threshold": {"TP": all_tp, "FP": all_fp, "FN": all_fn,
            "precision": overall_precision, "recall": overall_recall,
            "F1": 2 * overall_precision * overall_recall / (overall_precision + overall_recall) if overall_precision + overall_recall else 0.0,
            "mAP50": float(np.mean(supported_ap50)) if supported_ap50 else 0.0,
            "mAP50_95": float(np.mean(supported_ap95)) if supported_ap95 else 0.0,
            "mAP_classes_included": [name for name, item in per_class.items() if item.get("AP50") is not None]},
        "per_class": per_class,
        "benchmark": {"measured_batches": len(batch_latencies_ms), "warmup_batches_excluded": 1,
            "mean_batch_latency_ms": mean_batch_ms, "p95_batch_latency_ms": batch_p95_ms,
            "mean_per_image_latency_ms": mean_batch_ms / len(cameras) if cameras else 0.0,
            "inference_images_per_second": images_per_second,
            "inference_fps_per_camera_equivalent": timestamps_per_second,
            "end_to_end_elapsed_seconds": total_elapsed,
            "end_to_end_images_per_second": e2e_images_per_second,
            "end_to_end_fps_per_camera_equivalent": len(records) / total_elapsed if total_elapsed else 0.0,
            "gpu_peak_memory_allocated_mb": gpu_peak_allocated_mb,
            "gpu_peak_memory_reserved_mb": gpu_peak_reserved_mb},
        "visualization_frames": len(records), "predictions": total_predictions,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    first_frame = frames_dir / f"frame_000000_t{int(records[0]['timestamp'])}.jpg"
    if first_frame.is_file():
        import shutil
        shutil.copy2(first_frame, out / "preview_montage.jpg")
    write_evidence(metrics)
    print(json.dumps({"output": str(out), "images": total_images, "predictions": total_predictions,
                      "metrics": metrics["metrics_at_conf_threshold"], "benchmark": metrics["benchmark"]}, indent=2))


def write_evidence(metrics: dict[str, Any]) -> None:
    benchmark = metrics["benchmark"]
    score = metrics["metrics_at_conf_threshold"]
    lines = [
        "# Task 4 — YOLO11s COCO baseline on NavSim (8 cameras)", "",
        f"- Checkpoint: `{metrics['weights']}` (SHA-256 `{metrics['weights_sha256']}`)",
        f"- Device: `{metrics['device']}`; CUDA `{metrics['cuda_version']}`",
        f"- Ultralytics / PyTorch: `{metrics['ultralytics_version']}` / `{metrics['torch_version']}`",
        f"- Dataset: NavSim shard 1; `{metrics['timestamps']}` synchronized timestamps × 8 cameras = `{metrics['images']}` images",
        f"- Classes: COCO `person`, `bicycle`, and `motorcycle`; input size `{metrics['imgsz']}`, confidence `{metrics['confidence_threshold']}`, batch size `{metrics['batch_size']}`",
        "- Training: none; official COCO-pretrained `yolo11s.pt` checkpoint.", "",
        "## Ground-truth evaluation", "",
        metrics["ground_truth_note"],
        f"Projected GT counts: pedestrian/person `{metrics['ground_truth_boxes_projected']['pedestrian']}`, bicycle `{metrics['ground_truth_boxes_projected']['bicycle']}`, motorcycle `{metrics['ground_truth_boxes_projected']['motorcycle']}`.",
        f"Other NavSim GT instances ignored (3D annotation counts, before camera projection): `{json.dumps(metrics['ignored_navsim_gt_counts'], sort_keys=True)}`.",
        f"Precision and recall are measured at confidence {metrics['confidence_threshold']:.2f} and IoU 0.50. AP uses confidence-ranked predictions retained from score floor {metrics['inference_confidence_floor']:.3f}, with 101-point interpolation; mAP50-95 averages IoU thresholds 0.50 through 0.95 in 0.05 increments. The aggregate mAP averages classes with projected GT support; unsupported classes are excluded rather than treated as zero.", "",
        "| Class | GT boxes | Predictions | Precision @ conf | Recall @ conf | AP50 | AP50-95 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in metrics["per_class"].items():
        if item.get("AP50") is None:
            lines.append(f"| {name} | 0 | {item['predictions']} | N/A | N/A | N/A | N/A |")
        else:
            lines.append(f"| {name} | {item['ground_truth_boxes']} | {item['predictions']} | {item['precision_at_conf']:.4f} | {item['recall_at_conf']:.4f} | {item['AP50']:.4f} | {item['AP50_95']:.4f} |")
    lines += [
        "", "### Aggregate", "",
        "| Metric | Value |", "|---|---:|",
        f"| TP / FP / FN at confidence 0.25, IoU 0.50 | {score['TP']} / {score['FP']} / {score['FN']} |",
        f"| Precision | {score['precision']:.4f} |", f"| Recall | {score['recall']:.4f} |",
        f"| F1 | {score['F1']:.4f} |", f"| mAP50 | {score['mAP50']:.4f} |",
        f"| mAP50-95 | {score['mAP50_95']:.4f} |",
        f"| Classes included in mAP | {', '.join(score['mAP_classes_included'])} |", "",
        "## Throughput and resource benchmark", "",
        "Each measured batch consists of the eight camera images from one NavSim timestamp. Batch latency is end-to-end Ultralytics predict latency for that batch (preprocess + GPU inference + postprocess); one unmeasured warm-up batch is excluded. GPU timing is synchronized before and after each measured batch.", "",
        "| Metric | Measurement |", "|---|---:|",
        f"| Mean batch latency | {benchmark['mean_batch_latency_ms']:.2f} ms |",
        f"| P95 batch latency | {benchmark['p95_batch_latency_ms']:.2f} ms |",
        f"| Mean latency per camera image | {benchmark['mean_per_image_latency_ms']:.2f} ms |",
        f"| Inference throughput | {benchmark['inference_images_per_second']:.2f} images/s |",
        f"| Inference FPS per-camera equivalent | {benchmark['inference_fps_per_camera_equivalent']:.2f} timestamp batches/s |",
        f"| Peak GPU memory allocated / reserved | {benchmark['gpu_peak_memory_allocated_mb']:.1f} / {benchmark['gpu_peak_memory_reserved_mb']:.1f} MiB |",
        f"| End-to-end runtime (including GT projection and visualization writes) | {benchmark['end_to_end_elapsed_seconds']:.2f} s |",
        f"| End-to-end throughput | {benchmark['end_to_end_images_per_second']:.2f} images/s |",
        f"| End-to-end FPS per-camera equivalent | {benchmark['end_to_end_fps_per_camera_equivalent']:.2f} timestamp batches/s |", "",
        "## Deliverables", "",
        "- `run_yolo11s_navsim.py`: reproducible runner; defaults target the local NavSim shard and CUDA device 0.",
        "- `metrics.json`: configuration, per-class and aggregate quality metrics, and hardware benchmark.",
        "- `batch_benchmark.csv`: one measured row per timestamp batch.",
        "- `predictions.csv`: frame/timestamp/camera, COCO class, confidence, and image-space xyxy boxes.",
        f"- `visualized_frames/`: `{metrics['visualization_frames']}` synchronized 4×2 montages; every tile shows camera ID and every box shows class and confidence.",
        "- `preview_montage.jpg`: first timestamp for quick review.", "",
        "## Reproducibility notes", "",
        "No full BDD100K download is required. Evaluation uses only the available NavSim metadata and camera images. Since NavSim stores 3D annotations rather than native 2D camera labels, metrics are projection-based and are not directly comparable to benchmark results computed from native 2D annotations.", "",
    ]
    (Path(metrics["output_dir"]) / "evidence.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
