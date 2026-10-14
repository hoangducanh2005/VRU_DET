#!/usr/bin/env python3
"""Run YOLO pedestrian inference across all eight cameras in one NavSim shard.

For each metadata timestamp, save one synchronized 4x2 camera montage with
predicted boxes and confidence, a video, predictions CSV, and projected-GT
2D detection metrics (overall and per camera).
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from ultralytics import YOLO

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from evaluate_vru_metrics import average_precision, iou, project_box

CAMERA_ORDER = ("CAM_F0", "CAM_L0", "CAM_R0", "CAM_L1", "CAM_R1", "CAM_L2", "CAM_R2", "CAM_B0")
DEFAULT_WEIGHTS = Path("/home/anhhd/Downloads/Datasets/nuimages/runs/detect/runs/detect/runs/pedestrian/yolo26m_nuimages/weights/best.pt")
DEFAULT_METADATA = Path("/home/anhhd/navsim_workspace/dataset/downloads/openscene-v1.1/openscene-v1.1/meta_datas/mini/2021.06.28.16.29.11_veh-38_01415_01821.pkl")
DEFAULT_SENSOR_ROOT = Path("/home/anhhd/navsim_workspace/dataset/downloads/openscene-v1.1/openscene_sensor_mini_camera/openscene-v1.1/sensor_blobs/mini")
DEFAULT_OUTPUT = Path("task3_yolo26m/yolo26m_nuimages_20ep/results/navsim_shard1_8cam")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA)
    parser.add_argument("--sensor-root", type=Path, default=DEFAULT_SENSOR_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--video-fps", type=float, default=2.0)
    return parser.parse_args()


def summarize(scores: list[float], score_tp: list[bool], gt: int, tp: int, fp: int, fn: int, ious: list[float]) -> dict[str, Any]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "GT": gt, "predictions": tp + fp, "TP": tp, "FP": fp, "FN": fn,
        "precision": precision, "recall": recall,
        "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "AP_at_iou": average_precision(scores, score_tp, gt),
        "mean_IoU_TP": float(np.mean(ious)) if ious else 0.0,
    }


def draw_tile(image: np.ndarray, detections: list[dict[str, Any]], camera: str, timestamp: int) -> np.ndarray:
    height, width = image.shape[:2]
    tile_w, tile_h = 960, 540
    resized = cv2.resize(image, (tile_w, tile_h), interpolation=cv2.INTER_AREA)
    sx, sy = tile_w / width, tile_h / height
    color = (40, 220, 40)
    for detection in detections:
        x1, y1, x2, y2 = detection["bbox"]
        left, top = int(round(x1 * sx)), int(round(y1 * sy))
        right, bottom = int(round(x2 * sx)), int(round(y2 * sy))
        cv2.rectangle(resized, (left, top), (right, bottom), color, 2)
        label = f"pedestrian {detection['confidence']:.2f}"
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)
        label_top = max(0, top - th - baseline - 3)
        cv2.rectangle(resized, (left, label_top), (left + tw + 6, top), color, -1)
        cv2.putText(resized, label, (left + 3, max(th + 1, top - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 0, 0), 1, cv2.LINE_AA)
    header = np.full((32, tile_w, 3), (35, 35, 35), dtype=np.uint8)
    title = f"{camera} | timestamp {timestamp} | pedestrians {len(detections)}"
    cv2.putText(header, title, (12, 23), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    return np.vstack((header, resized))


def render_evidence(metrics: dict[str, Any]) -> str:
    det = metrics["detection"]
    lines = [
        "# YOLO26m NavSim 8-camera evaluation",
        "",
        f"- Metadata: `{metrics['metadata']}`",
        f"- Sensor root: `{metrics['sensor_root']}`",
        f"- Checkpoint: `{metrics['weights']}`",
        f"- Cameras: `{', '.join(metrics['cameras'])}`",
        f"- Timestamps: `{metrics['timestamps']}`; camera images: `{metrics['images']}`",
        f"- Image size / confidence threshold: `{metrics['imgsz']}` / `{metrics['confidence_threshold']}`",
        f"- IoU threshold: `{metrics['iou_threshold']}`",
        f"- Projected pedestrian GT boxes across camera views: `{det['GT']}`",
        "- Training lineage: initial 5-epoch nuImages checkpoint + 15 additional fine-tuning epochs = 20 cumulative epochs",
        "",
        "## Overall detection metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Predictions | {det['predictions']} |",
        f"| TP / FP / FN | {det['TP']} / {det['FP']} / {det['FN']} |",
        f"| Precision | {det['precision']:.4f} |",
        f"| Recall | {det['recall']:.4f} |",
        f"| F1 | {det['F1']:.4f} |",
        f"| AP at IoU {metrics['iou_threshold']} | {det['AP_at_iou']:.4f} |",
        f"| Mean IoU of TP | {det['mean_IoU_TP']:.4f} |",
        "",
        "Metrics pool confidence-ranked detections across all camera-view/timestamp pairs. GT 3D pedestrian boxes are projected into each camera using NavSim calibration. Other GT classes are ignored; predictions are pedestrian-only. This run reports detector metrics without a temporal tracker.",
        "",
        "## Per-camera detection metrics",
        "",
        "| Camera | GT | Pred | TP | FP | FN | Precision | Recall | F1 | AP@0.5 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for camera in metrics["cameras"]:
        item = metrics["per_camera"][camera]
        lines.append(f"| {camera} | {item['GT']} | {item['predictions']} | {item['TP']} | {item['FP']} | {item['FN']} | {item['precision']:.4f} | {item['recall']:.4f} | {item['F1']:.4f} | {item['AP_at_iou']:.4f} |")
    lines += [
        "",
        "## Runtime and visual outputs",
        "",
        f"- Inference mean per image: `{metrics['mean_inference_ms_per_image']:.2f} ms`",
        f"- Total elapsed time including visualization and metrics: `{metrics['elapsed_seconds']:.2f} s`",
        f"- Effective camera-image throughput: `{metrics['effective_images_per_second']:.2f} images/s`",
        f"- Synchronized montage frames: `{metrics['montage_frames']}`; each contains all 8 cameras at the same metadata timestamp in a 4×2 layout.",
        "- Montage JPEGs: `montage_frames/`",
        "- Montage video: `all_cameras_montage.mp4`",
        "- Per-detection CSV with camera: `predictions.csv`",
        "- Full metrics JSON: `metrics.json`",
        "",
        "The checkpoint metadata records the final fine-tuning stage (`train_args.epochs=15`). Per training provenance, this stage started from the earlier 5-epoch nuImages checkpoint, for 20 cumulative epochs.",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    with args.metadata.open("rb") as handle:
        records = pickle.load(handle)
    if not records:
        raise ValueError(f"No records in metadata: {args.metadata}")
    cameras = [camera for camera in CAMERA_ORDER if camera in records[0]["cams"]]
    missing = set(CAMERA_ORDER) - set(cameras)
    if missing or len(cameras) != 8:
        raise ValueError(f"Expected all 8 cameras, found {cameras}; missing {sorted(missing)}")
    for camera in cameras:
        for record in records:
            image_path = args.sensor_root / record["cams"][camera]["data_path"]
            if not image_path.is_file():
                raise FileNotFoundError(image_path)

    out = args.output_dir.resolve()
    montage_dir = out / "montage_frames"
    montage_dir.mkdir(parents=True, exist_ok=True)
    args.weights = args.weights.resolve()
    model = YOLO(str(args.weights))
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    pedestrian_ids = [int(key) for key, name in names.items() if str(name).lower() == "pedestrian"]
    if len(pedestrian_ids) != 1:
        raise ValueError(f"Expected one pedestrian class, found {names}")
    class_id = pedestrian_ids[0]
    checkpoint_epochs = model.ckpt.get("train_args", {}).get("epochs") if model.ckpt else None

    per_cam_state = {camera: {"scores": [], "score_tp": [], "gt": 0, "tp": 0, "fp": 0, "fn": 0, "ious": []} for camera in cameras}
    overall_scores: list[float] = []
    overall_score_tp: list[bool] = []
    total_gt = total_tp = total_fp = total_fn = ignored_gt = projected_gt = 0
    inference_ms: list[float] = []
    video = None
    csv_path = out / "predictions.csv"
    start_all = time.perf_counter()
    image_count = 0

    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=["frame_id", "timestamp", "camera", "class", "confidence", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        for frame_id, record in enumerate(records):
            timestamp = int(record["timestamp"])
            images: list[np.ndarray] = []
            shapes: dict[str, tuple[int, int]] = {}
            for camera in cameras:
                path = args.sensor_root / record["cams"][camera]["data_path"]
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f"Unreadable image: {path}")
                shapes[camera] = image.shape[:2]
                images.append(image)

            if str(args.device).lower() not in ("cpu", "mps") and torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_start = time.perf_counter()
            results = model.predict(images, imgsz=args.imgsz, conf=args.conf, iou=0.7, classes=pedestrian_ids, device=args.device, batch=8, verbose=False)
            if str(args.device).lower() not in ("cpu", "mps") and torch.cuda.is_available():
                torch.cuda.synchronize()
            batch_ms = (time.perf_counter() - inference_start) * 1000
            inference_ms.append(batch_ms / len(cameras))

            pred_by_camera: dict[str, list[dict[str, Any]]] = {}
            tiles: list[np.ndarray] = []
            for camera, image, result in zip(cameras, images, results):
                boxes = result.boxes
                detections: list[dict[str, Any]] = []
                if boxes is not None and len(boxes):
                    coords = boxes.xyxy.detach().cpu().numpy()
                    confs = boxes.conf.detach().cpu().numpy()
                    for xyxy, score in zip(coords, confs):
                        bbox = [float(value) for value in xyxy]
                        detections.append({"bbox": bbox, "confidence": float(score)})
                        writer.writerow({"frame_id": frame_id, "timestamp": timestamp, "camera": camera, "class": "pedestrian", "confidence": f"{float(score):.6f}", "x1": f"{bbox[0]:.2f}", "y1": f"{bbox[1]:.2f}", "x2": f"{bbox[2]:.2f}", "y2": f"{bbox[3]:.2f}"})
                pred_by_camera[camera] = detections
                tiles.append(draw_tile(image, detections, camera, timestamp))
                image_count += 1

                cam_state = per_cam_state[camera]
                gt_boxes: list[np.ndarray] = []
                anns = record.get("anns", {})
                gt_boxes_3d = np.asarray(anns.get("gt_boxes", np.empty((0, 7))), dtype=np.float64).reshape(-1, 7)
                gt_names = np.asarray(anns.get("gt_names", np.empty((0,), dtype=str))).astype(str).tolist()
                for gt_box, gt_name in zip(gt_boxes_3d, gt_names):
                    if gt_name != "pedestrian":
                        ignored_gt += 1
                        continue
                    projected = project_box(gt_box, record["cams"][camera], shapes[camera])
                    if projected is not None:
                        gt_boxes.append(projected)
                        projected_gt += 1
                cam_state["gt"] += len(gt_boxes)
                total_gt += len(gt_boxes)
                unmatched = set(range(len(gt_boxes)))
                for det in sorted(detections, key=lambda item: -item["confidence"]):
                    score = det["confidence"]
                    best_idx, best_iou = None, 0.0
                    for gt_idx in unmatched:
                        candidate = iou(np.asarray(det["bbox"], dtype=np.float64), gt_boxes[gt_idx])
                        if candidate > best_iou:
                            best_idx, best_iou = gt_idx, candidate
                    is_tp = best_idx is not None and best_iou >= args.iou_threshold
                    cam_state["scores"].append(score)
                    cam_state["score_tp"].append(is_tp)
                    overall_scores.append(score)
                    overall_score_tp.append(is_tp)
                    if is_tp:
                        unmatched.remove(best_idx)
                        cam_state["tp"] += 1
                        total_tp += 1
                        cam_state["ious"].append(best_iou)
                    else:
                        cam_state["fp"] += 1
                        total_fp += 1
                fn = len(unmatched)
                cam_state["fn"] += fn
                total_fn += fn

            canvas_w, canvas_h = 3840, 32 + 2 * 572
            canvas = np.full((canvas_h, canvas_w, 3), (15, 15, 15), dtype=np.uint8)
            global_header = f"NavSim shard 1 | frame {frame_id:04d}/{len(records)-1:04d} | timestamp {timestamp} | 8 synchronized cameras"
            cv2.putText(canvas, global_header, (16, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
            for tile_index, tile in enumerate(tiles):
                row, col = divmod(tile_index, 4)
                y, x = 32 + row * 572, col * 960
                canvas[y:y+572, x:x+960] = tile
            jpg_path = montage_dir / f"frame_{frame_id:06d}_t{timestamp}.jpg"
            if not cv2.imwrite(str(jpg_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 88]):
                raise OSError(f"Could not write montage: {jpg_path}")
            if video is None:
                video_path = out / "all_cameras_montage.mp4"
                video = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"), args.video_fps, (canvas_w, canvas_h))
                if not video.isOpened():
                    raise RuntimeError(f"Could not create video: {video_path}")
            video.write(canvas)
            if (frame_id + 1) % 100 == 0:
                print(f"timestamps={frame_id+1}/{len(records)} images={image_count}", flush=True)
    if video is not None:
        video.release()

    elapsed = time.perf_counter() - start_all
    per_camera = {}
    for camera, state in per_cam_state.items():
        per_camera[camera] = summarize(state["scores"], state["score_tp"], state["gt"], state["tp"], state["fp"], state["fn"], state["ious"])
    overall_ious = [value for state in per_cam_state.values() for value in state["ious"]]
    overall = summarize(overall_scores, overall_score_tp, total_gt, total_tp, total_fp, total_fn, overall_ious)
    metrics = {
        "metadata": str(args.metadata.resolve()), "sensor_root": str(args.sensor_root.resolve()), "weights": str(args.weights),
        "checkpoint_train_args_epochs": checkpoint_epochs, "initial_checkpoint_training_epochs": 5,
        "additional_finetuning_epochs": 15, "cumulative_training_epochs": 20,
        "training_lineage": "5-epoch nuImages checkpoint followed by 15 additional fine-tuning epochs",
        "cameras": list(cameras), "timestamps": len(records), "images": image_count,
        "imgsz": args.imgsz, "confidence_threshold": args.conf, "iou_threshold": args.iou_threshold,
        "projected_pedestrian_gt_boxes": projected_gt, "ignored_non_pedestrian_gt_camera_views": ignored_gt,
        "detection": overall, "per_camera": per_camera,
        "mean_inference_ms_per_image": float(np.mean(inference_ms)) if inference_ms else 0.0,
        "effective_images_per_second": image_count / elapsed if elapsed else 0.0, "elapsed_seconds": elapsed,
        "montage_frames": len(records), "montage_layout": "4x2", "video_fps": args.video_fps,
        "tracking_metrics": None,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "evidence.md").write_text(render_evidence(metrics), encoding="utf-8")
    print(json.dumps({"frames": len(records), "images": image_count, "detection": overall, "elapsed_seconds": elapsed, "output_dir": str(out)}, indent=2))


if __name__ == "__main__":
    main()
