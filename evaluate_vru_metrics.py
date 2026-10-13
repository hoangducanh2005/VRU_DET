"""Evaluate 2D YOLO + ByteTrack predictions against NavSim shard-5 GT.

NavSim stores 3D boxes in LiDAR coordinates.  This evaluator follows the
coordinate transform used by NavSim itself, projects the VRU GT boxes into
CAM_F0, matches detections by class and 2D IoU, and computes detection and
tracking metrics.  The local GT taxonomy has ``pedestrian`` and ``bicycle``
but no separate ``rider`` or ``motorcycle`` labels, so only those two classes
are scored.
"""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np


DEFAULT_METADATA = Path(
    r"D:\navsim_workspace\dataset\navsim_logs\mini\2021.06.09.14.58.55_veh-35_01894_02311.pkl"
)
DEFAULT_SENSOR_ROOT = Path(r"D:\navsim_workspace\dataset\sensor_blobs\mini")
GT_TO_PRED = {"pedestrian": "person", "bicycle": "bicycle"}
PRED_TO_GT = {value: key for key, value in GT_TO_PRED.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-csv", type=Path, required=True, help="task2_vru_tracking.py CSV output")
    parser.add_argument("--metadata", type=Path, default=DEFAULT_METADATA, help="NavSim shard .pkl")
    parser.add_argument("--sensor-root", type=Path, default=DEFAULT_SENSOR_ROOT)
    parser.add_argument("--camera-name", default="CAM_F0")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument(
        "--class-map",
        default="person:pedestrian,bicycle:bicycle",
        help="Prediction-to-GT map, e.g. Pedestrian:pedestrian,Cyclist:bicycle",
    )
    parser.add_argument("--output-json", type=Path, default=Path("output/task2_vru_evaluation.json"))
    parser.add_argument("--output-report", type=Path, default=Path("output/task2_vru_evaluation.md"))
    return parser.parse_args()


def parse_class_map(value: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for item in value.split(","):
        pair = [part.strip().lower() for part in item.split(":", 1)]
        if len(pair) != 2 or not pair[0] or not pair[1]:
            raise ValueError(f"Invalid --class-map item: {item!r}")
        result[pair[0]] = pair[1]
    if not result:
        raise ValueError("--class-map must contain at least one prediction:gt pair")
    return result


def load_predictions(path: Path, pred_to_gt: dict[str, str]) -> dict[int, list[dict[str, Any]]]:
    predictions: dict[int, list[dict[str, Any]]] = defaultdict(list)
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            class_name = row["class"].strip().lower()
            if "bbox" in row:
                bbox = np.asarray(json.loads(row["bbox"]), dtype=np.float64)
            else:
                bbox = np.asarray([float(row[key]) for key in ("x1", "y1", "x2", "y2")], dtype=np.float64)
            prediction = {
                "class": class_name,
                "gt_class": pred_to_gt.get(class_name),
                "track_id": str(row["track_id"]),
                "confidence": float(row["confidence"]),
                "bbox": bbox,
            }
            predictions[int(row["frame_id"])].append(prediction)
    return predictions


def load_metadata(path: Path) -> list[dict[str, Any]]:
    with path.open("rb") as handle:
        records = pickle.load(handle)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Expected a non-empty NavSim metadata list: {path}")
    return records


def resolve_image(sensor_root: Path, relative_path: str) -> Path:
    candidate = sensor_root / Path(relative_path)
    if candidate.is_file():
        return candidate
    fallback = sensor_root / Path(relative_path).parent.name / Path(relative_path).name
    if fallback.is_file():
        return fallback
    raise FileNotFoundError(f"Could not resolve camera image: {relative_path}")


def lidar_to_camera(points: np.ndarray, camera: dict[str, Any]) -> np.ndarray:
    """Use NavSim's row-vector LiDAR-to-camera homogeneous transform."""
    sensor2lidar_rotation = np.asarray(camera["sensor2lidar_rotation"], dtype=np.float64)
    sensor2lidar_translation = np.asarray(camera["sensor2lidar_translation"], dtype=np.float64)
    lidar2cam_rotation = np.linalg.inv(sensor2lidar_rotation)
    lidar2cam_translation = sensor2lidar_translation @ lidar2cam_rotation.T
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = lidar2cam_rotation.T
    transform[3, :3] = -lidar2cam_translation
    homogeneous = np.concatenate([points, np.ones((len(points), 1), dtype=np.float64)], axis=1)
    return (transform.T @ homogeneous.T).T[:, :3]


def box_corners_lidar(box: np.ndarray) -> np.ndarray:
    """Convert [x,y,z,length,width,height,heading] to eight LiDAR corners."""
    x, y, z, length, width, height, heading = box.tolist()
    signs = np.asarray(
        [
            [-1, -1, -1],
            [-1, -1, 1],
            [-1, 1, -1],
            [-1, 1, 1],
            [1, -1, -1],
            [1, -1, 1],
            [1, 1, -1],
            [1, 1, 1],
        ],
        dtype=np.float64,
    )
    corners = signs * np.asarray([length, width, height], dtype=np.float64) / 2.0
    cos_yaw, sin_yaw = np.cos(heading), np.sin(heading)
    rotation = np.asarray(
        [[cos_yaw, -sin_yaw, 0.0], [sin_yaw, cos_yaw, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return corners @ rotation.T + np.asarray([x, y, z], dtype=np.float64)


def project_box(box: np.ndarray, camera: dict[str, Any], image_shape: tuple[int, int]) -> np.ndarray | None:
    corners_camera = lidar_to_camera(box_corners_lidar(box), camera)
    visible = corners_camera[:, 2] > 1e-3
    if not np.any(visible):
        return None
    points = corners_camera[visible]
    intrinsic = np.asarray(camera["cam_intrinsic"], dtype=np.float64)
    image_points = (intrinsic @ points.T).T
    image_points = image_points[:, :2] / image_points[:, 2:3]
    height, width = image_shape
    x1, y1 = np.min(image_points, axis=0)
    x2, y2 = np.max(image_points, axis=0)
    clipped = np.asarray(
        [np.clip(x1, 0, width - 1), np.clip(y1, 0, height - 1), np.clip(x2, 0, width - 1), np.clip(y2, 0, height - 1)],
        dtype=np.float64,
    )
    if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
        return None
    return clipped


def iou(left: np.ndarray, right: np.ndarray) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_left = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    area_right = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = area_left + area_right - intersection
    return intersection / union if union > 0 else 0.0


def average_precision(scores: list[float], true_positives: list[bool], positives: int) -> float:
    if not scores or positives == 0:
        return 0.0
    order = np.argsort(-np.asarray(scores))
    tp = np.asarray([true_positives[index] for index in order], dtype=np.float64)
    fp = 1.0 - tp
    precision = np.cumsum(tp) / np.maximum(np.cumsum(tp) + np.cumsum(fp), 1e-12)
    recall = np.cumsum(tp) / positives
    precision_envelope = np.maximum.accumulate(precision[::-1])[::-1]
    recall_points = np.concatenate(([0.0], recall))
    precision_points = np.concatenate(([precision_envelope[0]], precision_envelope))
    return float(np.sum((recall_points[1:] - recall_points[:-1]) * precision_points[1:]))


def maximum_weight_assignment(weights: np.ndarray) -> list[tuple[int, int]]:
    """Hungarian maximum-weight assignment using only NumPy."""
    if weights.size == 0:
        return []
    transposed = weights.shape[0] > weights.shape[1]
    matrix = weights.T if transposed else weights
    rows, columns = matrix.shape
    cost = -matrix.astype(np.float64)
    u = np.zeros(rows + 1, dtype=np.float64)
    v = np.zeros(columns + 1, dtype=np.float64)
    matching = np.zeros(columns + 1, dtype=np.int64)
    way = np.zeros(columns + 1, dtype=np.int64)
    for row in range(1, rows + 1):
        matching[0] = row
        column_zero = 0
        minimum = np.full(columns + 1, np.inf, dtype=np.float64)
        used = np.zeros(columns + 1, dtype=bool)
        while True:
            used[column_zero] = True
            row_zero = matching[column_zero]
            delta = np.inf
            column_one = 0
            for column in range(1, columns + 1):
                if not used[column]:
                    current = cost[row_zero - 1, column - 1] - u[row_zero] - v[column]
                    if current < minimum[column]:
                        minimum[column] = current
                        way[column] = column_zero
                    if minimum[column] < delta:
                        delta = minimum[column]
                        column_one = column
            for column in range(columns + 1):
                if used[column]:
                    u[matching[column]] += delta
                    v[column] -= delta
                else:
                    minimum[column] -= delta
            column_zero = column_one
            if matching[column_zero] == 0:
                break
        while True:
            column_one = way[column_zero]
            matching[column_zero] = matching[column_one]
            column_zero = column_one
            if column_zero == 0:
                break
    pairs = [(int(matching[column] - 1), column - 1) for column in range(1, columns + 1) if matching[column]]
    if transposed:
        return [(column, row) for row, column in pairs]
    return pairs


def assignment_metrics(pair_counts: Counter[tuple[str, str]], gt_total: int, pred_total: int) -> dict[str, float | int]:
    gt_ids = sorted({pair[0] for pair in pair_counts})
    pred_ids = sorted({pair[1] for pair in pair_counts})
    idtp = 0
    if gt_ids and pred_ids:
        matrix = np.asarray([[pair_counts[(gt_id, pred_id)] for pred_id in pred_ids] for gt_id in gt_ids])
        idtp = int(sum(matrix[row, column] for row, column in maximum_weight_assignment(matrix)))
    idfp = pred_total - idtp
    idfn = gt_total - idtp
    denominator = 2 * idtp + idfp + idfn
    return {
        "IDTP": idtp,
        "IDFP": idfp,
        "IDFN": idfn,
        "IDF1": (2 * idtp / denominator) if denominator else 0.0,
    }


def evaluate(args: argparse.Namespace) -> dict[str, Any]:
    if not args.pred_csv.is_file():
        raise FileNotFoundError(args.pred_csv)
    records = load_metadata(args.metadata)
    pred_to_gt = parse_class_map(args.class_map)
    gt_to_pred = {gt_class: pred_class for pred_class, gt_class in pred_to_gt.items()}
    predictions = load_predictions(args.pred_csv, pred_to_gt)
    first_image = resolve_image(args.sensor_root, records[0]["cams"][args.camera_name]["data_path"])
    image = cv2.imread(str(first_image), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {first_image}")
    image_shape = image.shape[:2]

    total_gt = 0
    total_pred = 0
    true_positive = 0
    false_positive = 0
    false_negative = 0
    id_switches = 0
    matched_ious: list[float] = []
    scores: list[float] = []
    score_tp: list[bool] = []
    pair_counts: Counter[tuple[str, str]] = Counter()
    last_prediction_for_gt: dict[str, str] = {}
    gt_track_ids_seen: set[str] = set()
    prediction_track_ids_seen: set[str] = set()
    matched_gt_track_ids: set[str] = set()
    matched_prediction_track_ids: set[str] = set()
    tracked_prediction_detections = 0
    untracked_prediction_detections = 0
    identity_untracked_matches = 0
    per_class: dict[str, dict[str, Any]] = {
        gt_class: {"gt": 0, "pred": 0, "TP": 0, "FP": 0, "FN": 0, "ious": [], "scores": [], "score_tp": []}
        for gt_class in gt_to_pred
    }
    projected_boxes = 0
    ignored_gt_boxes = 0
    ignored_predictions = Counter()

    for frame_id, record in enumerate(records):
        camera = record["cams"][args.camera_name]
        anns = record.get("anns", {})
        boxes = np.asarray(anns.get("gt_boxes", np.empty((0, 7))), dtype=np.float64).reshape(-1, 7)
        names = np.asarray(anns.get("gt_names", np.empty((0,), dtype=str))).astype(str).tolist()
        track_tokens = [str(token) for token in anns.get("track_tokens", [])]
        gt_by_class: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for box, name, track_token in zip(boxes, names, track_tokens):
            if name not in gt_to_pred:
                ignored_gt_boxes += 1
                continue
            projected = project_box(box, camera, image_shape)
            if projected is None:
                continue
            projected_boxes += 1
            gt_track_ids_seen.add(track_token)
            gt_by_class[name].append({"bbox": projected, "track_id": track_token})

        eligible_predictions: list[dict[str, Any]] = []
        for prediction in predictions.get(frame_id, []):
            if prediction["gt_class"] is None:
                ignored_predictions[prediction["class"]] += 1
            else:
                if prediction["track_id"] == "-1":
                    untracked_prediction_detections += 1
                else:
                    tracked_prediction_detections += 1
                    prediction_track_ids_seen.add(prediction["track_id"])
                eligible_predictions.append(prediction)

        for gt_class, pred_class in gt_to_pred.items():
            gts = gt_by_class[gt_class]
            preds = sorted((p for p in eligible_predictions if p["gt_class"] == gt_class), key=lambda p: -p["confidence"])
            per_class[gt_class]["gt"] += len(gts)
            per_class[gt_class]["pred"] += len(preds)
            total_gt += len(gts)
            total_pred += len(preds)
            unmatched_gt = set(range(len(gts)))
            for prediction in preds:
                scores.append(prediction["confidence"])
                per_class[gt_class]["scores"].append(prediction["confidence"])
                best_index, best_iou = None, 0.0
                for gt_index in unmatched_gt:
                    candidate_iou = iou(prediction["bbox"], gts[gt_index]["bbox"])
                    if candidate_iou > best_iou:
                        best_iou, best_index = candidate_iou, gt_index
                if best_index is not None and best_iou >= args.iou_threshold:
                    unmatched_gt.remove(best_index)
                    gt_id = gts[best_index]["track_id"]
                    true_positive += 1
                    per_class[gt_class]["TP"] += 1
                    matched_ious.append(best_iou)
                    per_class[gt_class]["ious"].append(best_iou)
                    score_tp.append(True)
                    per_class[gt_class]["score_tp"].append(True)
                    if prediction["track_id"] == "-1":
                        identity_untracked_matches += 1
                    else:
                        pair_counts[(gt_id, prediction["track_id"])] += 1
                        matched_gt_track_ids.add(gt_id)
                        matched_prediction_track_ids.add(prediction["track_id"])
                        if gt_id in last_prediction_for_gt and last_prediction_for_gt[gt_id] != prediction["track_id"]:
                            id_switches += 1
                        last_prediction_for_gt[gt_id] = prediction["track_id"]
                else:
                    false_positive += 1
                    per_class[gt_class]["FP"] += 1
                    score_tp.append(False)
                    per_class[gt_class]["score_tp"].append(False)
            false_negative += len(unmatched_gt)
            per_class[gt_class]["FN"] += len(unmatched_gt)

    def summarize(item: dict[str, Any]) -> dict[str, Any]:
        tp, fp, fn = item["TP"], item["FP"], item["FN"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        return {
            "gt": item["gt"],
            "pred": item["pred"],
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "precision": precision,
            "recall": recall,
            "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "AP": average_precision(item["scores"], item["score_tp"], item["gt"]),
            "mean_IoU": float(np.mean(item["ious"])) if item["ious"] else 0.0,
        }

    precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    recall = true_positive / total_gt if total_gt else 0.0
    tracking = assignment_metrics(pair_counts, total_gt, total_pred)
    tracking.update(
        {
            "GT": total_gt,
            "predictions": total_pred,
            "FP": false_positive,
            "FN": false_negative,
            "ID_switches": id_switches,
            "MOTA": 1.0 - (false_positive + false_negative + id_switches) / total_gt if total_gt else 0.0,
            "ID_precision": tracking["IDTP"] / (tracking["IDTP"] + tracking["IDFP"]) if tracking["IDTP"] + tracking["IDFP"] else 0.0,
            "ID_recall": tracking["IDTP"] / (tracking["IDTP"] + tracking["IDFN"]) if tracking["IDTP"] + tracking["IDFN"] else 0.0,
            "unique_gt_track_tokens": len(gt_track_ids_seen),
            "unique_prediction_track_ids": len(prediction_track_ids_seen),
            "matched_gt_track_tokens": len(matched_gt_track_ids),
            "matched_prediction_track_ids": len(matched_prediction_track_ids),
            "tracked_prediction_detections": tracked_prediction_detections,
            "untracked_prediction_detections": untracked_prediction_detections,
            "untracked_matches": identity_untracked_matches,
            "track_id_coverage": tracked_prediction_detections / total_pred if total_pred else 0.0,
        }
    )
    timestamps = [int(record["timestamp"]) for record in records]
    return {
        "metadata": str(args.metadata),
        "pred_csv": str(args.pred_csv),
        "camera": args.camera_name,
        "iou_threshold": args.iou_threshold,
        "image_shape": {"height": image_shape[0], "width": image_shape[1]},
        "frames": len(records),
        "timestamp_fps": (len(timestamps) - 1) / ((timestamps[-1] - timestamps[0]) / 1e6),
        "prediction_to_gt_class": pred_to_gt,
        "scored_gt_classes": list(gt_to_pred),
        "ignored_gt_boxes_other_classes": ignored_gt_boxes,
        "projected_gt_boxes": projected_boxes,
        "ignored_prediction_classes": dict(ignored_predictions),
        "detection": {
            "GT": total_gt,
            "predictions": total_pred,
            "TP": true_positive,
            "FP": false_positive,
            "FN": false_negative,
            "precision": precision,
            "recall": recall,
            "F1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
            "AP_at_iou": average_precision(scores, score_tp, total_gt),
            "mean_IoU_TP": float(np.mean(matched_ious)) if matched_ious else 0.0,
        },
        "per_class": {gt_class: summarize(values) for gt_class, values in per_class.items()},
        "tracking": tracking,
    }


def render_report(metrics: dict[str, Any]) -> str:
    detection = metrics["detection"]
    tracking = metrics["tracking"]
    lines = [
        "# VRU detection and tracking evaluation",
        "",
        f"- Metadata: `{metrics['metadata']}`",
        f"- Prediction CSV: `{metrics['pred_csv']}`",
        f"- Camera: `{metrics['camera']}`",
        f"- IoU threshold: `{metrics['iou_threshold']}`",
        f"- Frames: `{metrics['frames']}` at `{metrics['timestamp_fps']:.6f} FPS`",
        "",
        "## Detection metrics",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| GT boxes scored | {detection['GT']} |",
        f"| Predictions scored | {detection['predictions']} |",
        f"| TP / FP / FN | {detection['TP']} / {detection['FP']} / {detection['FN']} |",
        f"| Precision | {detection['precision']:.4f} |",
        f"| Recall | {detection['recall']:.4f} |",
        f"| F1 | {detection['F1']:.4f} |",
        f"| AP at IoU {metrics['iou_threshold']} | {detection['AP_at_iou']:.4f} |",
        f"| Mean IoU of TP | {detection['mean_IoU_TP']:.4f} |",
        "",
        "| GT class | GT | Pred | TP | FP | FN | Precision | Recall | F1 | AP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, values in metrics["per_class"].items():
        lines.append(
            f"| {name} | {values['gt']} | {values['pred']} | {values['TP']} | {values['FP']} | {values['FN']} | "
            f"{values['precision']:.4f} | {values['recall']:.4f} | {values['F1']:.4f} | {values['AP']:.4f} |"
        )
    lines += [
        "",
        "## Tracking metrics",
        "",
        f"- MOTA: `{tracking['MOTA']:.4f}`",
        f"- IDF1: `{tracking['IDF1']:.4f}`",
        f"- IDTP / IDFP / IDFN: `{tracking['IDTP']} / {tracking['IDFP']} / {tracking['IDFN']}`",
        f"- ID switches: `{tracking['ID_switches']}`",
        "",
        "Only GT `pedestrian` and `bicycle` are scored. The shard has no separate",
        "`rider` or `motorcycle` GT label; unsupported prediction classes are ignored",
        "and reported in the JSON rather than counted as false positives.",
        "",
        f"Ignored GT boxes from other classes: `{metrics['ignored_gt_boxes_other_classes']}`.",
        f"Projected GT boxes: `{metrics['projected_gt_boxes']}`.",
    ]
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    metrics = evaluate(args)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    args.output_report.write_text(render_report(metrics), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved JSON report: {args.output_json}")
    print(f"Saved Markdown report: {args.output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
