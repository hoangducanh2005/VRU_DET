"""Task 2: VRU detection and tracking with Ultralytics + ByteTrack.

The detector is loaded from an external YOLO26n-BDD100K checkpoint.  This
runner intentionally does not change the model architecture: switching from
the FP32 PyTorch checkpoint to an ONNX export only changes ``--model``.

Input can be a video file or a directory of images.  For a directory, files
are processed by file modification timestamp by default.  The NavSim mini
image names are hashes, so lexicographic filename order is not temporal.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Iterable, Iterator

import cv2
import numpy as np


DEFAULT_INPUT = Path(
    r"D:\navsim_workspace\dataset\sensor_blobs\mini\2021.06.09.14.58.55_veh-35_01894_02311\CAM_F0"
)
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
VRU_CLASSES = ("person", "rider", "bicycle", "motorcycle")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="YOLO26n-BDD100K .pt or .onnx model")
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Video file or image directory (default: NavSim CAM_F0)",
    )
    parser.add_argument(
        "--image-sort",
        choices=("mtime", "name"),
        default="mtime",
        help="Ordering for image folders: file timestamp (default) or filename",
    )
    parser.add_argument("--output-video", type=Path, default=Path("output/task2_vru_tracking.mp4"))
    parser.add_argument("--output-csv", type=Path, default=Path("output/task2_tracks.csv"))
    parser.add_argument("--metrics-json", type=Path, default=Path("output/task2_metrics.json"))
    parser.add_argument(
        "--tracker",
        default="bytetrack",
        help="bytetrack, botsort, or a path to an Ultralytics tracker YAML",
    )
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument(
        "--imgsz",
        type=int,
        nargs="+",
        default=[640],
        help="Inference size: one value for square or HEIGHT WIDTH for rectangular ONNX",
    )
    parser.add_argument("--device", default=None, help="Ultralytics device, e.g. 0, cuda:0, or cpu")
    parser.add_argument("--fps", type=float, default=20.0, help="FPS used for an image directory")
    parser.add_argument("--max-frames", type=int, default=0, help="0 means process all frames")
    parser.add_argument("--show", action="store_true", help="Display the annotated stream")
    parser.add_argument(
        "--step",
        action="store_true",
        help="Show one frame and wait for a key before processing the next frame",
    )
    parser.add_argument(
        "--delay-ms",
        type=int,
        default=0,
        help="Playback delay in milliseconds; ignored in --step mode",
    )
    parser.add_argument(
        "--save-frames-dir",
        type=Path,
        default=None,
        help="Optional directory for individual annotated frame JPEGs",
    )
    return parser.parse_args()


def resolve_tracker(value: str) -> str:
    aliases = {"bytetrack": "bytetrack.yaml", "botsort": "botsort.yaml"}
    return aliases.get(value.lower(), value)


def iter_image_paths(folder: Path, sort_by: str) -> list[Path]:
    paths = [
        path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if sort_by == "mtime":
        # NavSim stores capture time in the extracted file mtime.  The hash
        # filename itself has no temporal meaning.  Name is only a tie-breaker
        # because this shard preserves mtime at one-second resolution.
        return sorted(paths, key=lambda path: (path.stat().st_mtime_ns, path.name.lower()))
    return sorted(paths, key=lambda path: path.name.lower())


def is_image_sequence(path: Path) -> bool:
    return path.is_dir()


def iter_frames(
    source: Path, directory_fps: float, image_sort: str
) -> Iterator[tuple[int, np.ndarray, float, float]]:
    """Yield frame number, BGR image, source FPS, and frame-start timestamp."""
    if is_image_sequence(source):
        paths = iter_image_paths(source, image_sort)
        if not paths:
            raise FileNotFoundError(f"No supported images found in {source}")
        for frame_id, image_path in enumerate(paths):
            frame_start = time.perf_counter()
            frame = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if frame is not None:
                yield frame_id, frame, directory_fps, frame_start
        return

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {source}")
    source_fps = capture.get(cv2.CAP_PROP_FPS)
    if not source_fps or source_fps <= 0:
        source_fps = directory_fps
    frame_id = 0
    try:
        while True:
            frame_start = time.perf_counter()
            ok, frame = capture.read()
            if not ok:
                break
            yield frame_id, frame, source_fps, frame_start
            frame_id += 1
    finally:
        capture.release()


def load_class_ids(model) -> list[int]:
    names = model.names
    if isinstance(names, dict):
        name_by_id = {int(index): str(name).lower() for index, name in names.items()}
    else:
        name_by_id = {index: str(name).lower() for index, name in enumerate(names)}
    missing = [name for name in VRU_CLASSES if name not in name_by_id.values()]
    if missing:
        raise RuntimeError(f"Model does not provide required classes: {', '.join(missing)}")
    return [index for index, name in name_by_id.items() if name in VRU_CLASSES]


def sync_cuda(device: object) -> None:
    """Synchronize CUDA when timing a PyTorch CUDA model."""
    try:
        import torch

        requested = "" if device is None else str(device).lower()
        if requested != "cpu" and torch.cuda.is_available():
            torch.cuda.synchronize()
    except ImportError:
        pass


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        return 0.0
    return float(np.percentile(np.asarray(values, dtype=np.float64), fraction))


def draw_tracks(frame: np.ndarray, result, names: dict[int, str]) -> list[dict[str, object]]:
    boxes = result.boxes
    if boxes is None or len(boxes) == 0:
        return []

    xyxy = boxes.xyxy.detach().cpu().numpy()
    classes = boxes.cls.detach().cpu().numpy().astype(int)
    confidence = boxes.conf.detach().cpu().numpy()
    if boxes.id is None:
        track_ids = np.full(len(xyxy), -1, dtype=np.int64)
    else:
        track_ids = boxes.id.detach().cpu().numpy().round().astype(np.int64)

    records: list[dict[str, object]] = []
    for coordinates, class_id, score, track_id in zip(xyxy, classes, confidence, track_ids):
        x1, y1, x2, y2 = [int(round(value)) for value in coordinates]
        class_name = names.get(int(class_id), str(class_id))
        # Stable, readable colours without relying on a model-specific plotter.
        colour = (50 + (int(class_id) * 67) % 180, 80 + (int(class_id) * 43) % 150, 230)
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 2)
        label = f"{class_name} {float(score):.2f} id={int(track_id)}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        text_top = max(0, y1 - text_height - baseline - 3)
        cv2.rectangle(frame, (x1, text_top), (x1 + text_width + 4, y1), colour, -1)
        cv2.putText(
            frame,
            label,
            (x1 + 2, max(text_height + 1, y1 - 3)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        records.append(
            {
                "track_id": int(track_id),
                "class": class_name,
                "confidence": float(score),
                "bbox": [x1, y1, x2, y2],
            }
        )
    return records


def main() -> int:
    args = parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"Input does not exist: {args.input}")
    if args.delay_ms < 0:
        raise ValueError("--delay-ms must be >= 0")
    args.output_video.parent.mkdir(parents=True, exist_ok=True)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_json.parent.mkdir(parents=True, exist_ok=True)
    if args.save_frames_dir is not None:
        args.save_frames_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError(
            "Ultralytics is not installed. Follow README_TASK2.md requirement 1 first."
        ) from exc

    model = YOLO(str(args.model))
    class_ids = load_class_ids(model)
    names = model.names if isinstance(model.names, dict) else dict(enumerate(model.names))
    names = {int(index): str(name) for index, name in names.items()}
    tracker = resolve_tracker(args.tracker)
    inference_size: int | list[int] = args.imgsz[0] if len(args.imgsz) == 1 else args.imgsz

    output_writer = None
    csv_file = None
    csv_writer = None
    inference_ms: list[float] = []
    end_to_end_ms: list[float] = []
    frame_count = 0
    detection_count = 0
    first_frame_time = time.perf_counter()

    try:
        csv_file = args.output_csv.open("w", newline="", encoding="utf-8")
        csv_writer = csv.DictWriter(
            csv_file,
            fieldnames=["frame_id", "track_id", "class", "confidence", "bbox"],
        )
        csv_writer.writeheader()

        for frame_id, frame, source_fps, frame_start in iter_frames(
            args.input, args.fps, args.image_sort
        ):
            if args.max_frames and frame_count >= args.max_frames:
                break
            if output_writer is None:
                height, width = frame.shape[:2]
                output_writer = cv2.VideoWriter(
                    str(args.output_video),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    source_fps,
                    (width, height),
                )
                if not output_writer.isOpened():
                    raise RuntimeError(f"Could not create output video: {args.output_video}")

            sync_cuda(args.device)
            inference_start = time.perf_counter()
            results = model.track(
                frame,
                persist=True,
                tracker=tracker,
                classes=class_ids,
                conf=args.conf,
                iou=args.iou,
                imgsz=inference_size,
                device=args.device,
                verbose=False,
            )
            sync_cuda(args.device)
            inference_ms.append((time.perf_counter() - inference_start) * 1000.0)

            result = results[0]
            records = draw_tracks(frame, result, names)
            for record in records:
                csv_writer.writerow(
                    {
                        "frame_id": frame_id,
                        "track_id": record["track_id"],
                        "class": record["class"],
                        "confidence": f"{record['confidence']:.6f}",
                        "bbox": json.dumps(record["bbox"], separators=(",", ":")),
                    }
                )
            detection_count += len(records)
            output_writer.write(frame)
            frame_count += 1
            end_to_end_ms.append((time.perf_counter() - frame_start) * 1000.0)

            if args.save_frames_dir is not None:
                frame_path = args.save_frames_dir / f"frame_{frame_id:06d}.jpg"
                if not cv2.imwrite(str(frame_path), frame):
                    raise RuntimeError(f"Could not save frame: {frame_path}")

            if args.show or args.step:
                cv2.imshow("VRU Detection + Tracking", frame)
                wait_ms = 0 if args.step else max(1, args.delay_ms)
                if cv2.waitKey(wait_ms) & 0xFF == ord("q"):
                    break
    finally:
        if output_writer is not None:
            output_writer.release()
        if csv_file is not None:
            csv_file.close()
        cv2.destroyAllWindows()

    elapsed_s = max(time.perf_counter() - first_frame_time, 1e-9)
    metrics = {
        "model": str(args.model),
        "input": str(args.input),
        "image_sort": args.image_sort if args.input.is_dir() else None,
        "tracker": tracker,
        "classes": list(VRU_CLASSES),
        "frames": frame_count,
        "detections": detection_count,
        "inference_latency_ms": {
            "mean": float(np.mean(inference_ms)) if inference_ms else 0.0,
            "p50": percentile(inference_ms, 50),
            "p95": percentile(inference_ms, 95),
        },
        "end_to_end_latency_ms": {
            "mean": float(np.mean(end_to_end_ms)) if end_to_end_ms else 0.0,
            "p50": percentile(end_to_end_ms, 50),
            "p95": percentile(end_to_end_ms, 95),
        },
        "effective_fps": frame_count / elapsed_s if frame_count else 0.0,
    }
    args.metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Saved video: {args.output_video}")
    print(f"Saved CSV:   {args.output_csv}")
    print(f"Saved stats: {args.metrics_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
