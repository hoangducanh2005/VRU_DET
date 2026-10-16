"""Overlay projected NavSim 3D VRU boxes and their 2D enclosing boxes."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np

from evaluate_vru_metrics import (
    box_corners_lidar,
    lidar_to_camera,
    project_box,
    resolve_image,
)


EDGES = (
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def project_points(points_camera: np.ndarray, intrinsic: np.ndarray) -> np.ndarray:
    homogeneous = points_camera @ intrinsic.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--camera-name", default="CAM_F0")
    parser.add_argument("--frame", type=int, default=0)
    parser.add_argument("--classes", default="pedestrian,bicycle")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    records = pickle.loads(args.metadata.read_bytes())
    if not records:
        raise RuntimeError("Metadata contains no records")
    if not 0 <= args.frame < len(records):
        raise IndexError(f"frame must be in [0, {len(records) - 1}]")

    record = records[args.frame]
    camera = record["cams"][args.camera_name]
    image_path = resolve_image(args.sensor_root, camera["data_path"])
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    selected = {name.strip() for name in args.classes.split(",") if name.strip()}
    anns = record.get("anns", {})
    boxes = np.asarray(anns.get("gt_boxes", np.empty((0, 7))), dtype=np.float64).reshape(-1, 7)
    names = np.asarray(anns.get("gt_names", np.empty((0,), dtype=str))).astype(str).tolist()
    tokens = [str(token) for token in anns.get("track_tokens", [])]

    drawn = 0
    for index, (box, name) in enumerate(zip(boxes, names)):
        if name not in selected:
            continue
        corners_camera = lidar_to_camera(box_corners_lidar(box), camera)
        visible = corners_camera[:, 2] > 1e-3
        if not visible.any():
            continue
        projected = project_box(box, camera, (height, width))
        if projected is None:
            continue
        intrinsic = np.asarray(camera["cam_intrinsic"], dtype=np.float64)
        points = corners_camera[visible]
        pixels = project_points(points, intrinsic)
        pixels = np.rint(pixels).astype(np.int32)
        all_pixels = project_points(corners_camera, intrinsic)

        # Green: projected 3D box edges; red: enclosing 2D bbox.
        for start, end in EDGES:
            if visible[start] and visible[end]:
                cv2.line(image, tuple(np.rint(all_pixels[start]).astype(int)), tuple(np.rint(all_pixels[end]).astype(int)), (0, 255, 0), 2)
        x1, y1, x2, y2 = np.rint(projected).astype(int).tolist()
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"{name} {tokens[index][:8] if index < len(tokens) else index}"
        cv2.putText(image, label, (x1, max(18, y1 - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2, cv2.LINE_AA)
        drawn += 1

    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / f"frame_{args.frame:06d}_projection.jpg"
    cv2.imwrite(str(output), image)
    print(f"Input: {image_path}")
    print(f"Frame: {record.get('frame_idx', args.frame)}, timestamp: {record.get('timestamp')}")
    print(f"Classes: {sorted(selected)}, projected boxes: {drawn}")
    print(f"Saved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
