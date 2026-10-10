"""Run the official CSP pedestrian detector on a folder of images.

The model, preprocessing and decoder are imported from the cloned
``csp.pytorch`` repository.  This file only adds folder traversal, device
selection, visualization and result reporting around that implementation.
"""

from __future__ import annotations

import argparse
import contextlib
import io
import sys
from pathlib import Path

import cv2
import numpy as np
import torch


REPO_DIR = Path(__file__).resolve().parent / "csp.pytorch"
if not REPO_DIR.is_dir():
    raise FileNotFoundError(
        f"CSP repository not found at {REPO_DIR}. Clone "
        "https://github.com/wang-xinyu/csp.pytorch into csp.pytorch first."
    )
sys.path.insert(0, str(REPO_DIR))

from models.cspnet import CSPNet_p3p4p5  # noqa: E402
from utils.keras_weights_loader import load_keras_weights  # noqa: E402
from utils.utils import format_img, parse_det_offset  # noqa: E402


DEFAULT_INPUT_DIR = Path(
    r"D:\navsim_workspace\dataset\sensor_blobs\mini"
) / "2021.06.09.14.58.55_veh-35_01894_02311"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="CSP ResNet-50 pedestrian inference for a local image folder."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help=f"Input folder (default: {DEFAULT_INPUT_DIR})",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("output"),
        help="Output folder; input subdirectories are preserved (default: output)",
    )
    parser.add_argument(
        "--weights",
        type=Path,
        default=Path("net_e121_l0.hdf5"),
        help="CSP Keras HDF5 checkpoint (default: net_e121_l0.hdf5)",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.1,
        help="Center confidence threshold before NMS (default: 0.1, as test_city.py)",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Inference device (default: auto; CUDA is preferred when available)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process at most N images; useful for a smoke test",
    )
    return parser.parse_args()


def select_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError(
                "--device cuda was requested, but torch.cuda.is_available() is False. "
                "Install a CUDA-enabled PyTorch wheel or use --device cpu."
            )
        return torch.device("cuda:0")
    if requested == "cpu":
        return torch.device("cpu")
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def image_files(input_dir: Path) -> list[Path]:
    return sorted(
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def draw_detections(image: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    rendered = image.copy()
    for x1, y1, x2, y2, score in boxes:
        left, top = int(round(x1)), int(round(y1))
        right, bottom = int(round(x2)), int(round(y2))
        cv2.rectangle(rendered, (left, top), (right, bottom), (0, 0, 255), 2)
        label = f"pedestrian {score:.3f}"
        (text_width, text_height), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1
        )
        label_top = max(0, top - text_height - baseline - 4)
        cv2.rectangle(
            rendered,
            (left, label_top),
            (left + text_width + 6, top),
            (0, 0, 255),
            thickness=-1,
        )
        cv2.putText(
            rendered,
            label,
            (left + 3, max(text_height + 1, top - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return rendered


def run_one(
    model: torch.nn.Module,
    image: np.ndarray,
    device: torch.device,
    score_threshold: float,
) -> np.ndarray:
    # This is the repository's exact BGR mean-subtraction preprocessing.
    formatted = format_img(image)
    input_dim = formatted.shape[1:3]
    tensor = torch.from_numpy(formatted).to(device=device)
    tensor = tensor.permute(0, 3, 1, 2)

    # The upstream model contains diagnostic print calls in every residual
    # block. Keep those logs out of per-image output without changing the model.
    with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
        x_cls, x_reg, x_off = model(tensor)
        outputs = [
            x_cls.detach().cpu().numpy(),
            x_reg.detach().cpu().numpy(),
            x_off.detach().cpu().numpy(),
        ]
        boxes = parse_det_offset(
            outputs, input_dim, score=score_threshold, down=4
        )
    if len(boxes) == 0:
        return np.empty((0, 5), dtype=np.float32)
    return np.asarray(boxes, dtype=np.float32)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    output_dir = args.output_dir.resolve()
    weights_path = args.weights.resolve()

    if not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not weights_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {weights_path}\n"
            "Download net_e121_l0.hdf5 yourself as described in README.md, "
            "then pass --weights with its path."
        )
    if not 0.0 <= args.score_threshold <= 1.0:
        raise ValueError("--score-threshold must be between 0 and 1")

    files = image_files(input_dir)
    if args.limit is not None:
        if args.limit <= 0:
            raise ValueError("--limit must be positive")
        files = files[: args.limit]
    if not files:
        raise FileNotFoundError(f"No supported images found below {input_dir}")

    device = select_device(args.device)
    print(f"device={device}")
    print(f"input_images={len(files)}")
    print(f"weights={weights_path}")

    model = CSPNet_p3p4p5()
    load_keras_weights(model, str(weights_path))
    model.to(device).eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    failures = 0
    for index, image_path in enumerate(files, start=1):
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            failures += 1
            print(f"[{index}/{len(files)}] {image_path} | ERROR: unreadable image")
            continue

        try:
            boxes = run_one(model, image, device, args.score_threshold)
            rendered = draw_detections(image, boxes)
            relative_path = image_path.relative_to(input_dir)
            output_path = output_dir / relative_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(output_path), rendered):
                raise OSError(f"could not write {output_path}")
            print(
                f"[{index}/{len(files)}] {relative_path} | "
                f"pedestrians={len(boxes)} | output={output_path}"
            )
        except Exception as exc:
            failures += 1
            print(f"[{index}/{len(files)}] {image_path} | ERROR: {exc}")

    print(f"completed={len(files) - failures} failed={failures} output={output_dir}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
