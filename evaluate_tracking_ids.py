"""Report only ByteTrack track-ID metrics against NavSim track_tokens."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import evaluate_vru_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pred-csv", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--sensor-root", type=Path, required=True)
    parser.add_argument("--camera-name", default="CAM_F0")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--class-map", default="person:pedestrian,bicycle:bicycle")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-report", type=Path, required=True)
    return parser.parse_args()


def render_report(metrics: dict) -> str:
    tracking = metrics["tracking"]
    rows = [
        ("ID Precision", tracking["ID_precision"]),
        ("ID Recall", tracking["ID_recall"]),
        ("IDF1", tracking["IDF1"]),
        ("MOTA", tracking["MOTA"]),
        ("IDTP / IDFP / IDFN", f"{tracking['IDTP']} / {tracking['IDFP']} / {tracking['IDFN']}"),
        ("ID switches", tracking["ID_switches"]),
        ("Untracked prediction detections", tracking["untracked_prediction_detections"]),
        ("Untracked GT matches", tracking["untracked_matches"]),
        ("Track-ID coverage", tracking["track_id_coverage"]),
        ("Unique GT track_tokens", tracking["unique_gt_track_tokens"]),
        ("Unique predicted track IDs", tracking["unique_prediction_track_ids"]),
        ("Matched GT track_tokens", tracking["matched_gt_track_tokens"]),
        ("Matched predicted track IDs", tracking["matched_prediction_track_ids"]),
    ]
    lines = [
        "# ByteTrack identity-only evaluation",
        "",
        f"- Prediction CSV: `{metrics['pred_csv']}`",
        f"- NavSim metadata: `{metrics['metadata']}`",
        f"- Camera: `{metrics['camera']}`",
        f"- Association IoU threshold: `{metrics['iou_threshold']}`",
        "- GT identity source: `anns.track_tokens`",
        "- Prediction identity source: ByteTrack `track_id`",
        "",
        "This report evaluates track identity only. Detection AP/precision/recall are intentionally excluded.",
        "A prediction with `track_id=-1` is treated as untracked and cannot contribute IDTP.",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for name, value in rows:
        if isinstance(value, float):
            value = f"{value:.4f}"
        lines.append(f"| {name} | {value} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    metrics = evaluate_vru_metrics.evaluate(args)
    tracking_only = {
        "metadata": metrics["metadata"],
        "pred_csv": metrics["pred_csv"],
        "camera": metrics["camera"],
        "iou_threshold": metrics["iou_threshold"],
        "frames": metrics["frames"],
        "prediction_to_gt_class": metrics["prediction_to_gt_class"],
        "tracking": metrics["tracking"],
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(tracking_only, indent=2), encoding="utf-8")
    args.output_report.write_text(render_report(tracking_only), encoding="utf-8")
    print(json.dumps(tracking_only, indent=2))
    print(f"Saved JSON report: {args.output_json}")
    print(f"Saved Markdown report: {args.output_report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
