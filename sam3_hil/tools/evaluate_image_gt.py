#!/usr/bin/env python3
"""Evaluate Image Set E COCO exports against COCO GT using an image manifest."""

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np

from tools.evaluate_video_gt import _annotation_to_mask, _image_maps, _match_iou


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_image_manifest(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    mapping: dict[str, dict[str, str]] = {}
    for row in rows:
        image_id = row["image_id"]
        filename = Path(row["original_filename"]).name
        mapping[filename] = {
            "image_id": image_id,
            "original_filename": filename,
            "stratum": row.get("stratum", ""),
        }
    return mapping


def _masks_by_filename(coco: dict[str, Any]) -> dict[str, list[np.ndarray]]:
    images, annotations_by_image = _image_maps(coco)
    masks_by_filename: dict[str, list[np.ndarray]] = {}

    for coco_image_id, image in images.items():
        filename = Path(str(image.get("file_name", ""))).name
        width = int(image["width"])
        height = int(image["height"])
        frame_masks = [
            _annotation_to_mask(annotation, width, height)
            for annotation in annotations_by_image.get(coco_image_id, [])
        ]
        masks_by_filename.setdefault(filename, []).extend(frame_masks)

    return masks_by_filename


def _write_per_image(rows: Iterable[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "original_filename",
        "stratum",
        "gt_count",
        "pred_count",
        "matched_count",
        "missed_count",
        "false_positive_count",
        "image_iou",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate_image_coco(
    pred_coco_path: Path,
    gt_coco_path: Path,
    manifest_path: Path,
    output_path: Path,
    summary_path: Optional[Path] = None,
) -> dict[str, Any]:
    manifest = _load_image_manifest(Path(manifest_path))
    pred_masks = _masks_by_filename(_load_json(Path(pred_coco_path)))
    gt_masks = _masks_by_filename(_load_json(Path(gt_coco_path)))

    rows: list[dict[str, Any]] = []
    total_gt = 0
    total_pred = 0
    total_matched = 0
    total_missed = 0
    total_false_positive = 0
    image_scores: list[float] = []

    for filename in sorted(manifest):
        info = manifest[filename]
        gt_image_masks = gt_masks.get(filename, [])
        pred_image_masks = pred_masks.get(filename, [])
        matched_ious = _match_iou(gt_image_masks, pred_image_masks)
        denominator = max(len(gt_image_masks), len(pred_image_masks))
        image_iou = sum(matched_ious) / denominator if denominator else ""
        if denominator:
            image_scores.append(float(image_iou))

        matched_count = len(matched_ious)
        missed_count = len(gt_image_masks) - matched_count
        false_positive_count = len(pred_image_masks) - matched_count

        total_gt += len(gt_image_masks)
        total_pred += len(pred_image_masks)
        total_matched += matched_count
        total_missed += missed_count
        total_false_positive += false_positive_count

        rows.append(
            {
                "image_id": info["image_id"],
                "original_filename": info["original_filename"],
                "stratum": info["stratum"],
                "gt_count": len(gt_image_masks),
                "pred_count": len(pred_image_masks),
                "matched_count": matched_count,
                "missed_count": missed_count,
                "false_positive_count": false_positive_count,
                "image_iou": "" if image_iou == "" else f"{image_iou:.6f}",
            }
        )

    _write_per_image(rows, Path(output_path))

    summary = {
        "images_evaluated": len(manifest),
        "mean_image_iou": round(float(np.mean(image_scores)), 6) if image_scores else None,
        "total_gt_instances": total_gt,
        "total_pred_instances": total_pred,
        "total_matched": total_matched,
        "total_missed": total_missed,
        "total_false_positive": total_false_positive,
    }
    if summary_path is not None:
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate Image Set E COCO predictions against COCO GT."
    )
    parser.add_argument("--pred-coco", type=Path, required=True)
    parser.add_argument("--gt-coco", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = evaluate_image_coco(
        pred_coco_path=args.pred_coco,
        gt_coco_path=args.gt_coco,
        manifest_path=args.manifest,
        output_path=args.output,
        summary_path=args.summary,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
