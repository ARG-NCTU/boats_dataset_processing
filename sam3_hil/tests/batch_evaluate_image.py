#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch evaluate Image Set E outputs using image-level foreground IoU.

For each image:
1. Union all GT dolphin annotations into one foreground mask.
2. Union all prediction annotations into one foreground mask.
3. Compute IoU between the two foreground masks.

GT files such as E_0003.jpg and prediction files such as frame_0003.jpg
are aligned using the numeric image key 0003.
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent

TOOLS = {
    "labelme": "LabelMe",
    "cvat": "CVAT+SAM",
    "stamp": "STAMP",
}

# Assignment from STAMP_assignment_table.xlsx.
ASSIGNMENTS = {
    1: {"labelme": "E1", "cvat": "E2", "stamp": "E3"},
    2: {"labelme": "E2", "cvat": "E3", "stamp": "E4"},
    3: {"labelme": "E3", "cvat": "E4", "stamp": "E1"},
    4: {"labelme": "E4", "cvat": "E1", "stamp": "E2"},
    5: {"labelme": "E1", "cvat": "E3", "stamp": "E4"},
    6: {"labelme": "E2", "cvat": "E4", "stamp": "E1"},
    7: {"labelme": "E3", "cvat": "E1", "stamp": "E2"},
    8: {"labelme": "E4", "cvat": "E2", "stamp": "E3"},
}

IMAGE_KEY_RE = re.compile(r"(\d+)$")


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_image_key(file_name: str) -> str:
    """Extract a normalized numeric key from an image filename.

    Examples:
        E_0003.jpg     -> 0003
        frame_0003.jpg -> 0003
    """
    stem = Path(file_name).stem
    match = IMAGE_KEY_RE.search(stem)

    if not match:
        raise ValueError(
            "Cannot extract numeric image key from filename: "
            f"{file_name}"
        )

    return match.group(1).zfill(4)


def normalize_polygons(segmentation: Any) -> List[List[float]]:
    if not isinstance(segmentation, list):
        return []

    if not segmentation:
        return []

    # Also support one flat COCO polygon.
    if all(isinstance(value, (int, float)) for value in segmentation):
        return [segmentation]

    return segmentation


def annotation_to_mask(
    annotation: Dict[str, Any],
    width: int,
    height: int,
) -> np.ndarray:
    """Convert one COCO polygon or RLE annotation into a boolean mask."""
    segmentation = annotation.get("segmentation", [])

    if isinstance(segmentation, list):
        mask = np.zeros((height, width), dtype=np.uint8)

        for polygon in normalize_polygons(segmentation):
            if len(polygon) < 6:
                continue

            points = np.asarray(
                polygon,
                dtype=np.float32,
            ).reshape(-1, 2)

            points = np.rint(points).astype(np.int32)
            cv2.fillPoly(mask, [points], 1)

        return mask.astype(bool)

    if isinstance(segmentation, dict):
        try:
            from pycocotools import mask as mask_utils
        except ImportError as exc:
            raise ImportError(
                "pycocotools is required for COCO RLE segmentation"
            ) from exc

        decoded = mask_utils.decode(segmentation)

        # Some RLE decoders may return H x W x N.
        if decoded.ndim == 3:
            decoded = np.any(decoded, axis=2)

        return decoded.astype(bool)

    return np.zeros((height, width), dtype=bool)


def coco_foreground_masks(
    coco: Dict[str, Any],
) -> Tuple[
    Dict[str, np.ndarray],
    Dict[str, str],
]:
    """Union all annotations in each image into one foreground mask."""
    images = {
        int(image["id"]): image
        for image in coco.get("images", [])
    }

    annotations_by_image: Dict[int, List[Dict[str, Any]]] = {
        image_id: []
        for image_id in images
    }

    for annotation in coco.get("annotations", []):
        image_id = int(annotation["image_id"])
        annotations_by_image.setdefault(image_id, []).append(annotation)

    masks_by_key: Dict[str, np.ndarray] = {}
    filenames_by_key: Dict[str, str] = {}

    for image_id, image in images.items():
        file_name = Path(
            str(image.get("file_name", ""))
        ).name

        image_key = parse_image_key(file_name)
        width = int(image["width"])
        height = int(image["height"])

        if image_key in masks_by_key:
            raise ValueError(
                f"Duplicate image key {image_key} in COCO file"
            )

        foreground = np.zeros(
            (height, width),
            dtype=bool,
        )

        for annotation in annotations_by_image.get(image_id, []):
            annotation_mask = annotation_to_mask(
                annotation,
                width,
                height,
            )
            foreground = np.logical_or(
                foreground,
                annotation_mask,
            )

        masks_by_key[image_key] = foreground
        filenames_by_key[image_key] = file_name

    return masks_by_key, filenames_by_key


def mask_iou(
    gt_mask: np.ndarray,
    pred_mask: np.ndarray,
) -> Optional[float]:
    if gt_mask.shape != pred_mask.shape:
        raise ValueError(
            "GT and prediction mask dimensions do not match: "
            f"{gt_mask.shape} vs {pred_mask.shape}"
        )

    intersection = np.logical_and(
        gt_mask,
        pred_mask,
    ).sum()

    union = np.logical_or(
        gt_mask,
        pred_mask,
    ).sum()

    if union == 0:
        return None

    return float(intersection / union)


def write_per_image(
    rows: Iterable[Dict[str, Any]],
    output_path: Path,
) -> None:
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "image_key",
        "gt_filename",
        "pred_filename",
        "stratum",
        "gt_present",
        "pred_present",
        "image_iou",
    ]

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def evaluate_image_foreground(
    pred_coco_path: Path,
    gt_coco_path: Path,
    stratum: str,
    output_path: Path,
    summary_path: Optional[Path] = None,
) -> Dict[str, Any]:
    pred_coco = load_json(Path(pred_coco_path))
    gt_coco = load_json(Path(gt_coco_path))

    pred_masks, pred_filenames = coco_foreground_masks(
        pred_coco
    )
    gt_masks, gt_filenames = coco_foreground_masks(
        gt_coco
    )

    # Include keys from both files so missing and extra images are visible.
    image_keys = sorted(
        set(gt_masks) | set(pred_masks)
    )

    rows: List[Dict[str, Any]] = []
    image_scores: List[float] = []

    gt_foreground_images = 0
    pred_foreground_images = 0
    missing_prediction_images = 0
    extra_prediction_images = 0

    for image_key in image_keys:
        gt_mask = gt_masks.get(image_key)
        pred_mask = pred_masks.get(image_key)

        if gt_mask is not None and pred_mask is not None:
            if gt_mask.shape != pred_mask.shape:
                raise ValueError(
                    f"Image {image_key} dimensions do not match: "
                    f"GT {gt_mask.shape}, prediction {pred_mask.shape}"
                )

            shape = gt_mask.shape

        elif gt_mask is not None:
            shape = gt_mask.shape
            pred_mask = np.zeros(shape, dtype=bool)

        elif pred_mask is not None:
            shape = pred_mask.shape
            gt_mask = np.zeros(shape, dtype=bool)

        else:
            continue

        gt_present = bool(np.any(gt_mask))
        pred_present = bool(np.any(pred_mask))

        if gt_present:
            gt_foreground_images += 1

        if pred_present:
            pred_foreground_images += 1

        if gt_present and not pred_present:
            missing_prediction_images += 1

        if pred_present and not gt_present:
            extra_prediction_images += 1

        image_iou = mask_iou(
            gt_mask,
            pred_mask,
        )

        if image_iou is not None:
            image_scores.append(image_iou)

        rows.append(
            {
                "image_key": image_key,
                "gt_filename": gt_filenames.get(
                    image_key,
                    "",
                ),
                "pred_filename": pred_filenames.get(
                    image_key,
                    "",
                ),
                "stratum": stratum,
                "gt_present": int(gt_present),
                "pred_present": int(pred_present),
                "image_iou": (
                    ""
                    if image_iou is None
                    else f"{image_iou:.6f}"
                ),
            }
        )

    write_per_image(
        rows,
        Path(output_path),
    )

    summary = {
        "images_evaluated": len(image_keys),
        "mean_image_iou": (
            round(float(np.mean(image_scores)), 6)
            if image_scores
            else None
        ),
        "gt_foreground_images": gt_foreground_images,
        "pred_foreground_images": pred_foreground_images,
        "missing_prediction_images": (
            missing_prediction_images
        ),
        "extra_prediction_images": (
            extra_prediction_images
        ),
    }

    if summary_path is not None:
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        summary_path.write_text(
            json.dumps(
                summary,
                indent=2,
            ),
            encoding="utf-8",
        )

    return summary


def find_prediction(
    participant_dir: Path,
    participant: int,
    tool: str,
) -> Optional[Path]:
    tagname = f"p{participant}ie"

    if tool == "labelme":
        path = (
            participant_dir
            / "LabelMe"
            / f"{tagname}_labelme_coco.json"
        )

    elif tool == "cvat":
        path = (
            participant_dir
            / "CVAT"
            / "annotations"
            / "instances_default.json"
        )

    elif tool == "stamp":
        path = (
            participant_dir
            / "STAMP"
            / "annotations"
            / "instances_train2024.json"
        )

    else:
        raise ValueError(f"Unknown tool: {tool}")

    return path if path.exists() else None


def resolve_gt(
    exp_dir: Path,
    stratum: str,
) -> Optional[Path]:
    path = (
        exp_dir
        / "GT"
        / f"{stratum}GT"
        / "annotations"
        / "instances_default.json"
    )

    return path if path.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Batch evaluate Image Set E using image-level "
            "foreground IoU."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
    )
    parser.add_argument(
        "--exp-dir",
        type=Path,
        default=None,
        help="Default: <root>/tests/exp",
    )
    parser.add_argument(
        "--participants",
        type=int,
        nargs="+",
        default=list(range(1, 9)),
    )
    parser.add_argument(
        "--aggregate-csv",
        type=Path,
        default=None,
        help="Default: <exp>/image_eval_summary.csv",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
    )

    args = parser.parse_args()

    root = Path(args.root)

    exp_dir = (
        Path(args.exp_dir)
        if args.exp_dir is not None
        else root / "tests" / "exp"
    )

    aggregate_path = (
        Path(args.aggregate_csv)
        if args.aggregate_csv is not None
        else exp_dir / "image_eval_summary.csv"
    )

    if not exp_dir.exists():
        print(
            f"[ERROR] 找不到 exp 資料夾：{exp_dir}"
        )
        return 1

    dry_run_tag = (
        " (DRY-RUN)"
        if args.dry_run
        else ""
    )

    print(
        f"\n=== Batch Image Set E evaluation"
        f"{dry_run_tag} ==="
    )
    print(f"exp = {exp_dir}\n")

    aggregate_rows: List[Dict[str, Any]] = []

    completed_count = 0
    skipped_count = 0

    for participant in args.participants:
        assignment = ASSIGNMENTS.get(participant)
        tagname = f"p{participant}ie"
        participant_dir = exp_dir / tagname

        if assignment is None:
            print(
                f"[SKIP] {tagname}: "
                "找不到 Image Set E assignment"
            )
            skipped_count += len(TOOLS)
            continue

        for tool, display_name in TOOLS.items():
            stratum = assignment[tool]

            pred_path = find_prediction(
                participant_dir=participant_dir,
                participant=participant,
                tool=tool,
            )

            if pred_path is None:
                print(
                    f"[SKIP] {tagname} {display_name}: "
                    "找不到 prediction COCO"
                )
                skipped_count += 1
                continue

            gt_path = resolve_gt(
                exp_dir=exp_dir,
                stratum=stratum,
            )

            if gt_path is None:
                print(
                    f"[SKIP] {tagname} {display_name}: "
                    f"找不到 {stratum}GT"
                )
                skipped_count += 1
                continue

            results_dir = (
                participant_dir / "results"
            )

            output_csv = (
                results_dir
                / f"{tagname}_{tool}_per_image.csv"
            )

            summary_json = (
                results_dir
                / f"{tagname}_{tool}_summary.json"
            )

            if args.dry_run:
                print(
                    f"[DRY-RUN] {tagname} "
                    f"tool={display_name} "
                    f"stratum={stratum}"
                )
                print(f"          pred    = {pred_path}")
                print(f"          gt      = {gt_path}")
                print(f"          output  = {output_csv}")
                print(f"          summary = {summary_json}")

                completed_count += 1
                continue

            try:
                summary = evaluate_image_foreground(
                    pred_coco_path=pred_path,
                    gt_coco_path=gt_path,
                    stratum=stratum,
                    output_path=output_csv,
                    summary_path=summary_json,
                )

            except Exception as exc:
                print(
                    f"[ERROR] {tagname} {display_name}: "
                    f"{type(exc).__name__}: {exc}"
                )
                skipped_count += 1
                continue

            mean_iou = summary.get(
                "mean_image_iou"
            )

            mean_iou_text = (
                f"{mean_iou:.4f}"
                if isinstance(
                    mean_iou,
                    (int, float),
                )
                else "n/a"
            )

            print(
                f"[OK] {tagname} "
                f"tool={display_name} "
                f"stratum={stratum} "
                f"images={summary['images_evaluated']} "
                f"mIoU={mean_iou_text} "
                f"missing={summary['missing_prediction_images']} "
                f"extra={summary['extra_prediction_images']}"
            )

            aggregate_rows.append(
                {
                    "participant": f"P{participant}",
                    "scene": "Image Set E",
                    "tool": tool,
                    "stratum": stratum,
                    **summary,
                }
            )

            completed_count += 1

    print("\n--- Summary ---")
    print(f"Completed: {completed_count}")
    print(f"Skipped:   {skipped_count}")

    if not args.dry_run and aggregate_rows:
        aggregate_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        fieldnames = [
            "participant",
            "scene",
            "tool",
            "stratum",
            "images_evaluated",
            "mean_image_iou",
            "gt_foreground_images",
            "pred_foreground_images",
            "missing_prediction_images",
            "extra_prediction_images",
        ]

        with aggregate_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fieldnames,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(aggregate_rows)

        print(
            f"\nAggregate CSV: {aggregate_path}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())