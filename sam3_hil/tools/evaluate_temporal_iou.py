#!/usr/bin/env python3
"""Compute temporal IoU from a single COCO instance-segmentation export."""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

import numpy as np

try:  # OpenCV is preferred for COCO polygon rasterization.
    import cv2
except ImportError:  # pragma: no cover - exercised only when cv2 is absent.
    cv2 = None


FRAME_RE = re.compile(r"(?:frame[_-])?(\d+)(?=\.[^.]+$)")
INSTANCE_KEY_FIELDS = ("instance_name", "instance_id", "obj_id", "track_id", "group_id")
GENERIC_CATEGORY_NAMES = {"boat", "ship", "vessel", "object"}


def _load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _parse_frame_index(image):
    for key in ("source_frame_idx", "clip_frame_idx", "frame_idx", "frame_index"):
        if key in image and image[key] not in (None, ""):
            return int(image[key])

    file_name = Path(str(image.get("file_name", ""))).name
    match = FRAME_RE.search(file_name)
    if not match:
        raise ValueError("Cannot parse frame index from file_name: {}".format(file_name))
    return int(match.group(1))


def _image_maps(coco):
    images = {int(image["id"]): image for image in coco.get("images", [])}
    annotations_by_image = {image_id: [] for image_id in images}
    for annotation in coco.get("annotations", []):
        image_id = int(annotation["image_id"])
        annotations_by_image.setdefault(image_id, []).append(annotation)
    return images, annotations_by_image


def _category_name_by_id(coco):
    return {
        int(category["id"]): str(category["name"])
        for category in coco.get("categories", [])
        if "id" in category and "name" in category
    }


def _annotation_instance_key(annotation, category_names=None):
    value = annotation.get("instance_name")
    if value not in (None, ""):
        return str(value)

    if category_names is not None and "category_id" in annotation:
        category_name = category_names.get(int(annotation["category_id"]))
        if category_name:
            normalized_name = category_name.strip().lower()
            if normalized_name and normalized_name not in GENERIC_CATEGORY_NAMES:
                return category_name

    for field in INSTANCE_KEY_FIELDS:
        if field == "instance_name":
            continue
        value = annotation.get(field)
        if value not in (None, ""):
            return str(value)

    return None


def _normalize_polygons(segmentation):
    if not isinstance(segmentation, list):
        return []
    if segmentation and all(isinstance(value, (int, float)) for value in segmentation):
        return [segmentation]
    return segmentation


def _fill_polygon(mask, polygon):
    if len(polygon) < 6:
        return

    points = np.array(polygon, dtype=np.float32).reshape(-1, 2)
    if cv2 is not None:
        cv_points = np.rint(points).astype(np.int32)
        cv2.fillPoly(mask, [cv_points], 1)
        return

    # Lightweight fallback for environments without OpenCV. It is sufficient
    # for metric computation and tests, while OpenCV remains the preferred path.
    height, width = mask.shape
    min_x = max(int(np.floor(np.min(points[:, 0]))), 0)
    max_x = min(int(np.ceil(np.max(points[:, 0]))), width - 1)
    min_y = max(int(np.floor(np.min(points[:, 1]))), 0)
    max_y = min(int(np.ceil(np.max(points[:, 1]))), height - 1)
    if min_x > max_x or min_y > max_y:
        return

    x_coords = np.arange(min_x, max_x + 1, dtype=np.float32) + 0.5
    y_coords = np.arange(min_y, max_y + 1, dtype=np.float32) + 0.5
    grid_x, grid_y = np.meshgrid(x_coords, y_coords)
    inside = np.zeros(grid_x.shape, dtype=bool)

    xs = points[:, 0]
    ys = points[:, 1]
    prev = len(points) - 1
    for curr in range(len(points)):
        yi = ys[curr]
        yj = ys[prev]
        xi = xs[curr]
        xj = xs[prev]
        crosses = ((yi > grid_y) != (yj > grid_y)) & (
            grid_x < (xj - xi) * (grid_y - yi) / ((yj - yi) + 1e-12) + xi
        )
        inside ^= crosses
        prev = curr

    mask[min_y : max_y + 1, min_x : max_x + 1][inside] = 1


def _annotation_to_mask(annotation, width, height):
    segmentation = annotation.get("segmentation", [])

    if isinstance(segmentation, list):
        mask = np.zeros((height, width), dtype=np.uint8)
        for polygon in _normalize_polygons(segmentation):
            _fill_polygon(mask, polygon)
        return mask.astype(bool)

    if isinstance(segmentation, dict):
        try:
            from pycocotools import mask as mask_utils
        except ImportError as exc:
            raise ImportError("pycocotools is required for COCO RLE segmentation") from exc
        decoded = mask_utils.decode(segmentation)
        return decoded.astype(bool)

    return np.zeros((height, width), dtype=bool)


def _merge_mask(existing, new_mask):
    if existing is None:
        return new_mask.copy()
    return np.logical_or(existing, new_mask)


def _frame_instance_masks(coco):
    images, annotations_by_image = _image_maps(coco)
    category_names = _category_name_by_id(coco)
    masks_by_frame = {}

    for image_id, image in images.items():
        frame_idx = _parse_frame_index(image)
        width = int(image["width"])
        height = int(image["height"])
        frame_instances = masks_by_frame.setdefault(frame_idx, {})

        for annotation in annotations_by_image.get(image_id, []):
            instance_key = _annotation_instance_key(annotation, category_names)
            if instance_key is None:
                continue
            mask = _annotation_to_mask(annotation, width, height)
            frame_instances[instance_key] = _merge_mask(frame_instances.get(instance_key), mask)

    return masks_by_frame


def _mask_iou(a, b):
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def _temporal_rows(masks_by_frame):
    frame_indices = sorted(masks_by_frame)
    rows = []

    for frame_t, frame_t1 in zip(frame_indices, frame_indices[1:]):
        instances_t = masks_by_frame.get(frame_t, {})
        instances_t1 = masks_by_frame.get(frame_t1, {})
        instance_keys = sorted(set(instances_t) | set(instances_t1))

        for instance_key in instance_keys:
            present_t = instance_key in instances_t
            present_t1 = instance_key in instances_t1
            if present_t and present_t1:
                iou = _mask_iou(instances_t[instance_key], instances_t1[instance_key])
            else:
                iou = 0.0
            rows.append(
                {
                    "instance_name": instance_key,
                    "frame_t": frame_t,
                    "frame_t1": frame_t1,
                    "present_t": 1 if present_t else 0,
                    "present_t1": 1 if present_t1 else 0,
                    "iou": iou,
                }
            )

    return rows


def _write_rows(rows, output_path):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["instance_name", "frame_t", "frame_t1", "present_t", "present_t1", "iou"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "instance_name": row["instance_name"],
                    "frame_t": row["frame_t"],
                    "frame_t1": row["frame_t1"],
                    "present_t": row["present_t"],
                    "present_t1": row["present_t1"],
                    "iou": "{:.6f}".format(row["iou"]),
                }
            )


def _summarize(rows, masks_by_frame):
    scores = [float(row["iou"]) for row in rows]
    instance_names = sorted({key for instances in masks_by_frame.values() for key in instances})
    missing_pair_count = sum(
        1 for row in rows if not (bool(row["present_t"]) and bool(row["present_t1"]))
    )
    return {
        "frame_count": len(masks_by_frame),
        "pair_count": len(rows),
        "adjacent_frame_pair_count": max(len(masks_by_frame) - 1, 0),
        "instance_count": len(instance_names),
        "instance_names": instance_names,
        "missing_pair_count": missing_pair_count,
        "mean_temporal_iou": round(float(np.mean(scores)), 6) if scores else None,
    }


def evaluate_temporal_iou(coco_path, output_path, summary_path=None):
    coco = _load_json(coco_path)
    masks_by_frame = _frame_instance_masks(coco)
    rows = _temporal_rows(masks_by_frame)
    _write_rows(rows, Path(output_path))

    summary = _summarize(rows, masks_by_frame)
    if summary_path is not None:
        summary_path = Path(summary_path)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute temporal IoU between same-instance masks in adjacent COCO frames."
    )
    parser.add_argument("--coco", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    summary = evaluate_temporal_iou(
        coco_path=args.coco,
        output_path=args.output,
        summary_path=args.summary,
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
