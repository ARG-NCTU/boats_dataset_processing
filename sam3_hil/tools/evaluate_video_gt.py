#!/usr/bin/env python3
"""Evaluate a clipped video COCO export against full-video COCO GT."""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np

try:
    from scipy.optimize import linear_sum_assignment
except ImportError:  # pragma: no cover - fallback is covered by behavior tests.
    linear_sum_assignment = None


FRAME_RE = re.compile(r"(?:frame[_-])?(\d+)(?=\.[^.]+$)")
INSTANCE_KEY_FIELDS = ("instance_name", "instance_id", "obj_id", "track_id", "group_id")
GENERIC_CATEGORY_NAMES = {"boat", "ship", "vessel", "object"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest(path: Path) -> dict[int, int]:
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    mapping: dict[int, int] = {}
    for row in rows:
        mapping[int(row["clip_frame_idx"])] = int(row["source_frame_idx"])
    return mapping


def _parse_frame_index(image: dict[str, Any]) -> int:
    for key in ("source_frame_idx", "clip_frame_idx", "frame_idx", "frame_index"):
        if key in image and image[key] not in (None, ""):
            return int(image[key])

    file_name = Path(str(image.get("file_name", ""))).name
    match = FRAME_RE.search(file_name)
    if not match:
        raise ValueError(f"Cannot parse frame index from file_name: {file_name}")
    return int(match.group(1))


def _image_maps(coco: dict[str, Any]) -> tuple[dict[int, dict[str, Any]], dict[int, list[dict[str, Any]]]]:
    images = {int(image["id"]): image for image in coco.get("images", [])}
    annotations_by_image: dict[int, list[dict[str, Any]]] = {image_id: [] for image_id in images}
    for annotation in coco.get("annotations", []):
        image_id = int(annotation["image_id"])
        annotations_by_image.setdefault(image_id, []).append(annotation)
    return images, annotations_by_image


def _category_name_by_id(coco: dict[str, Any]) -> dict[int, str]:
    return {
        int(category["id"]): str(category["name"])
        for category in coco.get("categories", [])
        if "id" in category and "name" in category
    }


def _annotation_instance_key(
    annotation: dict[str, Any],
    category_names: Optional[dict[int, str]] = None,
) -> Optional[str]:
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


def _annotation_to_mask(annotation: dict[str, Any], width: int, height: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=np.uint8)
    segmentation = annotation.get("segmentation", [])

    if isinstance(segmentation, list):
        for polygon in segmentation:
            if len(polygon) < 6:
                continue
            points = np.array(polygon, dtype=np.float32).reshape(-1, 2)
            points = np.rint(points).astype(np.int32)
            cv2.fillPoly(mask, [points], 1)
        return mask.astype(bool)

    if isinstance(segmentation, dict):
        try:
            from pycocotools import mask as mask_utils
        except ImportError as exc:
            raise ImportError("pycocotools is required for COCO RLE segmentation") from exc
        decoded = mask_utils.decode(segmentation)
        return decoded.astype(bool)

    return mask.astype(bool)


def _merge_mask(existing: Optional[np.ndarray], new_mask: np.ndarray) -> np.ndarray:
    if existing is None:
        return new_mask.copy()
    return np.logical_or(existing, new_mask)


def _frame_masks(
    coco: dict[str, Any],
    *,
    clip_to_source: Optional[dict[int, int]] = None,
) -> dict[int, list[np.ndarray]]:
    images, annotations_by_image = _image_maps(coco)
    masks_by_frame: dict[int, list[np.ndarray]] = {}

    for image_id, image in images.items():
        parsed_idx = _parse_frame_index(image)
        frame_idx = clip_to_source.get(parsed_idx, parsed_idx) if clip_to_source else parsed_idx
        width = int(image["width"])
        height = int(image["height"])
        frame_masks = [
            _annotation_to_mask(annotation, width, height)
            for annotation in annotations_by_image.get(image_id, [])
        ]
        masks_by_frame.setdefault(frame_idx, []).extend(frame_masks)

    return masks_by_frame


def _frame_instance_masks(
    coco: dict[str, Any],
    *,
    clip_to_source: Optional[dict[int, int]] = None,
) -> dict[int, dict[str, np.ndarray]]:
    images, annotations_by_image = _image_maps(coco)
    category_names = _category_name_by_id(coco)
    masks_by_frame: dict[int, dict[str, np.ndarray]] = {}

    for image_id, image in images.items():
        parsed_idx = _parse_frame_index(image)
        frame_idx = clip_to_source.get(parsed_idx, parsed_idx) if clip_to_source else parsed_idx
        width = int(image["width"])
        height = int(image["height"])

        for annotation in annotations_by_image.get(image_id, []):
            instance_key = _annotation_instance_key(annotation, category_names)
            if instance_key is None:
                continue
            mask = _annotation_to_mask(annotation, width, height)
            frame_instances = masks_by_frame.setdefault(frame_idx, {})
            frame_instances[instance_key] = _merge_mask(frame_instances.get(instance_key), mask)

    return masks_by_frame


def _mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    intersection = np.logical_and(a, b).sum()
    union = np.logical_or(a, b).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def _match_iou(gt_masks: list[np.ndarray], pred_masks: list[np.ndarray]) -> list[float]:
    if not gt_masks or not pred_masks:
        return []

    iou_matrix = np.zeros((len(gt_masks), len(pred_masks)), dtype=float)
    for gt_index, gt_mask in enumerate(gt_masks):
        for pred_index, pred_mask in enumerate(pred_masks):
            iou_matrix[gt_index, pred_index] = _mask_iou(gt_mask, pred_mask)

    if linear_sum_assignment is not None:
        gt_indices, pred_indices = linear_sum_assignment(-iou_matrix)
        return [
            float(iou_matrix[gt_index, pred_index])
            for gt_index, pred_index in zip(gt_indices, pred_indices)
            if iou_matrix[gt_index, pred_index] > 0
        ]

    pairs: list[tuple[float, int, int]] = []
    for gt_index in range(iou_matrix.shape[0]):
        for pred_index in range(iou_matrix.shape[1]):
            if iou_matrix[gt_index, pred_index] > 0:
                pairs.append((float(iou_matrix[gt_index, pred_index]), gt_index, pred_index))
    pairs.sort(reverse=True)

    used_gt: set[int] = set()
    used_pred: set[int] = set()
    matched: list[float] = []
    for iou, gt_index, pred_index in pairs:
        if gt_index in used_gt or pred_index in used_pred:
            continue
        used_gt.add(gt_index)
        used_pred.add(pred_index)
        matched.append(iou)
    return matched


def _match_instance_iou(
    gt_instances: dict[str, np.ndarray],
    pred_instances: dict[str, np.ndarray],
) -> tuple[list[float], int, int, int, int, int]:
    gt_keys = set(gt_instances)
    pred_keys = set(pred_instances)
    common_keys = sorted(gt_keys & pred_keys)
    matched_ious = [_mask_iou(gt_instances[key], pred_instances[key]) for key in common_keys]
    matched_count = len(common_keys)
    missed_count = len(gt_keys - pred_keys)
    false_positive_count = len(pred_keys - gt_keys)
    return (
        matched_ious,
        len(gt_keys),
        len(pred_keys),
        matched_count,
        missed_count,
        false_positive_count,
    )


def _write_per_frame(rows: Iterable[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "clip_frame_idx",
        "source_frame_idx",
        "gt_count",
        "pred_count",
        "matched_count",
        "missed_count",
        "false_positive_count",
        "frame_iou",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def evaluate_video_coco(
    pred_coco_path: Path,
    gt_coco_path: Path,
    manifest_path: Path,
    output_path: Path,
    summary_path: Optional[Path] = None,
) -> dict[str, Any]:
    clip_to_source = _load_manifest(Path(manifest_path))
    source_to_clip = {source: clip for clip, source in clip_to_source.items()}
    pred_coco = _load_json(Path(pred_coco_path))
    gt_coco = _load_json(Path(gt_coco_path))
    pred_masks = _frame_masks(pred_coco, clip_to_source=clip_to_source)
    gt_masks = _frame_masks(gt_coco)
    pred_instances = _frame_instance_masks(pred_coco, clip_to_source=clip_to_source)
    gt_instances = _frame_instance_masks(gt_coco)
    use_instance_matching = bool(pred_instances) and bool(gt_instances)

    # Full-video GT may contain frames outside the participant clip. Evaluate
    # only the source frames listed in the clip manifest.
    frame_indices = sorted(source_to_clip)
    rows: list[dict[str, Any]] = []
    total_gt = 0
    total_pred = 0
    total_matched = 0
    total_missed = 0
    total_false_positive = 0
    frame_scores: list[float] = []

    for source_frame_idx in frame_indices:
        gt_frame_instances = gt_instances.get(source_frame_idx, {})
        pred_frame_instances = pred_instances.get(source_frame_idx, {})

        if use_instance_matching and (gt_frame_instances or pred_frame_instances):
            (
                matched_ious,
                gt_count,
                pred_count,
                matched_count,
                missed_count,
                false_positive_count,
            ) = _match_instance_iou(gt_frame_instances, pred_frame_instances)
        else:
            gt_frame_masks = gt_masks.get(source_frame_idx, [])
            pred_frame_masks = pred_masks.get(source_frame_idx, [])
            matched_ious = _match_iou(gt_frame_masks, pred_frame_masks)
            gt_count = len(gt_frame_masks)
            pred_count = len(pred_frame_masks)
            matched_count = len(matched_ious)
            missed_count = gt_count - matched_count
            false_positive_count = pred_count - matched_count

        denominator = max(gt_count, pred_count)
        frame_iou = sum(matched_ious) / denominator if denominator else ""
        if denominator:
            frame_scores.append(float(frame_iou))

        total_gt += gt_count
        total_pred += pred_count
        total_matched += matched_count
        total_missed += missed_count
        total_false_positive += false_positive_count

        rows.append(
            {
                "clip_frame_idx": source_to_clip.get(source_frame_idx, ""),
                "source_frame_idx": source_frame_idx,
                "gt_count": gt_count,
                "pred_count": pred_count,
                "matched_count": matched_count,
                "missed_count": missed_count,
                "false_positive_count": false_positive_count,
                "frame_iou": "" if frame_iou == "" else f"{frame_iou:.6f}",
            }
        )

    _write_per_frame(rows, Path(output_path))

    summary = {
        "frames_evaluated": len(frame_indices),
        "mean_frame_iou": round(float(np.mean(frame_scores)), 6) if frame_scores else None,
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
        description="Evaluate clipped-video COCO predictions against full-video COCO GT."
    )
    parser.add_argument("--pred-coco", type=Path, required=True)
    parser.add_argument("--gt-coco", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    summary = evaluate_video_coco(
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
