#!/usr/bin/env python3
"""Visualize GT/prediction mask overlap for COCO segmentation files."""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np


FRAME_RE = re.compile(r"(?:frame[_-])?(\d+)(?=\.[^.]+$)")
INSTANCE_KEY_FIELDS = ("instance_name", "instance_id", "obj_id", "track_id", "group_id")
GENERIC_CATEGORY_NAMES = {"boat", "ship", "vessel", "object"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_manifest(path: Optional[Path]) -> Dict[int, int]:
    if path is None:
        return {}
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    return {int(row["clip_frame_idx"]): int(row["source_frame_idx"]) for row in rows}


def _parse_frame_index_from_name(file_name: str) -> int:
    match = FRAME_RE.search(Path(str(file_name)).name)
    if not match:
        raise ValueError("Cannot parse frame index from file_name: {}".format(file_name))
    return int(match.group(1))


def _parse_frame_index(image: Dict[str, Any]) -> int:
    for key in ("source_frame_idx", "clip_frame_idx", "frame_idx", "frame_index"):
        if key in image and image[key] not in (None, ""):
            return int(image[key])
    return _parse_frame_index_from_name(str(image.get("file_name", "")))


def _image_maps(coco: Dict[str, Any]) -> Tuple[Dict[int, Dict[str, Any]], Dict[int, List[Dict[str, Any]]]]:
    images = {int(image["id"]): image for image in coco.get("images", [])}
    annotations_by_image = {image_id: [] for image_id in images}
    for annotation in coco.get("annotations", []):
        image_id = int(annotation["image_id"])
        annotations_by_image.setdefault(image_id, []).append(annotation)
    return images, annotations_by_image


def _category_name_by_id(coco: Dict[str, Any]) -> Dict[int, str]:
    return {
        int(category["id"]): str(category["name"])
        for category in coco.get("categories", [])
        if "id" in category and "name" in category
    }


def _annotation_instance_key(
    annotation: Dict[str, Any],
    category_names: Optional[Dict[int, str]] = None,
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


def _annotation_to_mask(annotation: Dict[str, Any], width: int, height: int) -> np.ndarray:
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
        return mask_utils.decode(segmentation).astype(bool)

    return mask.astype(bool)


def _merge_mask(existing: Optional[np.ndarray], new_mask: np.ndarray) -> np.ndarray:
    if existing is None:
        return new_mask.copy()
    return np.logical_or(existing, new_mask)


def _frame_instance_masks(
    coco: Dict[str, Any],
    clip_to_source: Optional[Dict[int, int]] = None,
) -> Tuple[Dict[int, Dict[str, np.ndarray]], Dict[int, Tuple[int, int]]]:
    images, annotations_by_image = _image_maps(coco)
    category_names = _category_name_by_id(coco)
    masks_by_frame = {}
    sizes_by_frame = {}

    for image_id, image in images.items():
        parsed_idx = _parse_frame_index(image)
        frame_idx = clip_to_source.get(parsed_idx, parsed_idx) if clip_to_source else parsed_idx
        width = int(image["width"])
        height = int(image["height"])
        sizes_by_frame[frame_idx] = (width, height)

        for annotation in annotations_by_image.get(image_id, []):
            instance_key = _annotation_instance_key(annotation, category_names)
            if instance_key is None:
                continue
            mask = _annotation_to_mask(annotation, width, height)
            frame_instances = masks_by_frame.setdefault(frame_idx, {})
            frame_instances[instance_key] = _merge_mask(frame_instances.get(instance_key), mask)

    return masks_by_frame, sizes_by_frame


def _mask_iou(gt_mask: np.ndarray, pred_mask: np.ndarray) -> float:
    intersection = np.logical_and(gt_mask, pred_mask).sum()
    union = np.logical_or(gt_mask, pred_mask).sum()
    if union == 0:
        return 0.0
    return float(intersection / union)


def _resize_mask(mask: np.ndarray, width: int, height: int) -> np.ndarray:
    if mask.shape == (height, width):
        return mask
    return cv2.resize(mask.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST).astype(bool)


def _combined_mask(instances: Dict[str, np.ndarray], width: int, height: int) -> np.ndarray:
    combined = np.zeros((height, width), dtype=bool)
    for mask in instances.values():
        combined = np.logical_or(combined, _resize_mask(mask, width, height))
    return combined


def _find_images_by_frame(image_dir: Optional[Path]) -> Dict[int, Path]:
    if image_dir is None:
        return {}
    images = {}
    for path in sorted(image_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        try:
            frame_idx = _parse_frame_index_from_name(path.name)
        except ValueError:
            continue
        images.setdefault(frame_idx, path)
    return images


def _base_image(
    frame_idx: int,
    width: int,
    height: int,
    images_by_frame: Dict[int, Path],
) -> np.ndarray:
    image_path = images_by_frame.get(frame_idx)
    if image_path is None:
        return np.zeros((height, width, 3), dtype=np.uint8)

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        return np.zeros((height, width, 3), dtype=np.uint8)
    if image.shape[:2] != (height, width):
        image = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    return image


def _overlay_masks(image: np.ndarray, gt_mask: np.ndarray, pred_mask: np.ndarray) -> np.ndarray:
    gt_only = np.logical_and(gt_mask, np.logical_not(pred_mask))
    pred_only = np.logical_and(pred_mask, np.logical_not(gt_mask))
    overlap = np.logical_and(gt_mask, pred_mask)

    overlay = image.copy()
    overlay[gt_only] = (0, 180, 0)
    overlay[pred_only] = (0, 0, 255)
    overlay[overlap] = (0, 255, 255)
    return cv2.addWeighted(image, 0.45, overlay, 0.55, 0)


def _choose_size(frame_idx: int, gt_sizes: Dict[int, Tuple[int, int]], pred_sizes: Dict[int, Tuple[int, int]]) -> Tuple[int, int]:
    if frame_idx in gt_sizes:
        return gt_sizes[frame_idx]
    if frame_idx in pred_sizes:
        return pred_sizes[frame_idx]
    return (640, 480)


def create_iou_overlays(
    pred_coco_path: Path,
    gt_coco_path: Path,
    output_dir: Path,
    image_dir: Optional[Path] = None,
    manifest_path: Optional[Path] = None,
    frames: Optional[Iterable[int]] = None,
) -> List[Dict[str, Any]]:
    clip_to_source = _load_manifest(Path(manifest_path) if manifest_path else None)
    pred_coco = _load_json(Path(pred_coco_path))
    gt_coco = _load_json(Path(gt_coco_path))
    pred_instances, pred_sizes = _frame_instance_masks(pred_coco, clip_to_source=clip_to_source)
    gt_instances, gt_sizes = _frame_instance_masks(gt_coco)

    if frames is None:
        frame_indices = sorted(set(gt_instances) | set(pred_instances))
    else:
        frame_indices = sorted(int(frame) for frame in frames)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    images_by_frame = _find_images_by_frame(Path(image_dir) if image_dir else None)
    rows = []

    for frame_idx in frame_indices:
        width, height = _choose_size(frame_idx, gt_sizes, pred_sizes)
        gt_mask = _combined_mask(gt_instances.get(frame_idx, {}), width, height)
        pred_mask = _combined_mask(pred_instances.get(frame_idx, {}), width, height)
        frame_iou = _mask_iou(gt_mask, pred_mask)
        image = _base_image(frame_idx, width, height, images_by_frame)
        overlay = _overlay_masks(image, gt_mask, pred_mask)

        label = "GT green | Pred red | Overlap yellow | IoU {:.3f}".format(frame_iou)
        cv2.putText(overlay, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(overlay, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

        output_path = output_dir / "frame_{:06d}_iou_{:.3f}.png".format(frame_idx, frame_iou)
        cv2.imwrite(str(output_path), overlay)
        rows.append(
            {
                "frame_idx": frame_idx,
                "frame_iou": frame_iou,
                "output_path": str(output_path),
                "gt_instances": len(gt_instances.get(frame_idx, {})),
                "pred_instances": len(pred_instances.get(frame_idx, {})),
            }
        )

    return rows


def _write_summary(rows: List[Dict[str, Any]], output_path: Path) -> None:
    if not rows:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["frame_idx", "frame_iou", "gt_instances", "pred_instances", "output_path"],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create overlay images showing GT/prediction mask overlap."
    )
    parser.add_argument("--pred-coco", type=Path, required=True)
    parser.add_argument("--gt-coco", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--frames", type=int, nargs="*", default=None)
    parser.add_argument("--summary-csv", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = create_iou_overlays(
        pred_coco_path=args.pred_coco,
        gt_coco_path=args.gt_coco,
        output_dir=args.output_dir,
        image_dir=args.image_dir,
        manifest_path=args.manifest,
        frames=args.frames,
    )
    if args.summary_csv:
        _write_summary(rows, args.summary_csv)
    print("Saved {} overlay images to {}".format(len(rows), args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
