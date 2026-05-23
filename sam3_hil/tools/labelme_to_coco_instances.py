#!/usr/bin/env python3
"""Convert Labelme polygon JSON files to COCO instance segmentation JSON."""

import argparse
import json
import re
import zipfile
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Optional


TRAILING_INSTANCE_NUMBER_RE = re.compile(r"[_-]?\d+$")


def _derive_category_name(instance_name: str) -> str:
    category = TRAILING_INSTANCE_NUMBER_RE.sub("", instance_name).strip()
    return category or instance_name


def _flatten_polygon(points: list[Any]) -> Optional[list[float]]:
    flat: list[float] = []
    for point in points:
        if not isinstance(point, list) or len(point) < 2:
            return None
        flat.extend([float(point[0]), float(point[1])])
    if len(flat) < 6:
        return None
    return flat


def _polygon_area(segmentation: list[float]) -> float:
    points = list(zip(segmentation[0::2], segmentation[1::2]))
    if len(points) < 3:
        return 0.0

    area = 0.0
    for index, (x1, y1) in enumerate(points):
        x2, y2 = points[(index + 1) % len(points)]
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def _bbox_from_segmentations(segmentations: list[list[float]]) -> list[float]:
    xs: list[float] = []
    ys: list[float] = []
    for segmentation in segmentations:
        xs.extend(segmentation[0::2])
        ys.extend(segmentation[1::2])
    if not xs or not ys:
        return [0.0, 0.0, 0.0, 0.0]
    x_min = min(xs)
    y_min = min(ys)
    return [x_min, y_min, max(xs) - x_min, max(ys) - y_min]


def _normalise_zip_name(name: str) -> str:
    return name.replace("\\", "/").strip("/")


def _matches_include_prefix(name: str, include_prefix: Optional[str]) -> bool:
    if not include_prefix:
        return True
    normalized_name = _normalise_zip_name(name)
    normalized_prefix = _normalise_zip_name(include_prefix)
    return normalized_name == normalized_prefix or normalized_name.startswith(f"{normalized_prefix}/")


def _is_labelme_document(document: dict[str, Any]) -> bool:
    return (
        isinstance(document.get("shapes"), list)
        and "imagePath" in document
        and "imageWidth" in document
        and "imageHeight" in document
    )


def _iter_labelme_documents_from_dir(
    input_dir: Path,
    include_prefix: Optional[str] = None,
) -> Iterable[tuple[str, dict[str, Any]]]:
    for json_path in sorted(input_dir.rglob("*.json")):
        source_name = _normalise_zip_name(str(json_path.relative_to(input_dir)))
        if not _matches_include_prefix(source_name, include_prefix):
            continue
        yield source_name, json.loads(json_path.read_text(encoding="utf-8"))


def _iter_labelme_documents_from_zip(
    input_zip: Path,
    include_prefix: Optional[str] = None,
) -> Iterable[tuple[str, dict[str, Any]]]:
    with zipfile.ZipFile(input_zip) as archive:
        for name in sorted(archive.namelist()):
            if name.endswith("/") or not name.lower().endswith(".json"):
                continue
            if not _matches_include_prefix(name, include_prefix):
                continue
            with archive.open(name) as f:
                yield name, json.loads(f.read().decode("utf-8"))


def iter_labelme_documents(
    input_path: Path,
    *,
    include_prefix: Optional[str] = None,
) -> Iterable[tuple[str, dict[str, Any]]]:
    if input_path.is_dir():
        return _iter_labelme_documents_from_dir(input_path, include_prefix=include_prefix)
    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        return _iter_labelme_documents_from_zip(input_path, include_prefix=include_prefix)
    raise ValueError(f"Input must be a directory or .zip file: {input_path}")


def convert_labelme_documents(
    labelme_documents: Iterable[tuple[str, dict[str, Any]]],
    *,
    single_category: Optional[str] = None,
) -> dict[str, Any]:
    images: list[dict[str, Any]] = []
    annotations: list[dict[str, Any]] = []
    category_ids: OrderedDict[str, int] = OrderedDict()
    annotation_id = 1

    for source_name, labelme in labelme_documents:
        if not _is_labelme_document(labelme):
            continue
        image_id = len(images) + 1
        image_path = labelme.get("imagePath") or f"{Path(source_name).stem}.png"
        width = int(labelme.get("imageWidth") or 0)
        height = int(labelme.get("imageHeight") or 0)

        images.append(
            {
                "id": image_id,
                "file_name": Path(str(image_path)).name,
                "width": width,
                "height": height,
                "source_json": source_name,
            }
        )

        instances: OrderedDict[str, dict[str, Any]] = OrderedDict()
        for shape in labelme.get("shapes", []):
            if shape.get("shape_type", "polygon") != "polygon":
                continue
            instance_name = str(shape.get("label", "")).strip()
            if not instance_name:
                continue
            segmentation = _flatten_polygon(shape.get("points", []))
            if segmentation is None:
                continue

            category_name = single_category or _derive_category_name(instance_name)
            if category_name not in category_ids:
                category_ids[category_name] = len(category_ids) + 1

            instance = instances.setdefault(
                instance_name,
                {
                    "category_name": category_name,
                    "segmentations": [],
                    "group_id": shape.get("group_id"),
                },
            )
            instance["segmentations"].append(segmentation)

        for instance_name, instance in instances.items():
            segmentations = instance["segmentations"]
            category_name = instance["category_name"]
            annotation: dict[str, Any] = {
                "id": annotation_id,
                "image_id": image_id,
                "category_id": category_ids[category_name],
                "segmentation": segmentations,
                "area": sum(_polygon_area(segmentation) for segmentation in segmentations),
                "bbox": _bbox_from_segmentations(segmentations),
                "iscrowd": 0,
                "instance_name": instance_name,
            }
            if instance.get("group_id") is not None:
                annotation["group_id"] = instance["group_id"]
            annotations.append(annotation)
            annotation_id += 1

    categories = [
        {"id": category_id, "name": name, "supercategory": "object"}
        for name, category_id in category_ids.items()
    ]

    return {
        "info": {
            "description": "Labelme polygon annotations converted to COCO instance segmentation",
            "version": "1.0",
            "year": datetime.now().year,
            "date_created": datetime.now().isoformat(),
        },
        "licenses": [],
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Labelme polygon JSON files to COCO instance segmentation JSON."
    )
    parser.add_argument("--input", type=Path, required=True, help="Labelme JSON directory or .zip file.")
    parser.add_argument("--output", type=Path, required=True, help="Output COCO JSON path.")
    parser.add_argument(
        "--single-category",
        default=None,
        help="Force all instances to one category, e.g. 'boat'. By default boat1 -> boat.",
    )
    parser.add_argument(
        "--include-prefix",
        default=None,
        help="Only read JSON files under this zip/directory prefix, e.g. 'p1va/LabelMe'.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    coco = convert_labelme_documents(
        iter_labelme_documents(args.input, include_prefix=args.include_prefix),
        single_category=args.single_category,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(coco, indent=2), encoding="utf-8")
    print(
        f"Saved {len(coco['images'])} images and "
        f"{len(coco['annotations'])} annotations to {args.output}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
