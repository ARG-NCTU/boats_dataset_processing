#!/usr/bin/env python3
"""Generate Image Set E strata and task manifests.

The manifest gives every image a stable image_id, assigns it to E1-E4 with a
fixed random seed, and marks the first 20 images per stratum as task images.
"""

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_SEED = 20260511
DEFAULT_STRATA = 4
DEFAULT_ITEMS_PER_STRATUM = 50
DEFAULT_SELECTED_PER_STRATUM = 20


@dataclass(frozen=True)
class ImageManifestRow:
    image_id: str
    original_filename: str
    image_path: str
    stratum: str
    order_in_stratum: int
    selected_for_task: int


def _iter_image_files(input_dir: Path) -> list[Path]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")
    if not input_dir.is_dir():
        raise ValueError(f"Input path is not a directory: {input_dir}")

    image_files = [
        path
        for path in input_dir.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    ]
    if not image_files:
        raise ValueError(f"No supported image files found in: {input_dir}")
    return sorted(image_files, key=lambda path: path.name.lower())


def generate_rows(
    image_files: list[Path],
    *,
    seed: int = DEFAULT_SEED,
    strata: int = DEFAULT_STRATA,
    items_per_stratum: int = DEFAULT_ITEMS_PER_STRATUM,
    selected_per_stratum: int = DEFAULT_SELECTED_PER_STRATUM,
    image_prefix: str = "E",
) -> list[ImageManifestRow]:
    expected_total = strata * items_per_stratum
    if len(image_files) != expected_total:
        raise ValueError(
            f"Expected {expected_total} images for {strata} strata, got {len(image_files)}"
        )
    if selected_per_stratum > items_per_stratum:
        raise ValueError("selected_per_stratum must be <= items_per_stratum")

    rows_with_ids = [
        (f"{image_prefix}_{index:04d}", path)
        for index, path in enumerate(image_files, start=1)
    ]
    rng = random.Random(seed)
    shuffled = list(rows_with_ids)
    rng.shuffle(shuffled)

    rows: list[ImageManifestRow] = []
    for stratum_index in range(strata):
        stratum = f"{image_prefix}{stratum_index + 1}"
        start = stratum_index * items_per_stratum
        end = start + items_per_stratum
        stratum_items = shuffled[start:end]
        for order, (image_id, path) in enumerate(stratum_items, start=1):
            rows.append(
                ImageManifestRow(
                    image_id=image_id,
                    original_filename=path.name,
                    image_path=str(path),
                    stratum=stratum,
                    order_in_stratum=order,
                    selected_for_task=1 if order <= selected_per_stratum else 0,
                )
            )

    return sorted(rows, key=lambda row: (row.stratum, row.order_in_stratum))


def _write_rows(rows: Iterable[ImageManifestRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "image_id",
        "original_filename",
        "image_path",
        "stratum",
        "order_in_stratum",
        "selected_for_task",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)


def generate_image_set_manifest(
    input_dir: Path,
    output_path: Path,
    selected_output_path: Optional[Path] = None,
    *,
    seed: int = DEFAULT_SEED,
    strata: int = DEFAULT_STRATA,
    items_per_stratum: int = DEFAULT_ITEMS_PER_STRATUM,
    selected_per_stratum: int = DEFAULT_SELECTED_PER_STRATUM,
    image_prefix: str = "E",
) -> list[ImageManifestRow]:
    image_files = _iter_image_files(Path(input_dir))
    rows = generate_rows(
        image_files,
        seed=seed,
        strata=strata,
        items_per_stratum=items_per_stratum,
        selected_per_stratum=selected_per_stratum,
        image_prefix=image_prefix,
    )
    _write_rows(rows, Path(output_path))
    if selected_output_path is not None:
        selected_rows = [row for row in rows if row.selected_for_task]
        _write_rows(selected_rows, Path(selected_output_path))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Image Set E task manifests.")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--selected-output", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--strata", type=int, default=DEFAULT_STRATA)
    parser.add_argument("--items-per-stratum", type=int, default=DEFAULT_ITEMS_PER_STRATUM)
    parser.add_argument("--selected-per-stratum", type=int, default=DEFAULT_SELECTED_PER_STRATUM)
    parser.add_argument("--image-prefix", default="E")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = generate_image_set_manifest(
        input_dir=args.input_dir,
        output_path=args.output,
        selected_output_path=args.selected_output,
        seed=args.seed,
        strata=args.strata,
        items_per_stratum=args.items_per_stratum,
        selected_per_stratum=args.selected_per_stratum,
        image_prefix=args.image_prefix,
    )
    selected_count = sum(row.selected_for_task for row in rows)
    print(f"Wrote {len(rows)} image rows to {args.output}")
    if args.selected_output:
        print(f"Wrote {selected_count} selected task rows to {args.selected_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
