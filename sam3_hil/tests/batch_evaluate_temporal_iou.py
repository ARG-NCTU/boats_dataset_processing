#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch compute Temporal IoU for LabelMe, CVAT+SAM, and STAMP."""

import argparse
import csv
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

THIS_DIR = Path(__file__).resolve().parent
ROOT = THIS_DIR.parent

sys.path.insert(0, str(THIS_DIR))

from evaluate_temporal_iou import evaluate_temporal_iou  # noqa: E402


TOOLS = {
    "labelme": "LabelMe",
    "cvat": "CVAT+SAM",
    "stamp": "STAMP",
}

DEFAULT_VIDEOS = ["A", "B"]


def find_coco(
    participant_dir: Path,
    participant: int,
    video: str,
    tool: str,
) -> Optional[Path]:
    """Return the COCO path for one participant, video, and tool."""
    scene = f"v{video.lower()}"
    tagname = f"p{participant}{scene}"

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


def normalize_videos(videos: List[str]) -> List[str]:
    """Normalize and validate video names."""
    normalized = []

    for video in videos:
        name = video.strip().upper()

        if name not in {"A", "B", "C", "D"}:
            raise ValueError(
                f"Unsupported video: {video}. "
                "Allowed values are A, B, C, D."
            )

        if name not in normalized:
            normalized.append(name)

    return normalized


def write_aggregate_csv(
    rows: List[Dict[str, Any]],
    output_path: Path,
) -> None:
    """Write all Temporal IoU summaries into one CSV."""
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fieldnames = [
        "participant",
        "video",
        "scene",
        "tool",
        "tool_name",
        "coco_path",
        "frame_count",
        "pair_count",
        "adjacent_frame_pair_count",
        "instance_count",
        "instance_names",
        "missing_pair_count",
        "mean_temporal_iou",
    ]

    with output_path.open(
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
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Batch compute Temporal IoU for LabelMe, "
            "CVAT+SAM, and STAMP video annotations."
        )
    )

    parser.add_argument(
        "--root",
        type=Path,
        default=ROOT,
        help="Project root. Default: parent directory of tests/",
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
        help="Participant numbers. Default: 1 through 8",
    )

    parser.add_argument(
        "--videos",
        nargs="+",
        default=DEFAULT_VIDEOS,
        help="Videos to process. Default: A B",
    )

    parser.add_argument(
        "--aggregate-csv",
        type=Path,
        default=None,
        help="Default: <exp>/temporal_iou_summary.csv",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show jobs and paths without running evaluation.",
    )

    args = parser.parse_args()

    try:
        videos = normalize_videos(args.videos)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1

    root = Path(args.root)

    exp_dir = (
        Path(args.exp_dir)
        if args.exp_dir is not None
        else root / "tests" / "exp"
    )

    aggregate_path = (
        Path(args.aggregate_csv)
        if args.aggregate_csv is not None
        else exp_dir / "temporal_iou_summary.csv"
    )

    if not exp_dir.exists():
        print(f"[ERROR] 找不到 exp 資料夾：{exp_dir}")
        return 1

    dry_run_label = " (DRY-RUN)" if args.dry_run else ""

    print(
        f"\n=== Batch Temporal IoU evaluation"
        f"{dry_run_label} ==="
    )
    print(f"root         = {root}")
    print(f"exp          = {exp_dir}")
    print(f"participants = {args.participants}")
    print(f"videos       = {videos}")
    print(f"tools        = {list(TOOLS.values())}\n")

    aggregate_rows: List[Dict[str, Any]] = []

    completed_count = 0
    skipped_count = 0
    error_count = 0

    for participant in args.participants:
        for video in videos:
            scene = f"v{video.lower()}"
            tagname = f"p{participant}{scene}"
            participant_dir = exp_dir / tagname

            for tool, tool_name in TOOLS.items():
                coco_path = find_coco(
                    participant_dir=participant_dir,
                    participant=participant,
                    video=video,
                    tool=tool,
                )

                if coco_path is None:
                    print(
                        f"[SKIP] {tagname} {tool_name}: "
                        "找不到 COCO annotation"
                    )
                    skipped_count += 1
                    continue

                results_dir = participant_dir / "results"

                output_csv = (
                    results_dir
                    / f"{tagname}_{tool}_temporal_iou.csv"
                )

                summary_json = (
                    results_dir
                    / (
                        f"{tagname}_{tool}_"
                        "temporal_iou_summary.json"
                    )
                )

                if args.dry_run:
                    print(
                        f"[DRY-RUN] {tagname} "
                        f"tool={tool_name}"
                    )
                    print(f"          coco    = {coco_path}")
                    print(f"          output  = {output_csv}")
                    print(f"          summary = {summary_json}")

                    completed_count += 1
                    continue

                try:
                    summary = evaluate_temporal_iou(
                        coco_path=coco_path,
                        output_path=output_csv,
                        summary_path=summary_json,
                    )

                except Exception as exc:
                    print(
                        f"[ERROR] {tagname} {tool_name}: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    error_count += 1
                    continue

                mean_iou = summary.get(
                    "mean_temporal_iou"
                )

                mean_iou_text = (
                    f"{mean_iou:.4f}"
                    if isinstance(mean_iou, (int, float))
                    else "n/a"
                )

                print(
                    f"[OK] {tagname} "
                    f"tool={tool_name} "
                    f"frames={summary['frame_count']} "
                    f"pairs={summary['pair_count']} "
                    f"instances={summary['instance_count']} "
                    f"missing={summary['missing_pair_count']} "
                    f"TemporalIoU={mean_iou_text}"
                )

                instance_names = summary.get(
                    "instance_names",
                    [],
                )

                aggregate_rows.append(
                    {
                        "participant": f"P{participant}",
                        "video": f"Video {video}",
                        "scene": scene,
                        "tool": tool,
                        "tool_name": tool_name,
                        "coco_path": str(coco_path),
                        "frame_count": summary.get(
                            "frame_count"
                        ),
                        "pair_count": summary.get(
                            "pair_count"
                        ),
                        "adjacent_frame_pair_count": (
                            summary.get(
                                "adjacent_frame_pair_count"
                            )
                        ),
                        "instance_count": summary.get(
                            "instance_count"
                        ),
                        "instance_names": ",".join(
                            str(name)
                            for name in instance_names
                        ),
                        "missing_pair_count": summary.get(
                            "missing_pair_count"
                        ),
                        "mean_temporal_iou": mean_iou,
                    }
                )

                completed_count += 1

    print("\n--- Summary ---")
    print(f"Completed: {completed_count}")
    print(f"Skipped:   {skipped_count}")
    print(f"Errors:    {error_count}")

    if not args.dry_run:
        # Always overwrite the aggregate CSV, including when no job succeeds.
        write_aggregate_csv(
            aggregate_rows,
            aggregate_path,
        )

        print(f"\nAggregate CSV: {aggregate_path}")

    return 1 if error_count else 0


if __name__ == "__main__":
    raise SystemExit(main())