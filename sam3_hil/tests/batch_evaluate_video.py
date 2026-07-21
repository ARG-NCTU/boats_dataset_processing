#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch evaluate LabelMe, CVAT+SAM, and STAMP outputs against video GT."""

import argparse
import csv
import sys
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(THIS_DIR))

from evaluate_video_gt import evaluate_video_coco  # noqa: E402


ROOT = Path("/app")
VIDEOS = ["A", "B", "C", "D"]

TOOLS = {
    "labelme": {"display_name": "LabelMe"},
    "cvat": {"display_name": "CVAT+SAM"},
    "stamp": {"display_name": "STAMP"},
}

# Assignment from STAMP_assignment_table.xlsx.
# Video A/B: unchanged. Video C/D: read from the "Video C" / "Video D" sheets.
ASSIGNMENTS = {
    (1, "A"): {"labelme": "Q1", "cvat": "Q2", "stamp": "Q3"},
    (1, "B"): {"labelme": "Q2", "cvat": "Q3", "stamp": "Q4"},
    (1, "C"): {"labelme": "Q3", "cvat": "Q4", "stamp": "Q1"},
    (1, "D"): {"labelme": "Q4", "cvat": "Q1", "stamp": "Q2"},

    (2, "A"): {"labelme": "Q2", "cvat": "Q3", "stamp": "Q4"},
    (2, "B"): {"labelme": "Q3", "cvat": "Q4", "stamp": "Q1"},
    (2, "C"): {"labelme": "Q4", "cvat": "Q1", "stamp": "Q2"},
    (2, "D"): {"labelme": "Q1", "cvat": "Q2", "stamp": "Q3"},

    (3, "A"): {"labelme": "Q3", "cvat": "Q4", "stamp": "Q1"},
    (3, "B"): {"labelme": "Q4", "cvat": "Q1", "stamp": "Q2"},
    (3, "C"): {"labelme": "Q1", "cvat": "Q2", "stamp": "Q3"},
    (3, "D"): {"labelme": "Q2", "cvat": "Q3", "stamp": "Q4"},

    (4, "A"): {"labelme": "Q4", "cvat": "Q1", "stamp": "Q2"},
    (4, "B"): {"labelme": "Q1", "cvat": "Q2", "stamp": "Q3"},
    (4, "C"): {"labelme": "Q2", "cvat": "Q3", "stamp": "Q4"},
    (4, "D"): {"labelme": "Q3", "cvat": "Q4", "stamp": "Q1"},

    (5, "A"): {"labelme": "Q1", "cvat": "Q3", "stamp": "Q4"},
    (5, "B"): {"labelme": "Q2", "cvat": "Q4", "stamp": "Q1"},
    (5, "C"): {"labelme": "Q3", "cvat": "Q1", "stamp": "Q2"},
    (5, "D"): {"labelme": "Q4", "cvat": "Q2", "stamp": "Q3"},

    (6, "A"): {"labelme": "Q2", "cvat": "Q4", "stamp": "Q1"},
    (6, "B"): {"labelme": "Q3", "cvat": "Q1", "stamp": "Q2"},
    (6, "C"): {"labelme": "Q4", "cvat": "Q2", "stamp": "Q3"},
    (6, "D"): {"labelme": "Q1", "cvat": "Q3", "stamp": "Q4"},

    (7, "A"): {"labelme": "Q3", "cvat": "Q1", "stamp": "Q2"},
    (7, "B"): {"labelme": "Q4", "cvat": "Q2", "stamp": "Q3"},
    (7, "C"): {"labelme": "Q1", "cvat": "Q3", "stamp": "Q4"},
    (7, "D"): {"labelme": "Q2", "cvat": "Q4", "stamp": "Q1"},

    (8, "A"): {"labelme": "Q4", "cvat": "Q2", "stamp": "Q3"},
    (8, "B"): {"labelme": "Q1", "cvat": "Q3", "stamp": "Q4"},
    (8, "C"): {"labelme": "Q2", "cvat": "Q4", "stamp": "Q1"},
    (8, "D"): {"labelme": "Q3", "cvat": "Q1", "stamp": "Q2"},
}


def normalize_videos(videos):
    """Normalize and validate video names."""
    normalized = []

    for video in videos:
        name = str(video).strip().upper()

        if name not in VIDEOS:
            raise ValueError(
                f"Unsupported video: {video}. "
                f"Allowed values are {', '.join(VIDEOS)}."
            )

        if name not in normalized:
            normalized.append(name)

    return normalized


def resolve_gt(gt_dir: Path, video: str, seg: str):
    """Return (GT JSON path, GT folder name), or (None, reason)."""
    base = f"V{video}{seg}GT"
    exact_folder = gt_dir / base

    if exact_folder.exists():
        candidates = [exact_folder]
    else:
        candidates = sorted(gt_dir.glob(f"{base}*"))

    folders = [path for path in candidates if path.is_dir()]

    if not folders:
        return None, f"找不到 {gt_dir}/{base} 或 {base}*"

    if len(folders) > 1:
        names = [folder.name for folder in folders]
        return None, f"{base}* 對應到多個資料夾：{names}"

    folder = folders[0]
    annotations_dir = folder / "annotations"
    annotation_path = annotations_dir / "instances_default.json"

    if annotation_path.exists():
        return annotation_path, folder.name

    json_files = (
        sorted(annotations_dir.glob("*.json"))
        if annotations_dir.exists()
        else []
    )

    if len(json_files) == 1:
        return json_files[0], folder.name

    if not json_files:
        return None, f"{annotations_dir} 找不到 JSON"

    names = [path.name for path in json_files]
    return None, f"{annotations_dir} 有多個 JSON：{names}"


def find_pred(
    pxvy_dir: Path,
    participant: int,
    scene: str,
    tool: str,
):
    """Resolve the prediction COCO path for one annotation tool."""
    tagname = f"p{participant}{scene}"

    if tool == "labelme":
        candidates = [
            pxvy_dir / "LabelMe" / f"{tagname}_labelme_coco.json",
            # Support the previous filename format.
            pxvy_dir / "LabelMe" / f"p{participant}_labelme_coco.json",
        ]
    elif tool == "cvat":
        candidates = [
            pxvy_dir
            / "CVAT"
            / "annotations"
            / "instances_default.json"
        ]
    elif tool == "stamp":
        candidates = [
            pxvy_dir
            / "STAMP"
            / "annotations"
            / "instances_train2024.json"
        ]
    else:
        raise ValueError(f"Unknown tool: {tool}")

    for path in candidates:
        if path.exists():
            return path

    return None


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Batch evaluate LabelMe, CVAT+SAM, and STAMP "
            "video annotations against GT."
        )
    )
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument(
        "--exp-dir",
        default=None,
        help="Default: <root>/tests/exp",
    )
    parser.add_argument(
        "--gt-dir",
        default=None,
        help="Default: <root>/tests/exp/GT",
    )
    parser.add_argument(
        "--participants",
        type=int,
        nargs="+",
        default=list(range(1, 9)),
    )
    parser.add_argument(
        "--videos",
        nargs="+",
        default=VIDEOS,
        help=f"Videos to process. Default: {' '.join(VIDEOS)}",
    )
    parser.add_argument(
        "--aggregate-csv",
        default=None,
        help="Default: <exp>/video_eval_summary.csv",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        videos = normalize_videos(args.videos)
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        sys.exit(1)

    root = Path(args.root)
    exp_dir = (
        Path(args.exp_dir)
        if args.exp_dir
        else root / "tests" / "exp"
    )
    gt_dir = (
        Path(args.gt_dir)
        if args.gt_dir
        else root / "tests" / "exp" / "GT"
    )

    for label, directory in (("exp", exp_dir), ("GT", gt_dir)):
        if not directory.exists():
            print(f"[ERROR] 找不到 {label} 資料夾：{directory}")
            sys.exit(1)

    agg_path = (
        Path(args.aggregate_csv)
        if args.aggregate_csv
        # Covers all three tools (LabelMe / CVAT+SAM / STAMP), not just LabelMe.
        # The old name was a leftover from a LabelMe-only draft of this script.
        else exp_dir / "video_eval_summary.csv"
    )

    dry_run_tag = " (DRY-RUN)" if args.dry_run else ""

    print(f"\n=== Batch video evaluation{dry_run_tag} ===")
    print(f"exp = {exp_dir}")
    print(f"GT  = {gt_dir}")
    print(f"participants = {args.participants}")
    print(f"videos       = {videos}\n")

    aggregate_rows = []
    ok_count = 0
    skipped_count = 0

    for participant in args.participants:
        for video in videos:
            scene = f"v{video.lower()}"
            tagname = f"p{participant}{scene}"
            pxvy_dir = exp_dir / tagname

            assignment = ASSIGNMENTS.get((participant, video))

            if assignment is None:
                print(f"[SKIP] {tagname}: 找不到 Q assignment")
                skipped_count += len(TOOLS)
                continue

            for tool, tool_config in TOOLS.items():
                display_name = tool_config["display_name"]
                seg = assignment[tool]

                pred_path = find_pred(
                    pxvy_dir=pxvy_dir,
                    participant=participant,
                    scene=scene,
                    tool=tool,
                )

                if pred_path is None:
                    print(
                        f"[SKIP] {tagname} {display_name}: "
                        "找不到 prediction COCO"
                    )
                    skipped_count += 1
                    continue

                gt_path, gt_used = resolve_gt(
                    gt_dir=gt_dir,
                    video=video,
                    seg=seg,
                )

                if gt_path is None:
                    print(
                        f"[SKIP] {tagname} {display_name}: "
                        f"GT 錯誤：{gt_used}"
                    )
                    skipped_count += 1
                    continue

                results_dir = pxvy_dir / "results"
                output_csv = (
                    results_dir / f"{tagname}_{tool}_per_frame.csv"
                )
                summary_json = (
                    results_dir / f"{tagname}_{tool}_summary.json"
                )

                if args.dry_run:
                    print(
                        f"[DRY-RUN] {tagname} "
                        f"tool={display_name} "
                        f"seg={seg} "
                        f"GT={gt_used}"
                    )
                    print(f"          pred    = {pred_path}")
                    print(f"          gt      = {gt_path}")
                    print(f"          output  = {output_csv}")
                    print(f"          summary = {summary_json}")
                    ok_count += 1
                    continue

                try:
                    summary = evaluate_video_coco(
                        pred_coco_path=pred_path,
                        gt_coco_path=gt_path,
                        manifest_path=None,
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

                mean_iou = summary.get("mean_frame_iou")
                mean_iou_text = (
                    f"{mean_iou:.4f}"
                    if isinstance(mean_iou, (int, float))
                    else "n/a"
                )

                print(
                    f"[OK] {tagname} "
                    f"tool={display_name} "
                    f"seg={seg} "
                    f"GT={gt_used} "
                    f"frames={summary['frames_evaluated']} "
                    f"mIoU={mean_iou_text} "
                    f"gt={summary['total_gt_instances']} "
                    f"pred={summary['total_pred_instances']} "
                    f"matched={summary['total_matched']} "
                    f"miss={summary['total_missed']} "
                    f"fp={summary['total_false_positive']}"
                )

                aggregate_rows.append(
                    {
                        "participant": f"P{participant}",
                        "scene": scene,
                        "tool": tool,
                        "seg": seg,
                        "gt_folder": gt_used,
                        **summary,
                    }
                )
                ok_count += 1

    print("\n--- Summary ---")
    print(f"Completed: {ok_count}")
    print(f"Skipped:   {skipped_count}")

    if not args.dry_run and aggregate_rows:
        agg_path.parent.mkdir(parents=True, exist_ok=True)

        fields = [
            "participant",
            "scene",
            "tool",
            "seg",
            "gt_folder",
            "frames_evaluated",
            "mean_frame_iou",
            "total_gt_instances",
            "total_pred_instances",
            "total_matched",
            "total_missed",
            "total_false_positive",
        ]

        with agg_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as file:
            writer = csv.DictWriter(
                file,
                fieldnames=fields,
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(aggregate_rows)

        print(f"\nAggregate CSV: {agg_path}")


if __name__ == "__main__":
    main()