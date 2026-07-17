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

try:
    # ASSIGNMENTS lets each pred row record which segment it annotated, so GT
    # baseline rows can be joined to tool rows on (scene, seg).
    from batch_evaluate_video import ASSIGNMENTS, resolve_gt  # noqa: E402
except ImportError:  # pragma: no cover
    ASSIGNMENTS, resolve_gt = None, None


TOOLS = {
    "labelme": "LabelMe",
    "cvat": "CVAT+SAM",
    "stamp": "STAMP",
}

DEFAULT_VIDEOS = ["A", "B", "C", "D"]
SEGMENTS = ["Q1", "Q2", "Q3", "Q4"]


def gt_clips(gt_dir: Path, videos: List[str]):
    """Yield (video, seg, gt_json_path, folder_name) for each GT clip.

    GT Temporal IoU is a property of the *clip*, not of a participant: there are
    only 4 videos x 4 segments = 16 GT clips, versus 96 participant jobs. Each
    clip is measured once and serves as the baseline that every tool annotating
    that same clip is compared against.
    """
    for video in videos:
        for seg in SEGMENTS:
            if resolve_gt is None:
                continue
            gt_path, folder = resolve_gt(gt_dir=gt_dir, video=video, seg=seg)
            if gt_path is not None:
                yield video, seg, gt_path, folder


def seg_for(participant: int, video: str, tool: str) -> str:
    """Which segment this participant annotated with this tool."""
    if ASSIGNMENTS is None:
        return ""
    assignment = ASSIGNMENTS.get((participant, video))
    return assignment.get(tool, "") if assignment else ""


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
        "seg",
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
        help="Videos to process. Default: A B C D",
    )

    parser.add_argument(
        "--aggregate-csv",
        type=Path,
        default=None,
        help="Default: <exp>/temporal_iou_summary.csv",
    )

    parser.add_argument(
        "--gt-dir",
        type=Path,
        default=None,
        help="Default: <root>/tests/exp/GT",
    )

    parser.add_argument(
        "--include-gt",
        action="store_true",
        help=(
            "Also compute Temporal IoU on each GT clip as a baseline, and "
            "report each tool's deviation from it. Without a baseline, a raw "
            "Temporal IoU of 1.0 cannot be told apart from copy-pasted "
            "annotations in a static scene."
        ),
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

    gt_dir = (
        Path(args.gt_dir)
        if args.gt_dir is not None
        else exp_dir / "GT"
    )

    if args.include_gt:
        if resolve_gt is None:
            print("[ERROR] --include-gt 需要 batch_evaluate_video.py "
                  "在同一個資料夾（用來解析 GT 路徑與 seg 對應）")
            return 1
        if not gt_dir.exists():
            print(f"[ERROR] 找不到 GT 資料夾：{gt_dir}")
            return 1

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
                        "seg": seg_for(participant, video, tool),
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

    # ---- GT baseline: one Temporal IoU per clip, not per participant --------
    gt_by_clip = {}
    if args.include_gt:
        print("\n--- GT baseline ---")
        for video, seg, gt_path, folder in gt_clips(gt_dir, videos):
            scene = f"v{video.lower()}"
            if args.dry_run:
                print(f"[DRY-RUN] GT {scene}{seg} ({folder})")
                print(f"          coco = {gt_path}")
                continue
            try:
                summary = evaluate_temporal_iou(
                    coco_path=gt_path,
                    output_path=(exp_dir / "GT_temporal_iou"
                                 / f"{scene}{seg}_gt_temporal_iou.csv"),
                    summary_path=(exp_dir / "GT_temporal_iou"
                                  / f"{scene}{seg}_gt_temporal_iou_summary.json"),
                )
            except Exception as exc:  # noqa: BLE001
                print(f"[ERROR] GT {scene}{seg}: {type(exc).__name__}: {exc}")
                error_count += 1
                continue

            mean_iou = summary.get("mean_temporal_iou")
            gt_by_clip[(scene, seg)] = mean_iou
            mean_text = (f"{mean_iou:.4f}"
                         if isinstance(mean_iou, (int, float)) else "n/a")
            print(f"[GT] {scene}{seg} ({folder}) frames={summary['frame_count']} "
                  f"instances={summary['instance_count']} TemporalIoU={mean_text}")

            aggregate_rows.append({
                "participant": "GT",
                "video": f"Video {video}",
                "scene": scene,
                "seg": seg,
                "tool": "gt",
                "tool_name": "GroundTruth",
                "coco_path": str(gt_path),
                "frame_count": summary.get("frame_count"),
                "pair_count": summary.get("pair_count"),
                "adjacent_frame_pair_count": summary.get("adjacent_frame_pair_count"),
                "instance_count": summary.get("instance_count"),
                "instance_names": ",".join(
                    str(n) for n in summary.get("instance_names", [])),
                "missing_pair_count": summary.get("missing_pair_count"),
                "mean_temporal_iou": mean_iou,
            })

    print("\n--- Summary ---")
    print(f"Completed: {completed_count}")
    print(f"Skipped:   {skipped_count}")
    print(f"Errors:    {error_count}")

    # ---- pred vs GT deviation ---------------------------------------------
    if gt_by_clip and not args.dry_run:
        print("\n--- Temporal IoU：各工具與 GT 基準線的偏離 ---")
        print("  正值 = 比 GT 更平滑（可能是沿用/複製上一格）")
        print("  負值 = 比 GT 更跳動\n")
        by_clip_tool = {}
        for row in aggregate_rows:
            if row["tool"] == "gt":
                continue
            value = row.get("mean_temporal_iou")
            if not isinstance(value, (int, float)):
                continue
            by_clip_tool.setdefault((row["scene"], row["seg"]), {}) \
                        .setdefault(row["tool"], []).append(value)

        header = f"{'clip':<9}{'GT':>9}" + "".join(
            f"{TOOLS[t]:>12}" for t in TOOLS)
        print(header)
        deltas = {t: [] for t in TOOLS}
        for (scene, seg), gt_value in sorted(gt_by_clip.items()):
            if not isinstance(gt_value, (int, float)):
                continue
            line = f"{scene}{seg:<7}{gt_value:>9.4f}"
            for tool in TOOLS:
                values = by_clip_tool.get((scene, seg), {}).get(tool)
                if not values:
                    line += f"{'-':>12}"
                    continue
                delta = sum(values) / len(values) - gt_value
                deltas[tool].append(delta)
                line += f"{delta:>+12.4f}"
            print(line)

        print(f"\n{'tool':<12}{'n':>4}{'mean Δ':>10}{'mean |Δ|':>11}")
        for tool in TOOLS:
            values = deltas[tool]
            if not values:
                continue
            mean_delta = sum(values) / len(values)
            mean_abs = sum(abs(v) for v in values) / len(values)
            print(f"{TOOLS[tool]:<12}{len(values):>4}{mean_delta:>+10.4f}{mean_abs:>11.4f}")
        print("\n  mean |Δ| 越小 = 時序一致性越接近真實情況。")
        print("  這比原始 Temporal IoU 更適合跨工具比較：原始值會獎勵複製貼上。")

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