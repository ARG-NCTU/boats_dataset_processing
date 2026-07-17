#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Batch STAMP Layer 2 metrics analyzer (RFR / PTR / GER).

Wraps analyze_stamp_metrics.py's logic over every participant and scene instead
of one log file at a time. The analysis itself is unchanged: it calls the same
SessionAnalyzer.load_jsonl / analyze_layer2_actions.

What it automates
-----------------
* Finds each session log under <exp>/pXvy/STAMP/logs/*.jsonl (and pXie for the
  image-folder mode).
* Picks unit_mode per scene: "object" for video clips, "frame_object" for the
  Image Set E folder mode -- matching the original script's own help text.
* Derives --final-object-count and --total-frames from the exported COCO, so
  MAR does not fall back to its substitute denominator. The original script
  warns "Pass --final-object-count for thesis reporting"; doing that by hand for
  96 jobs is where mistakes come from.
* Writes one JSON per job plus an aggregate CSV across all jobs.

Object counting matches the rest of the pipeline: instance identity is read the
same way evaluate_video_gt reads it (instance_name first, then a non-generic
category name), because STAMP encodes object identity in categories[].name
while LabelMe puts it in annotations[].instance_name.

Usage
-----
    python tests/batch_analyze_stamp_metrics.py --dry-run
    python tests/batch_analyze_stamp_metrics.py
    python tests/batch_analyze_stamp_metrics.py --videos C D
    python tests/batch_analyze_stamp_metrics.py --participants 1 2 --mode video
"""

import argparse
import csv
import json
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path

THIS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = THIS_DIR.parent
for path in (str(PROJECT_ROOT), str(THIS_DIR)):
    if path not in sys.path:
        sys.path.insert(0, path)

from src.core.action_logger import SessionAnalyzer  # noqa: E402

try:
    # Used only to label each row with the segment it belongs to, and to help
    # disambiguate when a logs/ folder holds more than one session.
    from batch_evaluate_video import ASSIGNMENTS  # noqa: E402
except ImportError:  # pragma: no cover
    ASSIGNMENTS = None

ROOT = Path("/app")
VIDEOS = ["A", "B", "C", "D"]
GENERIC_CATEGORY_NAMES = {"boat", "ship", "vessel", "object"}


def normalize_videos(videos):
    normalized = []
    for video in videos:
        name = str(video).strip().upper()
        if name not in VIDEOS:
            raise ValueError(
                f"Unsupported video: {video}. Allowed: {', '.join(VIDEOS)}")
        if name not in normalized:
            normalized.append(name)
    return normalized


def seg_for(participant, video):
    """The segment this participant annotated with STAMP, e.g. 'Q3'."""
    if ASSIGNMENTS is None:
        return None
    assignment = ASSIGNMENTS.get((participant, video))
    return assignment.get("stamp") if assignment else None


def instance_keys_from_coco(coco_path: Path):
    """Distinct object identities in an exported COCO, pipeline-consistent."""
    coco = json.loads(coco_path.read_text(encoding="utf-8"))
    categories = {
        int(c["id"]): str(c["name"])
        for c in coco.get("categories", [])
        if "id" in c and "name" in c
    }
    keys = set()
    for annotation in coco.get("annotations", []):
        name = annotation.get("instance_name")
        if name not in (None, ""):
            keys.add(str(name))
            continue
        if "category_id" in annotation:
            category = categories.get(int(annotation["category_id"]))
            if category and category.strip().lower() not in GENERIC_CATEGORY_NAMES:
                keys.add(category)
                continue
        for field in ("instance_id", "obj_id", "track_id", "group_id"):
            value = annotation.get(field)
            if value not in (None, ""):
                keys.add(str(value))
                break
    return keys


def coco_facts(stamp_dir: Path):
    """Return (final_object_count, total_frames) from the exported COCO."""
    coco_path = stamp_dir / "annotations" / "instances_train2024.json"
    if not coco_path.exists():
        return None, None, f"找不到 {coco_path.name}"
    try:
        coco = json.loads(coco_path.read_text(encoding="utf-8"))
        frames = len(coco.get("images", []))
        objects = len(instance_keys_from_coco(coco_path))
    except Exception as exc:  # noqa: BLE001
        return None, None, f"讀取失敗: {type(exc).__name__}: {exc}"
    return (objects or None), (frames or None), None


def pick_log(logs_dir: Path, video, seg):
    """Choose the session log for this job. Returns (path, note) or (None, why)."""
    if not logs_dir.is_dir():
        return None, f"找不到 {logs_dir}"
    candidates = sorted(logs_dir.glob("*.jsonl"))
    if not candidates:
        return None, f"{logs_dir} 內沒有 .jsonl"
    if len(candidates) == 1:
        return candidates[0], ""

    # More than one session: prefer the one naming this clip, e.g. VideoA_Q2.
    if video and seg:
        tagged = [p for p in candidates if f"Video{video}_{seg}" in p.name]
        if len(tagged) == 1:
            return tagged[0], f"(從 {len(candidates)} 個 log 挑出 Video{video}_{seg})"
        if len(tagged) > 1:
            # Filenames start with a timestamp, so the last one is the latest.
            return tagged[-1], (f"⚠ {len(tagged)} 個符合 Video{video}_{seg}，"
                                f"取最新的 {tagged[-1].name}")
    return candidates[-1], (f"⚠ {len(candidates)} 個 log 無法判斷，"
                            f"取最新的 {candidates[-1].name}")


def main():
    parser = argparse.ArgumentParser(
        description="Batch analyze STAMP Layer 2 metrics (RFR/PTR/GER).")
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--exp-dir", default=None, help="Default: <root>/tests/exp")
    parser.add_argument("--participants", type=int, nargs="+",
                        default=list(range(1, 9)))
    parser.add_argument("--videos", nargs="+", default=None,
                        help=f"預設全部 {' '.join(VIDEOS)}；"
                             "指定時若未給 --mode 則自動只做 video")
    parser.add_argument("--mode", choices=["all", "video", "image"], default=None,
                        help="做 video / image / all。預設 all")
    parser.add_argument("--unit-mode-video", default="object",
                        choices=["object", "frame_object", "instance"],
                        help="video 場景的 unit mode（預設 object）")
    parser.add_argument("--unit-mode-image", default="frame_object",
                        choices=["object", "frame_object", "instance"],
                        help="Image Set E 的 unit mode（預設 frame_object）")
    parser.add_argument("--aggregate-csv", default=None,
                        help="預設 <exp>/stamp_layer2_summary.csv")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    try:
        videos = normalize_videos(args.videos) if args.videos else VIDEOS
    except ValueError as exc:
        print(f"[ERROR] {exc}")
        return 1
    mode = args.mode if args.mode is not None else ("video" if args.videos else "all")

    root = Path(args.root)
    exp_dir = Path(args.exp_dir) if args.exp_dir else root / "tests" / "exp"
    if not exp_dir.exists():
        print(f"[ERROR] 找不到 exp 資料夾：{exp_dir}")
        return 1
    agg_path = (Path(args.aggregate_csv) if args.aggregate_csv
                else exp_dir / "stamp_layer2_summary.csv")

    tag = " (DRY-RUN)" if args.dry_run else ""
    print(f"\n=== Batch STAMP Layer 2 metrics{tag} ===")
    print(f"exp          = {exp_dir}")
    print(f"participants = {args.participants}")
    print(f"mode         = {mode}   videos = {videos}\n")

    # Build the job list: (tagname, scene, video, seg, unit_mode)
    jobs = []
    for participant in args.participants:
        if mode in ("all", "video"):
            for video in videos:
                scene = f"v{video.lower()}"
                jobs.append((participant, f"p{participant}{scene}", scene, video,
                             seg_for(participant, video), args.unit_mode_video))
        if mode in ("all", "image"):
            jobs.append((participant, f"p{participant}ie", "ie", None, None,
                         args.unit_mode_image))

    rows = []
    ok = skipped = errors = fallback = 0
    mece_violations = 0

    for participant, tagname, scene, video, seg, unit_mode in jobs:
        stamp_dir = exp_dir / tagname / "STAMP"
        log_path, note = pick_log(stamp_dir / "logs", video, seg)
        if log_path is None:
            print(f"[SKIP] {tagname}: {note}")
            skipped += 1
            continue

        final_objects, total_frames, why = coco_facts(stamp_dir)
        if why:
            print(f"[WARN] {tagname}: {why} → MAR 會走 fallback")

        if args.dry_run:
            print(f"[DRY-RUN] {tagname}  unit_mode={unit_mode}  seg={seg}")
            print(f"          log          = {log_path}")
            print(f"          total_frames = {total_frames}")
            print(f"          final_objects= {final_objects}")
            if note:
                print(f"          {note}")
            ok += 1
            continue

        try:
            actions = SessionAnalyzer.load_jsonl(log_path)
            metrics = SessionAnalyzer.analyze_layer2_actions(
                actions,
                total_frames=total_frames,
                unit_mode=unit_mode,
                final_object_count=final_objects,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {tagname}: {type(exc).__name__}: {exc}")
            errors += 1
            continue

        data = asdict(metrics) if is_dataclass(metrics) else dict(vars(metrics))

        # Per-job JSON keeps everything, including the structured fields.
        out_json = exp_dir / tagname / "results" / f"{tagname}_stamp_layer2.json"
        out_json.parent.mkdir(parents=True, exist_ok=True)
        out_json.write_text(json.dumps(data, indent=2, ensure_ascii=False),
                            encoding="utf-8")

        # ---- shape the row for CSV ----------------------------------------
        row_data = dict(data)

        # outcome_counts is a dict; as a single CSV cell it is unusable, so
        # flatten it into one column per outcome.
        outcomes = row_data.pop("outcome_counts", {}) or {}
        for name, count in sorted(outcomes.items()):
            row_data[f"outcome_{name}"] = count

        # reviewed_frame_indices is a per-frame list (up to total_frames long).
        # reviewed_frames already carries the count; keep the list in the JSON.
        row_data.pop("reviewed_frame_indices", None)

        # MECE check: PTR + GER + Reject Rate should account for every reviewed
        # unit. Verify it per job rather than assuming it.
        mece_sum = None
        if all(isinstance(row_data.get(k), (int, float))
               for k in ("ptr", "ger", "reject_rate")):
            mece_sum = row_data["ptr"] + row_data["ger"] + row_data["reject_rate"]
            row_data["mece_sum"] = round(mece_sum, 6)

        used_fallback = bool(data.get("mar_fallback_used"))
        if used_fallback:
            fallback += 1

        mece_bad = (mece_sum is not None
                    and row_data.get("reviewed_units")
                    and abs(mece_sum - 1.0) > 1e-6)
        if mece_bad:
            mece_violations += 1

        flags = ""
        if used_fallback:
            flags += "  ⚠ MAR fallback"
        if mece_bad:
            flags += f"  ⚠ MECE 破損 (PTR+GER+Reject={mece_sum:.4f})"

        summary_bits = "  ".join(
            f"{k.upper()}={data[k]:.4f}" if isinstance(data.get(k), float)
            else f"{k.upper()}={data.get(k)}"
            for k in ("rfr", "ptr", "ger") if k in data
        )
        print(f"[OK] {tagname}  unit={unit_mode}  frames={total_frames} "
              f"objects={final_objects}  {summary_bits}{flags}")
        if note:
            print(f"       {note}")

        rows.append({
            "participant": f"P{participant}",
            "scene": scene,
            "video": f"Video {video}" if video else "ImageSetE",
            "seg": seg or "",
            "unit_mode": unit_mode,
            "log_file": str(log_path),
            **row_data,
        })
        ok += 1

    print(f"\n--- Summary ---\nCompleted: {ok}\nSkipped:   {skipped}\nErrors:    {errors}")
    if fallback:
        print(f"⚠ {fallback} 個 job 走了 MAR fallback（final_object_count 推不出來）"
              f"\n  論文報告前請確認這些 job 的 STAMP COCO 是否存在。")
    if mece_violations:
        print(f"⚠ {mece_violations} 個 job 的 PTR+GER+Reject Rate != 1.0"
              f"\n  MECE 約束是三層框架的前提，出現這個代表 log 或分析邏輯有問題，"
              f"\n  請先查清楚再拿去報告。")

    if rows and not args.dry_run:
        # Union of keys, so new metric fields flow through without editing this file.
        fields, seen = [], set()
        for row in rows:
            for key in row:
                if key not in seen:
                    seen.add(key)
                    fields.append(key)
        agg_path.parent.mkdir(parents=True, exist_ok=True)
        with agg_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nAggregate CSV: {agg_path}")

    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())