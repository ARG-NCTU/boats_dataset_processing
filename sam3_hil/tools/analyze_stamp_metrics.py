#!/usr/bin/env python3
"""Offline STAMP Layer 2 metrics analyzer."""

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.core.action_logger import SessionAnalyzer  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze STAMP Layer 2 metrics from a JSONL action log."
    )
    parser.add_argument("log_file", type=Path, help="Path to a STAMP JSONL session log.")
    parser.add_argument(
        "--unit-mode",
        choices=["object", "frame_object"],
        default="object",
        help="Use object for video mode or frame_object for image-folder mode.",
    )
    parser.add_argument(
        "--total-frames",
        type=int,
        default=None,
        help="Override total frame/image count when it is not available in the log.",
    )
    parser.add_argument(
        "--final-object-count",
        type=int,
        default=None,
        help="Final exported object count. Required for non-fallback MAR.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional JSON output path for the computed metrics.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not args.log_file.exists():
        print(f"Log file not found: {args.log_file}", file=sys.stderr)
        return 2

    actions = SessionAnalyzer.load_jsonl(args.log_file)
    metrics = SessionAnalyzer.analyze_layer2_actions(
        actions,
        total_frames=args.total_frames,
        unit_mode=args.unit_mode,
        final_object_count=args.final_object_count,
    )

    print(metrics)
    if metrics.mar_fallback_used:
        print("MAR used fallback denominator. Pass --final-object-count for thesis reporting.")

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(asdict(metrics), f, indent=2, ensure_ascii=False)
        print(f"Wrote metrics JSON: {args.output}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
