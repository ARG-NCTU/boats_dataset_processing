#!/usr/bin/env python3
"""Generate a clip-to-source frame manifest for an existing video clip."""

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


MANIFEST_FIELDS = [
    "source_video",
    "output_video",
    "stratum",
    "clip_frame_idx",
    "source_frame_idx",
    "start_frame",
    "end_frame_inclusive",
    "num_frames",
    "fps",
    "width",
    "height",
    "seed",
]


@dataclass(frozen=True)
class VideoClipManifestRow:
    source_video: str
    output_video: str
    stratum: str
    clip_frame_idx: int
    source_frame_idx: int
    start_frame: int
    end_frame_inclusive: int
    num_frames: int
    fps: str
    width: str
    height: str
    seed: str


def build_video_clip_manifest_rows(
    *,
    source_video: Path,
    output_video: Path,
    start_frame: int,
    num_frames: int,
    stratum: str = "",
    fps: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    seed: Optional[int] = None,
) -> list[VideoClipManifestRow]:
    if start_frame < 0:
        raise ValueError("start_frame must be greater than or equal to 0")
    if num_frames <= 0:
        raise ValueError("num_frames must be greater than 0")

    end_frame_inclusive = start_frame + num_frames - 1
    return [
        VideoClipManifestRow(
            source_video=str(source_video),
            output_video=str(output_video),
            stratum=stratum,
            clip_frame_idx=clip_frame_idx,
            source_frame_idx=start_frame + clip_frame_idx,
            start_frame=start_frame,
            end_frame_inclusive=end_frame_inclusive,
            num_frames=num_frames,
            fps="" if fps is None else str(fps),
            width="" if width is None else str(width),
            height="" if height is None else str(height),
            seed="" if seed is None else str(seed),
        )
        for clip_frame_idx in range(num_frames)
    ]


def write_video_clip_manifest(
    *,
    source_video: Path,
    output_video: Path,
    output_path: Path,
    start_frame: int,
    num_frames: int,
    stratum: str = "",
    fps: Optional[float] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    seed: Optional[int] = None,
) -> list[VideoClipManifestRow]:
    rows = build_video_clip_manifest_rows(
        source_video=source_video,
        output_video=output_video,
        start_frame=start_frame,
        num_frames=num_frames,
        stratum=stratum,
        fps=fps,
        width=width,
        height=height,
        seed=seed,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.__dict__)
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate clip-to-source frame CSV.")
    parser.add_argument("--source-video", type=Path, required=True)
    parser.add_argument("--output-video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, required=True)
    parser.add_argument("--num-frames", type=int, required=True)
    parser.add_argument("--stratum", default="")
    parser.add_argument("--fps", type=float, default=None)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = write_video_clip_manifest(
        source_video=args.source_video,
        output_video=args.output_video,
        output_path=args.output,
        start_frame=args.start_frame,
        num_frames=args.num_frames,
        stratum=args.stratum,
        fps=args.fps,
        width=args.width,
        height=args.height,
        seed=args.seed,
    )
    print(f"Wrote {len(rows)} frame mapping rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
