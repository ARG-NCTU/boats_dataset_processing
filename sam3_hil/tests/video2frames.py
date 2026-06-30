#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Split videos into per-frame PNG images.

For each input video, every frame is written as a PNG into an output folder
named after the video (its filename without extension). Modeled on
``videos_processing/mp42png.py`` but extracts *all* frames (each Exp video
only has 20).

Layout (default batch mode), with X = A..D and Y = 1..4::

    data/video/Exp/tasks/VideoX/VideoX_QY.mp4
        ->
    data/video/Exp/tasks/VideoX/VideoX_QY/VideoX_QY_1.png
    data/video/Exp/tasks/VideoX/VideoX_QY/VideoX_QY_2.png
    ...

Usage
-----
    # Process every VideoX/VideoX_QY.mp4 under the default tasks dir
    python3 tests/video2frames.py

    # Process a single video (output folder created next to it)
    python3 tests/video2frames.py --video data/video/Exp/tasks/VideoA/VideoA_Q1.mp4

    # Point at a different tasks dir
    python3 tests/video2frames.py --tasks-dir /path/to/tasks
"""

import argparse
import sys
from pathlib import Path

import cv2

THIS_DIR = Path(__file__).resolve().parent
REPO_ROOT = THIS_DIR.parent  # tests/ -> sam3_hil/
DEFAULT_TASKS_DIR = REPO_ROOT / "data" / "video" / "Exp" / "tasks"

VIDEO_LETTERS = ["A", "B", "C", "D"]
QUESTION_NUMS = [1, 2, 3, 4]
VIDEO_SUFFIXES = (".mp4", ".MP4", ".mov", ".MOV", ".avi")


def extract_frames(
    video_path: Path,
    output_dir: Path | None = None,
    start_index: int = 1,
    ext: str = "png",
    overwrite: bool = False,
) -> int:
    """Write every frame of ``video_path`` as an image.

    Frames are saved into ``output_dir`` (default: a folder named after the
    video, beside it) as ``<video_stem>_<n>.<ext>`` starting from
    ``start_index``. Returns the number of frames written.
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    stem = video_path.stem  # e.g. "VideoA_Q1"
    output_dir = video_path.parent / stem if output_dir is None else Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not overwrite and any(output_dir.glob(f"{stem}_*.{ext}")):
        print(f"  {video_path.name}: frames already exist in {output_dir}, "
              f"skipping (use --overwrite to redo)")
        return 0

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    written = 0
    index = start_index
    try:
        success, frame = cap.read()
        while success:
            out_path = output_dir / f"{stem}_{index}.{ext}"
            if not cv2.imwrite(str(out_path), frame):
                raise RuntimeError(f"Failed to write frame: {out_path}")
            written += 1
            index += 1
            success, frame = cap.read()
    finally:
        cap.release()

    print(f"  {video_path.name}: wrote {written} frame(s) -> {output_dir}")
    return written


def iter_default_videos(tasks_dir: Path):
    """Yield the VideoX/VideoX_QY.<ext> files that exist under ``tasks_dir``."""
    for letter in VIDEO_LETTERS:
        video_dir = tasks_dir / f"Video{letter}"
        for q in QUESTION_NUMS:
            stem = f"Video{letter}_Q{q}"
            for suffix in VIDEO_SUFFIXES:
                candidate = video_dir / f"{stem}{suffix}"
                if candidate.exists():
                    yield candidate
                    break


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description="Split video(s) into per-frame images, one folder per video."
    )
    parser.add_argument(
        "--video", type=Path, default=None,
        help="Process a single video file instead of the default batch.",
    )
    parser.add_argument(
        "--tasks-dir", type=Path, default=DEFAULT_TASKS_DIR,
        help="Root containing VideoX/VideoX_QY.mp4 (default: %(default)s).",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None,
        help="Override output folder (single-video mode only). "
             "Default is a folder named after the video, beside it.",
    )
    parser.add_argument(
        "--start-index", type=int, default=1,
        help="Frame numbering start (default: 1).",
    )
    parser.add_argument(
        "--ext", type=str, default="png",
        help="Output image extension (default: png).",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Re-extract even if the output folder already has frames.",
    )
    args = parser.parse_args(argv)

    if args.video is not None:
        total = extract_frames(
            args.video,
            output_dir=args.output_dir,
            start_index=args.start_index,
            ext=args.ext,
            overwrite=args.overwrite,
        )
        print(f"Done. {total} frame(s) extracted from {args.video.name}.")
        return 0

    if args.output_dir is not None:
        parser.error("--output-dir is only valid together with --video.")

    videos = list(iter_default_videos(args.tasks_dir))
    if not videos:
        print(f"No videos found under {args.tasks_dir} "
              f"(expected VideoX/VideoX_QY with X=A-D, Y=1-4).", file=sys.stderr)
        return 1

    print(f"Found {len(videos)} video(s) under {args.tasks_dir}")
    grand_total = 0
    for video in videos:
        grand_total += extract_frames(
            video,
            start_index=args.start_index,
            ext=args.ext,
            overwrite=args.overwrite,
        )
    print(f"Done. {grand_total} frame(s) extracted from {len(videos)} video(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
