#!/usr/bin/env python3
"""Extract a contiguous frame clip from a source video as MP4.

The tool name is kept for compatibility with existing thesis scripts, but the
behavior is now video clip extraction rather than participant sampling.
"""

import argparse
import csv
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2


DEFAULT_SEED = 20260511
MANIFEST_FIELDS = [
    "source_video",
    "output_video",
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
class VideoMetadata:
    total_frames: int
    fps: float
    width: int
    height: int


@dataclass(frozen=True)
class ClipExtractionResult:
    input_path: Path
    output_path: Path
    manifest_path: Path
    start_frame: int
    end_frame_inclusive: int
    num_frames: int
    metadata: VideoMetadata
    seed: Optional[int] = None


def read_video_metadata(input_path: Path) -> VideoMetadata:
    """Read source video metadata with OpenCV."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input video does not exist: {input_path}")
    if not input_path.is_file():
        raise ValueError(f"Input path is not a file: {input_path}")

    cap = cv2.VideoCapture(str(input_path))
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open input video: {input_path}")

        metadata = VideoMetadata(
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            fps=float(cap.get(cv2.CAP_PROP_FPS)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
    finally:
        cap.release()

    if metadata.total_frames <= 0:
        raise ValueError(f"Input video has no readable frames: {input_path}")
    if metadata.width <= 0 or metadata.height <= 0:
        raise ValueError(f"Input video has invalid dimensions: {input_path}")
    return metadata


def resolve_start_frame(
    *,
    total_frames: int,
    num_frames: int,
    start_frame: Optional[int],
    random_mode: bool,
    seed: Optional[int],
) -> int:
    """Validate clip parameters and resolve the 0-based start frame."""
    if num_frames <= 0:
        raise ValueError("num_frames must be greater than 0")
    if total_frames <= 0:
        raise ValueError("total_frames must be greater than 0")
    if num_frames > total_frames:
        raise ValueError("num_frames exceeds source video length")
    if random_mode and start_frame is not None:
        raise ValueError("--random and --start-frame cannot be used together")
    if not random_mode and start_frame is None:
        raise ValueError("--start-frame is required unless --random is used")

    max_start = total_frames - num_frames
    if random_mode:
        rng = random.Random(seed)
        return rng.randint(0, max_start)

    assert start_frame is not None
    if start_frame < 0:
        raise ValueError("start_frame must be greater than or equal to 0")
    if start_frame > max_start:
        raise ValueError("start_frame + num_frames exceeds source video length")
    return start_frame


def _writer_fps(metadata: VideoMetadata) -> float:
    return metadata.fps if metadata.fps > 0 else 30.0


def _write_manifest(result: ClipExtractionResult) -> None:
    result.manifest_path.parent.mkdir(parents=True, exist_ok=True)
    seed_value = "" if result.seed is None else str(result.seed)

    with result.manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for clip_frame_idx in range(result.num_frames):
            source_frame_idx = result.start_frame + clip_frame_idx
            writer.writerow(
                {
                    "source_video": str(result.input_path),
                    "output_video": str(result.output_path),
                    "clip_frame_idx": clip_frame_idx,
                    "source_frame_idx": source_frame_idx,
                    "start_frame": result.start_frame,
                    "end_frame_inclusive": result.end_frame_inclusive,
                    "num_frames": result.num_frames,
                    "fps": result.metadata.fps,
                    "width": result.metadata.width,
                    "height": result.metadata.height,
                    "seed": seed_value,
                }
            )


def extract_video_clip(
    *,
    input_path: Path,
    output_path: Path,
    start_frame: int,
    num_frames: int,
    manifest_path: Optional[Path] = None,
    seed: Optional[int] = None,
) -> ClipExtractionResult:
    """Extract a contiguous frame range from input_path into output_path."""
    input_path = Path(input_path)
    output_path = Path(output_path)
    manifest_path = Path(manifest_path) if manifest_path else output_path.with_suffix(".csv")

    if output_path.suffix.lower() != ".mp4":
        raise ValueError("output_path must end with .mp4")

    metadata = read_video_metadata(input_path)
    resolved_start = resolve_start_frame(
        total_frames=metadata.total_frames,
        num_frames=num_frames,
        start_frame=start_frame,
        random_mode=False,
        seed=seed,
    )
    end_frame_inclusive = resolved_start + num_frames - 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(input_path))
    writer = None

    try:
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open input video: {input_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, resolved_start)
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            _writer_fps(metadata),
            (metadata.width, metadata.height),
        )
        if not writer.isOpened():
            raise RuntimeError(f"Cannot open output video writer: {output_path}")

        for offset in range(num_frames):
            ok, frame = cap.read()
            if not ok:
                source_frame = resolved_start + offset
                raise RuntimeError(f"Failed to read source frame {source_frame}")
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()
        cap.release()

    result = ClipExtractionResult(
        input_path=input_path,
        output_path=output_path,
        manifest_path=manifest_path,
        start_frame=resolved_start,
        end_frame_inclusive=end_frame_inclusive,
        num_frames=num_frames,
        metadata=metadata,
        seed=seed,
    )
    _write_manifest(result)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract a contiguous frame range from a source video as MP4."
    )
    parser.add_argument("--input", type=Path, required=True, help="Source video path.")
    parser.add_argument("--output", type=Path, required=True, help="Output MP4 path.")
    parser.add_argument(
        "--num-frames",
        type=int,
        required=True,
        help="Number of contiguous frames to write.",
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--start-frame",
        type=int,
        default=None,
        help="0-based source frame index where the clip starts.",
    )
    selection.add_argument(
        "--random",
        action="store_true",
        help="Randomly choose a valid contiguous clip start frame.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=DEFAULT_SEED,
        help="Seed used with --random.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Output CSV manifest path. Defaults to the output MP4 path with .csv.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    metadata = read_video_metadata(args.input)
    start_frame = resolve_start_frame(
        total_frames=metadata.total_frames,
        num_frames=args.num_frames,
        start_frame=args.start_frame,
        random_mode=args.random,
        seed=args.seed,
    )
    result = extract_video_clip(
        input_path=args.input,
        output_path=args.output,
        start_frame=start_frame,
        num_frames=args.num_frames,
        manifest_path=args.manifest,
        seed=args.seed if args.random else None,
    )
    print(
        "Wrote "
        f"{result.num_frames} frames "
        f"({result.start_frame}-{result.end_frame_inclusive}) "
        f"to {result.output_path}"
    )
    print(f"Manifest: {result.manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
