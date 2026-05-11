import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from tools.generate_stratified_sampling import extract_video_clip, resolve_start_frame


def _write_test_video(path: Path, frame_count: int = 40, fps: float = 10.0) -> None:
    width, height = 32, 24
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:, :] = (index % 255, (index * 3) % 255, (index * 7) % 255)
        writer.write(frame)
    writer.release()


def _video_frame_count(path: Path) -> int:
    cap = cv2.VideoCapture(str(path))
    try:
        assert cap.isOpened()
        return int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        cap.release()


def _read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_extracts_requested_contiguous_frames_and_writes_manifest(tmp_path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "clip.mp4"
    manifest = tmp_path / "clip.csv"
    _write_test_video(source, frame_count=40, fps=12.0)

    result = extract_video_clip(
        input_path=source,
        output_path=output,
        start_frame=10,
        num_frames=20,
        manifest_path=manifest,
        seed=None,
    )

    assert result.start_frame == 10
    assert result.end_frame_inclusive == 29
    assert _video_frame_count(output) == 20

    rows = _read_manifest(manifest)
    assert [int(row["clip_frame_idx"]) for row in rows] == list(range(20))
    assert [int(row["source_frame_idx"]) for row in rows] == list(range(10, 30))
    assert {row["source_video"] for row in rows} == {str(source)}
    assert {row["output_video"] for row in rows} == {str(output)}
    assert {row["num_frames"] for row in rows} == {"20"}
    assert {row["seed"] for row in rows} == {""}


def test_random_start_frame_is_reproducible_with_same_seed():
    first = resolve_start_frame(
        total_frames=50,
        num_frames=20,
        start_frame=None,
        random_mode=True,
        seed=20260511,
    )
    second = resolve_start_frame(
        total_frames=50,
        num_frames=20,
        start_frame=None,
        random_mode=True,
        seed=20260511,
    )

    assert first == second
    assert 0 <= first <= 30


def test_rejects_clip_range_beyond_source_video(tmp_path):
    source = tmp_path / "source.mp4"
    output = tmp_path / "clip.mp4"
    _write_test_video(source, frame_count=25)

    with pytest.raises(ValueError, match="exceeds source video length"):
        extract_video_clip(
            input_path=source,
            output_path=output,
            start_frame=10,
            num_frames=20,
        )
