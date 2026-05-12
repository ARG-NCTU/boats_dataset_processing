import csv
from pathlib import Path

from tools.generate_video_clip_manifest import write_video_clip_manifest


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_writes_clip_to_source_frame_mapping(tmp_path):
    output = tmp_path / "VideoA_Q2.csv"

    write_video_clip_manifest(
        source_video=Path("VideoA.mp4"),
        output_video=Path("VideoA_Q2.mp4"),
        output_path=output,
        start_frame=60,
        num_frames=20,
        fps=10.0,
        width=640,
        height=480,
        stratum="Q2",
    )

    rows = _read_rows(output)
    assert len(rows) == 20
    assert [int(row["clip_frame_idx"]) for row in rows] == list(range(20))
    assert [int(row["source_frame_idx"]) for row in rows] == list(range(60, 80))
    assert {row["stratum"] for row in rows} == {"Q2"}
    assert {row["start_frame"] for row in rows} == {"60"}
    assert {row["end_frame_inclusive"] for row in rows} == {"79"}
