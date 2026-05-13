import csv
import json
from pathlib import Path

from tools.evaluate_video_gt import evaluate_video_coco


def _square(x1: int, y1: int, x2: int, y2: int) -> list[int]:
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_evaluates_clip_prediction_against_full_video_gt(tmp_path):
    manifest = tmp_path / "clip.csv"
    manifest.write_text(
        "clip_frame_idx,source_frame_idx\n"
        "0,60\n",
        encoding="utf-8",
    )
    pred = tmp_path / "pred.json"
    gt = tmp_path / "gt.json"
    per_frame = tmp_path / "per_frame.csv"

    _write_json(
        pred,
        {
            "images": [{"id": 1, "file_name": "frame_000000.jpg", "width": 20, "height": 20}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "segmentation": [_square(2, 2, 10, 10)]}
            ],
            "categories": [{"id": 1, "name": "vessel"}],
        },
    )
    _write_json(
        gt,
        {
            "images": [{"id": 61, "file_name": "frame_000060.jpg", "width": 20, "height": 20}],
            "annotations": [
                {"id": 1, "image_id": 61, "category_id": 1, "segmentation": [_square(2, 2, 10, 10)]}
            ],
            "categories": [{"id": 1, "name": "vessel"}],
        },
    )

    summary = evaluate_video_coco(pred, gt, manifest, per_frame)

    rows = _read_csv(per_frame)
    assert rows[0]["clip_frame_idx"] == "0"
    assert rows[0]["source_frame_idx"] == "60"
    assert rows[0]["gt_count"] == "1"
    assert rows[0]["pred_count"] == "1"
    assert float(rows[0]["frame_iou"]) == 1.0
    assert summary["mean_frame_iou"] == 1.0
    assert summary["total_matched"] == 1


def test_ignores_full_video_gt_frames_outside_manifest(tmp_path):
    manifest = tmp_path / "clip.csv"
    manifest.write_text(
        "clip_frame_idx,source_frame_idx\n"
        "0,60\n",
        encoding="utf-8",
    )
    pred = tmp_path / "pred.json"
    gt = tmp_path / "gt.json"
    per_frame = tmp_path / "per_frame.csv"

    _write_json(
        pred,
        {
            "images": [{"id": 1, "file_name": "frame_000000.jpg", "width": 20, "height": 20}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "segmentation": [_square(2, 2, 10, 10)]}
            ],
            "categories": [{"id": 1, "name": "vessel"}],
        },
    )
    _write_json(
        gt,
        {
            "images": [
                {"id": 61, "file_name": "frame_000060.jpg", "width": 20, "height": 20},
                {"id": 62, "file_name": "frame_000061.jpg", "width": 20, "height": 20},
            ],
            "annotations": [
                {"id": 1, "image_id": 61, "category_id": 1, "segmentation": [_square(2, 2, 10, 10)]},
                {"id": 2, "image_id": 62, "category_id": 1, "segmentation": [_square(2, 2, 10, 10)]},
            ],
            "categories": [{"id": 1, "name": "vessel"}],
        },
    )

    summary = evaluate_video_coco(pred, gt, manifest, per_frame)

    assert len(_read_csv(per_frame)) == 1
    assert summary["frames_evaluated"] == 1
    assert summary["total_missed"] == 0
