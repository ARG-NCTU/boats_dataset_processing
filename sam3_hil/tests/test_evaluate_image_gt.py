import csv
import json
from pathlib import Path

from tools.evaluate_image_gt import evaluate_image_coco


def _square(x1: int, y1: int, x2: int, y2: int) -> list[int]:
    return [x1, y1, x2, y1, x2, y2, x1, y2]


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data), encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_evaluates_image_prediction_against_gt_with_image_manifest(tmp_path):
    manifest = tmp_path / "image_tasks.csv"
    manifest.write_text(
        "image_id,original_filename,image_path,stratum,order_in_stratum,selected_for_task\n"
        "E_0041,boat_0041.jpg,/data/boat_0041.jpg,E3,1,1\n",
        encoding="utf-8",
    )
    pred = tmp_path / "pred.json"
    gt = tmp_path / "gt.json"
    per_image = tmp_path / "per_image.csv"

    _write_json(
        pred,
        {
            "images": [{"id": 1, "file_name": "boat_0041.jpg", "width": 20, "height": 20}],
            "annotations": [
                {"id": 1, "image_id": 1, "category_id": 1, "segmentation": [_square(2, 2, 10, 10)]}
            ],
            "categories": [{"id": 1, "name": "vessel"}],
        },
    )
    _write_json(
        gt,
        {
            "images": [{"id": 41, "file_name": "boat_0041.jpg", "width": 20, "height": 20}],
            "annotations": [
                {"id": 1, "image_id": 41, "category_id": 1, "segmentation": [_square(2, 2, 10, 10)]}
            ],
            "categories": [{"id": 1, "name": "vessel"}],
        },
    )

    summary = evaluate_image_coco(pred, gt, manifest, per_image)

    rows = _read_csv(per_image)
    assert rows[0]["image_id"] == "E_0041"
    assert rows[0]["original_filename"] == "boat_0041.jpg"
    assert rows[0]["gt_count"] == "1"
    assert rows[0]["pred_count"] == "1"
    assert float(rows[0]["image_iou"]) == 1.0
    assert summary["mean_image_iou"] == 1.0
    assert summary["total_matched"] == 1


def test_ignores_gt_images_outside_image_manifest(tmp_path):
    manifest = tmp_path / "image_tasks.csv"
    manifest.write_text(
        "image_id,original_filename,image_path,stratum,order_in_stratum,selected_for_task\n"
        "E_0041,boat_0041.jpg,/data/boat_0041.jpg,E3,1,1\n",
        encoding="utf-8",
    )
    pred = tmp_path / "pred.json"
    gt = tmp_path / "gt.json"
    per_image = tmp_path / "per_image.csv"

    _write_json(
        pred,
        {
            "images": [{"id": 1, "file_name": "boat_0041.jpg", "width": 20, "height": 20}],
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
                {"id": 41, "file_name": "boat_0041.jpg", "width": 20, "height": 20},
                {"id": 42, "file_name": "boat_0042.jpg", "width": 20, "height": 20},
            ],
            "annotations": [
                {"id": 1, "image_id": 41, "category_id": 1, "segmentation": [_square(2, 2, 10, 10)]},
                {"id": 2, "image_id": 42, "category_id": 1, "segmentation": [_square(2, 2, 10, 10)]},
            ],
            "categories": [{"id": 1, "name": "vessel"}],
        },
    )

    summary = evaluate_image_coco(pred, gt, manifest, per_image)

    assert len(_read_csv(per_image)) == 1
    assert summary["images_evaluated"] == 1
    assert summary["total_missed"] == 0
