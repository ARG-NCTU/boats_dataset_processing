import csv
from pathlib import Path

from tools.generate_image_set_manifest import generate_image_set_manifest


def _make_images(root: Path, count: int = 200) -> None:
    root.mkdir()
    for index in range(1, count + 1):
        (root / f"boat_{index:04d}.jpg").write_bytes(b"")


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_generates_four_strata_and_selected_rows(tmp_path):
    image_dir = tmp_path / "images"
    all_manifest = tmp_path / "image_manifest.csv"
    selected_manifest = tmp_path / "image_tasks.csv"
    _make_images(image_dir)

    generate_image_set_manifest(
        input_dir=image_dir,
        output_path=all_manifest,
        selected_output_path=selected_manifest,
        seed=20260511,
    )

    all_rows = _read_rows(all_manifest)
    selected_rows = _read_rows(selected_manifest)

    assert len(all_rows) == 200
    assert len(selected_rows) == 80
    assert sorted({row["stratum"] for row in all_rows}) == ["E1", "E2", "E3", "E4"]
    assert {row["image_id"] for row in all_rows} == {f"E_{i:04d}" for i in range(1, 201)}
    assert {row["selected_for_task"] for row in selected_rows} == {"1"}

    for stratum in ("E1", "E2", "E3", "E4"):
        assert sum(row["stratum"] == stratum for row in all_rows) == 50
        assert sum(row["stratum"] == stratum for row in selected_rows) == 20


def test_image_manifest_is_reproducible_with_same_seed(tmp_path):
    image_dir = tmp_path / "images"
    first = tmp_path / "first.csv"
    second = tmp_path / "second.csv"
    _make_images(image_dir)

    generate_image_set_manifest(image_dir, first, seed=7)
    generate_image_set_manifest(image_dir, second, seed=7)

    assert first.read_text(encoding="utf-8") == second.read_text(encoding="utf-8")


def test_copies_selected_images_into_stratum_task_folders(tmp_path):
    image_dir = tmp_path / "images"
    manifest = tmp_path / "image_manifest.csv"
    task_dir = tmp_path / "tasks"
    _make_images(image_dir)

    generate_image_set_manifest(
        input_dir=image_dir,
        output_path=manifest,
        task_output_dir=task_dir,
        seed=20260511,
    )

    for stratum in ("E1", "E2", "E3", "E4"):
        files = sorted((task_dir / f"{stratum}_20").iterdir())
        assert len(files) == 20
        assert all(path.name.startswith("E_") for path in files)
        assert all(path.suffix == ".jpg" for path in files)
