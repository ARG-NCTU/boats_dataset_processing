from pathlib import Path


def test_source_mode_default_export_dir_is_project_data_output():
    from src.gui.export_paths import get_default_export_dir

    project_root = Path(__file__).resolve().parents[1]

    assert get_default_export_dir(is_frozen=False) == project_root / "data" / "output"


def test_source_mode_default_log_dir_is_under_project_data_output():
    from src.gui.export_paths import get_default_log_dir

    project_root = Path(__file__).resolve().parents[1]

    assert get_default_log_dir(is_frozen=False) == project_root / "data" / "output" / "logs"


def test_frozen_mode_default_export_dir_stays_in_user_home(tmp_path):
    from src.gui.export_paths import get_default_export_dir

    assert get_default_export_dir(is_frozen=True, home=tmp_path) == tmp_path / "data" / "output"


def test_frozen_mode_default_log_dir_stays_under_user_home_output(tmp_path):
    from src.gui.export_paths import get_default_log_dir

    assert get_default_log_dir(is_frozen=True, home=tmp_path) == tmp_path / "data" / "output" / "logs"
