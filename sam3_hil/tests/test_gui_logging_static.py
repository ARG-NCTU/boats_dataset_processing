"""
Static regression checks for GUI action logging paths.
"""

from pathlib import Path


def test_add_object_propagation_logs_manual_add_before_propagate():
    source = Path("src/gui/main_window_server.py").read_text(encoding="utf-8")
    function_body = source.split("def _propagate_to_following_frames", 1)[1]
    function_body = function_body.split("def _simple_propagate", 1)[0]

    add_object_branch = function_body.split("if self.add_object_mode:", 1)[1]
    before_propagate_log = add_object_branch.split("self.action_logger.log_propagate", 1)[0]

    assert "self.action_logger.log_add_object" in before_propagate_log


def test_server_gui_action_logs_use_default_output_log_dir():
    source = Path("src/gui/main_window_server.py").read_text(encoding="utf-8")

    assert "from gui.export_paths import get_default_export_dir, get_default_log_dir" in source
    assert "logs_dir = get_default_log_dir()" in source
    assert 'self.action_logger.relocate_output_dir(Path(stats.output_dir) / "logs")' in source
    assert 'video_dir / "logs"' not in source
    assert 'Path(folder_path) / "logs"' not in source


def test_polygon_overlay_has_independent_checkbox_and_selected_object_filter():
    source = Path("src/gui/main_window_server.py").read_text(encoding="utf-8")

    assert 'QCheckBox("Show Polygon")' in source
    assert "self.show_polygons = self.polygon_checkbox.isChecked()" in source
    assert "def get_selected_polygon_obj_id" in source

    visualize_body = source.split("def visualize_frame", 1)[1]
    visualize_body = visualize_body.split("# =========================================================================", 1)[0]

    assert "selected_polygon_obj_id = self.get_selected_polygon_obj_id()" in visualize_body
    assert "selected_polygon_obj_id is not None" in visualize_body
    assert "det.obj_id != selected_polygon_obj_id" in visualize_body
    assert "cv2.findContours" in visualize_body
    assert "cv2.drawContours" in visualize_body
