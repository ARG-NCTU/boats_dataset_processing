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
