import json


def test_action_logger_relocates_active_jsonl_session_to_export_logs(tmp_path):
    from src.core.action_logger import ActionLogger

    initial_logs = tmp_path / "output" / "logs"
    export_logs = tmp_path / "output" / "VideoA_Q1" / "logs"

    action_logger = ActionLogger(output_dir=initial_logs, format="jsonl", auto_flush=True)
    session_id = action_logger.start_session(
        video_path="VideoA_Q1.mp4",
        total_frames=10,
        width=640,
        height=480,
        fps=30.0,
    )
    action_logger.log_click(frame_idx=1, x=10, y=20, positive=True, obj_id=1)

    old_log_path = initial_logs / f"{session_id}.jsonl"
    new_log_path = export_logs / f"{session_id}.jsonl"

    assert old_log_path.exists()

    action_logger.relocate_output_dir(export_logs)
    action_logger.log_export(
        format="coco",
        output_path=str(tmp_path / "output" / "VideoA_Q1"),
        num_frames=10,
        num_objects=1,
    )
    action_logger.end_session()

    assert not old_log_path.exists()
    assert new_log_path.exists()
    assert (export_logs / f"{session_id}_summary.json").exists()

    actions = [json.loads(line)["action_type"] for line in new_log_path.read_text().splitlines()]
    assert actions == ["session_start", "positive_click", "export", "session_end"]
