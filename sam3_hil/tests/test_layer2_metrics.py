"""
Tests for STAMP Layer 2 post-processing metrics.
"""

import pytest

from src.core.action_logger import ActionRecord, ActionType, SessionAnalyzer


def action(action_type, frame_idx=None, obj_id=None, **kwargs):
    return ActionRecord(
        action_type=action_type.value if isinstance(action_type, ActionType) else action_type,
        timestamp=1.0,
        frame_idx=frame_idx,
        obj_id=obj_id,
        **kwargs,
    )


def test_accept_only_increases_ptr():
    metrics = SessionAnalyzer.analyze_layer2_actions(
        [action(ActionType.APPROVE_OBJECT, frame_idx=0, obj_id=1)],
        total_frames=10,
    )

    assert metrics.reviewed_units == 1
    assert metrics.pass_through_count == 1
    assert metrics.ptr == pytest.approx(1.0)
    assert metrics.ger == pytest.approx(0.0)
    assert metrics.reject_rate == pytest.approx(0.0)


def test_apply_refine_increases_ger():
    metrics = SessionAnalyzer.analyze_layer2_actions(
        [action(ActionType.APPLY_REFINE, frame_idx=2, obj_id=7)],
        total_frames=10,
    )

    assert metrics.reviewed_units == 1
    assert metrics.geometry_edit_count == 1
    assert metrics.ger == pytest.approx(1.0)


def test_reject_and_delete_take_priority_over_accept_and_refine():
    metrics = SessionAnalyzer.analyze_layer2_actions(
        [
            action(ActionType.APPROVE_OBJECT, frame_idx=3, obj_id=4),
            action(ActionType.APPLY_REFINE, frame_idx=3, obj_id=4),
            action(ActionType.DELETE_OBJECT, frame_idx=3, obj_id=4),
        ],
        total_frames=10,
    )

    assert metrics.reviewed_units == 1
    assert metrics.reject_count == 1
    assert metrics.reject_rate == pytest.approx(1.0)
    assert metrics.pass_through_count == 0
    assert metrics.geometry_edit_count == 0


def test_add_object_increases_mar():
    metrics = SessionAnalyzer.analyze_layer2_actions(
        [action(ActionType.ADD_OBJECT, frame_idx=5, obj_id=9)],
        total_frames=10,
        final_object_count=4,
    )

    assert metrics.manual_add_count == 1
    assert metrics.final_object_count == 4
    assert metrics.mar == pytest.approx(0.25)
    assert metrics.mar_fallback_used is False
    assert metrics.reviewed_units == 0


def test_same_frame_multiple_review_events_count_once_for_rfr():
    metrics = SessionAnalyzer.analyze_layer2_actions(
        [
            action(ActionType.POSITIVE_CLICK, frame_idx=6, obj_id=2),
            action(ActionType.NEGATIVE_CLICK, frame_idx=6, obj_id=2),
            action(ActionType.APPLY_REFINE, frame_idx=6, obj_id=2),
        ],
        total_frames=12,
    )

    assert metrics.reviewed_frames == 1
    assert metrics.reviewed_frame_indices == [6]
    assert metrics.rfr == pytest.approx(1 / 12)


def test_unit_mode_object_and_frame_object_use_different_denominators():
    actions = [
        action(ActionType.APPROVE_OBJECT, frame_idx=0, obj_id=1),
        action(ActionType.APPROVE_OBJECT, frame_idx=1, obj_id=1),
    ]

    object_metrics = SessionAnalyzer.analyze_layer2_actions(
        actions,
        total_frames=10,
        unit_mode="object",
    )
    frame_object_metrics = SessionAnalyzer.analyze_layer2_actions(
        actions,
        total_frames=10,
        unit_mode="frame_object",
    )

    assert object_metrics.reviewed_units == 1
    assert frame_object_metrics.reviewed_units == 2


def test_invalid_unit_mode_is_rejected():
    with pytest.raises(ValueError, match="unit_mode"):
        SessionAnalyzer.analyze_layer2_actions(
            [action(ActionType.APPROVE_OBJECT, frame_idx=0, obj_id=1)],
            unit_mode="frame",
        )
