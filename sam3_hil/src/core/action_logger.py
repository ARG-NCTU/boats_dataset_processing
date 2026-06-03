#!/usr/bin/env python3
"""
STAMP Action Logger
====================

記錄使用者標註操作，用於計算效率指標（HIR、CPO、SPF）。
支援 MCAP 格式（Foxglove 相容）和 JSON Lines 格式。

Features:
- 記錄所有標註操作（點擊、修正、刪除、合併等）
- 自動計算效率指標
- 支援 session 重播分析
- 匯出為 MCAP 或 JSON Lines 格式

Author: Sonic (Assistive Robotics Lab, NYCU)
"""

import json
import time
import logging
import shutil
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Set, Any, Union
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Action Types
# =============================================================================

class ActionType(Enum):
    """標註操作類型"""

    # Session 操作
    SESSION_START = "session_start"
    SESSION_END = "session_end"

    # 影片操作
    VIDEO_LOADED = "video_loaded"
    FRAME_NAVIGATION = "frame_navigation"

    # 偵測操作
    DETECTION_STARTED = "detection_started"
    DETECTION_FINISHED = "detection_finished"

    # 互動修正操作（影響 CPO）
    POSITIVE_CLICK = "positive_click"
    NEGATIVE_CLICK = "negative_click"

    # 編輯操作（影響 HIR）
    APPLY_REFINE = "apply_refine"
    PROPAGATE = "propagate"
    ADD_OBJECT = "add_object"
    DELETE_OBJECT = "delete_object"
    IGNORE_OBJECT = "ignore_object"
    MERGE_OBJECTS = "merge_objects"
    SWAP_LABELS = "swap_labels"

    # 審核操作
    APPROVE_OBJECT = "approve_object"
    REJECT_OBJECT = "reject_object"

    # 匯出操作
    EXPORT = "export"


# 會影響 HIR 計算的編輯操作
EDIT_ACTIONS = {
    ActionType.APPLY_REFINE,
    ActionType.PROPAGATE,
    ActionType.ADD_OBJECT,
    ActionType.DELETE_OBJECT,
    ActionType.MERGE_OBJECTS,
    ActionType.SWAP_LABELS,
}

# 會影響 CPO 計算的點擊操作
CLICK_ACTIONS = {
    ActionType.POSITIVE_CLICK,
    ActionType.NEGATIVE_CLICK,
}


# =============================================================================
# Action Data Classes
# =============================================================================

@dataclass
class ActionRecord:
    """單一操作記錄"""

    action_type: str
    timestamp: float  # Unix timestamp (seconds)
    frame_idx: Optional[int] = None

    # 位置資訊（點擊操作）
    x: Optional[int] = None
    y: Optional[int] = None

    # 物件資訊
    obj_id: Optional[int] = None
    source_obj_id: Optional[int] = None
    target_obj_id: Optional[int] = None

    # 範圍資訊（propagate）
    start_frame: Optional[int] = None
    end_frame: Optional[int] = None

    # 額外資訊
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """轉換為字典，移除 None 值"""
        result = {}
        for key, value in asdict(self).items():
            if value is not None:
                if key == "metadata" and not value:
                    continue
                result[key] = value
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ActionRecord":
        """從字典建立"""
        return cls(**data)


@dataclass
class SessionInfo:
    """Session 資訊"""

    session_id: str
    video_path: Optional[str] = None
    total_frames: int = 0
    width: int = 0
    height: int = 0
    fps: float = 0.0
    prompt: str = ""
    start_time: float = field(default_factory=time.time)
    end_time: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class EfficiencyMetrics:
    """
    效率指標

    核心指標（反映工作量）：
    - TEO (Total Edit Operations): 總編輯操作次數 - 主要工作量指標
    - EOR (Edit Operation Rate): 編輯操作率 = TEO / total_frames - 每幀平均編輯次數

    輔助指標：
    - FCR (Frame Coverage Rate): 被編輯的幀比例（原 HIR）- 只是覆蓋率，不反映工作量
    - CPO (Clicks Per Object): 每個物件的平均點擊次數
    - SPF (Seconds Per Frame): 每幀平均花費時間
    """

    # 主要工作量指標
    total_edit_operations: int = 0   # TEO: 總編輯操作次數
    eor: float = 0.0                 # EOR: edit operation rate = TEO / total_frames

    # 輔助指標
    total_frames: int = 0
    edited_frame_count: int = 0      # 被編輯過的幀數（unique）
    fcr: float = 0.0                 # FCR: frame coverage rate = edited_frame_count / total_frames

    # Click metrics
    cpo: float = 0.0                 # CPO: clicks per object
    total_clicks: int = 0
    total_objects: int = 0

    # Time metrics
    spf: float = 0.0                 # SPF: seconds per frame
    total_seconds: float = 0.0

    # Detailed breakdown
    action_counts: Dict[str, int] = field(default_factory=dict)
    frame_edit_counts: Dict[int, int] = field(default_factory=dict)  # frame_idx -> edit count

    # Legacy aliases (for backward compatibility)
    @property
    def actual_hir(self) -> float:
        """Deprecated: use fcr instead"""
        return self.fcr

    @property
    def hir(self) -> float:
        """Deprecated: use fcr instead"""
        return self.fcr

    @property
    def edited_frames(self) -> int:
        """Deprecated: use edited_frame_count instead"""
        return self.edited_frame_count

    @property
    def epf(self) -> float:
        """Alias for eor (edits per frame = edit operation rate)"""
        return self.eor

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # 移除 frame_edit_counts 如果太大
        if len(d.get("frame_edit_counts", {})) > 100:
            d["frame_edit_counts"] = {"_truncated": True, "_count": len(d["frame_edit_counts"])}
        return d

    def __str__(self) -> str:
        # 找出編輯次數最多的幀
        hotspots = ""
        if self.frame_edit_counts:
            sorted_frames = sorted(self.frame_edit_counts.items(), key=lambda x: x[1], reverse=True)[:5]
            if sorted_frames:
                hotspots = "\nEdit Hotspots: " + ", ".join([f"F{f}({c}x)" for f, c in sorted_frames])

        return (
            f"=== Efficiency Metrics ===\n"
            f"TEO: {self.total_edit_operations} edit operations (primary workload metric)\n"
            f"EOR: {self.eor:.4f} edits/frame\n"
            f"FCR: {self.fcr:.1f}% ({self.edited_frame_count}/{self.total_frames} frames touched)\n"
            f"CPO: {self.cpo:.2f} clicks/object ({self.total_clicks} clicks, {self.total_objects} objects)\n"
            f"SPF: {self.spf:.2f} seconds/frame ({self.total_seconds:.1f}s total)"
            f"{hotspots}\n"
            f"Action Counts: {self.action_counts}"
        )


@dataclass
class Layer2Metrics:
    """Offline thesis metrics derived from JSONL action logs."""

    total_frames: int
    reviewed_frames: int
    rfr: float

    reviewed_units: int
    pass_through_count: int
    geometry_edit_count: int
    reject_count: int

    ptr: float
    ger: float
    reject_rate: float

    manual_add_count: int
    final_object_count: int
    mar: float

    outcome_counts: Dict[str, int] = field(default_factory=dict)
    reviewed_frame_indices: List[int] = field(default_factory=list)
    mar_fallback_used: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def __str__(self) -> str:
        fallback = " (fallback denominator)" if self.mar_fallback_used else ""
        return (
            "=== STAMP Layer 2 Metrics ===\n"
            f"RFR: {self.rfr:.4f} ({self.reviewed_frames}/{self.total_frames} frames)\n"
            f"PTR: {self.ptr:.4f} ({self.pass_through_count}/{self.reviewed_units} units)\n"
            f"GER: {self.ger:.4f} ({self.geometry_edit_count}/{self.reviewed_units} units)\n"
            f"Reject Rate: {self.reject_rate:.4f} ({self.reject_count}/{self.reviewed_units} units)\n"
            f"MAR: {self.mar:.4f} ({self.manual_add_count}/{self.final_object_count} objects){fallback}\n"
            f"Outcome Counts: {self.outcome_counts}"
        )


# =============================================================================
# Action Logger (JSON Lines)
# =============================================================================

class ActionLogger:
    """
    使用者操作記錄器

    支援兩種輸出格式：
    1. JSON Lines (.jsonl) - 簡單、易讀、易分析
    2. MCAP (.mcap) - Foxglove 相容，需要額外安裝 mcap 套件

    Usage:
        logger = ActionLogger(output_dir="./logs")
        logger.start_session(video_path="video.mp4", total_frames=500)

        # 記錄操作
        logger.log_click(frame_idx=10, x=100, y=200, positive=True, obj_id=1)
        logger.log_apply_refine(frame_idx=10, obj_id=1)
        logger.log_delete_object(frame_idx=10, obj_id=2)

        # 結束並計算指標
        metrics = logger.end_session()
        print(metrics)
    """

    def __init__(
        self,
        output_dir: Union[str, Path] = "./logs",
        format: str = "jsonl",  # "jsonl" or "mcap"
        auto_flush: bool = True,
    ):
        """
        初始化 ActionLogger

        Args:
            output_dir: 輸出目錄
            format: 輸出格式 ("jsonl" 或 "mcap")
            auto_flush: 是否自動 flush 每筆記錄
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.format = format
        self.auto_flush = auto_flush

        # Session 狀態
        self.session: Optional[SessionInfo] = None
        self.actions: List[ActionRecord] = []
        self._file = None
        self._mcap_writer = None

        # 用於指標計算的快取
        self._clicked_objects: Set[int] = set()
        self._frame_edit_counts: Dict[int, int] = {}  # frame_idx -> edit_count
        self._total_edit_operations: int = 0

        logger.info(f"ActionLogger initialized: dir={output_dir}, format={format}")

    # =========================================================================
    # Session Management
    # =========================================================================

    def start_session(
        self,
        video_path: Optional[str] = None,
        total_frames: int = 0,
        width: int = 0,
        height: int = 0,
        fps: float = 0.0,
        prompt: str = "",
    ) -> str:
        """
        開始新的標註 session

        Returns:
            session_id: 唯一的 session ID
        """
        # 如果有舊的 session，先結束
        if self.session is not None:
            self.end_session()

        # 產生 session ID
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(video_path).stem if video_path else "unknown"
        session_id = f"{timestamp}_{video_name}"

        # 建立 session info
        self.session = SessionInfo(
            session_id=session_id,
            video_path=video_path,
            total_frames=total_frames,
            width=width,
            height=height,
            fps=fps,
            prompt=prompt,
        )

        # 重設快取
        self.actions = []
        self._clicked_objects = set()
        self._frame_edit_counts = {}
        self._total_edit_operations = 0

        # 開啟輸出檔案
        self._open_output_file()

        # 記錄 session 開始
        self._log_action(ActionRecord(
            action_type=ActionType.SESSION_START.value,
            timestamp=time.time(),
            metadata=self.session.to_dict(),
        ))

        logger.info(f"Session started: {session_id}")
        return session_id

    def end_session(self) -> Optional[EfficiencyMetrics]:
        """
        結束當前 session 並計算效率指標

        Returns:
            EfficiencyMetrics: 效率指標，如果沒有 session 則回傳 None
        """
        if self.session is None:
            return None

        # 記錄結束時間
        if self.session.end_time is None:
            self.session.end_time = time.time()

        # 記錄 session 結束
        self._log_action(ActionRecord(
            action_type=ActionType.SESSION_END.value,
            timestamp=time.time(),
        ))

        # 計算效率指標
        metrics = self.calculate_metrics()

        # 寫入 metrics 摘要
        self._write_metrics_summary(metrics)

        # 關閉檔案
        self._close_output_file()

        logger.info(f"Session ended: {self.session.session_id}")
        logger.info(f"\n{metrics}")

        # 重設 session
        session_id = self.session.session_id
        self.session = None

        return metrics

    def mark_annotation_complete(self, completed_at: Optional[float] = None) -> Optional[float]:
        """Freeze annotation time before export I/O starts."""
        if self.session is None:
            return None

        self.session.end_time = completed_at if completed_at is not None else time.time()
        return self.session.end_time

    def clear_annotation_complete(self):
        """Resume timing when export is cancelled before it starts."""
        if self.session is not None:
            self.session.end_time = None

    def relocate_output_dir(self, output_dir: Union[str, Path]) -> Path:
        """
        Move the active JSONL session log to a new output directory.

        This keeps pre-export actions and post-export actions in one per-task
        log folder after the export dataset directory is known.
        """
        new_output_dir = Path(output_dir)
        new_output_dir.mkdir(parents=True, exist_ok=True)

        if self.output_dir.resolve() == new_output_dir.resolve():
            return self.output_dir

        if self.session is None:
            self.output_dir = new_output_dir
            return self.output_dir

        if self.format != "jsonl":
            logger.warning("Relocating action logs is only supported for jsonl format")
            return self.output_dir

        session_id = self.session.session_id
        was_open = self._file is not None
        self._close_output_file()

        for suffix in (".jsonl", "_summary.json"):
            src = self.output_dir / f"{session_id}{suffix}"
            dst = new_output_dir / f"{session_id}{suffix}"
            if src.exists() and src.resolve() != dst.resolve():
                if dst.exists():
                    dst.unlink()
                shutil.move(str(src), str(dst))

        self.output_dir = new_output_dir

        if was_open:
            filepath = self.output_dir / f"{session_id}.jsonl"
            self._file = open(filepath, "a", encoding="utf-8")
            logger.info(f"Relocated JSONL file: {filepath}")

        return self.output_dir

    # =========================================================================
    # Logging Methods
    # =========================================================================

    def log_video_loaded(
        self,
        video_path: str,
        total_frames: int,
        width: int,
        height: int,
        fps: float,
    ):
        """記錄影片載入"""
        if self.session:
            self.session.video_path = video_path
            self.session.total_frames = total_frames
            self.session.width = width
            self.session.height = height
            self.session.fps = fps

        self._log_action(ActionRecord(
            action_type=ActionType.VIDEO_LOADED.value,
            timestamp=time.time(),
            metadata={
                "video_path": video_path,
                "total_frames": total_frames,
                "width": width,
                "height": height,
                "fps": fps,
            }
        ))

    def log_frame_navigation(self, from_frame: int, to_frame: int):
        """記錄幀跳轉"""
        self._log_action(ActionRecord(
            action_type=ActionType.FRAME_NAVIGATION.value,
            timestamp=time.time(),
            frame_idx=to_frame,
            metadata={"from_frame": from_frame},
        ))

    def log_detection_started(self, prompt: str, frame_idx: int = 0):
        """記錄偵測開始"""
        if self.session:
            self.session.prompt = prompt

        self._log_action(ActionRecord(
            action_type=ActionType.DETECTION_STARTED.value,
            timestamp=time.time(),
            frame_idx=frame_idx,
            metadata={"prompt": prompt},
        ))

    def log_detection_finished(
        self,
        num_objects: int,
        duration_seconds: float,
        num_frames: int,
    ):
        """記錄偵測完成"""
        self._log_action(ActionRecord(
            action_type=ActionType.DETECTION_FINISHED.value,
            timestamp=time.time(),
            metadata={
                "num_objects": num_objects,
                "duration_seconds": duration_seconds,
                "num_frames": num_frames,
            }
        ))

    def log_click(
        self,
        frame_idx: int,
        x: int,
        y: int,
        positive: bool,
        obj_id: Optional[int] = None,
    ):
        """記錄點擊操作"""
        action_type = ActionType.POSITIVE_CLICK if positive else ActionType.NEGATIVE_CLICK

        # 記錄物件
        if obj_id is not None:
            self._clicked_objects.add(obj_id)

        self._log_action(ActionRecord(
            action_type=action_type.value,
            timestamp=time.time(),
            frame_idx=frame_idx,
            x=x,
            y=y,
            obj_id=obj_id,
        ))

    def _record_frame_edit(self, frame_idx: int):
        """記錄幀編輯（內部 helper）"""
        self._frame_edit_counts[frame_idx] = self._frame_edit_counts.get(frame_idx, 0) + 1

    def _record_edit_operation(self):
        """記錄一次編輯操作（內部 helper）"""
        self._total_edit_operations += 1

    def log_apply_refine(self, frame_idx: int, obj_id: int):
        """記錄套用修正"""
        self._record_frame_edit(frame_idx)
        self._record_edit_operation()

        self._log_action(ActionRecord(
            action_type=ActionType.APPLY_REFINE.value,
            timestamp=time.time(),
            frame_idx=frame_idx,
            obj_id=obj_id,
        ))

    def log_propagate(
        self,
        start_frame: int,
        end_frame: int,
        obj_id: int,
    ):
        """
        記錄傳播操作

        注意：HIR 只計算用戶「主動介入」的幀，propagate 只算起始幀。
        被傳播影響的幀範圍記錄在 metadata 中供分析用。
        """
        # 只記錄起始幀為「被編輯」（用戶主動介入的幀）
        self._record_frame_edit(start_frame)
        # 算 1 次編輯操作
        self._record_edit_operation()

        self._log_action(ActionRecord(
            action_type=ActionType.PROPAGATE.value,
            timestamp=time.time(),
            frame_idx=start_frame,
            obj_id=obj_id,
            start_frame=start_frame,
            end_frame=end_frame,
            metadata={
                "affected_frames": end_frame - start_frame + 1,  # 記錄影響了多少幀
            }
        ))

    def log_add_object(self, frame_idx: int, obj_id: int):
        """記錄新增物件"""
        self._record_frame_edit(frame_idx)
        self._record_edit_operation()
        self._clicked_objects.add(obj_id)

        self._log_action(ActionRecord(
            action_type=ActionType.ADD_OBJECT.value,
            timestamp=time.time(),
            frame_idx=frame_idx,
            obj_id=obj_id,
        ))

    def log_delete_object(
        self,
        frame_idx: int,
        obj_id: int,
        delete_type: str = "all",  # "all" or "from_current"
    ):
        """記錄刪除物件"""
        self._record_frame_edit(frame_idx)
        self._record_edit_operation()

        self._log_action(ActionRecord(
            action_type=ActionType.DELETE_OBJECT.value,
            timestamp=time.time(),
            frame_idx=frame_idx,
            obj_id=obj_id,
            metadata={"delete_type": delete_type},
        ))

    def log_ignore_object(
        self,
        frame_idx: int,
        obj_id: int,
        reason: str = "out_of_scope",
        scope: str = "all_frames",
        affected_frames: int = 0,
        affected_detections: int = 0,
    ):
        """Log a protocol-level object exclusion that is not an annotation edit."""
        self._log_action(ActionRecord(
            action_type=ActionType.IGNORE_OBJECT.value,
            timestamp=time.time(),
            frame_idx=frame_idx,
            obj_id=obj_id,
            metadata={
                "reason": reason,
                "scope": scope,
                "affected_frames": affected_frames,
                "affected_detections": affected_detections,
                "excluded_from_metrics": True,
            },
        ))

    def log_merge_objects(
        self,
        frame_idx: int,
        source_obj_id: int,
        target_obj_id: int,
    ):
        """記錄合併物件"""
        self._record_frame_edit(frame_idx)
        self._record_edit_operation()

        self._log_action(ActionRecord(
            action_type=ActionType.MERGE_OBJECTS.value,
            timestamp=time.time(),
            frame_idx=frame_idx,
            source_obj_id=source_obj_id,
            target_obj_id=target_obj_id,
        ))

    def log_swap_labels(
        self,
        frame_idx: int,
        obj_a: int,
        obj_b: int,
    ):
        """記錄交換標籤"""
        self._record_frame_edit(frame_idx)
        self._record_edit_operation()

        self._log_action(ActionRecord(
            action_type=ActionType.SWAP_LABELS.value,
            timestamp=time.time(),
            frame_idx=frame_idx,
            metadata={"obj_a": obj_a, "obj_b": obj_b},
        ))

    def log_approve_object(self, frame_idx: int, obj_id: int):
        """記錄核准物件"""
        self._log_action(ActionRecord(
            action_type=ActionType.APPROVE_OBJECT.value,
            timestamp=time.time(),
            frame_idx=frame_idx,
            obj_id=obj_id,
        ))

    def log_reject_object(self, frame_idx: int, obj_id: int):
        """記錄拒絕物件"""
        self._log_action(ActionRecord(
            action_type=ActionType.REJECT_OBJECT.value,
            timestamp=time.time(),
            frame_idx=frame_idx,
            obj_id=obj_id,
        ))

    def log_export(
        self,
        format: str,
        output_path: str,
        num_frames: int,
        num_objects: int,
    ):
        """記錄匯出操作"""
        self._log_action(ActionRecord(
            action_type=ActionType.EXPORT.value,
            timestamp=time.time(),
            metadata={
                "format": format,
                "output_path": output_path,
                "num_frames": num_frames,
                "num_objects": num_objects,
            }
        ))

    # =========================================================================
    # Metrics Calculation
    # =========================================================================

    def calculate_metrics(self) -> EfficiencyMetrics:
        """計算效率指標"""
        total_frames = self.session.total_frames if self.session else 0

        # 統計各操作次數
        action_counts: Dict[str, int] = {}
        total_clicks = 0

        for action in self.actions:
            action_type = action.action_type
            action_counts[action_type] = action_counts.get(action_type, 0) + 1

            if action_type in [ActionType.POSITIVE_CLICK.value, ActionType.NEGATIVE_CLICK.value]:
                total_clicks += 1

        # 計算 FCR (Frame Coverage Rate) - 輔助指標
        edited_frame_count = len(self._frame_edit_counts)
        fcr = (edited_frame_count / total_frames * 100) if total_frames > 0 else 0

        # 計算 EOR (Edit Operation Rate) - 主要工作量指標
        eor = (self._total_edit_operations / total_frames) if total_frames > 0 else 0

        # 計算 CPO
        total_objects = len(self._clicked_objects)
        cpo = (total_clicks / total_objects) if total_objects > 0 else 0

        # 計算 SPF
        if self.session and self.session.end_time:
            total_seconds = self.session.end_time - self.session.start_time
        else:
            total_seconds = 0
        spf = (total_seconds / total_frames) if total_frames > 0 else 0

        return EfficiencyMetrics(
            total_edit_operations=self._total_edit_operations,
            eor=eor,
            total_frames=total_frames,
            edited_frame_count=edited_frame_count,
            fcr=fcr,
            cpo=cpo,
            total_clicks=total_clicks,
            total_objects=total_objects,
            spf=spf,
            total_seconds=total_seconds,
            action_counts=action_counts,
            frame_edit_counts=self._frame_edit_counts.copy(),
        )

    # =========================================================================
    # File I/O
    # =========================================================================

    def _open_output_file(self):
        """開啟輸出檔案"""
        if self.session is None:
            return

        if self.format == "jsonl":
            filepath = self.output_dir / f"{self.session.session_id}.jsonl"
            self._file = open(filepath, "w", encoding="utf-8")
            logger.info(f"Opened JSONL file: {filepath}")

        elif self.format == "mcap":
            self._open_mcap_file()

    def _close_output_file(self):
        """關閉輸出檔案"""
        if self._file:
            self._file.close()
            self._file = None

        if self._mcap_writer:
            self._close_mcap_file()

    def _log_action(self, action: ActionRecord):
        """記錄操作到檔案"""
        self.actions.append(action)

        if self.format == "jsonl" and self._file:
            self._file.write(json.dumps(action.to_dict()) + "\n")
            if self.auto_flush:
                self._file.flush()

        elif self.format == "mcap" and self._mcap_writer:
            self._write_mcap_message(action)

    def _write_metrics_summary(self, metrics: EfficiencyMetrics):
        """寫入指標摘要"""
        if self.session is None:
            return

        # 寫入 JSON 摘要檔案
        summary_path = self.output_dir / f"{self.session.session_id}_summary.json"
        summary = {
            "session": self.session.to_dict(),
            "metrics": metrics.to_dict(),
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

        logger.info(f"Metrics summary saved: {summary_path}")

    # =========================================================================
    # MCAP Support
    # =========================================================================

    def _open_mcap_file(self):
        """開啟 MCAP 檔案"""
        try:
            from mcap.writer import Writer
        except ImportError:
            logger.warning("mcap not installed, falling back to jsonl format")
            logger.warning("Install with: pip install mcap")
            self.format = "jsonl"
            self._open_output_file()
            return

        filepath = self.output_dir / f"{self.session.session_id}.mcap"
        self._mcap_file = open(filepath, "wb")
        self._mcap_writer = Writer(self._mcap_file)
        self._mcap_writer.start()

        # 註冊 schema
        self._mcap_schema_id = self._mcap_writer.register_schema(
            name="stamp.UserAction",
            encoding="jsonschema",
            data=json.dumps({
                "type": "object",
                "properties": {
                    "action_type": {"type": "string"},
                    "timestamp": {"type": "number"},
                    "frame_idx": {"type": "integer"},
                    "x": {"type": "integer"},
                    "y": {"type": "integer"},
                    "obj_id": {"type": "integer"},
                    "metadata": {"type": "object"},
                }
            }).encode()
        )

        # 註冊 channel
        self._mcap_channel_id = self._mcap_writer.register_channel(
            schema_id=self._mcap_schema_id,
            topic="/stamp/user_actions",
            message_encoding="json",
        )

        logger.info(f"Opened MCAP file: {filepath}")

    def _write_mcap_message(self, action: ActionRecord):
        """寫入 MCAP 訊息"""
        if self._mcap_writer is None:
            return

        timestamp_ns = int(action.timestamp * 1e9)
        self._mcap_writer.add_message(
            channel_id=self._mcap_channel_id,
            log_time=timestamp_ns,
            publish_time=timestamp_ns,
            data=json.dumps(action.to_dict()).encode(),
        )

    def _close_mcap_file(self):
        """關閉 MCAP 檔案"""
        if self._mcap_writer:
            self._mcap_writer.finish()
            self._mcap_writer = None

        if hasattr(self, "_mcap_file") and self._mcap_file:
            self._mcap_file.close()
            self._mcap_file = None


# =============================================================================
# Session Analyzer
# =============================================================================

class SessionAnalyzer:
    """
    Session 分析器

    用於分析已記錄的 session，計算指標並產生報告。

    Usage:
        analyzer = SessionAnalyzer()
        metrics = analyzer.analyze_file("session.jsonl")
        print(metrics)
    """

    _LAYER2_REVIEW_ACTIONS = {
        ActionType.APPROVE_OBJECT.value,
        ActionType.REJECT_OBJECT.value,
        ActionType.APPLY_REFINE.value,
        ActionType.PROPAGATE.value,
        ActionType.ADD_OBJECT.value,
        ActionType.DELETE_OBJECT.value,
        ActionType.MERGE_OBJECTS.value,
        ActionType.SWAP_LABELS.value,
        ActionType.POSITIVE_CLICK.value,
        ActionType.NEGATIVE_CLICK.value,
    }
    _LAYER2_REJECT_ACTIONS = {
        ActionType.REJECT_OBJECT.value,
        ActionType.DELETE_OBJECT.value,
    }
    _LAYER2_REFINE_ACTIONS = {
        ActionType.APPLY_REFINE.value,
        ActionType.PROPAGATE.value,
        ActionType.MERGE_OBJECTS.value,
        ActionType.SWAP_LABELS.value,
    }

    @staticmethod
    def load_jsonl(filepath: Union[str, Path]) -> List[ActionRecord]:
        """載入 JSONL 檔案"""
        actions = []
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    actions.append(ActionRecord.from_dict(data))
        return actions

    @staticmethod
    def analyze_actions(
        actions: List[ActionRecord],
        total_frames: Optional[int] = None,
    ) -> EfficiencyMetrics:
        """分析操作記錄並計算指標"""

        # 從 session_start 取得 total_frames
        if total_frames is None:
            for action in actions:
                if action.action_type == ActionType.SESSION_START.value:
                    total_frames = action.metadata.get("total_frames", 0)
                    break

        if total_frames is None or total_frames == 0:
            # 嘗試從 video_loaded 取得
            for action in actions:
                if action.action_type == ActionType.VIDEO_LOADED.value:
                    total_frames = action.metadata.get("total_frames", 0)
                    break

        total_frames = total_frames or 0

        # 統計
        action_counts: Dict[str, int] = {}
        clicked_objects: Set[int] = set()
        frame_edit_counts: Dict[int, int] = {}  # frame_idx -> edit count
        total_edit_operations = 0
        total_clicks = 0

        session_start_time: Optional[float] = None
        session_end_time: Optional[float] = None

        for action in actions:
            action_type = action.action_type
            action_counts[action_type] = action_counts.get(action_type, 0) + 1

            # Session 時間
            if action_type == ActionType.SESSION_START.value:
                session_start_time = action.timestamp
            elif action_type == ActionType.SESSION_END.value:
                session_end_time = action.timestamp

            # 點擊統計
            if action_type in [ActionType.POSITIVE_CLICK.value, ActionType.NEGATIVE_CLICK.value]:
                total_clicks += 1
                if action.obj_id is not None:
                    clicked_objects.add(action.obj_id)

            # 編輯操作統計
            if action_type in [a.value for a in EDIT_ACTIONS]:
                total_edit_operations += 1

                # 只記錄用戶主動操作的幀（propagate 只算起始幀）
                if action.frame_idx is not None:
                    frame_edit_counts[action.frame_idx] = frame_edit_counts.get(action.frame_idx, 0) + 1

                # 注意：propagate 的 affected_frames 記錄在 metadata 中，不影響 FCR

            # Add object 也要記錄物件
            if action_type == ActionType.ADD_OBJECT.value and action.obj_id is not None:
                clicked_objects.add(action.obj_id)

        # 計算指標
        edited_frame_count = len(frame_edit_counts)
        fcr = (edited_frame_count / total_frames * 100) if total_frames > 0 else 0
        eor = (total_edit_operations / total_frames) if total_frames > 0 else 0

        total_objects = len(clicked_objects)
        cpo = (total_clicks / total_objects) if total_objects > 0 else 0

        if session_start_time and session_end_time:
            total_seconds = session_end_time - session_start_time
        else:
            total_seconds = 0
        spf = (total_seconds / total_frames) if total_frames > 0 else 0

        return EfficiencyMetrics(
            total_edit_operations=total_edit_operations,
            eor=eor,
            total_frames=total_frames,
            edited_frame_count=edited_frame_count,
            fcr=fcr,
            cpo=cpo,
            total_clicks=total_clicks,
            total_objects=total_objects,
            spf=spf,
            total_seconds=total_seconds,
            action_counts=action_counts,
            frame_edit_counts=frame_edit_counts,
        )

    @staticmethod
    def _infer_total_frames(actions: List[ActionRecord]) -> int:
        for action in actions:
            if action.action_type in {ActionType.SESSION_START.value, ActionType.VIDEO_LOADED.value}:
                total_frames = action.metadata.get("total_frames", 0)
                if total_frames:
                    return int(total_frames)
        return 0

    @staticmethod
    def _object_ids_for_action(action: ActionRecord) -> List[Any]:
        object_ids: List[Any] = []

        for value in (action.obj_id, action.source_obj_id, action.target_obj_id):
            if value is not None:
                object_ids.append(value)

        for key in ("obj_a", "obj_b", "source_obj_id", "target_obj_id"):
            value = action.metadata.get(key)
            if value is not None:
                object_ids.append(value)

        return list(dict.fromkeys(object_ids))

    @staticmethod
    def _layer2_unit_key(
        action: ActionRecord,
        obj_id: Any,
        unit_mode: str,
        object_generations: Optional[Dict[Any, int]] = None,
    ) -> Optional[Any]:
        if unit_mode == "object":
            return obj_id
        if unit_mode == "frame_object":
            if action.frame_idx is None:
                return None
            return (action.frame_idx, obj_id)
        if unit_mode == "instance":
            generation = 0
            if object_generations is not None:
                generation = object_generations.get(obj_id, 0)
            return (obj_id, generation)
        raise ValueError("unit_mode must be 'object', 'frame_object', or 'instance'")

    @classmethod
    def analyze_layer2_actions(
        cls,
        actions: List[ActionRecord],
        total_frames: Optional[int] = None,
        unit_mode: str = "object",
        final_object_count: Optional[int] = None,
    ) -> Layer2Metrics:
        """
        Analyze Layer 2 thesis metrics from JSONL action records.

        RFR is event-based because current frame navigation logs are sparse and
        cannot support reliable dwell-time analysis.
        """
        if unit_mode not in {"object", "frame_object", "instance"}:
            raise ValueError("unit_mode must be 'object', 'frame_object', or 'instance'")

        total_frames = total_frames if total_frames is not None else cls._infer_total_frames(actions)
        total_frames = total_frames or 0

        reviewed_frame_indices: Set[int] = set()
        unit_events: Dict[Any, Set[str]] = {}
        reviewed_non_manual_object_ids: Set[Any] = set()
        object_generations: Dict[Any, int] = {}
        manual_add_count = 0
        ignored_object_ids = {
            action.obj_id
            for action in actions
            if action.action_type == ActionType.IGNORE_OBJECT.value and action.obj_id is not None
        }

        for action in actions:
            action_type = action.action_type
            if action_type not in cls._LAYER2_REVIEW_ACTIONS:
                continue

            action_object_ids = cls._object_ids_for_action(action)
            if any(obj_id in ignored_object_ids for obj_id in action_object_ids):
                continue

            if action.frame_idx is not None:
                reviewed_frame_indices.add(action.frame_idx)

            if action_type == ActionType.ADD_OBJECT.value:
                manual_add_count += 1
                if unit_mode == "instance" and action.obj_id is not None:
                    object_generations[action.obj_id] = object_generations.get(action.obj_id, 0) + 1
                continue

            for obj_id in action_object_ids:
                key = cls._layer2_unit_key(action, obj_id, unit_mode, object_generations)
                if key is None:
                    continue
                unit_events.setdefault(key, set()).add(action_type)
                reviewed_non_manual_object_ids.add(obj_id)

        outcome_counts = {
            "accept": 0,
            "refine": 0,
            "reject": 0,
            "unresolved": 0,
        }

        for events in unit_events.values():
            if events & cls._LAYER2_REJECT_ACTIONS:
                outcome_counts["reject"] += 1
            elif events & cls._LAYER2_REFINE_ACTIONS:
                outcome_counts["refine"] += 1
            elif ActionType.APPROVE_OBJECT.value in events:
                outcome_counts["accept"] += 1
            else:
                outcome_counts["unresolved"] += 1

        reviewed_units = len(unit_events)
        pass_through_count = outcome_counts["accept"]
        geometry_edit_count = outcome_counts["refine"]
        reject_count = outcome_counts["reject"]

        ptr = (pass_through_count / reviewed_units) if reviewed_units else 0.0
        ger = (geometry_edit_count / reviewed_units) if reviewed_units else 0.0
        reject_rate = (reject_count / reviewed_units) if reviewed_units else 0.0

        if final_object_count is None:
            final_object_count_value = manual_add_count + len(reviewed_non_manual_object_ids)
            mar_fallback_used = True
        else:
            final_object_count_value = final_object_count
            mar_fallback_used = False
        mar = (manual_add_count / final_object_count_value) if final_object_count_value else 0.0

        reviewed_frames = len(reviewed_frame_indices)
        rfr = (reviewed_frames / total_frames) if total_frames else 0.0

        return Layer2Metrics(
            total_frames=total_frames,
            reviewed_frames=reviewed_frames,
            rfr=rfr,
            reviewed_units=reviewed_units,
            pass_through_count=pass_through_count,
            geometry_edit_count=geometry_edit_count,
            reject_count=reject_count,
            ptr=ptr,
            ger=ger,
            reject_rate=reject_rate,
            manual_add_count=manual_add_count,
            final_object_count=final_object_count_value,
            mar=mar,
            outcome_counts=outcome_counts,
            reviewed_frame_indices=sorted(reviewed_frame_indices),
            mar_fallback_used=mar_fallback_used,
        )

    def analyze_file(self, filepath: Union[str, Path]) -> EfficiencyMetrics:
        """分析單一檔案"""
        actions = self.load_jsonl(filepath)
        return self.analyze_actions(actions)

    def analyze_directory(self, dirpath: Union[str, Path]) -> Dict[str, EfficiencyMetrics]:
        """分析目錄下所有 session"""
        dirpath = Path(dirpath)
        results = {}

        for filepath in dirpath.glob("*.jsonl"):
            session_id = filepath.stem
            metrics = self.analyze_file(filepath)
            results[session_id] = metrics

        return results

    def generate_report(
        self,
        results: Dict[str, EfficiencyMetrics],
        output_path: Optional[Union[str, Path]] = None,
    ) -> str:
        """產生分析報告"""
        lines = [
            "=" * 60,
            "STAMP Session Analysis Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Total Sessions: {len(results)}",
            "=" * 60,
            "",
        ]

        # 個別 session
        for session_id, metrics in results.items():
            lines.extend([
                f"Session: {session_id}",
                "-" * 40,
                f"  TEO: {metrics.total_edit_operations} edit operations (workload)",
                f"  EOR: {metrics.eor:.4f} edits/frame",
                f"  FCR: {metrics.fcr:.1f}% ({metrics.edited_frame_count}/{metrics.total_frames} frames touched)",
                f"  CPO: {metrics.cpo:.2f} clicks/object",
                f"  SPF: {metrics.spf:.2f} seconds/frame",
                f"  Total Time: {metrics.total_seconds:.1f}s",
                "",
            ])

        # 總結
        if results:
            avg_teo = sum(m.total_edit_operations for m in results.values()) / len(results)
            avg_eor = sum(m.eor for m in results.values()) / len(results)
            avg_fcr = sum(m.fcr for m in results.values()) / len(results)
            avg_cpo = sum(m.cpo for m in results.values()) / len(results)
            avg_spf = sum(m.spf for m in results.values()) / len(results)

            lines.extend([
                "=" * 60,
                "Summary (Averages)",
                "=" * 60,
                f"Average TEO: {avg_teo:.1f} operations",
                f"Average EOR: {avg_eor:.4f} edits/frame",
                f"Average FCR: {avg_fcr:.1f}%",
                f"Average CPO: {avg_cpo:.2f} clicks/object",
                f"Average SPF: {avg_spf:.2f} seconds/frame",
            ])

        report = "\n".join(lines)

        if output_path:
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(report)

        return report


# =============================================================================
# Example Usage
# =============================================================================

if __name__ == "__main__":
    # 設定 logging
    logging.basicConfig(level=logging.INFO)

    # 建立 logger
    action_logger = ActionLogger(output_dir="./test_logs", format="jsonl")

    # 模擬標註 session
    action_logger.start_session(
        video_path="test_video.mp4",
        total_frames=500,
        width=1920,
        height=1080,
        fps=30.0,
        prompt="ship, boat",
    )

    # 模擬操作
    action_logger.log_detection_started(prompt="ship, boat", frame_idx=0)
    time.sleep(0.1)  # 模擬偵測時間
    action_logger.log_detection_finished(num_objects=5, duration_seconds=2.5, num_frames=500)

    # 模擬修正
    action_logger.log_click(frame_idx=10, x=100, y=200, positive=True, obj_id=1)
    action_logger.log_click(frame_idx=10, x=150, y=180, positive=False, obj_id=1)
    action_logger.log_apply_refine(frame_idx=10, obj_id=1)

    # 模擬傳播
    action_logger.log_propagate(start_frame=10, end_frame=50, obj_id=1)

    # 模擬物件管理
    action_logger.log_delete_object(frame_idx=100, obj_id=3, delete_type="all")
    action_logger.log_merge_objects(frame_idx=150, source_obj_id=4, target_obj_id=2)

    # 結束 session
    metrics = action_logger.end_session()
    print(metrics)

    # 分析已存在的 session
    print("\n--- Analyzing saved session ---\n")
    analyzer = SessionAnalyzer()
    results = analyzer.analyze_directory("./test_logs")
    report = analyzer.generate_report(results)
    print(report)
