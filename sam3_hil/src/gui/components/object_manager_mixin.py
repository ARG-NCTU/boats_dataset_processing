#!/usr/bin/env python3
"""
STAMP - Object Manager Mixin
==============================

Object management methods for HILAAMainWindow:
delete, merge, swap labels, context menu, jump to first frame.

Host requirements (attributes expected on self):
- object_list, sam3_results, object_status, current_frame
- video_loader, video_analysis, processing_mode_combo
- action_logger
- display_frame(), update_object_list(), update_analysis_display()
- update_timeline(), _reanalyze_with_preserved_edits()
- _track_human_intervention(), seek_to_frame(), statusBar()
"""

import logging
from typing import Dict, List, Tuple

from PyQt6.QtWidgets import QMenu, QMessageBox
from PyQt6.QtCore import QPoint

from gui.components.dialogs import ObjectSelectionDialog

logger = logging.getLogger(__name__)


class ObjectManagerMixin:
    """物件管理邏輯：刪除、合併、交換標籤。"""

    def show_object_context_menu(self, position: QPoint):
        """顯示物件右鍵選單。"""
        item = self.object_list.itemAt(position)
        if item is None:
            return
        
        widget = self.object_list.itemWidget(item)
        if widget is None:
            return
        
        obj_id = widget.property("obj_id")
        frame_idx = widget.property("frame_idx")  # 可能為 None（Video 模式）
        if obj_id is None:
            return
        
        # 建立右鍵選單
        menu = QMenu(self)
        
        # 判斷是否為 Independent Images 模式
        is_independent_mode = (self.processing_mode_combo.currentIndex() == 1)
        
        # 刪除物件
        if not is_independent_mode:
            delete_action = menu.addAction("Delete Object (All Frames)")
            delete_action.triggered.connect(lambda: self.delete_object(obj_id))
        
        # 只刪除當前幀（兩種模式都有）
        delete_this_action = menu.addAction("Delete This Detection Only")
        # Independent 模式用物件所在的幀，Video 模式用當前顯示的幀
        target_frame = frame_idx if frame_idx is not None else self.current_frame
        delete_this_action.triggered.connect(lambda: self.delete_object_single_frame(obj_id, target_frame))
        
        # Video 模式額外提供「從當前幀刪到最後」
        if not is_independent_mode:
            delete_from_action = menu.addAction("Delete From Current Frame Onwards")
            delete_from_action.triggered.connect(lambda: self.delete_object_from_frame(obj_id, self.current_frame))
        
        menu.addSeparator()
        
        # 合併物件
        merge_action = menu.addAction("Merge Into Another Object...")
        merge_action.triggered.connect(lambda: self.show_merge_dialog(obj_id))
        
        # 交換標籤
        swap_action = menu.addAction("Swap Label With...")
        swap_action.triggered.connect(lambda: self.show_swap_dialog(obj_id))
        
        menu.addSeparator()
        
        # 跳轉到物件首次出現的幀
        jump_action = menu.addAction("Jump to First Appearance")
        jump_action.triggered.connect(lambda: self.jump_to_object_first_frame(obj_id))
        
        # 顯示選單
        menu.exec(self.object_list.mapToGlobal(position))
    
    def delete_object(self, obj_id: int):
        """
        完全刪除物件（從所有幀中移除）。
        
        Args:
            obj_id: 要刪除的物件 ID
        """
        # 確認操作
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete Object {obj_id} from ALL frames?\n\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        edited_frame = self.current_frame  # 記錄操作發生的幀
        
        # 從所有幀中移除該物件
        deleted_count = 0
        for frame_idx, frame_result in self.sam3_results.items():
            original_count = len(frame_result.detections)
            frame_result.detections = [
                d for d in frame_result.detections if d.obj_id != obj_id
            ]
            deleted_count += original_count - len(frame_result.detections)
        
        # 移除物件狀態
        if obj_id in self.object_status:
            del self.object_status[obj_id]
        
        # === ActionLogger: 記錄刪除物件 ===
        self.action_logger.log_delete_object(
            frame_idx=edited_frame,
            obj_id=obj_id,
            delete_type="all"
        )
        
        logger.info(f"Deleted object {obj_id}: removed {deleted_count} detections")
        
        # 更新 UI（先 reanalyze，再 track）
        self._reanalyze_with_preserved_edits()
        self._track_human_intervention(edited_frame)
        self.update_object_list()
        self.update_analysis_display()
        self.update_timeline()
        self.display_frame(self.current_frame)
        
        self.statusBar().showMessage(f"Deleted Object {obj_id} ({deleted_count} detections removed)")
    
    def delete_object_from_frame(self, obj_id: int, from_frame: int):
        """
        從指定幀開始刪除物件（保留之前的幀）。
        
        Args:
            obj_id: 要刪除的物件 ID
            from_frame: 從此幀開始刪除
        """
        if self.video_loader is None:
            return
        
        total_frames = self.video_loader.metadata.total_frames
        
        # 確認操作
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete Object {obj_id} from frame {from_frame} to {total_frames - 1}?\n\n"
            f"Frames 0 to {from_frame - 1} will be preserved.\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # 從指定幀開始移除
        deleted_count = 0
        for frame_idx in range(from_frame, total_frames):
            if frame_idx in self.sam3_results:
                frame_result = self.sam3_results[frame_idx]
                original_count = len(frame_result.detections)
                frame_result.detections = [
                    d for d in frame_result.detections if d.obj_id != obj_id
                ]
                deleted_count += original_count - len(frame_result.detections)
        
        # === ActionLogger: 記錄刪除物件 ===
        self.action_logger.log_delete_object(
            frame_idx=from_frame,
            obj_id=obj_id,
            delete_type="from_current"
        )
        
        logger.info(f"Deleted object {obj_id} from frame {from_frame}: removed {deleted_count} detections")
        
        # 更新 UI
        self._reanalyze_with_preserved_edits()
        self._track_human_intervention(from_frame)
        self.update_object_list()
        self.update_analysis_display()
        self.update_timeline()
        self.display_frame(self.current_frame)
        
        self.statusBar().showMessage(
            f"Deleted Object {obj_id} from frame {from_frame} onwards ({deleted_count} detections removed)"
        )
    
    def delete_object_single_frame(self, obj_id: int, frame_idx: int):
        """
        只刪除指定幀中的該物件（不影響其他幀）。
        
        Args:
            obj_id: 要刪除的物件 ID
            frame_idx: 要刪除的幀索引
        """
        if self.video_loader is None:
            return
        
        # 確認操作
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete Object {obj_id} from frame {frame_idx} only?\n\n"
            "Other frames will not be affected.\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # 只刪除指定幀
        deleted_count = 0
        if frame_idx in self.sam3_results:
            frame_result = self.sam3_results[frame_idx]
            original_count = len(frame_result.detections)
            frame_result.detections = [
                d for d in frame_result.detections if d.obj_id != obj_id
            ]
            deleted_count = original_count - len(frame_result.detections)
        
        # === ActionLogger: 記錄刪除物件 ===
        self.action_logger.log_delete_object(
            frame_idx=frame_idx,
            obj_id=obj_id,
            delete_type="single_frame"
        )
        
        logger.info(f"Deleted object {obj_id} from frame {frame_idx} only: removed {deleted_count} detection(s)")
        
        # 更新 UI
        self._reanalyze_with_preserved_edits()
        self._track_human_intervention(frame_idx)
        self.update_object_list()
        self.update_analysis_display()
        self.update_timeline()
        self.display_frame(self.current_frame)
        
        self.statusBar().showMessage(
            f"Deleted Object {obj_id} from frame {frame_idx} ({deleted_count} detection removed)"
        )
    
    def show_merge_dialog(self, source_obj_id: int):
        """
        顯示合併物件對話框。
        
        Args:
            source_obj_id: 來源物件 ID（將被合併到目標物件）
        """
        objects = self._get_all_object_info()
        
        dialog = ObjectSelectionDialog(
            parent=self,
            title="Merge Object",
            message=f"Merge Object {source_obj_id} into:",
            objects=objects,
            exclude_obj_id=source_obj_id
        )
        
        from PyQt6.QtWidgets import QDialog
        if dialog.exec() == QDialog.DialogCode.Accepted:
            target_obj_id = dialog.get_selected_obj_id()
            if target_obj_id is not None:
                self.merge_objects(source_obj_id, target_obj_id)
    
    def merge_objects(self, source_obj_id: int, target_obj_id: int):
        """
        合併兩個物件：將 source 的所有 detection 改為 target 的 ID。
        
        如果同一幀中兩個物件都存在，保留 mask 面積較大的。
        
        Args:
            source_obj_id: 來源物件 ID（將被移除）
            target_obj_id: 目標物件 ID（將保留）
        """
        merge_count = 0
        conflict_count = 0
        
        for frame_idx, frame_result in self.sam3_results.items():
            # 找到 source 和 target
            source_det = None
            target_det = None
            
            for det in frame_result.detections:
                if det.obj_id == source_obj_id:
                    source_det = det
                elif det.obj_id == target_obj_id:
                    target_det = det
            
            if source_det is None:
                continue  # 這幀沒有 source
            
            if target_det is None:
                # 只有 source，直接改 ID
                source_det.obj_id = target_obj_id
                merge_count += 1
            else:
                # 兩者都存在 - 保留 mask 面積較大的
                source_area = source_det.mask.sum() if source_det.mask is not None else 0
                target_area = target_det.mask.sum() if target_det.mask is not None else 0
                
                if source_area > target_area:
                    # source 更大，用 source 取代 target
                    source_det.obj_id = target_obj_id
                    frame_result.detections = [
                        d for d in frame_result.detections if d.obj_id != target_obj_id or d is source_det
                    ]
                    # 注意：上面的 filter 可能有問題，改用更安全的方式
                    frame_result.detections = [
                        d for d in frame_result.detections if d is not target_det
                    ]
                else:
                    # target 更大，直接移除 source
                    frame_result.detections = [
                        d for d in frame_result.detections if d is not source_det
                    ]
                
                conflict_count += 1
                merge_count += 1
        
        # 移除 source 的物件狀態
        if source_obj_id in self.object_status:
            del self.object_status[source_obj_id]
        
        edited_frame = self.current_frame  # 記錄操作發生的幀
        
        # === ActionLogger: 記錄合併物件 ===
        self.action_logger.log_merge_objects(
            frame_idx=edited_frame,
            source_obj=source_obj_id,
            target_obj=target_obj_id
        )
        
        logger.info(
            f"Merged object {source_obj_id} → {target_obj_id}: "
            f"{merge_count} frames affected, {conflict_count} conflicts resolved"
        )
        
        # 更新 UI（先 reanalyze，再 track）
        self._reanalyze_with_preserved_edits()
        self._track_human_intervention(edited_frame)
        self.update_object_list()
        self.update_analysis_display()
        self.update_timeline()
        self.display_frame(self.current_frame)
        
        self.statusBar().showMessage(
            f"Merged Object {source_obj_id} → {target_obj_id} "
            f"({merge_count} frames, {conflict_count} conflicts)"
        )
    
    def show_swap_dialog(self, obj_id_a: int):
        """
        顯示交換標籤對話框。
        
        Args:
            obj_id_a: 第一個物件 ID
        """
        objects = self._get_all_object_info()
        
        dialog = ObjectSelectionDialog(
            parent=self,
            title="Swap Labels",
            message=f"Swap Object {obj_id_a}'s label with:",
            objects=objects,
            exclude_obj_id=obj_id_a
        )
        
        from PyQt6.QtWidgets import QDialog
        if dialog.exec() == QDialog.DialogCode.Accepted:
            obj_id_b = dialog.get_selected_obj_id()
            if obj_id_b is not None:
                self.swap_object_labels(obj_id_a, obj_id_b)
    
    def swap_object_labels(self, obj_id_a: int, obj_id_b: int):
        """
        交換兩個物件的 ID（標籤互換）。
        
        Args:
            obj_id_a: 第一個物件 ID
            obj_id_b: 第二個物件 ID
        """
        swap_count = 0
        temp_id = -9999  # 臨時 ID 避免衝突
        
        for frame_idx, frame_result in self.sam3_results.items():
            for det in frame_result.detections:
                if det.obj_id == obj_id_a:
                    det.obj_id = temp_id
                    swap_count += 1
            
            for det in frame_result.detections:
                if det.obj_id == obj_id_b:
                    det.obj_id = obj_id_a
            
            for det in frame_result.detections:
                if det.obj_id == temp_id:
                    det.obj_id = obj_id_b
        
        # 交換物件狀態
        status_a = self.object_status.get(obj_id_a, "pending")
        status_b = self.object_status.get(obj_id_b, "pending")
        self.object_status[obj_id_a] = status_b
        self.object_status[obj_id_b] = status_a
        
        edited_frame = self.current_frame  # 記錄操作發生的幀
        
        # === ActionLogger: 記錄交換標籤 ===
        self.action_logger.log_swap_labels(
            frame_idx=edited_frame,
            obj_a=obj_id_a,
            obj_b=obj_id_b
        )
        
        logger.info(f"Swapped labels: Object {obj_id_a} ↔ Object {obj_id_b} ({swap_count} detections affected)")
        
        # 更新 UI（先 reanalyze，再 track）
        self._reanalyze_with_preserved_edits()
        self._track_human_intervention(edited_frame)
        self.update_object_list()
        self.update_analysis_display()
        self.update_timeline()
        self.display_frame(self.current_frame)
        
        self.statusBar().showMessage(
            f"Swapped Object {obj_id_a} ↔ Object {obj_id_b}"
        )
    
    def jump_to_object_first_frame(self, obj_id: int):
        """跳轉到物件首次出現的幀。"""
        first_frame = None
        
        for frame_idx in sorted(self.sam3_results.keys()):
            frame_result = self.sam3_results[frame_idx]
            for det in frame_result.detections:
                if det.obj_id == obj_id:
                    first_frame = frame_idx
                    break
            if first_frame is not None:
                break
        
        if first_frame is not None:
            self.seek_to_frame(first_frame)
            self.statusBar().showMessage(f"Object {obj_id} first appears at frame {first_frame}")
        else:
            QMessageBox.information(self, "Info", f"Object {obj_id} not found in any frame")
    
    def _get_all_object_info(self) -> List[Tuple[int, float]]:
        """
        取得所有物件的資訊列表。
        
        Returns:
            List of (obj_id, avg_score) tuples
        """
        objects = []
        
        if self.video_analysis and self.video_analysis.object_summaries:
            for obj_id, summary in self.video_analysis.object_summaries.items():
                objects.append((obj_id, summary.avg_score))
        else:
            # Fallback: 從結果中收集
            obj_scores: Dict[int, List[float]] = {}
            for frame_result in self.sam3_results.values():
                for det in frame_result.detections:
                    if det.obj_id not in obj_scores:
                        obj_scores[det.obj_id] = []
                    obj_scores[det.obj_id].append(det.score)
            
            for obj_id, scores in obj_scores.items():
                avg_score = sum(scores) / len(scores) if scores else 0
                objects.append((obj_id, avg_score))
        
        return sorted(objects, key=lambda x: x[1], reverse=True)
