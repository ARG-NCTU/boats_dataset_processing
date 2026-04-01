#!/usr/bin/env python3
"""
STAMP - Refinement Mixin
==========================

Interactive mask refinement and object addition methods for HILAAMainWindow.

Host requirements (attributes expected on self):
- video_loader, sam3_results, video_analysis, current_frame
- refinement_active, refinement_obj_id, add_object_mode
- sam3_engine, object_status, action_logger
- video_canvas, refinement_panel, object_list, refine_btn
- processing_mode_combo
- display_frame(), stop_play(), _reanalyze_with_preserved_edits()
- update_object_list(), update_analysis_display(), update_timeline()
- statusBar()
"""

import logging
from typing import Optional

import numpy as np

from PyQt6.QtWidgets import QMessageBox, QProgressDialog
from PyQt6.QtCore import Qt

from core.sam3_engine import SAM3Engine, FrameResult, Detection

logger = logging.getLogger(__name__)


class RefinementMixin:
    """Interactive refinement 和新增物件邏輯。"""

    def on_object_selection_changed(self):
        """當物件選擇改變時，更新 Refine 按鈕狀態。"""
        selected_items = self.object_list.selectedItems()
        self.refine_btn.setEnabled(len(selected_items) > 0 and not self.refinement_active)
    

    def start_refinement_for_selected(self):
        """開始對選中的物件進行 refinement。"""
        selected_items = self.object_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select an object to refine")
            return
        
        # 取得選中物件的 ID
        item = selected_items[0]
        item_widget = self.object_list.itemWidget(item)
        if not item_widget:
            return
        
        # 從 widget 中取得 obj_id（存在 property 中）
        obj_id = item_widget.property("obj_id")
        if obj_id is None:
            return
        
        # 記住當前幀位置
        target_frame = self.current_frame
        
        # 取得該物件在當前幀的 mask
        frame_result = self.sam3_results.get(target_frame)
        if not frame_result:
            QMessageBox.warning(self, "Warning", "No detection result for current frame")
            return
        
        # 找到對應的 detection
        target_det = None
        for det in frame_result.detections:
            if det.obj_id == obj_id:
                target_det = det
                break
        
        if target_det is None:
            QMessageBox.warning(self, "Warning", f"Object {obj_id} not found in current frame")
            return
        
        # 停止播放（先停止，避免播放時改變 current_frame）
        self.stop_play()
        
        # 進入 refinement 模式
        self.refinement_active = True
        self.refinement_obj_id = obj_id
        
        # 設置 canvas 為 refinement 模式
        self.video_canvas.enter_refinement_mode(
            obj_id=obj_id,
            frame_idx=target_frame,
            mask=target_det.mask
        )
        
        # 顯示控制面板
        score = target_det.score
        self.refinement_panel.enter_refinement(obj_id, score)
        # 檢查是否為 Independent Images 模式
        is_independent_mode = (self.processing_mode_combo.currentIndex() == 1)
        self.refinement_panel.set_propagate_visible(not is_independent_mode)
        
        # 禁用其他控制
        self._set_controls_enabled(False)
        
        # 重新顯示當前幀（確保 canvas 顯示正確的幀）
        self.display_frame(target_frame)
        
        self.statusBar().showMessage(f"Refinement Mode: Object {obj_id} at Frame {target_frame} - Left click to include, Right click to exclude")
        logger.info(f"Started refinement for object {obj_id} at frame {target_frame}")
    

    def start_add_object(self):
        """開始手動新增物件模式。"""
        if self.video_loader is None:
            QMessageBox.warning(self, "Warning", "Please open a video first")
            return
        
        # 記住當前幀位置
        target_frame = self.current_frame
        
        # 取得當前幀圖像大小
        frame = self.video_loader.get_frame(target_frame)
        if frame is None:
            QMessageBox.warning(self, "Warning", "Cannot get current frame")
            return
        
        h, w = frame.shape[:2]
        
        # 停止播放（先停止，避免播放時改變 current_frame）
        self.stop_play()
        
        # 進入 add object 模式
        self.refinement_active = True
        self.add_object_mode = True
        self.refinement_obj_id = None
        
        # 設置 canvas 為 add object 模式
        self.video_canvas.enter_add_object_mode(
            frame_idx=target_frame,
            image_shape=(h, w)
        )
        
        # 顯示控制面板（add object 模式）
        self.refinement_panel.enter_add_object()
        # 檢查是否為 Independent Images 模式
        is_independent_mode = (self.processing_mode_combo.currentIndex() == 1)
        self.refinement_panel.set_propagate_visible(not is_independent_mode)
        
        # 禁用其他控制
        self._set_controls_enabled(False)
        
        # 重新顯示當前幀（關鍵！確保 canvas 顯示正確的幀）
        self.display_frame(target_frame)
        
        self.statusBar().showMessage(f"Add Object Mode (Frame {target_frame}): Left click to include, Right click to exclude")
        logger.info(f"Started add object mode at frame {target_frame}")
    

    def on_refinement_point_added(self, x: int, y: int, is_positive: bool):
        """處理 refinement 點擊。"""
        if not self.refinement_active:
            return
        
        # === ActionLogger: 記錄點擊 ===
        self.action_logger.log_click(
            frame_idx=self.current_frame,
            x=x,
            y=y,
            positive=is_positive,
            obj_id=self.refinement_obj_id  # 可能是 None（add object mode）
        )
        
        # 更新點數顯示
        if self.video_canvas.refinement_state:
            point_count = len(self.video_canvas.refinement_state.points)
            self.refinement_panel.set_point_count(point_count)
        
        # 執行 SAM3 refinement
        self._run_refinement()
    

    def _run_refinement(self):
        """執行 SAM3 refinement。"""
        if not self.video_canvas.refinement_state:
            return
        
        state = self.video_canvas.refinement_state
        
        # 取得當前幀圖像
        frame = self.video_loader.get_frame(self.current_frame)
        if frame is None:
            return
        
        # 取得 points 和 labels
        points, labels = state.get_sam_inputs()
        
        if len(points) == 0:
            # 沒有點，顯示原始 mask
            self.video_canvas.update_refined_mask(state.original_mask)
            return
        
        # 初始化 SAM3 engine（如果還沒有）
        if self.sam3_engine is None:
            try:
                self.sam3_engine = SAM3Engine(mode="auto")
            except Exception as e:
                logger.error(f"Failed to initialize SAM3 engine: {e}")
                # 使用 mock refinement
                self._run_mock_refinement(points, labels, state.original_mask)
                return
        
        # 執行 refinement（不傳 mask_input，讓 SAM3 純粹根據 point prompts 預測）
        # 注意：SAM3 的 mask_input 需要是 logits 格式，而我們只有 binary mask
        try:
            new_mask = self.sam3_engine.refine_mask(
                image=frame,
                points=points,
                labels=labels,
                mask_input=None  # 純粹使用 point prompts
            )
            
            # 更新顯示
            self.video_canvas.update_refined_mask(new_mask)
            
        except Exception as e:
            logger.error(f"Refinement error: {e}")
            # Fallback to mock
            self._run_mock_refinement(points, labels, state.original_mask)
    

    def _run_mock_refinement(self, points: np.ndarray, labels: np.ndarray, original_mask: np.ndarray):
        """Mock refinement for testing."""
        h, w = original_mask.shape
        result_mask = original_mask.astype(np.float32).copy()
        
        for point, label in zip(points, labels):
            x, y = int(point[0]), int(point[1])
            radius = 30
            
            yy, xx = np.ogrid[:h, :w]
            circle = ((xx - x) ** 2 + (yy - y) ** 2) <= radius ** 2
            
            if label == 1:
                result_mask = np.maximum(result_mask, circle.astype(np.float32))
            else:
                result_mask[circle] = 0
        
        self.video_canvas.update_refined_mask(result_mask > 0.5)
    

    def on_refinement_clear(self):
        """清除所有 refinement 點。"""
        self.video_canvas.clear_points()
        self.refinement_panel.set_point_count(0)
        self.display_frame(self.current_frame)  # 重新顯示原始 mask
    

    def on_refinement_undo(self):
        """撤銷上一個 refinement 點。"""
        self.video_canvas.undo_last_point()
        
        if self.video_canvas.refinement_state:
            point_count = len(self.video_canvas.refinement_state.points)
            self.refinement_panel.set_point_count(point_count)
        
        # 重新計算 mask
        self._run_refinement()
    

    def on_refinement_apply(self):
        """套用 refinement 結果或新增物件。"""
        if not self.refinement_active or not self.video_canvas.refinement_state:
            return
        
        state = self.video_canvas.refinement_state
        new_mask = state.current_mask
        
        # 檢查 mask 是否有效
        if new_mask is None or not np.any(new_mask):
            QMessageBox.warning(self, "Warning", "No valid mask to apply. Please add points first.")
            return
        
        edited_frame = self.current_frame
        
        if self.add_object_mode:
            # === Add New Object Mode ===
            new_obj_id = self._add_new_object(new_mask)
            
            # === ActionLogger: 記錄新增物件 ===
            if new_obj_id is not None:
                self.action_logger.log_add_object(
                    frame_idx=edited_frame,
                    obj_id=new_obj_id
                )
        else:
            # === Refinement Mode ===
            obj_id = state.object_id
            
            # 更新 sam3_results 中的 mask
            frame_result = self.sam3_results.get(self.current_frame)
            if frame_result:
                for det in frame_result.detections:
                    if det.obj_id == obj_id:
                        det.mask = new_mask.astype(np.uint8)
                        logger.info(f"Applied refined mask for object {obj_id}")
                        break
            
            # === ActionLogger: 記錄套用修正 ===
            self.action_logger.log_apply_refine(
                frame_idx=edited_frame,
                obj_id=obj_id
            )
            
            self.statusBar().showMessage(f"Refinement applied for object {obj_id}")
        
        # ====== 關鍵修復：正確的順序 ======
        # 1. 重新分析（保留已編輯的幀）
        self._reanalyze_with_preserved_edits()
        
        # 2. 追蹤人類介入（在 reanalyze 之後！）
        self._track_human_intervention(edited_frame)
        
        # 3. 更新 UI
        self.update_object_list()
        self.update_analysis_display()
        
        # 退出 refinement 模式
        self._exit_refinement_mode()
        
        # 重新顯示更新後的幀
        self.display_frame(self.current_frame)
    

    def _track_human_intervention(self, frame_idx: int):
        """
        追蹤人類介入的幀（用於計算實際 HIR）。
        
        注意：這個函數只負責記錄，不處理 UI 更新。
        UI 更新由調用者負責。
        """
        if self.video_analysis:
            self.video_analysis.frames_actually_edited.add(frame_idx)
            logger.info(f"Human intervention tracked at frame {frame_idx}, "
                       f"total edited: {len(self.video_analysis.frames_actually_edited)}")
        else:
            logger.warning(f"Cannot track intervention at frame {frame_idx}: video_analysis is None")
    

    def _add_new_object(self, mask: np.ndarray) -> Optional[int]:
        """
        新增一個新的物件到結果中。
        
        Returns:
            new_obj_id: 新建立的物件 ID，如果失敗則返回 None
        """
        # 計算新的 obj_id（找到最大的現有 ID + 1）
        max_obj_id = -1
        for frame_result in self.sam3_results.values():
            for det in frame_result.detections:
                max_obj_id = max(max_obj_id, det.obj_id)
        new_obj_id = max_obj_id + 1
        
        # 計算 bounding box
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            QMessageBox.warning(self, "Warning", "Empty mask, cannot add object.")
            return None
        
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        box = np.array([x_min, y_min, x_max - x_min, y_max - y_min])
        
        # 建立新的 Detection
        new_detection = Detection(
            obj_id=new_obj_id,
            mask=mask.astype(np.uint8),
            box=box,
            score=1.0  # 手動新增的給滿分
        )
        
        # 加入當前幀的結果
        if self.current_frame not in self.sam3_results:
            self.sam3_results[self.current_frame] = FrameResult(
                frame_index=self.current_frame,
                detections=[new_detection]
            )
        else:
            self.sam3_results[self.current_frame].detections.append(new_detection)
        
        # 設定新物件狀態為 accepted
        self.object_status[new_obj_id] = "accepted"
        
        logger.info(f"Added new object {new_obj_id} at frame {self.current_frame}")
        self.statusBar().showMessage(f"Added new object {new_obj_id}")
        
        return new_obj_id
        
        # 注意：不在這裡調用 analyze_video 和 UI 更新
        # 由調用者 on_refinement_apply 負責（避免順序問題）
    

    def on_refinement_propagate(self):
        """套用修改並傳播到後續所有幀。"""
        if not self.refinement_active or not self.video_canvas.refinement_state:
            return
        
        state = self.video_canvas.refinement_state
        new_mask = state.current_mask
        points, labels = state.get_sam_inputs()
        
        # 檢查是否有有效的 mask 和 points
        if new_mask is None or not np.any(new_mask):
            QMessageBox.warning(self, "Warning", "No valid mask to propagate. Please add points first.")
            return
        
        if len(points) == 0:
            QMessageBox.warning(self, "Warning", "No points added. Please click to define the object.")
            return
        
        # 確認操作
        total_frames = self.video_loader.metadata.total_frames
        remaining_frames = total_frames - self.current_frame - 1
        
        if remaining_frames <= 0:
            QMessageBox.information(self, "Info", "This is the last frame. Use 'Apply' instead.")
            return
        
        reply = QMessageBox.question(
            self, "Confirm Propagation",
            f"This will track the object from frame {self.current_frame} to frame {total_frames - 1} "
            f"({remaining_frames} frames).\n\n"
            f"This may take a while. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # 執行傳播
        self._propagate_to_following_frames(new_mask, points, labels)
    

    def _propagate_to_following_frames(self, mask: np.ndarray, points: np.ndarray, labels: np.ndarray):
        """使用 SAM3 Video Predictor 傳播到後續幀。"""
        state = self.video_canvas.refinement_state
        start_frame = self.current_frame
        
        # 確定 obj_id
        if self.add_object_mode:
            # 新增物件：分配新 ID
            max_obj_id = -1
            for frame_result in self.sam3_results.values():
                for det in frame_result.detections:
                    max_obj_id = max(max_obj_id, det.obj_id)
            obj_id = max_obj_id + 1
        else:
            obj_id = state.object_id
        
        # 顯示進度對話框
        total_frames = self.video_loader.metadata.total_frames
        remaining = total_frames - start_frame
        
        progress = QProgressDialog(
            f"Propagating object {obj_id} to following frames...",
            "Cancel", 0, remaining, self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        try:
            # 嘗試使用 SAM3 Video Predictor
            if self.sam3_engine is None:
                self.sam3_engine = SAM3Engine(mode="auto")
            
            # 檢查是否支援 video propagation
            if hasattr(self.sam3_engine, 'propagate_mask'):
                # 使用 SAM3 video predictor
                results = self.sam3_engine.propagate_mask(
                    video_path=str(self.video_loader.video_path),
                    start_frame=start_frame,
                    mask=mask,
                    points=points,
                    labels=labels,
                    obj_id=obj_id,
                    progress_callback=lambda i, n: progress.setValue(i) if not progress.wasCanceled() else None
                )
                
                if progress.wasCanceled():
                    self.statusBar().showMessage("Propagation cancelled")
                    return
                
                # 更新 sam3_results
                for frame_idx, frame_mask in results.items():
                    self._update_or_add_detection(frame_idx, obj_id, frame_mask)
            else:
                # Fallback: 簡易傳播（直接複製 mask）
                logger.warning("SAM3 video propagation not available, using simple copy")
                self._simple_propagate(obj_id, mask, start_frame, progress)
            
            progress.close()
            
            # 更新 object status
            if self.add_object_mode:
                self.object_status[obj_id] = "accepted"
            
            # === ActionLogger: 記錄傳播操作 ===
            end_frame = total_frames - 1
            
            self.action_logger.log_propagate(
                start_frame=start_frame,
                end_frame=end_frame,
                obj_id=obj_id
            )
            
            # ====== 關鍵修復：正確的順序 ======
            # 1. 重新分析（保留已編輯的幀）
            self._reanalyze_with_preserved_edits()
            
            # 2. 追蹤人類介入（在 reanalyze 之後！）
            self._track_human_intervention(start_frame)
            
            # 3. 更新 UI
            self.update_object_list()
            self.update_analysis_display()
            self.update_timeline()
            
            # 退出 refinement 模式
            self._exit_refinement_mode()
            self.display_frame(self.current_frame)
            
            self.statusBar().showMessage(
                f"Propagated object {obj_id} from frame {start_frame} to {total_frames - 1}"
            )
            
        except Exception as e:
            progress.close()
            logger.error(f"Propagation error: {e}")
            QMessageBox.warning(
                self, "Propagation Error",
                f"Failed to propagate: {e}\n\nFalling back to simple copy."
            )
            # Fallback
            self._simple_propagate(obj_id, mask, start_frame, None)
            
            # 更新 object status
            if self.add_object_mode:
                self.object_status[obj_id] = "accepted"
            
            # 同樣需要正確順序
            self._reanalyze_with_preserved_edits()
            self._track_human_intervention(start_frame)
            self.update_object_list()
            self._exit_refinement_mode()
            self.display_frame(self.current_frame)
    

    def _simple_propagate(self, obj_id: int, mask: np.ndarray, start_frame: int, progress: Optional[QProgressDialog]):
        """簡易傳播：將 mask 複製到後續所有幀（不追蹤）。"""
        total_frames = self.video_loader.metadata.total_frames
        
        for i, frame_idx in enumerate(range(start_frame, total_frames)):
            if progress and progress.wasCanceled():
                break
            if progress:
                progress.setValue(i)
            
            self._update_or_add_detection(frame_idx, obj_id, mask)
        
        # 注意：不在這裡調用 analyze_video 和 track_human_intervention
        # 由調用者 _propagate_to_following_frames 負責（避免重複調用和順序問題）
    

    def _update_or_add_detection(self, frame_idx: int, obj_id: int, mask: np.ndarray):
        """更新或新增特定幀的 detection。"""
        # 計算 bounding box
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return  # Empty mask, skip
        
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        box = np.array([x_min, y_min, x_max - x_min, y_max - y_min])
        
        new_detection = Detection(
            obj_id=obj_id,
            mask=mask.astype(np.uint8),
            box=box,
            score=1.0
        )
        
        if frame_idx not in self.sam3_results:
            self.sam3_results[frame_idx] = FrameResult(
                frame_index=frame_idx,
                detections=[new_detection]
            )
        else:
            # 檢查是否已存在該 obj_id
            found = False
            for i, det in enumerate(self.sam3_results[frame_idx].detections):
                if det.obj_id == obj_id:
                    self.sam3_results[frame_idx].detections[i] = new_detection
                    found = True
                    break
            if not found:
                self.sam3_results[frame_idx].detections.append(new_detection)
    

    def on_refinement_cancel(self):
        """取消 refinement。"""
        self._exit_refinement_mode()
        self.display_frame(self.current_frame)
        self.statusBar().showMessage("Refinement cancelled")
    

    def _exit_refinement_mode(self):
        """退出 refinement 或 add object 模式。"""
        self.refinement_active = False
        self.refinement_obj_id = None
        self.add_object_mode = False
        
        self.video_canvas.exit_refinement_mode()
        self.refinement_panel.exit_refinement()
        
        # 重新啟用控制
        self._set_controls_enabled(True)
    

    def _set_controls_enabled(self, enabled: bool):
        """啟用/禁用控制按鈕。"""
        self.prev_btn.setEnabled(enabled and self.video_loader is not None)
        self.next_btn.setEnabled(enabled and self.video_loader is not None)
        self.play_btn.setEnabled(enabled and self.video_loader is not None)
        self.timeline_slider.setEnabled(enabled and self.video_loader is not None)
        self.detect_btn.setEnabled(enabled and self.video_loader is not None)
        self.accept_all_btn.setEnabled(enabled and len(self.sam3_results) > 0)
        self.reset_all_btn.setEnabled(enabled and len(self.sam3_results) > 0)
        self.refine_btn.setEnabled(enabled and len(self.object_list.selectedItems()) > 0)
        # Add Object 按鈕在有影片時就可以用
        self.add_object_btn.setEnabled(enabled and self.video_loader is not None)

