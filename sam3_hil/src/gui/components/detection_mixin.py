#!/usr/bin/env python3
"""
STAMP - Detection Mixin
=========================

SAM3 detection workflow methods for HILAAMainWindow:
video detection, batch image detection, horizon detection,
jitter detection, and all detection callbacks.

Host requirements (attributes expected on self):
- video_loader, sam3_results, video_analysis, analyzer
- worker, progress_dialog, detect_btn, prompt_input, mode_combo
- processing_mode_combo, maritime_roi_checkbox, maritime_method_combo
- current_frame, object_status, action_logger
- _detection_start_time, sam3_engine, maritime_roi, horizon_result
- display_frame(), update_object_list(), update_analysis_display()
- update_timeline(), stop_play(), statusBar()
"""

import logging
import time

import cv2
import numpy as np

from PyQt6.QtWidgets import QMessageBox, QProgressDialog, QApplication
from PyQt6.QtCore import Qt

from gui.components.workers import SAM3Worker, ImageBatchWorker, clear_gpu_memory
from core.video_loader import ImageFolderLoader

logger = logging.getLogger(__name__)


class DetectionMixin:
    """SAM3 偵測流程邏輯。"""

    def run_detection(self):
        """執行 SAM3 偵測。"""
        if self.video_loader is None:
            return
            
        clear_gpu_memory()

        prompt = self.prompt_input.text().strip()
        if not prompt:
            QMessageBox.warning(self, "Warning", "Please enter a detection prompt")
            return
        
        mode = self.mode_combo.currentText()
        
        # 停止播放
        self.stop_play()
        
        # 判斷處理模式
        # 0 = Video Mode (propagate) - 影片追蹤
        # 1 = Images Mode (detect_image) - 獨立圖片處理
        processing_mode = self.processing_mode_combo.currentIndex()
        
        if processing_mode == 1:
            # === Images 模式：每張圖片獨立偵測 ===
            self._run_batch_detection(prompt, mode)
        else:
            # === Video 模式 ===
            self._run_video_detection(prompt, mode)
    

    def _run_video_detection(self, prompt: str, mode: str):
        """執行影片追蹤模式的偵測（原本的流程）。"""
        # Maritime ROI：偵測海平線
        if self.maritime_roi_checkbox.isChecked():
            self._run_horizon_detection()
        
        # 取得影片路徑（確保是字串）
        video_path = str(self.video_loader.video_path)
        logger.info(f"Starting video detection: video={video_path}, prompt={prompt}, mode={mode}")
        
        # === ActionLogger: 記錄偵測開始 ===
        self._detection_start_time = time.time()
        self.action_logger.log_detection_started(prompt=prompt, frame_idx=0)
        
        # 建立進度對話框
        self.progress_dialog = QProgressDialog(
            "Running SAM3 detection...", "Cancel", 0, 100, self
        )
        self.progress_dialog.setWindowTitle("Processing")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setAutoClose(False)  # 不要自動關閉
        self.progress_dialog.setAutoReset(False)  # 不要自動重設
        
        # 建立並啟動 worker 執行緒
        self.worker = SAM3Worker(video_path, prompt, mode)
        self.worker.progress.connect(self.on_detection_progress)
        self.worker.ready_to_propagate.connect(self.on_ready_to_propagate)
        self.worker.finished.connect(self.on_detection_finished)
        self.worker.error.connect(self.on_detection_error)
        self.worker.cancelled.connect(self.on_detection_cancelled)
        
        # 連接取消按鈕到 worker 的取消方法
        self.progress_dialog.canceled.connect(self._on_cancel_detection)
        
        self.detect_btn.setEnabled(False)
        self.worker.start()
    

    def _run_batch_detection(self, prompt: str, mode: str):
        """執行圖片獨立處理模式的偵測。"""
        # 取得圖片路徑列表
        if isinstance(self.video_loader, ImageFolderLoader):
            image_paths = self.video_loader.metadata.image_paths
        else:
            # 如果是 VideoLoader，需要先提取幀（這種情況較少見）
            QMessageBox.warning(
                self, "Warning", 
                "Please use 'Open Image Folder' for independent image processing,\n"
                "or switch to 'Video (Sequential Tracking)' mode."
            )
            return
        
        total = len(image_paths)
        logger.info(f"Starting batch detection: {total} images, prompt={prompt}, mode={mode}")
        
        # === ActionLogger: 記錄偵測開始 ===
        self._detection_start_time = time.time()
        self.action_logger.log_detection_started(prompt=prompt, frame_idx=0)
        
        # 建立進度對話框
        self.progress_dialog = QProgressDialog(
            f"Processing {total} images...", "Cancel", 0, total, self
        )
        self.progress_dialog.setWindowTitle("Batch Processing")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        
        # 建立並啟動 ImageBatchWorker
        self.worker = ImageBatchWorker(image_paths, prompt, mode)
        self.worker.progress.connect(self.on_batch_progress)
        self.worker.image_result.connect(self.on_batch_image_result)
        self.worker.finished.connect(self.on_batch_finished)
        self.worker.error.connect(self.on_batch_error)
        self.worker.cancelled.connect(self.on_batch_cancelled)
        
        # 連接取消按鈕
        self.progress_dialog.canceled.connect(self._on_cancel_detection)
        
        self.detect_btn.setEnabled(False)
        self.worker.start()
    
    # =========================================================================
    # Batch Processing Callbacks (for Independent Image Mode)
    # =========================================================================
    

    def on_batch_progress(self, current: int, total: int, message: str):
        """批次處理進度更新。"""
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.setValue(current)
            self.progress_dialog.setLabelText(message)
    

    def on_batch_image_result(self, frame_idx: int, frame_result):
        """單張圖片處理完成（即時回報）。"""
        # 儲存結果
        self.sam3_results[frame_idx] = frame_result
        
        # 如果是當前顯示的幀，立即更新顯示
        if frame_idx == self.current_frame:
            self.display_frame(frame_idx)
        
        logger.debug(f"Image {frame_idx} processed: {frame_result.num_objects} objects")
    

    def on_batch_finished(self, result: dict):
        """批次處理完成。"""
        # 清理對話框和 worker
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        if hasattr(self, 'worker') and self.worker:
            self.worker.wait(500)
            self.worker = None
        
        self.detect_btn.setEnabled(True)
        
        # 儲存結果（可能已經在 on_batch_image_result 中部分儲存）
        self.sam3_results = result.get("results", self.sam3_results)
        
        # 分析結果
        self.video_analysis = self.analyzer.analyze_video(self.sam3_results)
        
        # === ActionLogger: 記錄偵測完成 ===
        if self._detection_start_time:
            duration = time.time() - self._detection_start_time
        else:
            duration = 0.0
        self.action_logger.log_detection_finished(
            num_objects=self.video_analysis.unique_objects,
            duration_seconds=duration,
            num_frames=len(self.sam3_results)
        )
        self._detection_start_time = None
        
        # 注意：圖片模式不運行 Jitter Detection（因為沒有時序連續性）
        self.jitter_analysis = None
        
        # 更新物件列表
        self.update_object_list()
        
        # 更新分析顯示
        self.update_analysis_display()
        
        # 更新 Timeline
        self.update_timeline()
        
        # 重新顯示當前幀
        self.display_frame(self.current_frame)
        
        self.statusBar().showMessage(
            f"Batch detection completed: {self.video_analysis.unique_objects} objects "
            f"across {len(self.sam3_results)} images"
        )
    

    def on_batch_error(self, error_msg: str):
        """批次處理錯誤。"""
        # 清理對話框和 worker
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        if hasattr(self, 'worker') and self.worker:
            self.worker.wait(500)
            self.worker = None
        
        self.detect_btn.setEnabled(True)
        self._detection_start_time = None
        
        # 顯示錯誤
        display_msg = error_msg
        if len(error_msg) > 1000:
            display_msg = error_msg[:1000] + "\n\n... (see terminal for full error)"
        
        logger.error(f"Batch detection error:\n{error_msg}")
        QMessageBox.critical(self, "Batch Detection Error", f"Processing failed:\n\n{display_msg}")
    

    def on_batch_cancelled(self):
        """批次處理被取消。"""
        logger.info("Batch detection cancelled")
        
        # 清理
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        if hasattr(self, 'worker') and self.worker:
            self.worker.wait(1000)
            if self.worker.isRunning():
                self.worker.terminate()
                self.worker.wait(500)
            self.worker = None
        
        self.detect_btn.setEnabled(True)
        self._detection_start_time = None
        
        # 如果已經處理了一部分，保留那些結果
        if self.sam3_results:
            self.video_analysis = self.analyzer.analyze_video(self.sam3_results)
            self.update_object_list()
            self.update_analysis_display()
            self.statusBar().showMessage(
                f"Detection cancelled - {len(self.sam3_results)} images processed"
            )
        else:
            self.statusBar().showMessage("Detection cancelled")
    

    def _on_cancel_detection(self):
        """處理取消偵測請求"""
        logger.info("User requested detection cancellation")
        if hasattr(self, 'worker') and self.worker is not None:
            # 更新對話框顯示
            self.progress_dialog.setLabelText("Cancelling... please wait...")
            self.progress_dialog.setCancelButton(None)  # 隱藏取消按鈕，防止重複點擊
            
            # 請求取消
            self.worker.cancel()
    

    def on_detection_cancelled(self):
        """偵測被取消時的處理"""
        logger.info("Detection cancelled successfully")
        
        # 清理
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        if hasattr(self, 'worker') and self.worker:
            self.worker.wait(1000)  # 等待最多 1 秒
            if self.worker.isRunning():
                logger.warning("Worker still running after cancel, terminating...")
                self.worker.terminate()
                self.worker.wait(500)
            self.worker = None
        
        self.detect_btn.setEnabled(True)
        self._detection_start_time = None
        
        self.statusBar().showMessage("Detection cancelled")
    

    def on_ready_to_propagate(self, num_objects: int):
        """
        初步偵測完成，等待用戶確認是否繼續 propagate。
        
        這是一個關鍵暫停點，讓用戶可以在漫長的 propagate 開始前決定是否繼續。
        """
        logger.info(f"Ready to propagate: {num_objects} objects detected")
        
        # 暫時隱藏 progress dialog
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.hide()
        
        # 準備確認訊息
        if num_objects == 0:
            msg = (
                f"Detect with prompt: '{self.prompt_input.text()}'\n"
                f"Do you want to continue anyway?"
            )
        else:
            total_frames = self.video_loader.metadata.total_frames
            estimated_time = total_frames * 0.1  # 估算：每幀約 0.1 秒
            
            msg = (
                f"Initial Detection Complete!\n\n"
                f"Objects found: {num_objects}\n"
                f"Total frames: {total_frames}\n"
                f"Estimated time: ~{estimated_time:.0f} seconds\n\n"
                f"Warning: Once propagation starts, it cannot be interrupted.\n\n"
                f"Do you want to continue with propagation?"
            )
        
        # 顯示確認對話框
        reply = QMessageBox.question(
            self, 
            "Confirm Propagation",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes  # 預設選擇 Yes
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # 用戶確認，繼續 propagate
            logger.info("User confirmed propagation")
            
            # 重新顯示 progress dialog
            if hasattr(self, 'progress_dialog') and self.progress_dialog:
                self.progress_dialog.show()
                self.progress_dialog.setLabelText("Propagating masks (this may take a while)...")
                self.progress_dialog.setValue(50)
            
            # 通知 worker 繼續
            if hasattr(self, 'worker') and self.worker:
                self.worker.continue_propagation()
        else:
            # 用戶取消
            logger.info("User cancelled before propagation")
            
            # 通知 worker 取消
            if hasattr(self, 'worker') and self.worker:
                self.worker.cancel()
    

    def on_detection_progress(self, percent: int, message: str):
        """偵測進度更新。"""
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.setValue(percent)
            self.progress_dialog.setLabelText(message)
    

    def on_detection_finished(self, result: dict):
        """偵測完成。"""
        # 清理對話框和 worker
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        if hasattr(self, 'worker') and self.worker:
            self.worker.wait(500)
            self.worker = None
        
        self.detect_btn.setEnabled(True)
        
        self.sam3_results = result["results"]
        
        # Maritime ROI 後處理：過濾天空區域內的物件
        if self.maritime_roi_checkbox.isChecked() and self.horizon_result and self.horizon_result.valid:
            self._filter_sky_objects()
        
        # 分析結果
        self.video_analysis = self.analyzer.analyze_video(self.sam3_results)
        
        # === ActionLogger: 記錄偵測完成 ===
        if self._detection_start_time:
            duration = time.time() - self._detection_start_time
        else:
            duration = 0.0
        self.action_logger.log_detection_finished(
            num_objects=self.video_analysis.unique_objects,
            duration_seconds=duration,
            num_frames=len(self.sam3_results)
        )
        self._detection_start_time = None
        
        # 運行 Jitter Detection
        self._run_jitter_detection()
        
        # 更新物件列表
        self.update_object_list()
        
        # 更新分析顯示
        self.update_analysis_display()
        
        # 更新 Timeline
        self.update_timeline()
        
        # 重新顯示當前幀
        self.display_frame(self.current_frame)
        
        self.statusBar().showMessage(
            f"detection completed: {self.video_analysis.unique_objects} objects"
        )
    

    def _run_jitter_detection(self):
        """運行 Jitter Detection 分析時序穩定性。"""
        if not self.sam3_results:
            self.jitter_analysis = None
            return
        
        try:
            from core.jitter_detector import JitterDetector
            
            detector = JitterDetector(
                iou_threshold=0.85,
                area_change_threshold=0.15
            )
            self.jitter_analysis = detector.analyze_video(self.sam3_results)
            
            # 記錄結果
            ja = self.jitter_analysis
            logger.info(
                f"Jitter detection: {ja.total_jitter_events} events, "
                f"{ja.jitter_frame_count} frames, "
                f"stability: {ja.overall_stability:.1%}"
            )
            
            # 如果有 jitter，提示用戶
            if ja.jitter_frame_count > 0:
                jitter_frames = ja.get_all_jitter_frames()[:5]  # 前 5 個
                frames_str = ", ".join(str(f) for f in jitter_frames)
                if ja.jitter_frame_count > 5:
                    frames_str += f"... ({ja.jitter_frame_count} total)"
                logger.warning(f"Jitter detected at frames: {frames_str}")
                
        except Exception as e:
            logger.error(f"Jitter detection failed: {e}")
            self.jitter_analysis = None
    

    def _run_horizon_detection(self):
        """
        運行海平線偵測（Maritime ROI）。
        
        在第一幀上偵測海平線，用於排除天空區域。
        """
        if self.video_loader is None:
            self.horizon_result = None
            return
        
        try:
            from core.maritime_roi import MaritimeROI
            
            # 取得選擇的方法
            method = self._get_maritime_roi_method()
            
            # 初始化 MaritimeROI（如果需要）
            # MaritimeROI 會自動偵測 SegFormer 模型路徑（Docker 或 Host）
            if self.maritime_roi is None or getattr(self.maritime_roi, 'method', None) != method:
                self.maritime_roi = MaritimeROI(method=method)
            
            # 取得第一幀
            frame = self.video_loader.get_frame(0)
            if frame is None:
                logger.warning("Cannot read first frame for horizon detection")
                self.horizon_result = None
                return
            
            # 偵測海平線
            self.horizon_result = self.maritime_roi.detect_horizon(frame)
            
            if self.horizon_result.valid:
                logger.info(
                    f"Horizon detected: slope={self.horizon_result.slope:.4f}, "
                    f"center={self.horizon_result.center}, "
                    f"method={self.horizon_result.method_used}"
                )
                
                # 計算天空區域（供未來 SAM3 整合使用）
                sky_box = self.maritime_roi.get_sky_box_xyxy(frame, self.horizon_result)
                if sky_box:
                    logger.info(f"Sky box (negative region): {sky_box}")
            else:
                logger.warning("Horizon detection failed")
                
        except ImportError as e:
            logger.error(f"Maritime ROI module not found: {e}")
            self.horizon_result = None
        except Exception as e:
            logger.error(f"Horizon detection error: {e}")
            self.horizon_result = None
    

    def _draw_horizon_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        在畫面上繪製海平線和天空區域視覺化。
        
        - 黃色線：偵測到的海平線
        - 半透明紅色區域：天空區域（會被過濾的區域）
        - 顯示海平線資訊文字
        """
        if not self.horizon_result or not self.horizon_result.valid:
            return frame
        
        output = frame.copy()
        h, w = output.shape[:2]
        
        # 取得海平線參數
        slope = self.horizon_result.slope
        center_x, center_y = self.horizon_result.center
        
        # 計算海平線兩端點
        # y = slope * (x - center_x) + center_y
        x1, x2 = 0, w
        y1 = int(slope * (x1 - center_x) + center_y)
        y2 = int(slope * (x2 - center_x) + center_y)
        
        # 取得 sky box
        sky_box = None
        if self.maritime_roi:
            sky_box = self.maritime_roi.get_sky_box_xyxy(frame, self.horizon_result)
        
        # 繪製天空區域（半透明紅色）
        if sky_box:
            sky_x1, sky_y1, sky_x2, sky_y2 = sky_box
            overlay = output.copy()
            cv2.rectangle(overlay, (int(sky_x1), int(sky_y1)), (int(sky_x2), int(sky_y2)), 
                         (0, 0, 255), -1)  # 紅色填充
            cv2.addWeighted(overlay, 0.2, output, 0.8, 0, output)  # 20% 透明度
            
            # 繪製 sky box 邊框
            cv2.rectangle(output, (int(sky_x1), int(sky_y1)), (int(sky_x2), int(sky_y2)), 
                         (0, 0, 255), 2)  # 紅色邊框
        
        # 繪製海平線（黃色粗線）
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 255), 3)  # 黃色
        
        # 繪製海平線中心點
        cv2.circle(output, (int(center_x), int(center_y)), 8, (0, 255, 0), -1)  # 綠色圓點
        cv2.circle(output, (int(center_x), int(center_y)), 8, (0, 0, 0), 2)  # 黑色邊框
        
        # 顯示資訊文字
        info_lines = [
            f"Horizon: y={center_y}, slope={slope:.4f}",
            f"Method: {self.horizon_result.method_used}",
        ]
        if sky_box:
            info_lines.append(f"Sky region: y < {int(sky_box[3])}")
        
        # 繪製文字背景
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        padding = 10
        
        y_offset = 30
        for line in info_lines:
            (text_w, text_h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
            cv2.rectangle(output, (10, y_offset - text_h - padding), 
                         (10 + text_w + padding * 2, y_offset + baseline + padding), 
                         (0, 0, 0), -1)
            cv2.putText(output, line, (10 + padding, y_offset), font, font_scale, 
                       (0, 255, 255), thickness)
            y_offset += text_h + baseline + padding * 2
        
        return output
    

    def _filter_sky_objects(self):
        """
        後處理過濾：移除 bounding box 中心點位於天空區域的物件。
        
        這個方法在 SAM3 偵測完成後執行，用於過濾掉可能是天空中
        誤偵測的物件（例如雲、飛機、海平線上的遠處船隻等）。
        
        改進：檢查物件「首次有效出現」的幀，而不是只看第一幀。
        這樣可以正確處理影片中途才出現的物件。
        """
        if not self.sam3_results or not self.horizon_result or not self.horizon_result.valid:
            return
        
        if self.video_loader is None:
            return
        
        # 取得第一幀來計算 sky box
        frame = self.video_loader.get_frame(0)
        if frame is None:
            return
        
        sky_box = self.maritime_roi.get_sky_box_xyxy(frame, self.horizon_result)
        if sky_box is None:
            return
        
        sky_x1, sky_y1, sky_x2, sky_y2 = sky_box
        
        # 收集所有物件的「首次有效出現」資訊
        # 有效出現 = bounding box 面積大於閾值
        MIN_BOX_AREA = 100  # 最小有效面積（像素²）
        
        object_first_appearance = {}  # obj_id -> (frame_idx, center_x, center_y)
        
        for frame_idx in sorted(self.sam3_results.keys()):
            frame_result = self.sam3_results[frame_idx]
            for detection in frame_result.detections:
                obj_id = detection.obj_id
                
                # 如果已經記錄過這個物件的首次出現，跳過
                if obj_id in object_first_appearance:
                    continue
                
                # 取得 bounding box (xywh 格式)
                x, y, w, h = detection.box
                box_area = w * h
                
                # 檢查是否為有效的 bounding box
                if box_area < MIN_BOX_AREA:
                    continue
                
                # 計算中心點
                center_x = x + w / 2
                center_y = y + h / 2
                
                object_first_appearance[obj_id] = (frame_idx, center_x, center_y)
        
        # 找出需要過濾的物件 ID
        objects_to_remove = set()
        
        for obj_id, (frame_idx, center_x, center_y) in object_first_appearance.items():
            # 檢查中心點是否在 sky box 內
            if sky_x1 <= center_x <= sky_x2 and sky_y1 <= center_y <= sky_y2:
                objects_to_remove.add(obj_id)
                logger.info(
                    f"Filtering object {obj_id}: "
                    f"first valid appearance at frame {frame_idx}, "
                    f"center=({center_x:.0f}, {center_y:.0f}) is in sky region (y < {sky_y2})"
                )
        
        if not objects_to_remove:
            logger.info("Maritime ROI: No objects filtered (all objects are below horizon)")
            return
        
        # 從所有幀中移除這些物件
        filtered_count = 0
        for frame_idx, frame_result in self.sam3_results.items():
            original_count = len(frame_result.detections)
            frame_result.detections = [
                d for d in frame_result.detections 
                if d.obj_id not in objects_to_remove
            ]
            filtered_count += original_count - len(frame_result.detections)
        
        logger.info(
            f"Maritime ROI filter: removed {len(objects_to_remove)} objects "
            f"({filtered_count} detections total) - sky region y < {sky_y2}"
        )
    

    def _reanalyze_with_preserved_edits(self):
        """
        重新分析 SAM3 結果，同時保留已編輯的幀記錄。
        
        解決問題：analyze_video() 會創建新的 VideoAnalysis 物件，
        導致 frames_actually_edited 被重置。
        """
        # 1. 保存現有的 edited frames
        preserved_edits = set()
        if self.video_analysis and self.video_analysis.frames_actually_edited:
            preserved_edits = self.video_analysis.frames_actually_edited.copy()
            logger.debug(f"Preserving {len(preserved_edits)} edited frames before reanalysis")
        
        # 2. 重新分析
        self.video_analysis = self.analyzer.analyze_video(self.sam3_results)
        
        # 3. 恢復 edited frames
        self.video_analysis.frames_actually_edited = preserved_edits
        
        # 4. 重新運行 Jitter Detection
        self._run_jitter_detection()
        
        logger.debug(f"Reanalysis complete, {len(preserved_edits)} edited frames restored")
    

    def on_detection_error(self, error_msg: str):
        """偵測錯誤。"""
        # 清理對話框和 worker
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        if hasattr(self, 'worker') and self.worker:
            self.worker.wait(500)
            self.worker = None
        
        self.detect_btn.setEnabled(True)
        self._detection_start_time = None
        
        # 顯示錯誤（如果太長就截斷）
        display_msg = error_msg
        if len(error_msg) > 1000:
            display_msg = error_msg[:1000] + "\n\n... (see terminal for full error)"
        
        logger.error(f"Detection error:\n{error_msg}")
        QMessageBox.critical(self, "Detection Error", f"SAM3 detection failed:\n\n{display_msg}")
    
    # =========================================================================
    # 物件列表管理
    # =========================================================================
    
