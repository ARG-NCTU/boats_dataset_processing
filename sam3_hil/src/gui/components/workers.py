#!/usr/bin/env python3
"""
STAMP - Background Workers
===========================

SAM3 processing workers that run in background threads
to keep the GUI responsive.

Classes:
- SAM3Worker: Video-based SAM3 detection + propagation
- ImageBatchWorker: Independent per-image detection

Utility:
- clear_gpu_memory(): GPU memory cleanup
"""

import gc
import logging
import threading
from pathlib import Path

import cv2
import numpy as np
import torch

from PyQt6.QtCore import QThread, pyqtSignal

# Import engine (assumes project path is already set up by main)
from core.sam3_engine import SAM3Engine

logger = logging.getLogger(__name__)


# =============================================================================
# GPU Memory Cleanup Utility
# =============================================================================

def clear_gpu_memory():
    """
    清理 GPU 記憶體，避免連續執行偵測時發生 OOM。
    
    調用時機：
    - 每次偵測開始前
    - Worker 執行緒開始時
    """
    # 清理 Python 垃圾收集
    gc.collect()
    
    # 清理 PyTorch CUDA 快取
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 記錄清理後的記憶體狀態
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU memory cleared: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")


# =============================================================================
# Worker Thread for SAM3 Processing
# =============================================================================

class SAM3Worker(QThread):
    """
    背景執行緒處理 SAM3 推理。
    
    為什麼需要執行緒？
    - SAM3 推理需要 1-2 分鐘
    - 如果在主執行緒執行，GUI 會凍結
    - 使用 QThread 讓 GUI 保持響應
    
    工作流程：
    1. 載入 SAM3 模型
    2. 啟動 video session
    3. 添加 prompt 進行初步偵測
    4. **暫停** - 發送 ready_to_propagate 信號，等待用戶確認
    5. 用戶確認後，執行 propagate
    
    信號 (Signals):
    - progress: 回報進度 (0-100)
    - ready_to_propagate: 準備好 propagate，等待確認 (num_objects)
    - finished: 完成時發出結果
    - error: 發生錯誤時發出
    - cancelled: 取消時發出
    """
    progress = pyqtSignal(int, str)           # (百分比, 訊息)
    ready_to_propagate = pyqtSignal(int)      # (偵測到的物件數量)
    finished = pyqtSignal(dict)               # 結果字典
    error = pyqtSignal(str)                   # 錯誤訊息
    cancelled = pyqtSignal()                  # 取消信號
    
    def __init__(self, video_path: str, prompt: str, mode: str = "gpu"):
        super().__init__()
        self.video_path = str(video_path)  # 確保是字串
        self.prompt = prompt
        self.mode = mode
        self._cancelled = False
        self._continue_event = threading.Event()
        self._engine = None
        self._session_id = None
    
    def cancel(self):
        """請求取消處理"""
        logger.info("SAM3Worker: Cancel requested")
        self._cancelled = True
        self._continue_event.set()  # 解除等待狀態
    
    def continue_propagation(self):
        """用戶確認後，繼續執行 propagate"""
        logger.info("SAM3Worker: User confirmed, continuing propagation")
        self._continue_event.set()
    
    def _check_cancelled(self) -> bool:
        """檢查是否被取消，如果是則清理資源"""
        if self._cancelled:
            logger.info("SAM3Worker: Cancellation detected, cleaning up...")
            self._cleanup()
            return True
        return False
    
    def _cleanup(self):
        """清理資源"""
        try:
            if self._session_id and self._engine:
                logger.info(f"SAM3Worker: Closing session {self._session_id}")
                self._engine.close_session(self._session_id)
                self._session_id = None
            if self._engine:
                logger.info("SAM3Worker: Shutting down engine")
                self._engine.shutdown()
                self._engine = None
        except Exception as e:
            logger.warning(f"SAM3Worker: Cleanup error (ignored): {e}")
        
        # 清理 GPU 記憶體
        try:
            clear_gpu_memory()
        except:
            pass
    
    def run(self):
        """執行緒主函數。"""
        try:
            # 檢查取消
            if self._check_cancelled():
                self.cancelled.emit()
                return
            
            # 清理 GPU 記憶體
            self.progress.emit(5, "Clearing GPU memory...")
            clear_gpu_memory()

            # 檢查取消
            if self._check_cancelled():
                self.cancelled.emit()
                return

            self.progress.emit(10, "Loading SAM3 model...")
            logger.info(f"Worker starting: video={self.video_path}, prompt={self.prompt}, mode={self.mode}")
            
            self._engine = SAM3Engine(mode=self.mode)
            
            # 檢查取消
            if self._check_cancelled():
                self.cancelled.emit()
                return
            
            self.progress.emit(30, "Starting video session...")
            self._session_id = self._engine.start_video_session(self.video_path)
            logger.info(f"Session started: {self._session_id}")
            
            # 檢查取消
            if self._check_cancelled():
                self.cancelled.emit()
                return
            
            self.progress.emit(40, f"Detecting objects (prompt: {self.prompt})...")
            try:
                result = self._engine.add_prompt(self._session_id, 0, self.prompt)
                # add_prompt 可能返回不同格式，嘗試解析
                if isinstance(result, tuple) and len(result) >= 2:
                    obj_ids = result[1]
                    num_objects = len(obj_ids) if obj_ids is not None else 0
                else:
                    num_objects = 0  # 無法確定，設為 0
            except Exception as e:
                logger.warning(f"Could not get object count from add_prompt: {e}")
                num_objects = 0
            logger.info(f"Initial detection: {num_objects} objects found")
            
            # 檢查取消
            if self._check_cancelled():
                self.cancelled.emit()
                return
            
            # ================================================================
            # 暫停點：等待用戶確認
            # ================================================================
            self.progress.emit(45, f"Found {num_objects} objects. Waiting for confirmation...")
            self.ready_to_propagate.emit(num_objects)
            
            # 等待用戶確認或取消
            logger.info("SAM3Worker: Waiting for user confirmation...")
            self._continue_event.wait()  # 阻塞直到 set()
            
            # 檢查是取消還是確認
            if self._cancelled:
                logger.info("SAM3Worker: User cancelled after detection")
                self._cleanup()
                self.cancelled.emit()
                return
            
            # ================================================================
            # 用戶確認，繼續 propagate
            # ================================================================
            # 注意：propagate() 是一個長時間操作，無法在中間中斷
            # 取消請求會在 propagate 完成後立即生效
            self.progress.emit(50, "Propagating masks (this may take a while)...")
            results = self._engine.propagate(self._session_id)
            logger.info(f"Propagation done: {len(results)} frames")
            
            # 檢查取消（在處理完成後）
            if self._check_cancelled():
                self.cancelled.emit()
                return
            
            self.progress.emit(90, "Closing session...")
            self._engine.close_session(self._session_id)
            self._session_id = None
            
            self._engine.shutdown()
            self._engine = None
            
            # 最後再檢查一次取消
            if self._cancelled:
                self.cancelled.emit()
                return
            
            self.progress.emit(100, "Done!")
            self.finished.emit({"results": results})
            
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}"
            logger.error(f"Worker error: {error_msg}")
            
            # 清理資源
            self._cleanup()
            
            # 如果是取消導致的錯誤，發送取消信號而不是錯誤
            if self._cancelled:
                self.cancelled.emit()
            else:
                self.error.emit(error_msg)


# =============================================================================
# Image Batch Worker (for Independent Image Processing)
# =============================================================================

class ImageBatchWorker(QThread):
    """
    背景執行緒處理批次獨立圖片。
    
    與 SAM3Worker 不同，這個 Worker：
    - 每張圖片獨立使用 detect_image()
    - 不使用 propagate（因為場景不連續）
    - 適用於不同場景的多張照片
    
    信號 (Signals):
    - progress: 回報進度 (current_idx, total, message)
    - image_result: 單張圖片處理完成 (frame_idx, FrameResult)
    - finished: 全部完成 (Dict[int, FrameResult])
    - error: 發生錯誤
    - cancelled: 取消時發出
    """
    progress = pyqtSignal(int, int, str)      # (當前索引, 總數, 訊息)
    image_result = pyqtSignal(int, object)    # (frame_idx, FrameResult) - 即時回報
    finished = pyqtSignal(dict)               # 全部結果
    error = pyqtSignal(str)                   # 錯誤訊息
    cancelled = pyqtSignal()                  # 取消信號
    
    def __init__(self, image_paths: list, prompt: str, mode: str = "gpu"):
        """
        初始化 ImageBatchWorker。
        
        Args:
            image_paths: 圖片路徑列表
            prompt: 文字提示
            mode: SAM3 模式 ("gpu" 或 "mock")
        """
        super().__init__()
        self.image_paths = image_paths
        self.prompt = prompt
        self.mode = mode
        self._cancelled = False
        self._engine = None
    
    def cancel(self):
        """請求取消處理"""
        logger.info("ImageBatchWorker: Cancel requested")
        self._cancelled = True
    
    def _cleanup(self):
        """清理資源"""
        try:
            if self._engine:
                self._engine.shutdown()
                self._engine = None
        except Exception as e:
            logger.warning(f"ImageBatchWorker: Cleanup error (ignored): {e}")
        
        try:
            clear_gpu_memory()
        except:
            pass
    
    def run(self):
        """執行緒主函數 - 逐張處理圖片"""
        results = {}
        total = len(self.image_paths)
        
        try:
            # 清理 GPU
            self.progress.emit(0, total, "Clearing GPU memory...")
            clear_gpu_memory()
            
            if self._cancelled:
                self.cancelled.emit()
                return
            
            # 載入 SAM3 模型
            self.progress.emit(0, total, "Loading SAM3 model...")
            logger.info(f"ImageBatchWorker starting: {total} images, prompt={self.prompt}")
            
            self._engine = SAM3Engine(mode=self.mode)
            
            if self._cancelled:
                self._cleanup()
                self.cancelled.emit()
                return
            
            # 逐張處理
            for idx, image_path in enumerate(self.image_paths):
                if self._cancelled:
                    self._cleanup()
                    self.cancelled.emit()
                    return
                
                filename = Path(image_path).name
                self.progress.emit(idx, total, f"Processing {filename} ({idx+1}/{total})...")
                
                try:
                    # 讀取圖片
                    image = cv2.imread(str(image_path))
                    if image is None:
                        logger.warning(f"Cannot read image: {image_path}")
                        continue
                    
                    # 使用 detect_image 進行獨立偵測
                    frame_result = self._engine.detect_image(image, self.prompt)
                    
                    # 更新 frame_index 為正確的索引
                    frame_result.frame_index = idx
                    
                    results[idx] = frame_result
                    
                    # 即時回報結果
                    self.image_result.emit(idx, frame_result)
                    
                    logger.info(f"Processed {filename}: {frame_result.num_objects} objects")
                    
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    # 繼續處理下一張，不中斷整個流程
            
            # 完成
            self._cleanup()
            
            if self._cancelled:
                self.cancelled.emit()
                return
            
            self.progress.emit(total, total, "Done!")
            self.finished.emit({"results": results})
            
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}"
            logger.error(f"ImageBatchWorker error: {error_msg}")
            self._cleanup()
            
            if self._cancelled:
                self.cancelled.emit()
            else:
                self.error.emit(error_msg)
