"""
Server Workers
==============

背景執行緒 Workers，透過 HTTP/WebSocket 與遠端 Server 通訊。
取代原本的 SAM3Worker（本地 GPU 處理）。

使用方式：
    from src.gui.workers import ServerVideoDetectionWorker
    
    worker = ServerVideoDetectionWorker(
        server_url="http://192.168.1.100:8000",
        video_path="/app/data/video.mp4",
        prompt="ship"
    )
    
    # 連接信號
    worker.progress.connect(on_progress)
    worker.ready_to_propagate.connect(on_ready)
    worker.finished.connect(on_finished)
    worker.error.connect(on_error)
    
    # 開始
    worker.start()
    
    # 確認繼續
    worker.confirm()
    
    # 或取消
    worker.cancel()

信號 (Signals):
    progress(int, str)          進度百分比和訊息
    ready_to_propagate(int, dict)  準備好傳播，等待確認（物件數, 資料）
    finished(dict)              完成，結果資料
    error(str)                  錯誤訊息
    cancelled()                 已取消
"""

import json
import threading
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field

import numpy as np
from PyQt6.QtCore import QThread, pyqtSignal
from loguru import logger

# 嘗試導入 WebSocket
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False

# 導入 API Client
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.api_client import StampAPIClient, JobStatus


# =============================================================================
# 資料類別（與 sam3_engine.py 相容）
# =============================================================================

@dataclass
class Detection:
    """
    偵測結果（與 sam3_engine.Detection 相容）
    """
    obj_id: int
    mask: np.ndarray  # boolean mask
    score: float
    box: np.ndarray   # [x, y, w, h]
    
    @property
    def box_xyxy(self) -> np.ndarray:
        x, y, w, h = self.box
        return np.array([x, y, x + w, y + h])


@dataclass  
class FrameResult:
    """
    單幀結果（與 sam3_engine.FrameResult 相容）
    """
    frame_index: int
    detections: List[Detection] = field(default_factory=list)
    
    @property
    def num_objects(self) -> int:
        return len(self.detections)
    
    @property
    def masks(self) -> List[np.ndarray]:
        return [d.mask for d in self.detections]
    
    @property
    def boxes(self) -> List[np.ndarray]:
        return [d.box for d in self.detections]
    
    @property
    def scores(self) -> List[float]:
        return [d.score for d in self.detections]
    
    @property
    def obj_ids(self) -> List[int]:
        return [d.obj_id for d in self.detections]


def deserialize_results(data: dict) -> Dict[int, FrameResult]:
    """
    反序列化 Server 返回的結果（包含 RLE 編碼的 mask）
    
    Args:
        data: Server 返回的 {"results": {...}} 或 {"frames": {...}}
        
    Returns:
        Dict[int, FrameResult]: 與 SAM3Worker 相同的格式
    """
    from pycocotools import mask as mask_utils
    
    results = {}
    
    # 取得幀數據（相容兩種 key）
    frames_data = data.get("results") or data.get("frames") or {}
    
    for frame_idx_str, frame_data in frames_data.items():
        frame_idx = int(frame_idx_str)
        
        detections = []
        for det_data in frame_data.get("detections", []):
            # 解碼 RLE mask
            if "mask_rle" in det_data:
                rle = det_data["mask_rle"]
                # 確保 counts 是 bytes
                if isinstance(rle["counts"], str):
                    rle["counts"] = rle["counts"].encode("utf-8")
                mask = mask_utils.decode(rle).astype(bool)
            else:
                # Fallback：用 bbox 創建矩形 mask
                mask_shape = det_data.get("mask_shape", [480, 640])
                if mask_shape and len(mask_shape) >= 2:
                    mask = np.zeros((mask_shape[0], mask_shape[1]), dtype=bool)
                else:
                    mask = np.zeros((480, 640), dtype=bool)
                
                # 從 bbox 創建矩形 mask
                bbox = det_data.get("bbox_xyxy") or det_data.get("bbox", [0, 0, 10, 10])
                if len(bbox) == 4:
                    if det_data.get("bbox_xyxy"):
                        x1, y1, x2, y2 = [int(v) for v in bbox]
                    else:
                        x, y, w, h = [int(v) for v in bbox]
                        x1, y1, x2, y2 = x, y, x + w, y + h
                    
                    y1_clip = max(0, min(y1, mask.shape[0]))
                    y2_clip = max(0, min(y2, mask.shape[0]))
                    x1_clip = max(0, min(x1, mask.shape[1]))
                    x2_clip = max(0, min(x2, mask.shape[1]))
                    mask[y1_clip:y2_clip, x1_clip:x2_clip] = True
            
            # 取得 bbox
            bbox = det_data.get("bbox", [0, 0, 10, 10])
            box = np.array(bbox)
            
            detection = Detection(
                obj_id=det_data.get("obj_id", 0),
                mask=mask,
                score=float(det_data.get("score", 0.5)),
                box=box,
            )
            detections.append(detection)
        
        results[frame_idx] = FrameResult(
            frame_index=frame_idx,
            detections=detections,
        )
    
    return results


# =============================================================================
# Base Worker
# =============================================================================

class BaseServerWorker(QThread):
    """
    Server Worker 基類
    
    提供共用的信號和方法。
    """
    
    # 信號
    progress = pyqtSignal(int, str)      # (百分比, 訊息)
    error = pyqtSignal(str)              # 錯誤訊息
    cancelled = pyqtSignal()             # 已取消
    
    def __init__(self, server_url: str = "http://localhost:8000"):
        super().__init__()
        self.server_url = server_url
        self.client = StampAPIClient(server_url)
        
        self._task_id: Optional[str] = None
        self._cancelled = False
        self._confirm_event = threading.Event()
        self._ws: Optional[Any] = None
    
    def cancel(self):
        """請求取消任務"""
        self._cancelled = True
        self._confirm_event.set()  # 喚醒等待中的確認
        
        # 如果有任務 ID，也透過 API 取消
        if self._task_id:
            try:
                self.client.cancel_job(self._task_id)
            except Exception as e:
                logger.warning(f"Failed to cancel job via API: {e}")
        
        logger.info("Worker cancel requested")
    
    def confirm(self):
        """確認繼續執行"""
        self._confirm_event.set()
        
        # 透過 API 確認
        if self._task_id:
            try:
                self.client.confirm_job(self._task_id)
            except Exception as e:
                logger.warning(f"Failed to confirm job via API: {e}")
        
        logger.info("Worker confirmed")
    
    def _check_cancelled(self) -> bool:
        """檢查是否已取消"""
        return self._cancelled


# =============================================================================
# Video Detection Worker
# =============================================================================

class ServerVideoDetectionWorker(BaseServerWorker):
    """
    影片偵測 Worker
    
    流程：
    1. 建立 video-detection 任務
    2. 連接 WebSocket 監聽進度
    3. 收到 waiting_confirm 時發送信號，等待用戶確認
    4. 確認後繼續傳播
    5. 完成時發送結果
    
    信號：
        progress(int, str)              進度
        ready_to_propagate(int, dict)   等待確認（物件數, 初步結果）
        finished(dict)                  完成
        error(str)                      錯誤
        cancelled()                     取消
    """
    
    # 額外信號
    ready_to_propagate = pyqtSignal(int, dict)  # (物件數, 初步結果)
    finished = pyqtSignal(dict)                  # 完成結果
    upload_progress = pyqtSignal(int, str)       # 新增：上傳進度 (percent, message)
    
    def __init__(
        self,
        server_url: str,
        video_path: str,
        prompt: str,
    ):
        super().__init__(server_url)
        self.video_path = video_path
        self.prompt = prompt
        self._server_video_path = None  # 上傳後的 Server 路徑
    
    def continue_propagation(self):
        """
        繼續執行傳播（與 SAM3Worker 相容的別名）
        
        這是 confirm() 的別名，讓 main_window 可以用相同的介面。
        """
        self.confirm()
    
    def run(self):
        """執行任務（含上傳）"""
        try:
            # =================================================================
            # Step 0: 檢查是否需要上傳
            # =================================================================
            video_path_to_use = self.video_path
            
            # 判斷是否為本地路徑（Server 路徑是 /app/... 開頭）
            is_local_path = not self.video_path.startswith("/app/")
            
            if is_local_path:
                self.upload_progress.emit(0, "Preparing to upload video...")
                self.progress.emit(0, "Uploading video to server...")
                
                logger.info(f"Local video detected, uploading: {self.video_path}")
                
                # 上傳影片
                server_path = self.client.upload_video_with_progress(
                    self.video_path,
                    progress_callback=lambda p, m: self.upload_progress.emit(p, m)
                )
                
                if server_path is None:
                    self.error.emit("Failed to upload video to server")
                    return
                
                if self._check_cancelled():
                    self.cancelled.emit()
                    return
                
                video_path_to_use = server_path
                self._server_video_path = server_path  # 儲存供後續使用
                logger.info(f"Upload complete, server path: {video_path_to_use}")
                self.progress.emit(5, "Upload complete. Starting detection...")
            
            # =================================================================
            # Step 1: 建立任務（使用 Server 端路徑）
            # =================================================================
            self.progress.emit(10, "Creating detection job...")
            
            self._task_id = self.client.create_video_detection_job(
                video_path=video_path_to_use,
                prompt=self.prompt,
            )
            
            logger.info(f"Created job: {self._task_id}")
            
            if self._check_cancelled():
                self.cancelled.emit()
                return
            
            # =================================================================
            # Step 2: 監聽進度
            # =================================================================
            if WEBSOCKET_AVAILABLE:
                self._run_with_websocket()
            else:
                self._run_with_polling()
            
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Worker error: {error_msg}\n{traceback.format_exc()}")
            self.error.emit(error_msg)
    
    def _run_with_websocket(self):
        """使用 WebSocket 監聽進度"""
        try:
            ws = self.client.connect_job_websocket(self._task_id)
            self._ws = ws
        except Exception as e:
            logger.warning(f"WebSocket connection failed, falling back to polling: {e}")
            self._run_with_polling()
            return
        
        try:
            for event in self.client.iter_job_events(ws):
                if self._check_cancelled():
                    self.cancelled.emit()
                    return
                
                event_type = event.get("type")
                
                if event_type == "status":
                    # 初始狀態
                    progress = event.get("progress", 0)
                    message = event.get("message", "")
                    self.progress.emit(progress, message)
                
                elif event_type == "progress":
                    progress = event.get("progress", 0)
                    message = event.get("message", "")
                    self.progress.emit(progress, message)
                
                elif event_type == "waiting_confirm":
                    # 等待用戶確認
                    data = event.get("data", {})
                    num_objects = data.get("num_objects", 0)
                    
                    self.progress.emit(45, f"Found {num_objects} objects. Waiting for confirmation...")
                    self.ready_to_propagate.emit(num_objects, data)
                    
                    # 等待確認或取消（使用超時等待，每 10 秒發送 ping 保持連接）
                    logger.info("Waiting for user confirmation...")
                    ws_alive = True
                    while not self._confirm_event.wait(timeout=10.0):
                        if self._cancelled:
                            self.cancelled.emit()
                            return
                        # 發送 ping 保持連接
                        try:
                            ws.send('{"action": "ping"}')
                            logger.debug("Sent ping to keep WebSocket alive")
                        except Exception as e:
                            logger.warning(f"Failed to send ping: {e}")
                            ws_alive = False
                            break
                    
                    if not ws_alive:
                        self.error.emit("WebSocket connection lost while waiting for confirmation")
                        return

                    if self._cancelled:
                        self.cancelled.emit()
                        return
                    
                    # 用戶已確認，WebSocket 會收到後續進度
                    self.progress.emit(50, "Confirmed. Propagating...")
                
                elif event_type == "intermediate":
                    # 中間結果（批量處理時）
                    pass
                
                elif event_type == "completed":
                    logger.info("=== Received completed event ===")
                    raw_result = event.get("result", {})
                    logger.info(f"Raw result keys: {raw_result.keys() if raw_result else 'None'}")
                    
                    self.progress.emit(100, "Completed!")
                    
                    # 反序列化結果
                    try:
                        deserialized = deserialize_results(raw_result)
                        logger.info(f"Deserialized {len(deserialized)} frames")
                    except Exception as e:
                        logger.error(f"Failed to deserialize results: {e}")
                        import traceback
                        traceback.print_exc()
                        self.error.emit(f"Failed to deserialize results: {e}")
                        return
                    
                    # 發送與 SAM3Worker 相容的格式
                    # 包含 server_video_path 供後續 propagate 使用
                    logger.info("Emitting finished signal...")
                    self.finished.emit({
                        "results": deserialized,
                        "server_video_path": getattr(self, '_server_video_path', None),
                    })
                    logger.info("Finished signal emitted")
                    return
                
                elif event_type == "failed":
                    error_msg = event.get("error", "Unknown error")
                    self.error.emit(error_msg)
                    return
                
                elif event_type == "cancelled":
                    self.cancelled.emit()
                    return
                
                elif event_type == "heartbeat":
                    # 心跳，忽略
                    pass
                
                elif event_type == "pong":
                    # ping 回應，忽略
                    pass
                
        finally:
            if self._ws:
                try:
                    self._ws.close()
                except:
                    pass
                self._ws = None
    
    def _run_with_polling(self):
        """使用輪詢監聽進度（WebSocket 不可用時的 fallback）"""
        import time
        
        logger.info("Using polling mode")
        
        while True:
            if self._check_cancelled():
                self.cancelled.emit()
                return
            
            try:
                job = self.client.get_job(self._task_id)
                
                self.progress.emit(job.progress, job.message)
                
                if job.status == JobStatus.WAITING_CONFIRM:
                    # 等待確認
                    num_objects = 0
                    data = {}
                    
                    # 嘗試從之前的結果取得物件數
                    if job.result:
                        data = job.result
                        num_objects = data.get("num_objects", 0)
                    
                    self.ready_to_propagate.emit(num_objects, data)
                    
                    # 等待用戶確認
                    self._confirm_event.wait()
                    
                    if self._cancelled:
                        self.cancelled.emit()
                        return
                
                elif job.status == JobStatus.COMPLETED:
                    self.progress.emit(100, "Completed!")
                    
                    # 反序列化結果
                    raw_result = job.result or {}
                    deserialized = deserialize_results(raw_result)
                    
                    # 發送與 SAM3Worker 相容的格式
                    self.finished.emit({"results": deserialized})
                    return
                
                elif job.status == JobStatus.FAILED:
                    self.error.emit(job.error or "Unknown error")
                    return
                
                elif job.status == JobStatus.CANCELLED:
                    self.cancelled.emit()
                    return
                
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Polling error: {e}")
                time.sleep(1.0)


# =============================================================================
# Batch Detection Worker
# =============================================================================

class ServerBatchDetectionWorker(BaseServerWorker):
    """
    批量圖片偵測 Worker
    
    信號：
        progress(int, int, str)         當前進度, 總數, 訊息
        image_result(int, dict)         單張圖片結果（索引, 結果）
        finished(dict)                  完成
        error(str)                      錯誤
        cancelled()                     取消
    """
    
    # 信號（覆寫）
    progress = pyqtSignal(int, int, str)  # (當前, 總數, 訊息)
    image_result = pyqtSignal(int, dict)  # (索引, 結果)
    finished = pyqtSignal(dict)
    
    def __init__(
        self,
        server_url: str,
        image_paths: List[str],
        prompt: str,
    ):
        super().__init__(server_url)
        self.image_paths = image_paths
        self.prompt = prompt
    
    def run(self):
        """執行任務"""
        try:
            total = len(self.image_paths)
            
            # Step 1: 建立任務
            self.progress.emit(0, total, "Creating batch detection job...")
            
            self._task_id = self.client.create_batch_detection_job(
                image_paths=self.image_paths,
                prompt=self.prompt,
            )
            
            logger.info(f"Created batch job: {self._task_id}")
            
            if self._check_cancelled():
                self.cancelled.emit()
                return
            
            # Step 2: 監聽進度
            if WEBSOCKET_AVAILABLE:
                self._run_with_websocket(total)
            else:
                self._run_with_polling(total)
            
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"Worker error: {error_msg}")
            self.error.emit(error_msg)
    
    def _run_with_websocket(self, total: int):
        """使用 WebSocket"""
        try:
            ws = self.client.connect_job_websocket(self._task_id)
            self._ws = ws
        except Exception as e:
            logger.warning(f"WebSocket failed: {e}")
            self._run_with_polling(total)
            return
        
        try:
            for event in self.client.iter_job_events(ws):
                if self._check_cancelled():
                    self.cancelled.emit()
                    return
                
                event_type = event.get("type")
                
                if event_type == "progress":
                    progress = event.get("progress", 0)
                    message = event.get("message", "")
                    current = int(progress * total / 100)
                    self.progress.emit(current, total, message)
                
                elif event_type == "intermediate":
                    # 單張結果
                    data = event.get("data", {})
                    index = data.get("index", 0)
                    result = data.get("result", {})
                    self.image_result.emit(index, result)
                
                elif event_type == "completed":
                    result = event.get("result", {})
                    self.progress.emit(total, total, "Done!")
                    self.finished.emit(result)
                    return
                
                elif event_type == "failed":
                    self.error.emit(event.get("error", "Unknown error"))
                    return
                
                elif event_type == "cancelled":
                    self.cancelled.emit()
                    return
                
        finally:
            if self._ws:
                try:
                    self._ws.close()
                except:
                    pass
    
    def _run_with_polling(self, total: int):
        """使用輪詢"""
        import time
        
        while True:
            if self._check_cancelled():
                self.cancelled.emit()
                return
            
            try:
                job = self.client.get_job(self._task_id)
                
                current = int(job.progress * total / 100)
                self.progress.emit(current, total, job.message)
                
                if job.status == JobStatus.COMPLETED:
                    self.progress.emit(total, total, "Done!")
                    self.finished.emit(job.result or {})
                    return
                
                elif job.status == JobStatus.FAILED:
                    self.error.emit(job.error or "Unknown error")
                    return
                
                elif job.status == JobStatus.CANCELLED:
                    self.cancelled.emit()
                    return
                
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Polling error: {e}")
                time.sleep(1.0)


# =============================================================================
# Propagate Worker
# =============================================================================

class ServerPropagateWorker(BaseServerWorker):
    """
    傳播 Worker
    
    信號：
        progress(int, str)    進度
        finished(dict)        完成
        error(str)            錯誤
        cancelled()           取消
    """
    
    finished = pyqtSignal(dict)
    
    def __init__(
        self,
        server_url: str,
        video_path: str,
        start_frame: int,
        mask,  # np.ndarray
        points,  # np.ndarray
        labels,  # np.ndarray
        obj_id: int = 0,
    ):
        super().__init__(server_url)
        self.video_path = video_path
        self.start_frame = start_frame
        self.mask = mask
        self.points = points
        self.labels = labels
        self.obj_id = obj_id
    
    def run(self):
        """執行任務"""
        try:
            # Step 1: 建立任務
            self.progress.emit(5, "Creating propagate job...")
            
            self._task_id = self.client.create_propagate_job(
                video_path=self.video_path,
                start_frame=self.start_frame,
                mask=self.mask,
                points=self.points,
                labels=self.labels,
                obj_id=self.obj_id,
            )
            
            logger.info(f"Created propagate job: {self._task_id}")
            
            if self._check_cancelled():
                self.cancelled.emit()
                return
            
            # Step 2: 監聽進度
            if WEBSOCKET_AVAILABLE:
                self._run_with_websocket()
            else:
                self._run_with_polling()
            
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}\n{traceback.format_exc()}"
            logger.error(f"Worker error: {error_msg}")
            self.error.emit(error_msg)
    
    def _run_with_websocket(self):
        """使用 WebSocket"""
        try:
            ws = self.client.connect_job_websocket(self._task_id)
            self._ws = ws
        except Exception as e:
            logger.warning(f"WebSocket failed: {e}")
            self._run_with_polling()
            return
        
        try:
            for event in self.client.iter_job_events(ws):
                if self._check_cancelled():
                    self.cancelled.emit()
                    return
                
                event_type = event.get("type")
                
                if event_type == "progress":
                    progress = event.get("progress", 0)
                    message = event.get("message", "")
                    self.progress.emit(progress, message)
                
                elif event_type == "completed":
                    result = event.get("result", {})
                    self.progress.emit(100, "Propagation completed!")
                    self.finished.emit(result)
                    return
                
                elif event_type == "failed":
                    self.error.emit(event.get("error", "Unknown error"))
                    return
                
                elif event_type == "cancelled":
                    self.cancelled.emit()
                    return
                
        finally:
            if self._ws:
                try:
                    self._ws.close()
                except:
                    pass
    
    def _run_with_polling(self):
        """使用輪詢"""
        import time
        
        while True:
            if self._check_cancelled():
                self.cancelled.emit()
                return
            
            try:
                job = self.client.get_job(self._task_id)
                
                self.progress.emit(job.progress, job.message)
                
                if job.status == JobStatus.COMPLETED:
                    self.progress.emit(100, "Propagation completed!")
                    self.finished.emit(job.result or {})
                    return
                
                elif job.status == JobStatus.FAILED:
                    self.error.emit(job.error or "Unknown error")
                    return
                
                elif job.status == JobStatus.CANCELLED:
                    self.cancelled.emit()
                    return
                
                time.sleep(1.0)
                
            except Exception as e:
                logger.error(f"Polling error: {e}")
                time.sleep(1.0)


# =============================================================================
# 測試
# =============================================================================

if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication
    
    app = QApplication(sys.argv)
    
    print("=" * 50)
    print("Server Worker Test")
    print("=" * 50)
    
    # 測試 VideoDetectionWorker
    worker = ServerVideoDetectionWorker(
        server_url="http://localhost:8000",
        video_path="/app/data/video/test.mp4",
        prompt="ship"
    )
    
    def on_progress(progress, message):
        print(f"[{progress:3d}%] {message}")
    
    def on_ready(num_objects, data):
        print(f"Ready! Found {num_objects} objects")
        print("Confirming in 2 seconds...")
        import time
        time.sleep(2)
        worker.confirm()
    
    def on_finished(result):
        print(f"Finished! Result: {result}")
        app.quit()
    
    def on_error(error):
        print(f"Error: {error}")
        app.quit()
    
    def on_cancelled():
        print("Cancelled!")
        app.quit()
    
    worker.progress.connect(on_progress)
    worker.ready_to_propagate.connect(on_ready)
    worker.finished.connect(on_finished)
    worker.error.connect(on_error)
    worker.cancelled.connect(on_cancelled)
    
    print("Starting worker...")
    worker.start()
    
    sys.exit(app.exec())
