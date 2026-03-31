"""
STAMP Task Manager
==================

任務佇列管理系統，支援：
- 任務建立、執行、取消
- 進度回報（透過 callback）
- 用戶確認機制（偵測完等確認才傳播）
- 多任務排隊（GPU 一次只跑一個）

架構：
┌─────────────────────────────────────────────────────────┐
│                    TaskManager                          │
│  ┌─────────┐    ┌─────────┐    ┌─────────┐             │
│  │ Queue   │───→│ Worker  │───→│ SAM3    │             │
│  │ (待執行) │    │ (執行中) │    │ Engine  │             │
│  └─────────┘    └─────────┘    └─────────┘             │
│       ↑              │                                  │
│       │              ↓                                  │
│  ┌─────────┐    ┌─────────┐                            │
│  │ Tasks   │←───│ Events  │───→ WebSocket 推送         │
│  │ (狀態)  │    │ (進度)  │                            │
│  └─────────┘    └─────────┘                            │
└─────────────────────────────────────────────────────────┘

使用方式：
    manager = TaskManager()
    
    # 建立任務
    task_id = manager.create_task(
        task_type=TaskType.VIDEO_DETECTION,
        params={"video_path": "/path/to/video.mp4", "prompt": "ship"}
    )
    
    # 訂閱進度更新
    manager.subscribe(task_id, lambda event: print(event))
    
    # 任務會自動開始執行（如果沒有其他任務在跑）
    
    # 等待確認
    # ... WebSocket 收到 "waiting_confirm" 事件
    # ... 用戶點擊確認
    manager.confirm_task(task_id)
    
    # 取消任務
    manager.cancel_task(task_id)
"""

import asyncio
import threading
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set
from queue import Queue, Empty
import traceback

import numpy as np
from loguru import logger


# =============================================================================
# 列舉類型
# =============================================================================

class TaskType(Enum):
    """任務類型"""
    VIDEO_DETECTION = "video_detection"    # 影片：偵測 + 傳播
    IMAGE_DETECTION = "image_detection"    # 單張圖片偵測
    BATCH_DETECTION = "batch_detection"    # 批量圖片偵測
    PROPAGATE = "propagate"                # 單獨傳播（refinement 後）
    REFINE = "refine"                      # Mask 修正


class TaskStatus(Enum):
    """任務狀態"""
    PENDING = "pending"                    # 在佇列中等待
    RUNNING = "running"                    # 執行中
    WAITING_CONFIRM = "waiting_confirm"    # 等待用戶確認
    COMPLETED = "completed"                # 完成
    FAILED = "failed"                      # 失敗
    CANCELLED = "cancelled"                # 已取消


class EventType(Enum):
    """事件類型（推送給 WebSocket）"""
    PROGRESS = "progress"                  # 進度更新
    WAITING_CONFIRM = "waiting_confirm"    # 等待確認
    INTERMEDIATE_RESULT = "intermediate"   # 中間結果（例如初步偵測結果）
    COMPLETED = "completed"                # 完成
    FAILED = "failed"                      # 失敗
    CANCELLED = "cancelled"                # 已取消


# =============================================================================
# 資料類別
# =============================================================================

@dataclass
class TaskEvent:
    """
    任務事件（推送給 WebSocket）
    
    事件類型：
    - progress: {"type": "progress", "value": 45, "message": "Detecting..."}
    - waiting_confirm: {"type": "waiting_confirm", "data": {...初步結果...}}
    - completed: {"type": "completed", "result": {...}}
    - failed: {"type": "failed", "error": "..."}
    - cancelled: {"type": "cancelled"}
    """
    event_type: EventType
    task_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    # Progress 事件
    progress_value: Optional[int] = None      # 0-100
    progress_message: Optional[str] = None
    
    # 資料（中間結果、最終結果）
    data: Optional[Dict[str, Any]] = None
    
    # 錯誤訊息
    error_message: Optional[str] = None
    
    def to_dict(self) -> dict:
        """轉換成 dict（給 JSON 序列化）"""
        result = {
            "type": self.event_type.value,
            "task_id": self.task_id,
            "timestamp": self.timestamp.isoformat(),
        }
        
        if self.progress_value is not None:
            result["progress"] = self.progress_value
        if self.progress_message is not None:
            result["message"] = self.progress_message
        if self.data is not None:
            result["data"] = self.data
        if self.error_message is not None:
            result["error"] = self.error_message
            
        return result


@dataclass
class Task:
    """
    任務資料
    """
    task_id: str
    task_type: TaskType
    params: Dict[str, Any]
    
    # 狀態
    status: TaskStatus = TaskStatus.PENDING
    progress: int = 0
    message: str = ""
    
    # 結果
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    
    # 時間戳
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # 內部狀態
    _confirm_event: threading.Event = field(default_factory=threading.Event)
    _cancel_requested: bool = False
    
    def to_dict(self) -> dict:
        """轉換成 dict（給 API 回傳）"""
        # 過濾 params 中的 numpy array（無法 JSON 序列化）
        safe_params = {}
        for key, value in self.params.items():
            if isinstance(value, np.ndarray):
                # 只保存形狀資訊
                safe_params[key] = f"<ndarray shape={value.shape}>"
            elif isinstance(value, (str, int, float, bool, list, dict, type(None))):
                safe_params[key] = value
            else:
                safe_params[key] = str(value)
        
        return {
            "task_id": self.task_id,
            "task_type": self.task_type.value,
            "status": self.status.value,
            "progress": self.progress,
            "message": self.message,
            "params": safe_params,
            "result": self.result,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


# =============================================================================
# 任務管理器
# =============================================================================

class TaskManager:
    """
    任務管理器
    
    負責：
    - 管理任務佇列
    - 執行任務（一次一個，因為 GPU 只有一張）
    - 發送進度事件
    - 處理確認和取消
    """
    
    def __init__(self, sam3_engine=None):
        """
        初始化任務管理器
        
        Args:
            sam3_engine: SAM3Engine 實例（如果沒提供，會在第一次執行時載入）
        """
        # 任務儲存
        self._tasks: Dict[str, Task] = {}
        self._task_lock = threading.Lock()
        
        # 事件訂閱者
        self._subscribers: Dict[str, List[Callable[[TaskEvent], None]]] = {}
        self._subscriber_lock = threading.Lock()
        
        # 任務佇列
        self._task_queue: Queue[str] = Queue()
        
        # SAM3 Engine（懶載入）
        self._sam3_engine = sam3_engine
        self._engine_lock = threading.Lock()
        
        # Worker 執行緒
        self._worker_thread: Optional[threading.Thread] = None
        self._shutdown_requested = False
        
        # 啟動 worker
        self._start_worker()
        
        logger.info("TaskManager initialized")
    
    # =========================================================================
    # 公開 API
    # =========================================================================
    
    def create_task(
        self,
        task_type: TaskType,
        params: Dict[str, Any]
    ) -> str:
        """
        建立新任務
        
        Args:
            task_type: 任務類型
            params: 任務參數
            
        Returns:
            task_id
        """
        task_id = str(uuid.uuid4())[:8]
        
        task = Task(
            task_id=task_id,
            task_type=task_type,
            params=params,
        )
        
        with self._task_lock:
            self._tasks[task_id] = task
        
        # 加入佇列
        self._task_queue.put(task_id)
        
        logger.info(f"Task created: {task_id} ({task_type.value})")
        return task_id
    
    def get_task(self, task_id: str) -> Optional[Task]:
        """取得任務資訊"""
        with self._task_lock:
            return self._tasks.get(task_id)
    
    def get_all_tasks(self) -> List[Task]:
        """取得所有任務"""
        with self._task_lock:
            return list(self._tasks.values())
    
    def cancel_task(self, task_id: str) -> bool:
        """
        取消任務
        
        Returns:
            是否成功取消
        """
        task = self.get_task(task_id)
        if task is None:
            return False
        
        # 只能取消還沒完成的任務
        if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
            return False
        
        # 標記取消
        task._cancel_requested = True
        
        # 如果在等待確認，喚醒它
        task._confirm_event.set()
        
        logger.info(f"Task cancel requested: {task_id}")
        return True
    
    def confirm_task(self, task_id: str) -> bool:
        """
        確認任務（用戶確認繼續執行）
        
        Returns:
            是否成功確認
        """
        task = self.get_task(task_id)
        if task is None:
            return False
        
        if task.status != TaskStatus.WAITING_CONFIRM:
            return False
        
        # 喚醒等待中的任務
        task._confirm_event.set()
        
        logger.info(f"Task confirmed: {task_id}")
        return True
    
    def subscribe(
        self,
        task_id: str,
        callback: Callable[[TaskEvent], None]
    ) -> Callable[[], None]:
        """
        訂閱任務事件
        
        Args:
            task_id: 任務 ID
            callback: 收到事件時呼叫的函數
            
        Returns:
            取消訂閱的函數
        """
        with self._subscriber_lock:
            if task_id not in self._subscribers:
                self._subscribers[task_id] = []
            self._subscribers[task_id].append(callback)
        
        # 返回取消訂閱的函數
        def unsubscribe():
            with self._subscriber_lock:
                if task_id in self._subscribers:
                    try:
                        self._subscribers[task_id].remove(callback)
                    except ValueError:
                        pass
        
        return unsubscribe
    
    def shutdown(self):
        """關閉任務管理器"""
        self._shutdown_requested = True
        
        # 放一個 None 進佇列讓 worker 醒來
        self._task_queue.put(None)
        
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=5)
        
        logger.info("TaskManager shutdown")
    
    # =========================================================================
    # 內部方法
    # =========================================================================
    
    def _start_worker(self):
        """啟動背景 worker"""
        self._worker_thread = threading.Thread(
            target=self._worker_loop,
            daemon=True,
            name="TaskManager-Worker"
        )
        self._worker_thread.start()
    
    def _worker_loop(self):
        """Worker 主迴圈"""
        logger.info("TaskManager worker started")
        
        while not self._shutdown_requested:
            try:
                # 等待任務（timeout 讓 shutdown 可以生效）
                try:
                    task_id = self._task_queue.get(timeout=1.0)
                except Empty:
                    continue
                
                if task_id is None:  # shutdown 信號
                    break
                
                # 執行任務
                self._execute_task(task_id)
                
            except Exception as e:
                logger.error(f"Worker loop error: {e}\n{traceback.format_exc()}")
        
        logger.info("TaskManager worker stopped")
    
    def _execute_task(self, task_id: str):
        """執行單一任務"""
        task = self.get_task(task_id)
        if task is None:
            return
        
        # 檢查是否已取消
        if task._cancel_requested:
            self._complete_task(task, TaskStatus.CANCELLED)
            return
        
        # 更新狀態
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        logger.info(f"Task started: {task_id} ({task.task_type.value})")
        
        try:
            # 根據任務類型執行
            if task.task_type == TaskType.VIDEO_DETECTION:
                self._run_video_detection(task)
            elif task.task_type == TaskType.IMAGE_DETECTION:
                self._run_image_detection(task)
            elif task.task_type == TaskType.BATCH_DETECTION:
                self._run_batch_detection(task)
            elif task.task_type == TaskType.PROPAGATE:
                self._run_propagate(task)
            elif task.task_type == TaskType.REFINE:
                self._run_refine(task)
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
        except Exception as e:
            error_msg = f"{type(e).__name__}: {str(e)}"
            logger.error(f"Task failed: {task_id} - {error_msg}\n{traceback.format_exc()}")
            self._complete_task(task, TaskStatus.FAILED, error=error_msg)
    
    def _complete_task(
        self,
        task: Task,
        status: TaskStatus,
        result: Optional[Dict] = None,
        error: Optional[str] = None
    ):
        """完成任務（成功、失敗、或取消）"""
        task.status = status
        task.completed_at = datetime.now()
        task.result = result
        task.error = error
        
        # 發送事件
        if status == TaskStatus.COMPLETED:
            self._emit_event(TaskEvent(
                event_type=EventType.COMPLETED,
                task_id=task.task_id,
                data=result,
            ))
        elif status == TaskStatus.FAILED:
            self._emit_event(TaskEvent(
                event_type=EventType.FAILED,
                task_id=task.task_id,
                error_message=error,
            ))
        elif status == TaskStatus.CANCELLED:
            self._emit_event(TaskEvent(
                event_type=EventType.CANCELLED,
                task_id=task.task_id,
            ))
        
        logger.info(f"Task completed: {task.task_id} ({status.value})")
    
    def _emit_event(self, event: TaskEvent):
        """發送事件給訂閱者"""
        with self._subscriber_lock:
            subscribers = self._subscribers.get(event.task_id, [])
        
        for callback in subscribers:
            try:
                callback(event)
            except Exception as e:
                logger.error(f"Event callback error: {e}")
    
    def _emit_progress(self, task: Task, progress: int, message: str):
        """發送進度更新"""
        task.progress = progress
        task.message = message
        
        self._emit_event(TaskEvent(
            event_type=EventType.PROGRESS,
            task_id=task.task_id,
            progress_value=progress,
            progress_message=message,
        ))
    
    def _wait_for_confirm(self, task: Task, data: Optional[Dict] = None) -> bool:
        """
        等待用戶確認
        
        Returns:
            True = 確認，False = 取消
        """
        task.status = TaskStatus.WAITING_CONFIRM
        task._confirm_event.clear()
        
        # 發送等待確認事件
        self._emit_event(TaskEvent(
            event_type=EventType.WAITING_CONFIRM,
            task_id=task.task_id,
            data=data,
        ))
        
        logger.info(f"Task waiting for confirm: {task.task_id}")
        
        # 等待確認或取消
        task._confirm_event.wait()
        
        # 檢查是確認還是取消
        if task._cancel_requested:
            return False
        
        task.status = TaskStatus.RUNNING
        return True
    
    def _get_engine(self):
        """取得 SAM3 Engine（懶載入）"""
        with self._engine_lock:
            if self._sam3_engine is None:
                logger.info("Loading SAM3 engine...")
                from src.core.sam3_engine import SAM3Engine
                self._sam3_engine = SAM3Engine(mode="auto")
                logger.info("SAM3 engine loaded")
            return self._sam3_engine
    
    # =========================================================================
    # 任務執行邏輯
    # =========================================================================
    
    def _run_video_detection(self, task: Task):
        """
        執行影片偵測任務
        
        流程：
        1. 載入影片
        2. 偵測第一幀
        3. 等待用戶確認
        4. 傳播到所有幀
        5. 回傳結果
        """
        params = task.params
        video_path = params["video_path"]
        prompt = params["prompt"]
        
        engine = self._get_engine()
        
        # Step 1: 載入影片
        self._emit_progress(task, 10, "Loading video...")
        
        if task._cancel_requested:
            self._complete_task(task, TaskStatus.CANCELLED)
            return
        
        # Step 2: 開始 session
        self._emit_progress(task, 20, "Starting video session...")
        session_id = engine.start_video_session(video_path)
        
        try:
            if task._cancel_requested:
                engine.close_session(session_id)
                self._complete_task(task, TaskStatus.CANCELLED)
                return
            
            # Step 3: 偵測第一幀
            self._emit_progress(task, 30, f"Detecting objects (prompt: {prompt})...")
            frame_result = engine.add_prompt(session_id, 0, prompt)
            
            # 解析結果
            if hasattr(frame_result, 'detections'):
                num_objects = len(frame_result.detections)
                initial_result = {
                    "num_objects": num_objects,
                    "detections": [
                        {
                            "obj_id": d.obj_id,
                            "score": d.score,
                            "bbox": d.box.tolist() if hasattr(d.box, 'tolist') else list(d.box),
                        }
                        for d in frame_result.detections
                    ]
                }
            else:
                num_objects = 0
                initial_result = {"num_objects": 0, "detections": []}
            
            self._emit_progress(task, 45, f"Found {num_objects} objects")
            
            if task._cancel_requested:
                engine.close_session(session_id)
                self._complete_task(task, TaskStatus.CANCELLED)
                return
            
            # Step 4: 等待用戶確認
            confirmed = self._wait_for_confirm(task, data=initial_result)
            
            if not confirmed:
                engine.close_session(session_id)
                self._complete_task(task, TaskStatus.CANCELLED)
                return
            
            # Step 5: 傳播
            self._emit_progress(task, 50, "Propagating masks to all frames...")
            
            all_results = engine.propagate(session_id)
            
            self._emit_progress(task, 90, f"Processed {len(all_results)} frames")
            
            # Step 6: 關閉 session
            engine.close_session(session_id)
            
            # 清理 GPU 記憶體
            import torch
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Step 7: 轉換結果
            self._emit_progress(task, 95, "Preparing results...")
            
            # 把 FrameResult 轉成可序列化的格式
            serializable_results = self._serialize_video_results(all_results)
            
            # 完成
            self._complete_task(task, TaskStatus.COMPLETED, result={
                "total_frames": len(all_results),
                "results": serializable_results,  # 與 SAM3Worker 相容
            })
            
        except Exception as e:
            # 確保 session 被關閉
            try:
                engine.close_session(session_id)
            except:
                pass
            raise
    
    def _run_image_detection(self, task: Task):
        """
        執行單張圖片偵測
        """
        params = task.params
        image = params["image"]  # numpy array
        prompt = params["prompt"]
        
        engine = self._get_engine()
        
        self._emit_progress(task, 30, f"Detecting '{prompt}'...")
        
        if task._cancel_requested:
            self._complete_task(task, TaskStatus.CANCELLED)
            return
        
        # 執行偵測
        frame_result = engine.detect_image(image, prompt)
        
        self._emit_progress(task, 90, f"Found {frame_result.num_objects} objects")
        
        # 轉換結果
        result = self._serialize_frame_result(frame_result)
        
        self._complete_task(task, TaskStatus.COMPLETED, result=result)
    
    def _run_batch_detection(self, task: Task):
        """
        執行批量圖片偵測
        """
        params = task.params
        image_paths = params["image_paths"]  # List[str]
        prompt = params["prompt"]
        
        engine = self._get_engine()
        total = len(image_paths)
        results = {}
        
        for idx, image_path in enumerate(image_paths):
            if task._cancel_requested:
                self._complete_task(task, TaskStatus.CANCELLED)
                return
            
            progress = int(10 + (idx / total) * 80)
            self._emit_progress(task, progress, f"Processing {idx + 1}/{total}...")
            
            # 讀取圖片
            import cv2
            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning(f"Cannot read image: {image_path}")
                continue
            
            # 偵測
            frame_result = engine.detect_image(image, prompt)
            results[idx] = self._serialize_frame_result(frame_result)
            
            # 發送中間結果
            self._emit_event(TaskEvent(
                event_type=EventType.INTERMEDIATE_RESULT,
                task_id=task.task_id,
                data={"index": idx, "result": results[idx]},
            ))
        
        self._complete_task(task, TaskStatus.COMPLETED, result={
            "total": total,
            "results": results,
        })
    
    def _run_propagate(self, task: Task):
        """
        執行 mask 傳播
        """
        import torch
        import gc
        from pycocotools import mask as mask_utils
        
        # 強制徹底清理 GPU 記憶體
        logger.info("Cleaning GPU memory before propagate...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            # 再次清理
            gc.collect()
            torch.cuda.empty_cache()
            
            # 記錄當前記憶體狀態
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            logger.info(f"GPU memory: {allocated:.2f} GiB allocated, {reserved:.2f} GiB reserved")
        
        params = task.params
        video_path = params["video_path"]
        start_frame = params["start_frame"]
        mask = params["mask"]  # numpy array
        points = params["points"]  # numpy array
        labels = params["labels"]  # numpy array
        obj_id = params.get("obj_id", 0)
        
        engine = self._get_engine()
        
        self._emit_progress(task, 10, "Starting propagation...")
        
        if task._cancel_requested:
            self._complete_task(task, TaskStatus.CANCELLED)
            return
        
        # 執行傳播
        def progress_callback(current, total):
            progress = int(10 + (current / total) * 80)
            self._emit_progress(task, progress, f"Frame {current}/{total}")
        
        results = engine.propagate_mask(
            video_path=video_path,
            start_frame=start_frame,
            mask=mask,
            points=points,
            labels=labels,
            obj_id=obj_id,
            progress_callback=progress_callback,
        )
        
        # 把 mask 用 RLE 編碼
        self._emit_progress(task, 95, "Encoding results...")
        encoded_results = {}
        for frame_idx, frame_mask in results.items():
            mask_fortran = np.asfortranarray(frame_mask.astype(np.uint8))
            rle = mask_utils.encode(mask_fortran)
            encoded_results[str(frame_idx)] = {
                "counts": rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"],
                "size": rle["size"],
            }
        
        # 完成
        self._complete_task(task, TaskStatus.COMPLETED, result={
            "num_frames": len(results),
            "frame_indices": list(results.keys()),
            "masks": encoded_results,  # RLE 編碼的 mask
        })
    
    def _run_refine(self, task: Task):
        """
        執行 mask 修正
        """
        params = task.params
        image = params["image"]  # numpy array
        points = params["points"]  # numpy array
        labels = params["labels"]  # numpy array
        mask_input = params.get("mask_input")  # optional numpy array
        
        engine = self._get_engine()
        
        self._emit_progress(task, 30, "Refining mask...")
        
        if task._cancel_requested:
            self._complete_task(task, TaskStatus.CANCELLED)
            return
        
        # 執行修正
        new_mask = engine.refine_mask(
            image=image,
            points=points,
            labels=labels,
            mask_input=mask_input,
        )
        
        # 完成（mask 需要另外處理）
        self._complete_task(task, TaskStatus.COMPLETED, result={
            "mask_shape": list(new_mask.shape),
            # mask 本身透過另一個 API 取得
        })
    
    # =========================================================================
    # 結果序列化
    # =========================================================================
    
    def _serialize_frame_result(self, frame_result) -> dict:
        """把 FrameResult 轉成可 JSON 序列化的格式（使用 RLE 編碼 mask）"""
        from pycocotools import mask as mask_utils
        
        detections_data = []
        for d in frame_result.detections:
            det_dict = {
                "obj_id": d.obj_id,
                "score": float(d.score),
                "bbox": d.box.tolist() if hasattr(d.box, 'tolist') else list(d.box),
                "bbox_xyxy": d.box_xyxy.tolist() if hasattr(d.box_xyxy, 'tolist') else list(d.box_xyxy),
            }
            
            # RLE 編碼 mask
            if d.mask is not None:
                # 確保 mask 是 Fortran order（pycocotools 要求）
                mask_fortran = np.asfortranarray(d.mask.astype(np.uint8))
                rle = mask_utils.encode(mask_fortran)
                # RLE 的 counts 是 bytes，需要轉成 string
                det_dict["mask_rle"] = {
                    "counts": rle["counts"].decode("utf-8") if isinstance(rle["counts"], bytes) else rle["counts"],
                    "size": rle["size"],
                }
            
            detections_data.append(det_dict)
        
        return {
            "frame_index": frame_result.frame_index,
            "num_objects": frame_result.num_objects,
            "detections": detections_data,
        }
    
    def _serialize_video_results(self, all_results: dict) -> dict:
        """把影片所有幀的結果序列化"""
        return {
            str(frame_idx): self._serialize_frame_result(frame_result)
            for frame_idx, frame_result in all_results.items()
        }


# =============================================================================
# 單例模式（全局只有一個 TaskManager）
# =============================================================================

_task_manager: Optional[TaskManager] = None
_task_manager_lock = threading.Lock()


def get_task_manager(sam3_engine=None) -> TaskManager:
    """
    取得全局 TaskManager 實例
    
    Args:
        sam3_engine: 可選，SAM3Engine 實例。只在第一次呼叫時有效。
        
    Returns:
        TaskManager 實例
    """
    global _task_manager
    
    with _task_manager_lock:
        if _task_manager is None:
            _task_manager = TaskManager(sam3_engine=sam3_engine)
        return _task_manager


def init_task_manager(sam3_engine=None) -> TaskManager:
    """
    初始化全局 TaskManager（用於 server 啟動時）
    
    Args:
        sam3_engine: SAM3Engine 實例
        
    Returns:
        TaskManager 實例
    """
    global _task_manager
    
    with _task_manager_lock:
        if _task_manager is not None:
            logger.warning("TaskManager already initialized, returning existing instance")
            return _task_manager
        
        _task_manager = TaskManager(sam3_engine=sam3_engine)
        return _task_manager


def shutdown_task_manager():
    """關閉全局 TaskManager"""
    global _task_manager
    
    with _task_manager_lock:
        if _task_manager is not None:
            _task_manager.shutdown()
            _task_manager = None


# =============================================================================
# 測試
# =============================================================================

if __name__ == "__main__":
    import time
    
    # 設定 logging
    import logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 50)
    print("TaskManager Test")
    print("=" * 50)
    
    manager = TaskManager()
    
    # 測試訂閱
    def on_event(event: TaskEvent):
        print(f"Event: {event.to_dict()}")
    
    # 建立測試任務（會失敗因為沒有真正的影片）
    task_id = manager.create_task(
        task_type=TaskType.IMAGE_DETECTION,
        params={
            "image": np.zeros((100, 100, 3), dtype=np.uint8),
            "prompt": "test",
        }
    )
    
    print(f"Created task: {task_id}")
    
    # 訂閱
    unsubscribe = manager.subscribe(task_id, on_event)
    
    # 等待任務完成
    time.sleep(5)
    
    # 查看任務狀態
    task = manager.get_task(task_id)
    print(f"Task status: {task.status.value}")
    
    # 清理
    manager.shutdown()
    
    print("\n" + "=" * 50)
    print("Test complete!")
    print("=" * 50)
