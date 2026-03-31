"""
STAMP API Client
================

HTTP + WebSocket 客戶端，提供與 SAM3Engine 相同的介面。
讓 GUI 可以透過網路呼叫遠端 Server 的 SAM3。

功能：
- 同步 API：detect_image, refine_mask（快速操作）
- 異步 Jobs API：video_detection, batch_detection（長時間操作）
- WebSocket：即時進度監聽

用法：
    # === 方式 1：同步 API（簡單操作）===
    client = StampAPIClient("http://192.168.1.100:8000")
    result = client.detect_image(image, "dolphin")
    
    # === 方式 2：異步 Jobs API（長時間操作）===
    client = StampAPIClient("http://192.168.1.100:8000")
    
    # 建立任務
    task_id = client.create_video_detection_job(video_path, prompt)
    
    # 監聽進度（會阻塞直到完成）
    result = client.wait_for_job(
        task_id,
        on_progress=lambda p, m: print(f"{p}% - {m}"),
        on_confirm=lambda data: True,  # 自動確認
    )
    
    # === 方式 3：手動 WebSocket 控制 ===
    client = StampAPIClient("http://192.168.1.100:8000")
    
    task_id = client.create_video_detection_job(video_path, prompt)
    
    # 連接 WebSocket
    ws = client.connect_job_websocket(task_id)
    
    for event in client.iter_job_events(ws):
        if event["type"] == "progress":
            print(f"Progress: {event['progress']}%")
        elif event["type"] == "waiting_confirm":
            # 顯示給用戶，等待確認
            client.confirm_job(task_id)
        elif event["type"] == "completed":
            result = event["result"]
            break
    
    ws.close()
"""

import base64
import io
import json
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Union

import cv2
import numpy as np
from PIL import Image
import requests
from loguru import logger

# WebSocket 支援
try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    logger.warning("websocket-client not installed. Install with: pip install websocket-client")


# =============================================================================
# 資料類別（與 sam3_engine.py 相同）
# =============================================================================

@dataclass
class Detection:
    """單一物件偵測結果"""
    obj_id: int
    mask: np.ndarray          # 二值 mask (H, W)
    box: np.ndarray           # Bounding box [x, y, w, h]
    score: float              # 信心分數 (0-1)
    
    @property
    def box_xyxy(self) -> np.ndarray:
        """轉換 xywh 成 xyxy 格式"""
        x, y, w, h = self.box
        return np.array([x, y, x + w, y + h])


@dataclass
class FrameResult:
    """單一幀的結果"""
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


@dataclass
class VideoSessionInfo:
    """影片 Session 資訊"""
    session_id: str
    video_path: str
    total_frames: int
    width: int
    height: int
    fps: float


class JobStatus(Enum):
    """任務狀態"""
    PENDING = "pending"
    RUNNING = "running"
    WAITING_CONFIRM = "waiting_confirm"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class JobInfo:
    """任務資訊"""
    task_id: str
    task_type: str
    status: JobStatus
    progress: int
    message: str
    result: Optional[Dict] = None
    error: Optional[str] = None


# =============================================================================
# 編解碼工具
# =============================================================================

def encode_image_to_base64(image: np.ndarray) -> str:
    """把 numpy 圖片轉成 base64 字串"""
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    pil_image = Image.fromarray(image)
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=90)
    buffer.seek(0)
    
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def encode_mask_to_base64(mask: np.ndarray) -> str:
    """把 numpy mask 轉成 base64 字串"""
    mask_uint8 = (mask.astype(np.uint8) * 255)
    pil_image = Image.fromarray(mask_uint8, mode='L')
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def decode_base64_to_mask(base64_str: str) -> np.ndarray:
    """把 base64 字串轉回 numpy mask"""
    img_bytes = base64.b64decode(base64_str)
    pil_image = Image.open(io.BytesIO(img_bytes))
    mask = np.array(pil_image)
    
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    
    return mask > 127


def decode_base64_to_image(base64_str: str) -> np.ndarray:
    """把 base64 字串轉回 numpy 圖片"""
    img_bytes = base64.b64decode(base64_str)
    pil_image = Image.open(io.BytesIO(img_bytes))
    image = np.array(pil_image)
    
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image


# =============================================================================
# API Client
# =============================================================================

class StampAPIClient:
    """
    STAMP API 客戶端
    
    提供與 SAM3Engine 相同的介面，透過 HTTP/WebSocket 呼叫遠端 Server。
    """
    
    def __init__(
        self,
        server_url: str = "http://localhost:8000",
        timeout: int = 120,
        websocket_timeout: int = 30,
    ):
        """
        初始化 API Client
        
        Args:
            server_url: Server 網址，例如 "http://192.168.1.100:8000"
            timeout: HTTP 請求超時秒數
            websocket_timeout: WebSocket 操作超時秒數
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self.websocket_timeout = websocket_timeout
        
        # 計算 WebSocket URL
        if self.server_url.startswith("https://"):
            self.ws_url = "wss://" + self.server_url[8:]
        else:
            self.ws_url = "ws://" + self.server_url[7:]
        
        # Video sessions
        self._sessions: Dict[str, VideoSessionInfo] = {}
        
        logger.info(f"STAMP API Client initialized: {self.server_url}")
    
    # =========================================================================
    # 連線檢查
    # =========================================================================
    
    def check_connection(self) -> bool:
        """檢查 Server 是否可連線"""
        try:
            response = requests.get(f"{self.server_url}/health", timeout=5)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False
    
    def get_server_status(self) -> dict:
        """取得 Server 狀態"""
        try:
            response = requests.get(f"{self.server_url}/status", timeout=5)
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get server status: {e}")
            return {"status": "error", "message": str(e)}
    
    # =========================================================================
    # 同步 API：圖片偵測
    # =========================================================================
    
    def detect_image(
        self,
        image: np.ndarray,
        prompt: str,
        threshold_high: float = 0.8,
        threshold_low: float = 0.5
    ) -> FrameResult:
        """
        偵測圖片中的物件（同步）
        
        Args:
            image: 輸入圖片 (H, W, 3) BGR numpy array
            prompt: 文字提示
            threshold_high: HIGH 類別的門檻
            threshold_low: LOW 類別的門檻
            
        Returns:
            FrameResult 物件
        """
        # 轉成 JPEG bytes
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        pil_image = Image.fromarray(image_rgb)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=90)
        buffer.seek(0)
        
        try:
            response = requests.post(
                f"{self.server_url}/api/detect",
                files={"image": ("image.jpg", buffer, "image/jpeg")},
                data={
                    "prompt": prompt,
                    "threshold_high": threshold_high,
                    "threshold_low": threshold_low
                },
                timeout=self.timeout
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Detection request failed: {e}")
            return FrameResult(frame_index=0, detections=[])
        
        data = response.json()
        
        if not data.get("success", False):
            logger.warning(f"Detection failed: {data.get('message')}")
            return FrameResult(frame_index=0, detections=[])
        
        # 轉換成 Detection 物件
        detections = []
        for det_data in data.get("detections", []):
            mask = decode_base64_to_mask(det_data["mask"])
            bbox = det_data["bbox"]
            x1, y1, x2, y2 = bbox
            box_xywh = np.array([x1, y1, x2 - x1, y2 - y1])
            
            detection = Detection(
                obj_id=det_data["obj_id"],
                mask=mask,
                box=box_xywh,
                score=det_data["score"]
            )
            detections.append(detection)
        
        return FrameResult(frame_index=0, detections=detections)
    
    # =========================================================================
    # 同步 API：Mask 修正
    # =========================================================================
    
    def refine_mask(
        self,
        image: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
        mask_input: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        用點擊修正 mask（同步）
        
        Args:
            image: 輸入圖片 (H, W, 3) BGR
            points: 點座標 (N, 2)
            labels: 點標籤 (N,)，1=正向，0=負向
            mask_input: 可選，目前的 mask
            
        Returns:
            修正後的 mask (H, W) boolean array
        """
        request_data = {
            "image": encode_image_to_base64(image),
            "points": [{"x": int(p[0]), "y": int(p[1])} for p in points],
            "labels": [int(l) for l in labels],
        }
        
        if mask_input is not None:
            request_data["current_mask"] = encode_mask_to_base64(mask_input)
        
        try:
            response = requests.post(
                f"{self.server_url}/api/refine",
                json=request_data,
                timeout=self.timeout
            )
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            logger.error(f"Refine request failed: {e}")
            if mask_input is not None:
                return mask_input
            return np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        
        data = response.json()
        
        if not data.get("success", False):
            logger.warning(f"Refine failed: {data.get('message')}")
            if mask_input is not None:
                return mask_input
            return np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        
        return decode_base64_to_mask(data["mask"])
    
    # =========================================================================
    # Jobs API：建立任務
    # =========================================================================
    
    def create_video_detection_job(self, video_path: str, prompt: str) -> str:
        """
        建立影片偵測任務
        
        Args:
            video_path: 影片路徑（Server 端的路徑）
            prompt: 偵測提示詞
            
        Returns:
            task_id
        """
        try:
            response = requests.post(
                f"{self.server_url}/api/jobs/video-detection",
                json={"video_path": video_path, "prompt": prompt},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data["task_id"]
        except Exception as e:
            logger.error(f"Failed to create video detection job: {e}")
            raise
    
    def create_batch_detection_job(self, image_paths: List[str], prompt: str) -> str:
        """
        建立批量圖片偵測任務
        
        Args:
            image_paths: 圖片路徑列表（Server 端）
            prompt: 偵測提示詞
            
        Returns:
            task_id
        """
        try:
            response = requests.post(
                f"{self.server_url}/api/jobs/batch-detection",
                json={"image_paths": image_paths, "prompt": prompt},
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data["task_id"]
        except Exception as e:
            logger.error(f"Failed to create batch detection job: {e}")
            raise
    
    def create_propagate_job(
        self,
        video_path: str,
        start_frame: int,
        mask: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
        obj_id: int = 0
    ) -> str:
        """
        建立傳播任務
        
        Args:
            video_path: 影片路徑
            start_frame: 起始幀
            mask: 初始 mask
            points: 點座標
            labels: 點標籤
            obj_id: 物件 ID
            
        Returns:
            task_id
        """
        try:
            response = requests.post(
                f"{self.server_url}/api/jobs/propagate",
                json={
                    "video_path": video_path,
                    "start_frame": start_frame,
                    "mask": encode_mask_to_base64(mask),
                    "points": [[int(p[0]), int(p[1])] for p in points],
                    "labels": [int(l) for l in labels],
                    "obj_id": obj_id,
                },
                timeout=30
            )
            response.raise_for_status()
            data = response.json()
            return data["task_id"]
        except Exception as e:
            logger.error(f"Failed to create propagate job: {e}")
            raise
    
    # =========================================================================
    # Jobs API：查詢和控制
    # =========================================================================
    
    def get_job(self, task_id: str) -> JobInfo:
        """取得任務資訊"""
        try:
            response = requests.get(
                f"{self.server_url}/api/jobs/{task_id}",
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            return JobInfo(
                task_id=data["task_id"],
                task_type=data["task_type"],
                status=JobStatus(data["status"]),
                progress=data["progress"],
                message=data["message"],
                result=data.get("result"),
                error=data.get("error"),
            )
        except Exception as e:
            logger.error(f"Failed to get job: {e}")
            raise
    
    def list_jobs(self, status: Optional[str] = None, limit: int = 50) -> List[JobInfo]:
        """列出任務"""
        try:
            params = {"limit": limit}
            if status:
                params["status"] = status
            
            response = requests.get(
                f"{self.server_url}/api/jobs",
                params=params,
                timeout=10
            )
            response.raise_for_status()
            data = response.json()
            
            return [
                JobInfo(
                    task_id=d["task_id"],
                    task_type=d["task_type"],
                    status=JobStatus(d["status"]),
                    progress=d["progress"],
                    message=d["message"],
                    result=d.get("result"),
                    error=d.get("error"),
                )
                for d in data
            ]
        except Exception as e:
            logger.error(f"Failed to list jobs: {e}")
            raise
    
    def confirm_job(self, task_id: str) -> bool:
        """確認任務（繼續執行）"""
        try:
            response = requests.post(
                f"{self.server_url}/api/jobs/{task_id}/confirm",
                timeout=10
            )
            response.raise_for_status()
            return response.json().get("success", False)
        except Exception as e:
            logger.error(f"Failed to confirm job: {e}")
            return False
    
    def cancel_job(self, task_id: str) -> bool:
        """取消任務"""
        try:
            response = requests.post(
                f"{self.server_url}/api/jobs/{task_id}/cancel",
                timeout=10
            )
            response.raise_for_status()
            return response.json().get("success", False)
        except Exception as e:
            logger.error(f"Failed to cancel job: {e}")
            return False
    
    # =========================================================================
    # WebSocket：進度監聽
    # =========================================================================
    
    def connect_job_websocket(self, task_id: str) -> "websocket.WebSocket":
        """
        連接任務的 WebSocket
        
        Args:
            task_id: 任務 ID
            
        Returns:
            WebSocket 連線物件
        """
        if not WEBSOCKET_AVAILABLE:
            raise RuntimeError("websocket-client not installed")
        
        ws_url = f"{self.ws_url}/ws/jobs/{task_id}"
        logger.info(f"Connecting to WebSocket: {ws_url}")
        
        ws = websocket.create_connection(
            ws_url,
            timeout=self.websocket_timeout
        )
        
        return ws
    
    def iter_job_events(
        self,
        ws: "websocket.WebSocket"
    ) -> Generator[Dict[str, Any], None, None]:
        """
        迭代 WebSocket 事件
        
        Args:
            ws: WebSocket 連線
            
        Yields:
            事件字典，例如：
            {"type": "progress", "progress": 45, "message": "Detecting..."}
            {"type": "waiting_confirm", "data": {...}}
            {"type": "completed", "result": {...}}
        """
        while True:
            try:
                message = ws.recv()
                if not message:
                    break
                
                event = json.loads(message)
                yield event
                
                # 終結事件
                if event.get("type") in ["completed", "failed", "cancelled"]:
                    break
                    
            except websocket.WebSocketTimeoutException:
                # 發送 ping 保持連線
                ws.send(json.dumps({"action": "ping"}))
            except Exception as e:
                logger.error(f"WebSocket error: {e}")
                break
    
    def wait_for_job(
        self,
        task_id: str,
        on_progress: Optional[Callable[[int, str], None]] = None,
        on_confirm: Optional[Callable[[Dict], bool]] = None,
        on_intermediate: Optional[Callable[[Dict], None]] = None,
        auto_confirm: bool = False,
    ) -> Optional[Dict]:
        """
        等待任務完成（阻塞式）
        
        Args:
            task_id: 任務 ID
            on_progress: 進度回調 callback(progress, message)
            on_confirm: 確認回調 callback(data) -> bool，回傳 True 確認，False 取消
            on_intermediate: 中間結果回調 callback(data)
            auto_confirm: 是否自動確認（不等待用戶）
            
        Returns:
            完成時的結果，失敗或取消時回傳 None
        """
        if not WEBSOCKET_AVAILABLE:
            # Fallback：輪詢
            return self._wait_for_job_polling(task_id, on_progress, auto_confirm)
        
        try:
            ws = self.connect_job_websocket(task_id)
        except Exception as e:
            logger.error(f"Failed to connect WebSocket: {e}")
            return self._wait_for_job_polling(task_id, on_progress, auto_confirm)
        
        try:
            for event in self.iter_job_events(ws):
                event_type = event.get("type")
                
                if event_type == "progress":
                    if on_progress:
                        on_progress(event.get("progress", 0), event.get("message", ""))
                
                elif event_type == "waiting_confirm":
                    if auto_confirm:
                        self.confirm_job(task_id)
                        ws.send(json.dumps({"action": "confirm"}))
                    elif on_confirm:
                        should_continue = on_confirm(event.get("data", {}))
                        if should_continue:
                            self.confirm_job(task_id)
                            ws.send(json.dumps({"action": "confirm"}))
                        else:
                            self.cancel_job(task_id)
                            ws.send(json.dumps({"action": "cancel"}))
                            return None
                
                elif event_type == "intermediate":
                    if on_intermediate:
                        on_intermediate(event.get("data", {}))
                
                elif event_type == "completed":
                    return event.get("result")
                
                elif event_type == "failed":
                    logger.error(f"Job failed: {event.get('error')}")
                    return None
                
                elif event_type == "cancelled":
                    logger.info("Job cancelled")
                    return None
                
        finally:
            ws.close()
        
        return None
    
    def _wait_for_job_polling(
        self,
        task_id: str,
        on_progress: Optional[Callable[[int, str], None]] = None,
        auto_confirm: bool = False,
        poll_interval: float = 1.0,
    ) -> Optional[Dict]:
        """輪詢方式等待任務完成（WebSocket 不可用時的 fallback）"""
        logger.info(f"Using polling mode for job: {task_id}")
        
        while True:
            try:
                job = self.get_job(task_id)
                
                if on_progress:
                    on_progress(job.progress, job.message)
                
                if job.status == JobStatus.WAITING_CONFIRM:
                    if auto_confirm:
                        self.confirm_job(task_id)
                    else:
                        # 沒有 WebSocket，無法互動，自動確認
                        logger.warning("No WebSocket, auto-confirming job")
                        self.confirm_job(task_id)
                
                elif job.status == JobStatus.COMPLETED:
                    return job.result
                
                elif job.status == JobStatus.FAILED:
                    logger.error(f"Job failed: {job.error}")
                    return None
                
                elif job.status == JobStatus.CANCELLED:
                    logger.info("Job cancelled")
                    return None
                
                time.sleep(poll_interval)
                
            except Exception as e:
                logger.error(f"Polling error: {e}")
                time.sleep(poll_interval)
    
    # =========================================================================
    # 影片 Session API（相容 SAM3Engine 介面）
    # =========================================================================
    
    def start_video_session(self, video_path: str) -> str:
        """開始影片 session"""
        try:
            response = requests.post(
                f"{self.server_url}/api/video/load",
                json={"video_path": video_path},
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            
            session_id = data["session_id"]
            self._sessions[session_id] = VideoSessionInfo(
                session_id=session_id,
                video_path=video_path,
                total_frames=data["total_frames"],
                width=data["width"],
                height=data["height"],
                fps=data["fps"]
            )
            
            return session_id
        except Exception as e:
            logger.error(f"Failed to start video session: {e}")
            raise
    
    def get_session_info(self, session_id: str) -> Optional[VideoSessionInfo]:
        """取得 session 資訊"""
        return self._sessions.get(session_id)
    
    def get_frame(self, session_id: str, frame_idx: int) -> np.ndarray:
        """取得影片的某一幀"""
        try:
            response = requests.get(
                f"{self.server_url}/api/video/{session_id}/frame/{frame_idx}",
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()
            return decode_base64_to_image(data["image"])
        except Exception as e:
            logger.error(f"Failed to get frame: {e}")
            raise
    
    def close_session(self, session_id: str) -> None:
        """關閉影片 session"""
        try:
            requests.delete(
                f"{self.server_url}/api/video/{session_id}",
                timeout=10
            )
        except Exception as e:
            logger.warning(f"Failed to close session: {e}")
        
        if session_id in self._sessions:
            del self._sessions[session_id]
    
    def propagate_mask(
        self,
        video_path: str,
        start_frame: int,
        mask: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
        obj_id: int = 0,
        progress_callback: Optional[Callable[[int, int], None]] = None
    ) -> Dict[int, np.ndarray]:
        """
        傳播 mask 到後續的幀（同步版，使用 Jobs API）
        
        Args:
            video_path: 影片路徑（Server 端）
            start_frame: 起始幀
            mask: 初始 mask
            points: 點座標
            labels: 點標籤
            obj_id: 物件 ID
            progress_callback: 進度回調
            
        Returns:
            Dict[frame_idx, mask]
        """
        # 建立傳播任務
        task_id = self.create_propagate_job(
            video_path=video_path,
            start_frame=start_frame,
            mask=mask,
            points=points,
            labels=labels,
            obj_id=obj_id,
        )
        
        # 等待完成
        def on_progress(progress, message):
            if progress_callback:
                # 估算幀數
                total = 100  # 假設
                current = int(progress * total / 100)
                progress_callback(current, total)
        
        result = self.wait_for_job(
            task_id,
            on_progress=on_progress,
            auto_confirm=True,
        )
        
        if result is None:
            return {}
        
        # 解碼 RLE mask
        from pycocotools import mask as mask_utils
        
        masks_data = result.get("masks", {})
        decoded_results = {}
        
        for frame_idx_str, rle in masks_data.items():
            frame_idx = int(frame_idx_str)
            # 確保 counts 是 bytes
            if isinstance(rle["counts"], str):
                rle["counts"] = rle["counts"].encode("utf-8")
            mask = mask_utils.decode(rle).astype(bool)
            decoded_results[frame_idx] = mask
        
        return decoded_results
    
    # =========================================================================
    # Context Manager
    # =========================================================================
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        for session_id in list(self._sessions.keys()):
            self.close_session(session_id)
    
    @property
    def is_mock(self) -> bool:
        """是否為 mock 模式"""
        return False
    
    def shutdown(self) -> None:
        """關閉 client"""
        for session_id in list(self._sessions.keys()):
            self.close_session(session_id)


# =============================================================================
# 測試
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test STAMP API Client")
    parser.add_argument("--server", type=str, default="http://localhost:8000")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--prompt", type=str, default="dolphin")
    args = parser.parse_args()
    
    print("=" * 50)
    print("STAMP API Client Test")
    print("=" * 50)
    
    client = StampAPIClient(args.server)
    
    # 測試連線
    print("\n[1] Testing connection...")
    if client.check_connection():
        print("    ✅ Connected!")
        status = client.get_server_status()
        print(f"    Status: {status}")
    else:
        print("    ❌ Connection failed!")
        exit(1)
    
    # 測試同步偵測
    if args.image:
        print(f"\n[2] Testing image detection...")
        image = cv2.imread(args.image)
        if image is not None:
            result = client.detect_image(image, args.prompt)
            print(f"    ✅ Detected {result.num_objects} objects")
    
    # 測試 Jobs API
    if args.video:
        print(f"\n[3] Testing video detection job...")
        
        task_id = client.create_video_detection_job(args.video, args.prompt)
        print(f"    Task ID: {task_id}")
        
        def on_progress(progress, message):
            print(f"    [{progress:3d}%] {message}")
        
        def on_confirm(data):
            print(f"    Found {data.get('num_objects', 0)} objects. Confirming...")
            return True
        
        result = client.wait_for_job(
            task_id,
            on_progress=on_progress,
            on_confirm=on_confirm,
        )
        
        if result:
            print(f"    ✅ Completed! {result}")
        else:
            print("    ❌ Failed or cancelled")
    
    print("\n" + "=" * 50)
    print("✅ Test complete!")
    print("=" * 50)
