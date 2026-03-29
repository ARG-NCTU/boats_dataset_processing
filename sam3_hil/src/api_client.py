"""
STAMP API Client
================

HTTP 客戶端，提供與 SAM3Engine 相同的介面。
讓 GUI 可以透過 HTTP 呼叫遠端 Server 的 SAM3。

用法：
    # 原本（直接用 SAM3Engine）
    from src.core.sam3_engine import SAM3Engine
    engine = SAM3Engine()
    result = engine.detect_image(image, "dolphin")

    # 改成（用 API Client）
    from src.api_client import StampAPIClient
    client = StampAPIClient("http://192.168.1.100:8000")
    result = client.detect_image(image, "dolphin")
    
    # 介面一樣，GUI 幾乎不用改！
"""

import base64
import io
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Callable

import cv2
import numpy as np
from PIL import Image
import requests
from loguru import logger


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


# =============================================================================
# 編解碼工具
# =============================================================================

def encode_image_to_base64(image: np.ndarray) -> str:
    """把 numpy 圖片轉成 base64 字串"""
    # 確保是 uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8)
    
    # BGR to RGB（如果需要）
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # 轉成 PIL Image
    pil_image = Image.fromarray(image)
    
    # 存成 JPEG bytes
    buffer = io.BytesIO()
    pil_image.save(buffer, format='JPEG', quality=90)
    buffer.seek(0)
    
    # Base64 編碼
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def encode_mask_to_base64(mask: np.ndarray) -> str:
    """把 numpy mask 轉成 base64 字串"""
    # 轉成 uint8 (0 或 255)
    mask_uint8 = (mask.astype(np.uint8) * 255)
    
    # 轉成 PIL Image（灰階）
    pil_image = Image.fromarray(mask_uint8, mode='L')
    
    # 存成 PNG bytes
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Base64 編碼
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def decode_base64_to_mask(base64_str: str) -> np.ndarray:
    """把 base64 字串轉回 numpy mask"""
    # Base64 解碼
    img_bytes = base64.b64decode(base64_str)
    
    # 讀成 PIL Image
    pil_image = Image.open(io.BytesIO(img_bytes))
    
    # 轉成 numpy array
    mask = np.array(pil_image)
    
    # 確保是 2D
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    
    # 轉成 boolean
    return mask > 127


def decode_base64_to_image(base64_str: str) -> np.ndarray:
    """把 base64 字串轉回 numpy 圖片"""
    img_bytes = base64.b64decode(base64_str)
    pil_image = Image.open(io.BytesIO(img_bytes))
    image = np.array(pil_image)
    
    # RGB to BGR（OpenCV 格式）
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    return image


# =============================================================================
# API Client
# =============================================================================

class StampAPIClient:
    """
    STAMP API 客戶端
    
    提供與 SAM3Engine 相同的介面，但透過 HTTP 呼叫遠端 Server。
    """
    
    def __init__(self, server_url: str = "http://localhost:8000", timeout: int = 120):
        """
        初始化 API Client
        
        Args:
            server_url: Server 網址，例如 "http://192.168.1.100:8000"
            timeout: HTTP 請求超時秒數（偵測/傳播可能較久）
        """
        self.server_url = server_url.rstrip('/')
        self.timeout = timeout
        self._sessions: Dict[str, VideoSessionInfo] = {}
        
        logger.info(f"STAMP API Client initialized: {self.server_url}")
    
    # =========================================================================
    # 連線檢查
    # =========================================================================
    
    def check_connection(self) -> bool:
        """檢查 Server 是否可連線"""
        try:
            response = requests.get(
                f"{self.server_url}/health",
                timeout=5
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Connection check failed: {e}")
            return False
    
    def get_server_status(self) -> dict:
        """取得 Server 狀態"""
        try:
            response = requests.get(
                f"{self.server_url}/",
                timeout=5
            )
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get server status: {e}")
            return {"status": "error", "message": str(e)}
    
    # =========================================================================
    # 圖片偵測（對應 sam3_engine.detect_image）
    # =========================================================================
    
    def detect_image(
        self,
        image: np.ndarray,
        prompt: str,
        threshold_high: float = 0.8,
        threshold_low: float = 0.5
    ) -> FrameResult:
        """
        偵測圖片中的物件
        
        Args:
            image: 輸入圖片 (H, W, 3) BGR numpy array
            prompt: 文字提示，例如 "dolphin"
            threshold_high: HIGH 類別的門檻
            threshold_low: LOW 類別的門檻
            
        Returns:
            FrameResult 物件，包含偵測到的物件列表
        """
        # 把圖片存成暫存檔（因為要用 multipart/form-data 上傳）
        # 或者直接用 bytes
        
        # 轉成 JPEG bytes
        if len(image.shape) == 3 and image.shape[2] == 3:
            # BGR to RGB for PIL
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        pil_image = Image.fromarray(image_rgb)
        buffer = io.BytesIO()
        pil_image.save(buffer, format='JPEG', quality=90)
        buffer.seek(0)
        
        # 發送請求
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
            # 回傳空結果
            return FrameResult(frame_index=0, detections=[])
        
        # 解析回應
        data = response.json()
        
        if not data.get("success", False):
            logger.warning(f"Detection failed: {data.get('message', 'Unknown error')}")
            return FrameResult(frame_index=0, detections=[])
        
        # 轉換成 Detection 物件
        detections = []
        for det_data in data.get("detections", []):
            # 解碼 mask
            mask = decode_base64_to_mask(det_data["mask"])
            
            # bbox 從 xyxy 轉成 xywh
            bbox = det_data["bbox"]  # [x1, y1, x2, y2]
            x1, y1, x2, y2 = bbox
            box_xywh = np.array([x1, y1, x2 - x1, y2 - y1])
            
            detection = Detection(
                obj_id=det_data["obj_id"],
                mask=mask,
                box=box_xywh,
                score=det_data["score"]
            )
            detections.append(detection)
        
        logger.info(f"Detected {len(detections)} objects")
        return FrameResult(frame_index=0, detections=detections)
    
    # =========================================================================
    # Mask 修正（對應 sam3_engine.refine_mask）
    # =========================================================================
    
    def refine_mask(
        self,
        image: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
        mask_input: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        用點擊修正 mask
        
        Args:
            image: 輸入圖片 (H, W, 3) BGR numpy array
            points: 點座標 (N, 2) array of [x, y]
            labels: 點標籤 (N,) array，1=正向（包含），0=負向（排除）
            mask_input: 可選，目前的 mask，用於迭代修正
            
        Returns:
            修正後的 mask (H, W) boolean array
        """
        # 準備請求資料
        request_data = {
            "image": encode_image_to_base64(image),
            "points": [{"x": int(p[0]), "y": int(p[1])} for p in points],
            "labels": [int(l) for l in labels],
        }
        
        if mask_input is not None:
            request_data["current_mask"] = encode_mask_to_base64(mask_input)
        
        # 發送請求
        try:
            response = requests.post(
                f"{self.server_url}/api/refine",
                json=request_data,
                timeout=self.timeout
            )
            response.raise_for_status()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Refine request failed: {e}")
            # 回傳原本的 mask 或空 mask
            if mask_input is not None:
                return mask_input
            return np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        
        # 解析回應
        data = response.json()
        
        if not data.get("success", False):
            logger.warning(f"Refine failed: {data.get('message', 'Unknown error')}")
            if mask_input is not None:
                return mask_input
            return np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        
        # 解碼 mask
        mask = decode_base64_to_mask(data["mask"])
        
        logger.info(f"Mask refined, score: {data.get('score', 'N/A')}")
        return mask
    
    # =========================================================================
    # 影片 Session 管理
    # =========================================================================
    
    def start_video_session(self, video_path: str) -> str:
        """
        開始影片 session
        
        Args:
            video_path: 影片路徑（Server 端的路徑）
            
        Returns:
            Session ID
        """
        try:
            response = requests.post(
                f"{self.server_url}/api/video/load",
                json={"video_path": video_path},
                timeout=self.timeout
            )
            response.raise_for_status()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to start video session: {e}")
            raise RuntimeError(f"Failed to start video session: {e}")
        
        data = response.json()
        
        if not data.get("success", False):
            raise RuntimeError(f"Failed to load video: {data.get('message', 'Unknown error')}")
        
        # 儲存 session 資訊
        session_id = data["session_id"]
        self._sessions[session_id] = VideoSessionInfo(
            session_id=session_id,
            video_path=video_path,
            total_frames=data["total_frames"],
            width=data["width"],
            height=data["height"],
            fps=data["fps"]
        )
        
        logger.info(f"Video session started: {session_id}")
        return session_id
    
    def get_session_info(self, session_id: str) -> Optional[VideoSessionInfo]:
        """取得 session 資訊"""
        return self._sessions.get(session_id)
    
    def get_frame(self, session_id: str, frame_idx: int) -> np.ndarray:
        """
        取得影片的某一幀
        
        Args:
            session_id: Session ID
            frame_idx: 幀索引
            
        Returns:
            圖片 (H, W, 3) BGR numpy array
        """
        try:
            response = requests.get(
                f"{self.server_url}/api/video/{session_id}/frame/{frame_idx}",
                timeout=self.timeout
            )
            response.raise_for_status()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to get frame: {e}")
            raise RuntimeError(f"Failed to get frame: {e}")
        
        data = response.json()
        
        if not data.get("success", False):
            raise RuntimeError(f"Failed to get frame: {data.get('message', 'Unknown error')}")
        
        # 解碼圖片
        image = decode_base64_to_image(data["image"])
        return image
    
    def close_session(self, session_id: str) -> None:
        """關閉影片 session"""
        try:
            response = requests.delete(
                f"{self.server_url}/api/video/{session_id}",
                timeout=10
            )
            response.raise_for_status()
            
        except requests.exceptions.RequestException as e:
            logger.warning(f"Failed to close session: {e}")
        
        # 移除本地記錄
        if session_id in self._sessions:
            del self._sessions[session_id]
        
        logger.info(f"Video session closed: {session_id}")
    
    # =========================================================================
    # Mask 傳播（對應 sam3_engine.propagate_mask）
    # =========================================================================
    
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
        傳播 mask 到後續的幀
        
        Args:
            video_path: 影片路徑（Server 端）
            start_frame: 起始幀
            mask: 初始 mask (H, W)
            points: 點座標 (N, 2)
            labels: 點標籤 (N,)
            obj_id: 物件 ID
            progress_callback: 進度回調函數 callback(current, total)
            
        Returns:
            Dict[frame_idx, mask]，每一幀的 mask
        """
        # 找到對應的 session
        session_id = None
        for sid, info in self._sessions.items():
            if info.video_path == video_path:
                session_id = sid
                break
        
        if session_id is None:
            # 沒有 session，先建立
            session_id = self.start_video_session(video_path)
        
        # 準備請求
        request_data = {
            "session_id": session_id,
            "start_frame": start_frame,
            "mask": encode_mask_to_base64(mask),
            "points": [[int(p[0]), int(p[1])] for p in points],
            "labels": [int(l) for l in labels],
            "obj_id": obj_id
        }
        
        # 發送請求（這個可能要很久）
        try:
            response = requests.post(
                f"{self.server_url}/api/video/propagate",
                json=request_data,
                timeout=600  # 10 分鐘，傳播可能很久
            )
            response.raise_for_status()
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Propagate request failed: {e}")
            return {}
        
        # 解析回應
        data = response.json()
        
        if not data.get("success", False):
            logger.warning(f"Propagate failed: {data.get('message', 'Unknown error')}")
            return {}
        
        # 解碼所有 mask
        results = {}
        encoded_results = data.get("results", {})
        total = len(encoded_results)
        
        for i, (frame_idx_str, mask_base64) in enumerate(encoded_results.items()):
            frame_idx = int(frame_idx_str)
            results[frame_idx] = decode_base64_to_mask(mask_base64)
            
            # 回報進度
            if progress_callback:
                progress_callback(i + 1, total)
        
        logger.info(f"Propagated to {len(results)} frames")
        return results
    
    # =========================================================================
    # Context Manager（支援 with 語法）
    # =========================================================================
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        # 關閉所有 session
        for session_id in list(self._sessions.keys()):
            self.close_session(session_id)
    
    # =========================================================================
    # 屬性（相容性）
    # =========================================================================
    
    @property
    def is_mock(self) -> bool:
        """是否為 mock 模式（API Client 永遠不是 mock）"""
        return False
    
    def shutdown(self) -> None:
        """關閉 client（關閉所有 session）"""
        for session_id in list(self._sessions.keys()):
            self.close_session(session_id)


# =============================================================================
# 測試
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Test STAMP API Client")
    parser.add_argument("--server", type=str, default="http://localhost:8000",
                        help="Server URL")
    parser.add_argument("--image", type=str, default=None,
                        help="Test image path")
    parser.add_argument("--prompt", type=str, default="dolphin",
                        help="Detection prompt")
    args = parser.parse_args()
    
    print("=" * 50)
    print("STAMP API Client Test")
    print("=" * 50)
    print(f"Server: {args.server}")
    
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
    
    # 測試偵測
    if args.image:
        print(f"\n[2] Testing detection...")
        print(f"    Image: {args.image}")
        print(f"    Prompt: {args.prompt}")
        
        image = cv2.imread(args.image)
        if image is None:
            print(f"    ❌ Cannot read image!")
        else:
            result = client.detect_image(image, args.prompt)
            print(f"    ✅ Detected {result.num_objects} objects")
            for det in result.detections:
                print(f"       - Object {det.obj_id}: score={det.score:.2f}")
    
    print("\n" + "=" * 50)
    print("✅ Test complete!")
    print("=" * 50)
