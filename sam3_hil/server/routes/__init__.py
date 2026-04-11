"""
API Routes
==========

STAMP API 路由模組：

直接 API（同步，快速操作）：
- detection: 圖片偵測 (/api/detect)
- refinement: Mask 修正 (/api/refine)
- video: 影片處理 (/api/video/*)

任務 API（異步，長時間操作）：
- jobs: 任務管理 (/api/jobs/*)
- websocket: 即時進度 (/ws/*)
"""

from server.routes import detection
from server.routes import refinement
from server.routes import video
from server.routes import jobs
from server.routes import websocket

__all__ = [
    "detection",
    "refinement", 
    "video",
    "jobs",
    "websocket",
]
