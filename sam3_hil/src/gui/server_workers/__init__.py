"""
GUI Workers
===========

背景執行緒 Workers，用於處理長時間任務。

Workers:
- LocalWorker: 本地 SAM3 處理（需要 GPU）
- ServerWorker: 遠端 Server 處理（透過 HTTP/WebSocket）
"""

from .server_worker import (
    ServerVideoDetectionWorker,
    ServerBatchDetectionWorker,
    ServerPropagateWorker,
)

__all__ = [
    "ServerVideoDetectionWorker",
    "ServerBatchDetectionWorker",
    "ServerPropagateWorker",
]
