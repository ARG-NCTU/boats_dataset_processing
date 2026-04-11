"""
WebSocket Routes
================

WebSocket 連線處理，用於：
- 即時推送任務進度
- 接收用戶操作（確認、取消）
- 雙向通訊

端點：
    WS /ws/jobs/{task_id}    監聽特定任務的進度

訊息格式（Server → Client）：
    {"type": "progress", "progress": 45, "message": "Detecting..."}
    {"type": "waiting_confirm", "data": {...}}
    {"type": "completed", "result": {...}}
    {"type": "failed", "error": "..."}
    {"type": "cancelled"}
    {"type": "pong"}  # 心跳回應

訊息格式（Client → Server）：
    {"action": "confirm"}   確認繼續執行
    {"action": "cancel"}    取消任務
    {"action": "ping"}      心跳
"""

import asyncio
import json
from typing import Dict, Set
from datetime import datetime

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from loguru import logger

from server.task_manager import (
    get_task_manager,
    TaskEvent,
    TaskStatus,
    EventType,
)


router = APIRouter()


# =============================================================================
# 連線管理
# =============================================================================

class ConnectionManager:
    """
    WebSocket 連線管理器
    
    負責：
    - 追蹤所有活躍的 WebSocket 連線
    - 把 TaskManager 的事件推送給對應的連線
    - 處理連線的生命週期
    """
    
    def __init__(self):
        # task_id -> set of websockets
        self._connections: Dict[str, Set[WebSocket]] = {}
        # websocket -> task_id
        self._websocket_tasks: Dict[WebSocket, str] = {}
        # websocket -> asyncio.Queue (用於從同步 callback 傳遞事件到異步 WebSocket)
        self._queues: Dict[WebSocket, asyncio.Queue] = {}
    
    async def connect(self, websocket: WebSocket, task_id: str):
        """建立連線"""
        await websocket.accept()
        
        # 記錄連線
        if task_id not in self._connections:
            self._connections[task_id] = set()
        self._connections[task_id].add(websocket)
        self._websocket_tasks[websocket] = task_id
        
        # 建立事件佇列
        self._queues[websocket] = asyncio.Queue()
        
        logger.info(f"WebSocket connected: task={task_id}")
    
    def disconnect(self, websocket: WebSocket):
        """斷開連線"""
        task_id = self._websocket_tasks.get(websocket)
        
        if task_id and task_id in self._connections:
            self._connections[task_id].discard(websocket)
            if not self._connections[task_id]:
                del self._connections[task_id]
        
        if websocket in self._websocket_tasks:
            del self._websocket_tasks[websocket]
        
        if websocket in self._queues:
            del self._queues[websocket]
        
        logger.info(f"WebSocket disconnected: task={task_id}")
    
    def get_queue(self, websocket: WebSocket) -> asyncio.Queue:
        """取得 WebSocket 的事件佇列"""
        return self._queues.get(websocket)
    
    def push_event(self, task_id: str, event: TaskEvent):
        """
        把事件推送到對應的 WebSocket 佇列
        
        注意：這個方法會被 TaskManager 的同步 callback 呼叫，
        所以需要用 thread-safe 的方式把事件放到 asyncio 佇列。
        """
        if task_id not in self._connections:
            return
        
        event_data = event.to_dict()
        
        for websocket in self._connections[task_id]:
            queue = self._queues.get(websocket)
            if queue:
                # 使用 call_soon_threadsafe 從同步環境放入異步佇列
                try:
                    loop = asyncio.get_event_loop()
                    loop.call_soon_threadsafe(queue.put_nowait, event_data)
                except RuntimeError:
                    # 如果沒有 event loop，嘗試直接放入
                    try:
                        queue.put_nowait(event_data)
                    except Exception as e:
                        logger.warning(f"Failed to push event: {e}")
    
    async def send_json(self, websocket: WebSocket, data: dict):
        """發送 JSON 訊息"""
        try:
            await websocket.send_json(data)
        except Exception as e:
            logger.warning(f"Failed to send WebSocket message: {e}")


# 全局連線管理器
manager = ConnectionManager()


# =============================================================================
# WebSocket 端點
# =============================================================================

@router.websocket("/ws/jobs/{task_id}")
async def websocket_job_progress(websocket: WebSocket, task_id: str):
    """
    監聽任務進度的 WebSocket 端點
    
    連線後會收到該任務的所有事件：
    - progress: 進度更新
    - waiting_confirm: 等待確認
    - intermediate: 中間結果
    - completed: 完成
    - failed: 失敗
    - cancelled: 取消
    
    Client 可以發送：
    - {"action": "confirm"}: 確認繼續執行
    - {"action": "cancel"}: 取消任務
    - {"action": "ping"}: 心跳（Server 會回 {"type": "pong"}）
    """
    task_manager = get_task_manager()
    
    # 檢查任務是否存在
    task = task_manager.get_task(task_id)
    if task is None:
        await websocket.close(code=4004, reason=f"Task not found: {task_id}")
        return
    
    # 建立連線
    await manager.connect(websocket, task_id)
    
    # 訂閱 TaskManager 事件
    def on_event(event: TaskEvent):
        manager.push_event(task_id, event)
    
    unsubscribe = task_manager.subscribe(task_id, on_event)
    
    try:
        # 發送目前狀態
        await send_current_status(websocket, task)
        
        # 取得事件佇列
        queue = manager.get_queue(websocket)
        
        # 同時監聽：事件佇列 和 Client 訊息
        await asyncio.gather(
            handle_outgoing_events(websocket, queue, task_id),
            handle_incoming_messages(websocket, task_id),
        )
        
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected by client: task={task_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        # 清理
        unsubscribe()
        manager.disconnect(websocket)


async def send_current_status(websocket: WebSocket, task):
    """發送任務的目前狀態"""
    status_message = {
        "type": "status",
        "task_id": task.task_id,
        "status": task.status.value,
        "progress": task.progress,
        "message": task.message,
    }
    
    # 如果任務已經有結果，一起發送
    if task.status == TaskStatus.COMPLETED and task.result:
        status_message["result"] = task.result
    elif task.status == TaskStatus.FAILED and task.error:
        status_message["error"] = task.error
    
    await manager.send_json(websocket, status_message)


async def handle_outgoing_events(websocket: WebSocket, queue: asyncio.Queue, task_id: str):
    """
    處理從 TaskManager 來的事件，推送給 Client
    """
    while True:
        try:
            # 等待事件（有 timeout 以便檢查連線狀態）
            try:
                event_data = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # 發送心跳
                await manager.send_json(websocket, {"type": "heartbeat"})
                continue
            
            # 發送事件給 Client
            await manager.send_json(websocket, event_data)
            
            # 如果是終結事件，結束監聽
            if event_data.get("type") in ["completed", "failed", "cancelled"]:
                logger.info(f"Task {task_id} ended with {event_data.get('type')}")
                # 給 Client 一點時間收到訊息
                await asyncio.sleep(0.5)
                break
                
        except Exception as e:
            logger.error(f"Error handling outgoing event: {e}")
            break


async def handle_incoming_messages(websocket: WebSocket, task_id: str):
    """
    處理從 Client 來的訊息
    """
    task_manager = get_task_manager()
    
    while True:
        try:
            # 等待 Client 訊息
            data = await websocket.receive_json()
            
            action = data.get("action")
            
            if action == "ping":
                # 心跳回應
                await manager.send_json(websocket, {"type": "pong"})
                
            elif action == "confirm":
                # 確認任務
                success = task_manager.confirm_task(task_id)
                await manager.send_json(websocket, {
                    "type": "confirm_response",
                    "success": success,
                })
                
            elif action == "cancel":
                # 取消任務
                success = task_manager.cancel_task(task_id)
                await manager.send_json(websocket, {
                    "type": "cancel_response",
                    "success": success,
                })
                
            else:
                # 未知動作
                await manager.send_json(websocket, {
                    "type": "error",
                    "message": f"Unknown action: {action}",
                })
                
        except WebSocketDisconnect:
            raise
        except json.JSONDecodeError as e:
            await manager.send_json(websocket, {
                "type": "error",
                "message": f"Invalid JSON: {e}",
            })
        except Exception as e:
            logger.error(f"Error handling incoming message: {e}")
            break


# =============================================================================
# 輔助端點
# =============================================================================

@router.websocket("/ws/ping")
async def websocket_ping(websocket: WebSocket):
    """
    簡單的 ping-pong 端點，用於測試 WebSocket 連線
    """
    await websocket.accept()
    
    try:
        while True:
            data = await websocket.receive_text()
            
            if data == "ping":
                await websocket.send_text("pong")
            else:
                await websocket.send_json({
                    "received": data,
                    "timestamp": datetime.now().isoformat(),
                })
                
    except WebSocketDisconnect:
        pass


# =============================================================================
# 廣播功能（未來擴展用）
# =============================================================================

@router.websocket("/ws/broadcast")
async def websocket_broadcast(websocket: WebSocket):
    """
    廣播端點 - 接收所有任務的事件
    
    用於監控面板、管理介面等需要看到所有任務狀態的場景。
    
    注意：目前尚未實作，預留介面。
    """
    await websocket.accept()
    
    await websocket.send_json({
        "type": "info",
        "message": "Broadcast endpoint not implemented yet",
    })
    
    await websocket.close()
