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
import contextlib
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
        # websocket -> asyncio.Queue
        self._queues: Dict[WebSocket, asyncio.Queue] = {}
        # websocket -> owning event loop
        self._loops: Dict[WebSocket, asyncio.AbstractEventLoop] = {}
    
    async def connect(self, websocket: WebSocket, task_id: str):
        """建立連線"""
        await websocket.accept()

        if task_id not in self._connections:
            self._connections[task_id] = set()
        self._connections[task_id].add(websocket)
        self._websocket_tasks[websocket] = task_id

        self._queues[websocket] = asyncio.Queue()
        self._loops[websocket] = asyncio.get_running_loop()

        logger.info(f"WebSocket connected: task={task_id}")
    
    def disconnect(self, websocket: WebSocket):
        """斷開連線"""
        task_id = self._websocket_tasks.get(websocket)

        if task_id and task_id in self._connections:
            self._connections[task_id].discard(websocket)
            if not self._connections[task_id]:
                del self._connections[task_id]

        self._websocket_tasks.pop(websocket, None)
        self._queues.pop(websocket, None)
        self._loops.pop(websocket, None)

        logger.info(f"WebSocket disconnected: task={task_id}")
    
    def get_queue(self, websocket: WebSocket) -> asyncio.Queue:
        """取得 WebSocket 的事件佇列"""
        return self._queues.get(websocket)
    
    def push_event(self, task_id: str, event: TaskEvent):
        """
        把事件推送到對應的 WebSocket 佇列
        這個 callback 可能從非 async / 非同一 thread 被呼叫，
        所以一定要用該 websocket 所屬的 event loop 做 thread-safe put。
        """
        if task_id not in self._connections:
            return

        event_data = event.to_dict()

        dead_websockets = []

        for websocket in list(self._connections[task_id]):
            queue = self._queues.get(websocket)
            loop = self._loops.get(websocket)

            if queue is None or loop is None:
                dead_websockets.append(websocket)
                continue

            try:
                loop.call_soon_threadsafe(queue.put_nowait, event_data)
            except Exception as e:
                logger.warning(f"Failed to enqueue event for websocket: {e}")
                dead_websockets.append(websocket)

        for websocket in dead_websockets:
            self.disconnect(websocket)
    
    async def send_json(self, websocket: WebSocket, data: dict) -> bool:
        """
        發送 JSON 訊息
        回傳：
            True  = 成功送出
            False = websocket 已不可用
        """
        try:
            await websocket.send_json(data)
            return True
        except Exception as e:
            logger.warning(f"Failed to send WebSocket message: {e}")
            self.disconnect(websocket)
            return False


# 全局連線管理器
manager = ConnectionManager()


# =============================================================================
# WebSocket 端點
# =============================================================================

@router.websocket("/ws/jobs/{task_id}")
async def websocket_job_progress(websocket: WebSocket, task_id: str):
    """
    監聽任務進度的 WebSocket 端點
    """
    task_manager = get_task_manager()

    task = task_manager.get_task(task_id)
    if task is None:
        await websocket.close(code=4004, reason=f"Task not found: {task_id}")
        return

    await manager.connect(websocket, task_id)

    def on_event(event: TaskEvent):
        manager.push_event(task_id, event)

    unsubscribe = task_manager.subscribe(task_id, on_event)

    outgoing_task = None
    incoming_task = None

    try:
        ok = await send_current_status(websocket, task)
        if not ok:
            return

        queue = manager.get_queue(websocket)
        if queue is None:
            logger.warning(f"No queue found for websocket: task={task_id}")
            return

        outgoing_task = asyncio.create_task(handle_outgoing_events(websocket, queue, task_id))
        incoming_task = asyncio.create_task(handle_incoming_messages(websocket, task_id))

        done, pending = await asyncio.wait(
            {outgoing_task, incoming_task},
            return_when=asyncio.FIRST_COMPLETED,
        )

        for t in pending:
            t.cancel()

        for t in pending:
            with contextlib.suppress(asyncio.CancelledError):
                await t

        for t in done:
            exc = t.exception()
            if exc:
                raise exc

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected by client: task={task_id}")
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
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

    if task.status == TaskStatus.COMPLETED and task.result:
        status_message["result"] = task.result
    elif task.status == TaskStatus.FAILED and task.error:
        status_message["error"] = task.error

    return await manager.send_json(websocket, status_message)


async def handle_outgoing_events(websocket: WebSocket, queue: asyncio.Queue, task_id: str):
    """
    處理從 TaskManager 來的事件，推送給 Client
    """
    while True:
        try:
            try:
                event_data = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                ok = await manager.send_json(websocket, {"type": "heartbeat"})
                if not ok:
                    logger.info(f"Stop heartbeat because websocket is closed: task={task_id}")
                    break
                continue

            ok = await manager.send_json(websocket, event_data)
            if not ok:
                logger.info(f"Stop outgoing events because websocket is closed: task={task_id}")
                break

            if event_data.get("type") in ["completed", "failed", "cancelled"]:
                logger.info(f"Task {task_id} ended with {event_data.get('type')}")
                await asyncio.sleep(0.2)
                break

        except asyncio.CancelledError:
            raise
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
            data = await websocket.receive_json()
            action = data.get("action")

            if action == "ping":
                ok = await manager.send_json(websocket, {"type": "pong"})
                if not ok:
                    break

            elif action == "confirm":
                success = task_manager.confirm_task(task_id)
                ok = await manager.send_json(websocket, {
                    "type": "confirm_response",
                    "success": success,
                })
                if not ok:
                    break

            elif action == "cancel":
                success = task_manager.cancel_task(task_id)
                ok = await manager.send_json(websocket, {
                    "type": "cancel_response",
                    "success": success,
                })
                if not ok:
                    break

            else:
                ok = await manager.send_json(websocket, {
                    "type": "error",
                    "message": f"Unknown action: {action}",
                })
                if not ok:
                    break

        except WebSocketDisconnect:
            raise
        except json.JSONDecodeError as e:
            ok = await manager.send_json(websocket, {
                "type": "error",
                "message": f"Invalid JSON: {e}",
            })
            if not ok:
                break
        except asyncio.CancelledError:
            raise
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
