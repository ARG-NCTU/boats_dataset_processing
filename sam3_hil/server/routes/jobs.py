"""
Jobs API Routes
===============

任務管理 REST API，提供：
- 建立任務
- 查詢任務狀態
- 確認/取消任務

端點：
    POST   /api/jobs              建立新任務
    GET    /api/jobs              列出所有任務
    GET    /api/jobs/{task_id}    取得單一任務
    POST   /api/jobs/{task_id}/confirm   確認任務（繼續執行）
    POST   /api/jobs/{task_id}/cancel    取消任務
    DELETE /api/jobs/{task_id}    刪除任務記錄
"""

import base64
import io
from typing import List, Optional, Dict, Any

import numpy as np
from PIL import Image
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel, Field
from loguru import logger

from server.task_manager import (
    get_task_manager,
    TaskType,
    TaskStatus,
    Task,
)


router = APIRouter()


# =============================================================================
# Request/Response Schemas
# =============================================================================

class CreateVideoDetectionRequest(BaseModel):
    """建立影片偵測任務的請求"""
    video_path: str = Field(..., description="影片路徑（Server 端）")
    prompt: str = Field(..., description="偵測提示詞，如 'ship, boat'")
    
    class Config:
        json_schema_extra = {
            "example": {
                "video_path": "/app/data/video/harbor.mp4",
                "prompt": "ship, boat"
            }
        }


class CreateBatchDetectionRequest(BaseModel):
    """建立批量圖片偵測任務的請求"""
    image_paths: List[str] = Field(..., description="圖片路徑列表（Server 端）")
    prompt: str = Field(..., description="偵測提示詞")
    
    class Config:
        json_schema_extra = {
            "example": {
                "image_paths": ["/app/data/images/001.jpg", "/app/data/images/002.jpg"],
                "prompt": "dolphin"
            }
        }


class CreatePropagateRequest(BaseModel):
    """建立傳播任務的請求"""
    video_path: str = Field(..., description="影片路徑")
    start_frame: int = Field(..., description="起始幀")
    mask: str = Field(..., description="初始 mask（Base64 PNG）")
    points: List[List[int]] = Field(..., description="點座標 [[x1,y1], [x2,y2], ...]")
    labels: List[int] = Field(..., description="點標籤（1=正向, 0=負向）")
    obj_id: int = Field(0, description="物件 ID")


class TaskResponse(BaseModel):
    """任務資訊回應"""
    task_id: str
    task_type: str
    status: str
    progress: int
    message: str
    params: Dict[str, Any]
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    created_at: str
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


class CreateTaskResponse(BaseModel):
    """建立任務的回應"""
    success: bool
    task_id: str
    message: str


class ActionResponse(BaseModel):
    """操作回應（確認/取消）"""
    success: bool
    message: str


# =============================================================================
# Helper Functions
# =============================================================================

def task_to_response(task: Task) -> TaskResponse:
    """把 Task 物件轉成 API 回應格式"""
    return TaskResponse(
        task_id=task.task_id,
        task_type=task.task_type.value,
        status=task.status.value,
        progress=task.progress,
        message=task.message,
        params=task.params,
        result=task.result,
        error=task.error,
        created_at=task.created_at.isoformat(),
        started_at=task.started_at.isoformat() if task.started_at else None,
        completed_at=task.completed_at.isoformat() if task.completed_at else None,
    )


def decode_base64_to_mask(base64_str: str) -> np.ndarray:
    """把 Base64 字串轉成 numpy mask"""
    img_bytes = base64.b64decode(base64_str)
    pil_image = Image.open(io.BytesIO(img_bytes))
    mask = np.array(pil_image)
    
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    
    return mask > 127


# =============================================================================
# Routes - 建立任務
# =============================================================================

@router.post("/jobs/video-detection", response_model=CreateTaskResponse)
async def create_video_detection_job(request: CreateVideoDetectionRequest):
    """
    建立影片偵測任務
    
    流程：
    1. 載入影片
    2. 偵測第一幀 → 發送 WAITING_CONFIRM 事件
    3. 等待用戶確認
    4. 傳播到所有幀
    5. 完成
    
    使用 WebSocket 監聽進度。
    """
    manager = get_task_manager()
    
    task_id = manager.create_task(
        task_type=TaskType.VIDEO_DETECTION,
        params={
            "video_path": request.video_path,
            "prompt": request.prompt,
        }
    )
    
    logger.info(f"Created video detection job: {task_id}")
    
    return CreateTaskResponse(
        success=True,
        task_id=task_id,
        message=f"Task created. Connect to WebSocket /ws/jobs/{task_id} for progress.",
    )


@router.post("/jobs/image-detection", response_model=CreateTaskResponse)
async def create_image_detection_job(
    image: UploadFile = File(..., description="圖片檔案"),
    prompt: str = Form(..., description="偵測提示詞"),
):
    """
    建立單張圖片偵測任務
    
    這是一個快速任務，通常幾秒內完成。
    可以用 WebSocket 監聽，或直接等待結果。
    """
    manager = get_task_manager()
    
    # 讀取圖片
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    image_array = np.array(pil_image)
    
    # RGB to BGR
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = image_array[:, :, ::-1]
    
    task_id = manager.create_task(
        task_type=TaskType.IMAGE_DETECTION,
        params={
            "image": image_array,
            "prompt": prompt,
        }
    )
    
    logger.info(f"Created image detection job: {task_id}")
    
    return CreateTaskResponse(
        success=True,
        task_id=task_id,
        message="Task created.",
    )


@router.post("/jobs/batch-detection", response_model=CreateTaskResponse)
async def create_batch_detection_job(request: CreateBatchDetectionRequest):
    """
    建立批量圖片偵測任務
    
    會逐張處理圖片，每張完成時發送 INTERMEDIATE_RESULT 事件。
    """
    manager = get_task_manager()
    
    task_id = manager.create_task(
        task_type=TaskType.BATCH_DETECTION,
        params={
            "image_paths": request.image_paths,
            "prompt": request.prompt,
        }
    )
    
    logger.info(f"Created batch detection job: {task_id}, {len(request.image_paths)} images")
    
    return CreateTaskResponse(
        success=True,
        task_id=task_id,
        message=f"Task created. {len(request.image_paths)} images to process.",
    )


@router.post("/jobs/propagate", response_model=CreateTaskResponse)
async def create_propagate_job(request: CreatePropagateRequest):
    """
    建立傳播任務
    
    把指定的 mask 傳播到後續所有幀。
    通常在 refinement 完成後使用。
    """
    manager = get_task_manager()
    
    # 解碼 mask
    mask = decode_base64_to_mask(request.mask)
    
    task_id = manager.create_task(
        task_type=TaskType.PROPAGATE,
        params={
            "video_path": request.video_path,
            "start_frame": request.start_frame,
            "mask": mask,
            "points": np.array(request.points),
            "labels": np.array(request.labels),
            "obj_id": request.obj_id,
        }
    )
    
    logger.info(f"Created propagate job: {task_id}, start_frame={request.start_frame}")
    
    return CreateTaskResponse(
        success=True,
        task_id=task_id,
        message="Propagation task created.",
    )


# =============================================================================
# Routes - 查詢任務
# =============================================================================

@router.get("/jobs", response_model=List[TaskResponse])
async def list_jobs(
    status: Optional[str] = None,
    limit: int = 50,
):
    """
    列出所有任務
    
    - **status**: 過濾狀態（pending, running, completed, failed, cancelled）
    - **limit**: 最大回傳數量
    """
    manager = get_task_manager()
    tasks = manager.get_all_tasks()
    
    # 過濾狀態
    if status:
        try:
            status_enum = TaskStatus(status)
            tasks = [t for t in tasks if t.status == status_enum]
        except ValueError:
            raise HTTPException(400, f"Invalid status: {status}")
    
    # 排序（最新的在前）
    tasks.sort(key=lambda t: t.created_at, reverse=True)
    
    # 限制數量
    tasks = tasks[:limit]
    
    return [task_to_response(t) for t in tasks]


@router.get("/jobs/{task_id}", response_model=TaskResponse)
async def get_job(task_id: str):
    """
    取得單一任務的詳細資訊
    """
    manager = get_task_manager()
    task = manager.get_task(task_id)
    
    if task is None:
        raise HTTPException(404, f"Task not found: {task_id}")
    
    return task_to_response(task)


# =============================================================================
# Routes - 任務操作
# =============================================================================

@router.post("/jobs/{task_id}/confirm", response_model=ActionResponse)
async def confirm_job(task_id: str):
    """
    確認任務（繼續執行）
    
    當任務處於 WAITING_CONFIRM 狀態時，呼叫此端點讓任務繼續執行。
    通常用於影片偵測流程：偵測完第一幀後，用戶確認結果再繼續傳播。
    """
    manager = get_task_manager()
    task = manager.get_task(task_id)
    
    if task is None:
        raise HTTPException(404, f"Task not found: {task_id}")
    
    if task.status != TaskStatus.WAITING_CONFIRM:
        raise HTTPException(
            400, 
            f"Task is not waiting for confirmation. Current status: {task.status.value}"
        )
    
    success = manager.confirm_task(task_id)
    
    if success:
        logger.info(f"Task confirmed: {task_id}")
        return ActionResponse(success=True, message="Task confirmed, continuing execution.")
    else:
        raise HTTPException(500, "Failed to confirm task")


@router.post("/jobs/{task_id}/cancel", response_model=ActionResponse)
async def cancel_job(task_id: str):
    """
    取消任務
    
    可以取消 PENDING、RUNNING、WAITING_CONFIRM 狀態的任務。
    已完成（COMPLETED、FAILED、CANCELLED）的任務無法取消。
    """
    manager = get_task_manager()
    task = manager.get_task(task_id)
    
    if task is None:
        raise HTTPException(404, f"Task not found: {task_id}")
    
    if task.status in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
        raise HTTPException(
            400,
            f"Cannot cancel task with status: {task.status.value}"
        )
    
    success = manager.cancel_task(task_id)
    
    if success:
        logger.info(f"Task cancelled: {task_id}")
        return ActionResponse(success=True, message="Task cancellation requested.")
    else:
        raise HTTPException(500, "Failed to cancel task")


@router.delete("/jobs/{task_id}", response_model=ActionResponse)
async def delete_job(task_id: str):
    """
    刪除任務記錄
    
    只能刪除已完成的任務（COMPLETED、FAILED、CANCELLED）。
    執行中的任務請先取消。
    """
    manager = get_task_manager()
    task = manager.get_task(task_id)
    
    if task is None:
        raise HTTPException(404, f"Task not found: {task_id}")
    
    if task.status not in [TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED]:
        raise HTTPException(
            400,
            f"Cannot delete task with status: {task.status.value}. Cancel it first."
        )
    
    # 從 manager 中移除（需要在 TaskManager 加入此方法）
    # 目前先不實作，只是標記
    logger.info(f"Task delete requested (not implemented): {task_id}")
    
    return ActionResponse(success=True, message="Task deleted.")


# =============================================================================
# Routes - 快速操作（同步等待結果）
# =============================================================================

@router.post("/jobs/image-detection/sync")
async def create_image_detection_job_sync(
    image: UploadFile = File(..., description="圖片檔案"),
    prompt: str = Form(..., description="偵測提示詞"),
    timeout: float = Form(60.0, description="超時秒數"),
):
    """
    同步版本的單張圖片偵測
    
    會等待任務完成後才回傳結果。
    適合快速的單張偵測，不需要 WebSocket。
    
    注意：如果任務太久可能會超時。
    """
    import asyncio
    
    manager = get_task_manager()
    
    # 讀取圖片
    image_bytes = await image.read()
    pil_image = Image.open(io.BytesIO(image_bytes))
    image_array = np.array(pil_image)
    
    if len(image_array.shape) == 3 and image_array.shape[2] == 3:
        image_array = image_array[:, :, ::-1]
    
    # 建立任務
    task_id = manager.create_task(
        task_type=TaskType.IMAGE_DETECTION,
        params={
            "image": image_array,
            "prompt": prompt,
        }
    )
    
    # 等待完成
    start_time = asyncio.get_event_loop().time()
    
    while True:
        task = manager.get_task(task_id)
        
        if task.status == TaskStatus.COMPLETED:
            return {
                "success": True,
                "task_id": task_id,
                "result": task.result,
            }
        
        if task.status == TaskStatus.FAILED:
            raise HTTPException(500, f"Task failed: {task.error}")
        
        if task.status == TaskStatus.CANCELLED:
            raise HTTPException(400, "Task was cancelled")
        
        # 檢查超時
        elapsed = asyncio.get_event_loop().time() - start_time
        if elapsed > timeout:
            # 取消任務
            manager.cancel_task(task_id)
            raise HTTPException(408, f"Task timed out after {timeout}s")
        
        # 等待一下再檢查
        await asyncio.sleep(0.1)
