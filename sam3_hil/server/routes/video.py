"""
Video API Routes
================

Handles video loading, frame extraction, and propagation.
"""

import base64
import io
import asyncio
from typing import Dict, List, Optional
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from pydantic import BaseModel
from loguru import logger


router = APIRouter()


# =============================================================================
# In-memory session storage (simplified for now)
# =============================================================================

video_sessions: Dict[str, dict] = {}


# =============================================================================
# Schemas
# =============================================================================

class LoadVideoRequest(BaseModel):
    """Request to load a video."""
    video_path: str
    session_id: Optional[str] = None


class LoadVideoResponse(BaseModel):
    """Response for video loading."""
    success: bool
    session_id: str
    total_frames: int
    fps: float
    width: int
    height: int
    message: Optional[str] = None


class FrameResponse(BaseModel):
    """Response for frame request."""
    success: bool
    frame_idx: int
    image: str  # base64 encoded
    width: int
    height: int


class PropagateRequest(BaseModel):
    """Request for mask propagation."""
    session_id: str
    start_frame: int
    mask: str  # base64 encoded
    points: List[List[int]]  # [[x, y], ...]
    labels: List[int]
    obj_id: int


class PropagateResponse(BaseModel):
    """Response for propagation."""
    success: bool
    num_frames: int
    results: Dict[int, str]  # frame_idx -> base64 mask
    message: Optional[str] = None


# =============================================================================
# Helper Functions
# =============================================================================

def encode_image_to_base64(img: np.ndarray) -> str:
    """Convert numpy image to base64 string."""
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)
    
    pil_img = Image.fromarray(img)
    buffer = io.BytesIO()
    pil_img.save(buffer, format='JPEG', quality=85)
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def encode_mask_to_base64(mask: np.ndarray) -> str:
    """Convert numpy mask to base64 string."""
    mask_uint8 = (mask.astype(np.uint8) * 255)
    img = Image.fromarray(mask_uint8, mode='L')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def decode_base64_to_mask(base64_str: str) -> np.ndarray:
    """Convert base64 string to boolean mask."""
    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes))
    mask = np.array(img)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask > 127


def generate_session_id() -> str:
    """Generate unique session ID."""
    import uuid
    return str(uuid.uuid4())[:8]


# =============================================================================
# Routes
# =============================================================================

@router.post("/video/load", response_model=LoadVideoResponse)
async def load_video(request: LoadVideoRequest):
    """
    Load a video file and create a session.
    
    - **video_path**: Path to video file on server
    - **session_id**: Optional session ID (auto-generated if not provided)
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent))
    
    try:
        from src.core.video_loader import VideoLoader
        
        video_path = Path(request.video_path)
        if not video_path.exists():
            raise HTTPException(status_code=404, detail=f"Video not found: {video_path}")
        
        # Create session
        session_id = request.session_id or generate_session_id()
        
        # Load video
        loader = VideoLoader(video_path)
        metadata = loader.metadata
        
        # Store session
        video_sessions[session_id] = {
            "loader": loader,
            "video_path": str(video_path),
            "metadata": metadata,
        }
        
        logger.info(f"Loaded video: {video_path}, session: {session_id}")
        
        return LoadVideoResponse(
            success=True,
            session_id=session_id,
            total_frames=metadata.total_frames,
            fps=metadata.fps,
            width=metadata.width,
            height=metadata.height,
        )
        
    except Exception as e:
        logger.error(f"Failed to load video: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/video/{session_id}/frame/{frame_idx}", response_model=FrameResponse)
async def get_frame(session_id: str, frame_idx: int):
    """
    Get a specific frame from loaded video.
    
    - **session_id**: Video session ID
    - **frame_idx**: Frame index (0-based)
    """
    if session_id not in video_sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    
    session = video_sessions[session_id]
    loader = session["loader"]
    
    if frame_idx < 0 or frame_idx >= session["metadata"].total_frames:
        raise HTTPException(
            status_code=400,
            detail=f"Frame index out of range: {frame_idx} (total: {session['metadata'].total_frames})"
        )
    
    try:
        frame = loader.get_frame(frame_idx)
        
        return FrameResponse(
            success=True,
            frame_idx=frame_idx,
            image=encode_image_to_base64(frame),
            width=frame.shape[1],
            height=frame.shape[0],
        )
        
    except Exception as e:
        logger.error(f"Failed to get frame: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/video/propagate", response_model=PropagateResponse)
async def propagate(request: PropagateRequest):
    """
    Propagate mask to following frames using SAM3 Video Predictor.
    
    Note: This is a long-running operation. Use WebSocket for progress updates.
    """
    from server.main import get_sam3_engine
    
    if request.session_id not in video_sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {request.session_id}")
    
    session = video_sessions[request.session_id]
    engine = get_sam3_engine()
    
    if engine is None:
        raise HTTPException(status_code=503, detail="SAM3 engine not loaded")
    
    try:
        # Decode mask
        mask = decode_base64_to_mask(request.mask)
        points = np.array(request.points)
        labels = np.array(request.labels)
        
        logger.info(f"Propagating from frame {request.start_frame}")
        
        # Run propagation
        results = engine.propagate_mask(
            video_path=session["video_path"],
            start_frame=request.start_frame,
            mask=mask,
            points=points,
            labels=labels,
            obj_id=request.obj_id,
        )
        
        # Encode results
        encoded_results = {
            frame_idx: encode_mask_to_base64(frame_mask)
            for frame_idx, frame_mask in results.items()
        }
        
        return PropagateResponse(
            success=True,
            num_frames=len(encoded_results),
            results=encoded_results,
        )
        
    except Exception as e:
        logger.error(f"Propagation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/video/{session_id}")
async def close_video(session_id: str):
    """
    Close video session and release resources.
    """
    if session_id not in video_sessions:
        raise HTTPException(status_code=404, detail=f"Session not found: {session_id}")
    
    del video_sessions[session_id]
    logger.info(f"Closed video session: {session_id}")
    
    return {"success": True, "message": f"Session {session_id} closed"}


# =============================================================================
# WebSocket for Progress Updates
# =============================================================================

@router.websocket("/ws/progress/{task_id}")
async def progress_websocket(websocket: WebSocket, task_id: str):
    """
    WebSocket for receiving progress updates during long operations.
    
    Messages format:
    - {"type": "progress", "value": 45, "message": "Processing frame 45/100"}
    - {"type": "complete", "message": "Done"}
    - {"type": "error", "message": "Error details"}
    """
    await websocket.accept()
    logger.info(f"WebSocket connected for task: {task_id}")
    
    try:
        while True:
            # Keep connection alive, send heartbeat
            await asyncio.sleep(1)
            await websocket.send_json({"type": "heartbeat"})
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for task: {task_id}")
