"""
Refinement API Routes
=====================

Handles point-based mask refinement requests.
"""

import base64
import io
from typing import List, Optional

import numpy as np
from PIL import Image
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger


router = APIRouter()


# =============================================================================
# Schemas
# =============================================================================

class Point(BaseModel):
    """A point with x, y coordinates."""
    x: int
    y: int


class RefineRequest(BaseModel):
    """Request for mask refinement."""
    image: str  # base64 encoded
    points: List[Point]
    labels: List[int]  # 1=positive, 0=negative
    obj_id: Optional[int] = None
    current_mask: Optional[str] = None  # base64 encoded, for iterative refinement


class RefineResponse(BaseModel):
    """Response for refinement request."""
    success: bool
    mask: str  # base64 encoded
    score: float
    bbox: List[int]
    message: Optional[str] = None


# =============================================================================
# Helper Functions
# =============================================================================

def encode_mask_to_base64(mask: np.ndarray) -> str:
    """Convert numpy mask to base64 string."""
    mask_uint8 = (mask.astype(np.uint8) * 255)
    img = Image.fromarray(mask_uint8, mode='L')
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def decode_base64_to_image(base64_str: str) -> np.ndarray:
    """Convert base64 string to numpy array."""
    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes))
    return np.array(img)


def decode_base64_to_mask(base64_str: str) -> np.ndarray:
    """Convert base64 string to boolean mask."""
    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes))
    mask = np.array(img)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask > 127  # Convert to boolean


def get_bbox_from_mask(mask: np.ndarray) -> List[int]:
    """Get bounding box from mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return [0, 0, 0, 0]
    
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    
    return [int(x1), int(y1), int(x2), int(y2)]


# =============================================================================
# Routes
# =============================================================================

@router.post("/refine", response_model=RefineResponse)
async def refine(request: RefineRequest):
    """
    Refine mask using point prompts.
    
    - **image**: Base64 encoded image
    - **points**: List of points [{x, y}, ...]
    - **labels**: List of labels (1=positive, 0=negative)
    - **obj_id**: Optional object ID for tracking
    - **current_mask**: Optional current mask for iterative refinement
    """
    from server.main import get_sam3_engine
    
    engine = get_sam3_engine()
    
    if engine is None:
        logger.warning("SAM3 engine not loaded, returning mock response")
        # Create mock mask
        img = decode_base64_to_image(request.image)
        mock_mask = np.zeros((img.shape[0], img.shape[1]), dtype=bool)
        
        # Create a circle around the first positive point
        for point, label in zip(request.points, request.labels):
            if label == 1:
                y, x = np.ogrid[:img.shape[0], :img.shape[1]]
                dist = np.sqrt((x - point.x)**2 + (y - point.y)**2)
                mock_mask |= (dist < 50)
        
        return RefineResponse(
            success=True,
            mask=encode_mask_to_base64(mock_mask),
            score=0.85,
            bbox=get_bbox_from_mask(mock_mask),
            message="Mock mode - SAM3 not loaded",
        )
    
    try:
        # Decode image
        img_array = decode_base64_to_image(request.image)
        
        # Prepare points and labels
        points = np.array([[p.x, p.y] for p in request.points])
        labels = np.array(request.labels)
        
        logger.info(f"Refining with {len(points)} points")
        
        # Decode current mask if provided
        current_mask = None
        if request.current_mask:
            current_mask = decode_base64_to_mask(request.current_mask)
        
        # Run refinement
        result = engine.refine_mask(
            image=img_array,
            points=points,
            labels=labels,
            mask=current_mask,
        )
        
        mask = result.get('mask', np.zeros((img_array.shape[0], img_array.shape[1]), dtype=bool))
        score = result.get('score', 0.0)
        
        return RefineResponse(
            success=True,
            mask=encode_mask_to_base64(mask),
            score=float(score),
            bbox=get_bbox_from_mask(mask),
        )
        
    except Exception as e:
        logger.error(f"Refinement error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/apply")
async def apply_refinement(
    session_id: str,
    obj_id: int,
    mask: str,  # base64 encoded
):
    """
    Apply refined mask to object.
    This updates the server-side state.
    """
    # TODO: Implement session management
    return {
        "success": True,
        "message": f"Applied mask to object {obj_id}",
    }
