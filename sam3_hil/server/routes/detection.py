"""
Detection API Routes
====================

Handles SAM3 detection requests.
"""

import base64
import io
from typing import List, Optional

import numpy as np
from PIL import Image
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel
from loguru import logger


router = APIRouter()


# =============================================================================
# Schemas
# =============================================================================

class Detection(BaseModel):
    """Single detection result."""
    obj_id: int
    mask: str  # base64 encoded
    score: float
    category: str  # HIGH / UNCERTAIN / LOW
    bbox: List[int]  # [x1, y1, x2, y2]


class DetectionResponse(BaseModel):
    """Response for detection request."""
    success: bool
    num_detections: int
    detections: List[Detection]
    message: Optional[str] = None


# =============================================================================
# Helper Functions
# =============================================================================

def encode_mask_to_base64(mask: np.ndarray) -> str:
    """Convert numpy mask to base64 string."""
    # Convert to uint8
    mask_uint8 = (mask.astype(np.uint8) * 255)
    
    # Convert to PIL Image
    img = Image.fromarray(mask_uint8, mode='L')
    
    # Save to bytes
    buffer = io.BytesIO()
    img.save(buffer, format='PNG')
    buffer.seek(0)
    
    # Encode to base64
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def decode_base64_to_image(base64_str: str) -> np.ndarray:
    """Convert base64 string to numpy array."""
    img_bytes = base64.b64decode(base64_str)
    img = Image.open(io.BytesIO(img_bytes))
    return np.array(img)


def get_bbox_from_mask(mask: np.ndarray) -> List[int]:
    """Get bounding box from mask."""
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return [0, 0, 0, 0]
    
    y1, y2 = np.where(rows)[0][[0, -1]]
    x1, x2 = np.where(cols)[0][[0, -1]]
    
    return [int(x1), int(y1), int(x2), int(y2)]


def categorize_score(score: float, high_threshold: float, low_threshold: float) -> str:
    """Categorize score into HIGH/UNCERTAIN/LOW."""
    if score >= high_threshold:
        return "HIGH"
    elif score < low_threshold:
        return "LOW"
    else:
        return "UNCERTAIN"


# =============================================================================
# Routes
# =============================================================================

@router.post("/detect", response_model=DetectionResponse)
async def detect(
    image: UploadFile = File(..., description="Image file to detect"),
    prompt: str = Form(..., description="Text prompt for detection"),
    threshold_high: float = Form(0.8, description="High confidence threshold"),
    threshold_low: float = Form(0.5, description="Low confidence threshold"),
):
    """
    Detect objects in image using SAM3.
    
    - **image**: Image file (PNG, JPG)
    - **prompt**: Text prompt (e.g., "ship", "dolphin")
    - **threshold_high**: Score threshold for HIGH category
    - **threshold_low**: Score threshold for LOW category
    """
    from server.main import get_sam3_engine
    
    engine = get_sam3_engine()
    
    if engine is None:
        # Mock mode for testing
        logger.warning("SAM3 engine not loaded, returning mock response")
        return DetectionResponse(
            success=True,
            num_detections=0,
            detections=[],
            message="Mock mode - SAM3 not loaded",
        )
    
    try:
        # Read image
        image_bytes = await image.read()
        img = Image.open(io.BytesIO(image_bytes))
        img_array = np.array(img)
        
        # Convert RGB to BGR if needed (SAM3 expects BGR)
        if len(img_array.shape) == 3 and img_array.shape[2] == 3:
            img_array = img_array[:, :, ::-1]  # RGB to BGR
        
        logger.info(f"Detecting '{prompt}' in image {img_array.shape}")
        
        # Run detection - returns FrameResult object
        frame_result = engine.detect_image(img_array, prompt)
        
        # Convert FrameResult to response format
        detections = []
        for det in frame_result.detections:
            # det is a Detection object from sam3_engine
            detection = Detection(
                obj_id=det.obj_id,
                mask=encode_mask_to_base64(det.mask),
                score=det.score,
                category=categorize_score(det.score, threshold_high, threshold_low),
                bbox=[int(x) for x in det.box_xyxy],  # Convert to list of ints
            )
            detections.append(detection)
        
        logger.info(f"Detected {len(detections)} objects")
        
        return DetectionResponse(
            success=True,
            num_detections=len(detections),
            detections=detections,
        )
        
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/detect/batch", response_model=List[DetectionResponse])
async def detect_batch(
    images: List[UploadFile] = File(..., description="Image files to detect"),
    prompt: str = Form(..., description="Text prompt for detection"),
    threshold_high: float = Form(0.8),
    threshold_low: float = Form(0.5),
):
    """
    Batch detection for multiple images.
    """
    results = []
    for image in images:
        result = await detect(image, prompt, threshold_high, threshold_low)
        results.append(result)
    
    return results
