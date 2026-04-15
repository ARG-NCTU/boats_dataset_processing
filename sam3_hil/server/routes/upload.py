"""
Upload API
==========

處理檔案上傳，讓 Client 可以傳送影片到 Server。

API:
    POST /api/upload/video     上傳影片
    GET  /api/upload/list      列出已上傳的檔案
    DELETE /api/upload/{filename}  刪除已上傳的檔案
"""

import os
import shutil
import hashlib
from pathlib import Path
from typing import Optional
from datetime import datetime

from fastapi import APIRouter, UploadFile, File, HTTPException, Form
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel


router = APIRouter(prefix="/upload")


# =============================================================================
# Configuration
# =============================================================================

# 上傳目錄（Docker 容器內）
UPLOAD_DIR = Path("/app/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# 允許的影片副檔名
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm"}

# 允許的圖片副檔名
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# 最大檔案大小（5GB）
MAX_FILE_SIZE = 5 * 1024 * 1024 * 1024


# =============================================================================
# Response Models
# =============================================================================

class UploadResponse(BaseModel):
    """上傳結果"""
    success: bool
    filename: str
    server_path: str  # Server 內的路徑
    size_bytes: int
    md5: Optional[str] = None
    message: str


class FileInfo(BaseModel):
    """檔案資訊"""
    filename: str
    server_path: str
    size_bytes: int
    uploaded_at: str


class FileListResponse(BaseModel):
    """檔案列表"""
    files: list[FileInfo]
    total_size_bytes: int
    upload_dir: str


# =============================================================================
# Helper Functions
# =============================================================================

def get_file_md5(filepath: Path) -> str:
    """計算檔案 MD5"""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def sanitize_filename(filename: str) -> str:
    """清理檔案名稱，移除不安全字元"""
    # 只保留字母、數字、底線、連字號、點
    safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-.")
    sanitized = "".join(c if c in safe_chars else "_" for c in filename)
    
    # 確保有副檔名
    if "." not in sanitized:
        sanitized += ".mp4"
    
    return sanitized


def generate_unique_filename(original_filename: str) -> str:
    """生成唯一的檔案名稱（加入時間戳記）"""
    name = Path(original_filename).stem
    ext = Path(original_filename).suffix.lower()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{name}_{timestamp}{ext}"


# =============================================================================
# Upload Endpoints
# =============================================================================

@router.post("/video", response_model=UploadResponse)
async def upload_video(
    file: UploadFile = File(...),
    compute_md5: bool = Form(default=False),
):
    """
    上傳影片檔案
    
    Args:
        file: 影片檔案
        compute_md5: 是否計算 MD5（大檔案會較慢）
    
    Returns:
        UploadResponse: 包含 server_path 供後續 API 使用
    
    Example:
        ```python
        import requests
        
        with open("video.mp4", "rb") as f:
            response = requests.post(
                "http://server:8000/api/upload/video",
                files={"file": f}
            )
        
        result = response.json()
        server_path = result["server_path"]  # 用於 video-detection API
        ```
    """
    # 檢查副檔名
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_VIDEO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支援的檔案格式: {ext}。允許的格式: {ALLOWED_VIDEO_EXTENSIONS}"
        )
    
    # 清理並生成唯一檔名
    safe_filename = sanitize_filename(file.filename)
    unique_filename = generate_unique_filename(safe_filename)
    
    # 目標路徑
    target_path = UPLOAD_DIR / unique_filename
    
    logger.info(f"Uploading video: {file.filename} -> {target_path}")
    
    try:
        # 寫入檔案（串流方式，支援大檔案）
        total_size = 0
        with open(target_path, "wb") as buffer:
            while True:
                chunk = await file.read(1024 * 1024)  # 1MB chunks
                if not chunk:
                    break
                buffer.write(chunk)
                total_size += len(chunk)
                
                # 檢查大小限制
                if total_size > MAX_FILE_SIZE:
                    # 刪除已寫入的部分
                    buffer.close()
                    target_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=413,
                        detail=f"檔案太大。最大限制: {MAX_FILE_SIZE / 1024**3:.1f} GB"
                    )
        
        # 計算 MD5（可選）
        md5_hash = None
        if compute_md5:
            md5_hash = get_file_md5(target_path)
        
        logger.info(f"Upload complete: {unique_filename} ({total_size / 1024**2:.1f} MB)")
        
        return UploadResponse(
            success=True,
            filename=unique_filename,
            server_path=str(target_path),
            size_bytes=total_size,
            md5=md5_hash,
            message="上傳成功"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        # 清理失敗的檔案
        target_path.unlink(missing_ok=True)
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"上傳失敗: {str(e)}")

@router.post("/image")
async def upload_image(
    file: UploadFile = File(...),
):
    """上傳單張圖片"""
    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(400, f"不支援的圖片格式: {ext}")

    safe_filename = sanitize_filename(file.filename)
    unique_filename = generate_unique_filename(safe_filename)
    target_path = UPLOAD_DIR / unique_filename

    total_size = 0
    with open(target_path, "wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            buffer.write(chunk)
            total_size += len(chunk)

    return UploadResponse(
        success=True,
        filename=unique_filename,
        server_path=str(target_path),
        size_bytes=total_size,
        message="上傳成功",
    )

@router.get("/list", response_model=FileListResponse)
async def list_uploaded_files():
    """
    列出所有已上傳的檔案
    """
    files = []
    total_size = 0
    
    for filepath in UPLOAD_DIR.iterdir():
        if filepath.is_file():
            stat = filepath.stat()
            files.append(FileInfo(
                filename=filepath.name,
                server_path=str(filepath),
                size_bytes=stat.st_size,
                uploaded_at=datetime.fromtimestamp(stat.st_mtime).isoformat()
            ))
            total_size += stat.st_size
    
    # 按上傳時間排序（最新的在前）
    files.sort(key=lambda x: x.uploaded_at, reverse=True)
    
    return FileListResponse(
        files=files,
        total_size_bytes=total_size,
        upload_dir=str(UPLOAD_DIR)
    )


@router.delete("/{filename}")
async def delete_uploaded_file(filename: str):
    """
    刪除已上傳的檔案
    """
    # 安全檢查：不允許路徑穿越
    if "/" in filename or "\\" in filename or ".." in filename:
        raise HTTPException(status_code=400, detail="無效的檔案名稱")
    
    filepath = UPLOAD_DIR / filename
    
    if not filepath.exists():
        raise HTTPException(status_code=404, detail="檔案不存在")
    
    try:
        filepath.unlink()
        logger.info(f"Deleted uploaded file: {filename}")
        return {"success": True, "message": f"已刪除: {filename}"}
    except Exception as e:
        logger.error(f"Failed to delete {filename}: {e}")
        raise HTTPException(status_code=500, detail=f"刪除失敗: {str(e)}")


@router.delete("/")
async def clear_uploads():
    """
    清空所有上傳的檔案
    """
    try:
        count = 0
        for filepath in UPLOAD_DIR.iterdir():
            if filepath.is_file():
                filepath.unlink()
                count += 1
        
        logger.info(f"Cleared {count} uploaded files")
        return {"success": True, "message": f"已清空 {count} 個檔案"}
    except Exception as e:
        logger.error(f"Failed to clear uploads: {e}")
        raise HTTPException(status_code=500, detail=f"清空失敗: {str(e)}")


