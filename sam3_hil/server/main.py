"""
STAMP API Server
================

FastAPI server for STAMP annotation system.
Handles SAM3 inference requests from remote clients.

架構：
┌─────────────────────────────────────────────────────────────┐
│                      STAMP API Server                        │
│                                                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │   REST API   │  │  WebSocket   │  │ TaskManager  │       │
│  │              │  │              │  │              │       │
│  │ /api/detect  │  │ /ws/jobs/*   │  │ 任務佇列     │       │
│  │ /api/refine  │  │              │  │ 背景執行     │       │
│  │ /api/jobs/*  │  │              │  │              │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
│         │                 │                 │                │
│         └─────────────────┴─────────────────┘                │
│                           │                                  │
│                    ┌──────────────┐                          │
│                    │  SAM3 Engine │                          │
│                    │     (GPU)    │                          │
│                    └──────────────┘                          │
└─────────────────────────────────────────────────────────────┘

Usage:
    # 開發模式（熱重載）
    uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
    
    # 生產模式
    uvicorn server.main:app --host 0.0.0.0 --port 8000 --workers 1
    
    # 透過 run_server.py
    python run_server.py --host 0.0.0.0 --port 8000
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


# =============================================================================
# Global State
# =============================================================================

# SAM3 engine instance (for direct API calls like /api/detect)
sam3_engine = None

# TaskManager instance
task_manager = None

# Server start time
server_start_time = None


# =============================================================================
# Lifespan Management
# =============================================================================

@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events.
    
    Startup:
    - Load SAM3 model
    - Initialize TaskManager
    
    Shutdown:
    - Cleanup TaskManager
    - Release SAM3 resources
    """
    global sam3_engine, task_manager, server_start_time
    
    logger.info("=" * 60)
    logger.info("Starting STAMP API Server...")
    logger.info("=" * 60)
    
    server_start_time = datetime.now()
    
    # =========================================================================
    # Step 1: Load SAM3 Engine
    # =========================================================================
    try:
        logger.info("[1/2] Loading SAM3 engine...")
        from src.core.sam3_engine import SAM3Engine
        sam3_engine = SAM3Engine(mode="auto")
        logger.info("      SAM3 engine loaded successfully ✓")
    except Exception as e:
        logger.error(f"      Failed to load SAM3 engine: {e}")
        logger.warning("      Running in mock mode (no GPU inference)")
        sam3_engine = None
    
    # =========================================================================
    # Step 2: Initialize TaskManager (使用單例)
    # =========================================================================
    try:
        logger.info("[2/2] Initializing TaskManager...")
        from server.task_manager import init_task_manager
        task_manager = init_task_manager(sam3_engine=sam3_engine)
        logger.info("      TaskManager initialized ✓")
    except Exception as e:
        logger.error(f"      Failed to initialize TaskManager: {e}")
    
    logger.info("-" * 60)
    logger.info("STAMP API Server is ready!")
    logger.info(f"  - SAM3 Engine: {'Loaded' if sam3_engine else 'Not available'}")
    logger.info(f"  - API Docs: http://localhost:8000/docs")
    logger.info("-" * 60)
    
    yield
    
    # =========================================================================
    # Shutdown
    # =========================================================================
    logger.info("=" * 60)
    logger.info("Shutting down STAMP API Server...")
    logger.info("=" * 60)
    
    # Shutdown TaskManager
    from server.task_manager import shutdown_task_manager
    logger.info("Shutting down TaskManager...")
    try:
        shutdown_task_manager()
    except Exception as e:
        logger.error(f"Error shutting down TaskManager: {e}")
    
    # Release SAM3 engine
    if sam3_engine is not None:
        logger.info("Releasing SAM3 engine...")
        try:
            if hasattr(sam3_engine, 'shutdown'):
                sam3_engine.shutdown()
        except Exception as e:
            logger.error(f"Error releasing SAM3 engine: {e}")
    
    logger.info("STAMP API Server stopped.")


# =============================================================================
# Create FastAPI App
# =============================================================================

app = FastAPI(
    title="STAMP API",
    description="""
## SAM Tracking Annotation with Minimal Processing

STAMP API Server 提供以下功能：

### 直接 API（快速操作）
- **Detection**: 圖片偵測
- **Refinement**: Mask 修正
- **Video**: 影片處理

### 任務 API（長時間操作）
- **Jobs**: 建立、查詢、取消任務
- **WebSocket**: 即時進度推送

### 使用方式

1. **快速偵測**（同步）
   ```
   POST /api/detect
   ```

2. **長時間任務**（異步）
   ```
   POST /api/jobs/video-detection  → 取得 task_id
   WS   /ws/jobs/{task_id}          → 監聽進度
   POST /api/jobs/{task_id}/confirm → 確認繼續
   ```
    """,
    version="0.2.0",
    lifespan=lifespan,
)


# =============================================================================
# CORS Middleware
# =============================================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 開發環境允許所有來源
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health Check Endpoints
# =============================================================================

@app.get("/", tags=["Health"])
async def root():
    """
    Root endpoint - 服務狀態總覽
    """
    uptime = None
    if server_start_time:
        uptime = str(datetime.now() - server_start_time)
    
    return {
        "service": "STAMP API",
        "version": "0.2.0",
        "status": "running",
        "sam3_loaded": sam3_engine is not None,
        "task_manager_ready": task_manager is not None,
        "uptime": uptime,
    }


@app.get("/health", tags=["Health"])
async def health():
    """
    Health check endpoint
    """
    return {
        "status": "healthy",
        "sam3": "loaded" if sam3_engine else "not_loaded",
        "task_manager": "ready" if task_manager else "not_ready",
    }


@app.get("/status", tags=["Health"])
async def status():
    """
    詳細狀態資訊
    """
    import torch
    from server.task_manager import get_task_manager, TaskStatus
    
    gpu_info = None
    if torch.cuda.is_available():
        gpu_info = {
            "device": torch.cuda.get_device_name(0),
            "memory_allocated_gb": round(torch.cuda.memory_allocated() / 1024**3, 2),
            "memory_reserved_gb": round(torch.cuda.memory_reserved() / 1024**3, 2),
        }
    
    task_stats = None
    try:
        task_mgr = get_task_manager()
        tasks = task_mgr.get_all_tasks()
        task_stats = {
            "total": len(tasks),
            "pending": len([t for t in tasks if t.status == TaskStatus.PENDING]),
            "running": len([t for t in tasks if t.status == TaskStatus.RUNNING]),
            "waiting_confirm": len([t for t in tasks if t.status == TaskStatus.WAITING_CONFIRM]),
            "completed": len([t for t in tasks if t.status == TaskStatus.COMPLETED]),
            "failed": len([t for t in tasks if t.status == TaskStatus.FAILED]),
            "cancelled": len([t for t in tasks if t.status == TaskStatus.CANCELLED]),
        }
    except Exception:
        pass
    
    return {
        "service": "STAMP API",
        "version": "0.2.0",
        "uptime": str(datetime.now() - server_start_time) if server_start_time else None,
        "sam3_engine": "loaded" if sam3_engine else "not_loaded",
        "gpu": gpu_info,
        "tasks": task_stats,
    }


# =============================================================================
# Import and Include Routers
# =============================================================================

# 直接 API（原有的）
from server.routes import detection, refinement, video

app.include_router(detection.router, prefix="/api", tags=["Detection"])
app.include_router(refinement.router, prefix="/api", tags=["Refinement"])
app.include_router(video.router, prefix="/api", tags=["Video"])

# 任務 API（新增的）
from server.routes import jobs, websocket

app.include_router(jobs.router, prefix="/api", tags=["Jobs"])
app.include_router(websocket.router, tags=["WebSocket"])


# =============================================================================
# Utility Functions for Routes
# =============================================================================

def get_sam3_engine():
    """
    Get the global SAM3 engine instance.
    Used by detection.py, refinement.py, video.py
    """
    return sam3_engine


# Note: get_task_manager() is imported from server.task_manager
# Used by jobs.py, websocket.py


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
