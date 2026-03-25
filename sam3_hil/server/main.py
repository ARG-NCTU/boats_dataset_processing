"""
STAMP API Server
================

FastAPI server for STAMP annotation system.
Handles SAM3 inference requests from remote clients.

Usage:
    uvicorn server.main:app --host 0.0.0.0 --port 8000 --reload
"""

import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Global SAM3 engine instance
sam3_engine = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Startup and shutdown events.
    Load SAM3 model on startup, cleanup on shutdown.
    """
    global sam3_engine
    
    logger.info("Starting STAMP API Server...")
    
    # Load SAM3 engine
    try:
        from src.core.sam3_engine import SAM3Engine
        sam3_engine = SAM3Engine(mode="auto")
        logger.info("SAM3 engine loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load SAM3 engine: {e}")
        logger.warning("Running in mock mode")
        sam3_engine = None
    
    yield
    
    # Cleanup
    logger.info("Shutting down STAMP API Server...")
    if sam3_engine is not None:
        del sam3_engine


# Create FastAPI app
app = FastAPI(
    title="STAMP API",
    description="SAM Tracking Annotation with Minimal Processing - API Server",
    version="0.1.0",
    lifespan=lifespan,
)

# CORS middleware (allow all origins for development)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# =============================================================================
# Health Check
# =============================================================================

@app.get("/")
async def root():
    """Root endpoint - health check."""
    return {
        "service": "STAMP API",
        "status": "running",
        "sam3_loaded": sam3_engine is not None,
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}


# =============================================================================
# Import and include routers
# =============================================================================

from server.routes import detection, refinement, video

app.include_router(detection.router, prefix="/api", tags=["Detection"])
app.include_router(refinement.router, prefix="/api", tags=["Refinement"])
app.include_router(video.router, prefix="/api", tags=["Video"])


# =============================================================================
# Utility function for routes to access SAM3 engine
# =============================================================================

def get_sam3_engine():
    """Get the global SAM3 engine instance."""
    return sam3_engine


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "server.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
    )
