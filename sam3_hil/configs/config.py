"""
HIL-AA Maritime Annotation System - Configuration
=================================================

All configurable parameters in one place.
NO hardcoded thresholds elsewhere in the codebase.

Usage:
    from src.config import config
    
    threshold = config.confidence.high_threshold
"""

from pathlib import Path
from typing import Literal, Optional, Tuple

from pydantic import Field
from pydantic_settings import BaseSettings


# =============================================================================
# Path Configuration
# =============================================================================
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"


# =============================================================================
# Confidence Thresholds (Core Active Learning)
# =============================================================================
class ConfidenceConfig(BaseSettings):
    """
    Confidence-Driven Active Learning thresholds.
    
    SAM 3's Presence Head outputs a score [0, 1] indicating
    semantic confidence (not just geometric precision).
    """
    
    # High confidence → Auto-save (GREEN)
    high_threshold: float = Field(
        default=0.9,
        ge=0.0, le=1.0,
        description="Score >= this → auto-save without human review"
    )
    
    # Low confidence → Needs Review (RED)
    low_threshold: float = Field(
        default=0.7,
        ge=0.0, le=1.0,
        description="Score < this → requires human review"
    )
    
    # Between low and high → Uncertain (YELLOW)
    # No explicit field needed, computed from high - low range
    
    class Config:
        env_prefix = "HIL_CONF_"


# =============================================================================
# Temporal Tracking Configuration
# =============================================================================
class TemporalConfig(BaseSettings):
    """
    Temporal propagation and jitter detection settings.
    """
    
    # Jitter Detection: aspect ratio change threshold
    jitter_threshold: float = Field(
        default=0.15,  # 15% shape change triggers review
        ge=0.0, le=1.0,
        description="Aspect ratio change > this → tracking may have failed"
    )
    
    # Maximum frames to propagate from a keyframe
    max_propagation_frames: int = Field(
        default=100,
        ge=1,
        description="Force new keyframe after this many frames"
    )
    
    # Confidence drop threshold for keyframe creation
    confidence_drop_threshold: float = Field(
        default=0.2,
        ge=0.0, le=1.0,
        description="If confidence drops by this much, create new keyframe"
    )
    
    # IoU threshold for temporal consistency
    temporal_iou_threshold: float = Field(
        default=0.5,
        ge=0.0, le=1.0,
        description="Minimum IoU between consecutive frames"
    )
    
    class Config:
        env_prefix = "HIL_TEMP_"


# =============================================================================
# SAM 3 Model Configuration
# =============================================================================
class SAM3Config(BaseSettings):
    """
    SAM 3 model settings.
    """
    
    # Model variant
    model_type: Literal["sam3_base", "sam3_large", "sam3_huge"] = Field(
        default="sam3_large",
        description="SAM 3 model size variant"
    )
    
    # Model checkpoint path (if local)
    checkpoint_path: Optional[Path] = Field(
        default=None,
        description="Local checkpoint path, or None for auto-download"
    )
    
    # Device configuration
    device: str = Field(
        default="cuda",
        description="Compute device: cuda, cpu, or mps"
    )
    
    # Mock mode for development without GPU
    mock_mode: bool = Field(
        default=False,
        description="Use mock predictions for GUI development"
    )
    
    # Inference settings
    points_per_side: int = Field(
        default=32,
        ge=1,
        description="Points per side for automatic mask generation"
    )
    
    pred_iou_thresh: float = Field(
        default=0.88,
        ge=0.0, le=1.0,
        description="Predicted IoU threshold for filtering masks"
    )
    
    stability_score_thresh: float = Field(
        default=0.95,
        ge=0.0, le=1.0,
        description="Stability score threshold for filtering masks"
    )
    
    class Config:
        env_prefix = "HIL_SAM3_"


# =============================================================================
# Maritime ROI Configuration
# =============================================================================
class MaritimeConfig(BaseSettings):
    """
    Maritime-specific settings (horizon detection, ROI).
    """
    
    # Enable horizon-based ROI filtering
    enable_horizon_roi: bool = Field(
        default=True,
        description="Filter detections to below horizon line"
    )
    
    # Horizon detection method
    horizon_method: Literal["hough", "edge", "learned", "manual"] = Field(
        default="edge",
        description="Algorithm for horizon line detection"
    )
    
    # Horizon margin (pixels above horizon to include)
    horizon_margin: int = Field(
        default=50,
        ge=0,
        description="Include this many pixels above detected horizon"
    )
    
    # Minimum object size (filter tiny detections)
    min_object_area: int = Field(
        default=100,  # pixels
        ge=0,
        description="Minimum mask area in pixels"
    )
    
    # Maximum object area ratio (filter too-large detections)
    max_object_ratio: float = Field(
        default=0.8,  # 80% of image
        ge=0.0, le=1.0,
        description="Maximum mask area as ratio of image area"
    )
    
    class Config:
        env_prefix = "HIL_MARITIME_"


# =============================================================================
# GUI Configuration
# =============================================================================
class GUIConfig(BaseSettings):
    """
    GUI appearance and behavior settings.
    """
    
    # Window defaults
    window_width: int = Field(default=1600, ge=800)
    window_height: int = Field(default=900, ge=600)
    window_title: str = Field(default="HIL-AA Maritime Annotation")
    
    # Colors (RGB tuples as comma-separated strings for env compatibility)
    color_positive: Tuple[int, int, int] = Field(
        default=(0, 255, 0),
        description="Positive point marker color (GREEN)"
    )
    color_negative: Tuple[int, int, int] = Field(
        default=(255, 0, 0),
        description="Negative point marker color (RED)"
    )
    color_mask_overlay: Tuple[int, int, int] = Field(
        default=(0, 255, 0),
        description="Mask overlay color"
    )
    
    # Timeline colors
    color_auto_save: str = Field(default="#22c55e", description="Green for high confidence")
    color_needs_review: str = Field(default="#ef4444", description="Red for low confidence")
    color_uncertain: str = Field(default="#eab308", description="Yellow for uncertain")
    
    # Marker sizes
    point_marker_radius: int = Field(default=8, ge=2, le=20)
    mask_overlay_alpha: float = Field(default=0.3, ge=0.0, le=1.0)
    
    # Playback
    default_fps: int = Field(default=30, ge=1, le=120)
    
    class Config:
        env_prefix = "HIL_GUI_"


# =============================================================================
# Export Configuration
# =============================================================================
class ExportConfig(BaseSettings):
    """
    Export format settings.
    """
    
    # Default export format
    default_format: Literal["coco", "parquet", "both"] = Field(
        default="coco",
        description="Default annotation export format"
    )
    
    # COCO settings
    coco_include_crowd: bool = Field(default=False)
    coco_category_names: list[str] = Field(
        default=["ship", "buoy", "obstacle", "person", "unknown"],
        description="Default category names for COCO export"
    )
    
    # Output paths
    output_dir: Path = Field(
        default=OUTPUT_DIR,
        description="Directory for exported annotations"
    )
    
    class Config:
        env_prefix = "HIL_EXPORT_"


# =============================================================================
# Logging Configuration
# =============================================================================
class LoggingConfig(BaseSettings):
    """
    Logging settings.
    """
    
    level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(default="INFO")
    log_to_file: bool = Field(default=True)
    log_dir: Path = Field(default=PROJECT_ROOT / "logs")
    
    class Config:
        env_prefix = "HIL_LOG_"


# =============================================================================
# Main Configuration Class
# =============================================================================
class AppConfig(BaseSettings):
    """
    Main application configuration combining all sub-configs.
    """
    
    confidence: ConfidenceConfig = Field(default_factory=ConfidenceConfig)
    temporal: TemporalConfig = Field(default_factory=TemporalConfig)
    sam3: SAM3Config = Field(default_factory=SAM3Config)
    maritime: MaritimeConfig = Field(default_factory=MaritimeConfig)
    gui: GUIConfig = Field(default_factory=GUIConfig)
    export: ExportConfig = Field(default_factory=ExportConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# =============================================================================
# Global Config Instance
# =============================================================================
config = AppConfig()


# =============================================================================
# Utility Functions
# =============================================================================
def ensure_directories() -> None:
    """Create required directories if they don't exist."""
    for dir_path in [DATA_DIR, MODELS_DIR, OUTPUT_DIR, config.logging.log_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)


def print_config() -> None:
    """Print current configuration for debugging."""
    import json
    print("=" * 60)
    print("HIL-AA Configuration")
    print("=" * 60)
    print(json.dumps(config.model_dump(), indent=2, default=str))
    print("=" * 60)


# =============================================================================
# Quick Access Aliases
# =============================================================================
# For convenience in other modules
HIGH_THRESHOLD = config.confidence.high_threshold
LOW_THRESHOLD = config.confidence.low_threshold
JITTER_THRESHOLD = config.temporal.jitter_threshold


if __name__ == "__main__":
    ensure_directories()
    print_config()