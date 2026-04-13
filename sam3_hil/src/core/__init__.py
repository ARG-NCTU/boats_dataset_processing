"""
Core modules for STAMP Annotation System.

- video_loader: Video I/O and frame extraction
- horizon_detector: Maritime horizon line detection
- sam3_engine: SAM 3 inference wrapper
- confidence_analyzer: Active learning logic
- temporal_tracker: Video propagation and jitter detection
- exporter: Export annotations to COCO, HuggingFace Parquet, Labelme JSON
"""

from .video_loader import VideoLoader, VideoMetadata, load_video, extract_frames
try:
    from .sam3_engine import (
        SAM3Engine, 
        Detection, 
        FrameResult, 
        VideoSessionInfo,
        visualize_frame_results,
        save_visualization_video,
    )
    SAM3_AVAILABLE = True
except ImportError:
    SAM3_AVAILABLE = False
    SAM3Engine = None
    Detection = None
    FrameResult = None
    VideoSessionInfo = None
    visualize_frame_results = None
    save_visualization_video = None
    
from .confidence_analyzer import (
    ConfidenceAnalyzer,
    ConfidenceCategory,
    DetectionAnalysis,
    FrameAnalysis,
    VideoAnalysis,
    ObjectSummary,
)
from .exporter import (
    AnnotationExporter,
    ExportConfig,
    ExportStats,
    COCOExporter,
    HuggingFaceExporter,
    LabelmeExporter,
)

__all__ = [
    # video_loader
    "VideoLoader",
    "VideoMetadata",
    "load_video",
    "extract_frames",
    # sam3_engine
    "SAM3Engine",
    "Detection",
    "FrameResult",
    "VideoSessionInfo",
    "visualize_frame_results",
    "save_visualization_video",
    # confidence_analyzer
    "ConfidenceAnalyzer",
    "ConfidenceCategory",
    "DetectionAnalysis",
    "FrameAnalysis",
    "VideoAnalysis",
    "ObjectSummary",
    # exporter
    "AnnotationExporter",
    "ExportConfig",
    "ExportStats",
    "COCOExporter",
    "HuggingFaceExporter",
    "LabelmeExporter",
]
