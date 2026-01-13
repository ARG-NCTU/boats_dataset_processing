#!/usr/bin/env python3
"""
Video Loader Module
===================

Provides efficient video loading, frame extraction, and navigation
for the HIL-AA Maritime Annotation System.

Features:
- Random access to any frame
- Frame range extraction
- Thumbnail generation for timeline
- LRU cache for frequently accessed frames
- Metadata extraction

Usage:
    loader = VideoLoader("/path/to/video.mp4")
    frame = loader.get_frame(100)
    thumbnail = loader.get_thumbnail(100, size=(160, 90))
    metadata = loader.get_metadata()
"""

import cv2
import numpy as np
from pathlib import Path
from functools import lru_cache
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class VideoMetadata:
    """Video metadata container."""
    path: str
    filename: str
    width: int
    height: int
    fps: float
    total_frames: int
    duration_seconds: float
    codec: str
    
    def __str__(self) -> str:
        return (
            f"{self.filename}: {self.width}x{self.height} @ {self.fps:.2f}fps, "
            f"{self.total_frames} frames ({self.duration_seconds:.1f}s)"
        )


class VideoLoader:
    """
    Efficient video loader with random access and caching.
    
    Attributes:
        video_path: Path to the video file
        metadata: VideoMetadata object with video properties
    """
    
    SUPPORTED_FORMATS = {'.mp4', '.avi', '.mov', '.mkv', '.webm', '.m4v'}
    
    def __init__(self, video_path: str, cache_size: int = 128):
        """
        Initialize VideoLoader.
        
        Args:
            video_path: Path to video file
            cache_size: Number of frames to cache (default: 128)
        """
        self.video_path = Path(video_path)
        self._validate_video()
        
        # Initialize video capture
        self._cap = cv2.VideoCapture(str(self.video_path))
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        # Extract metadata
        self.metadata = self._extract_metadata()
        
        # Setup frame cache
        self._cache_size = cache_size
        self._frame_cache: Dict[int, np.ndarray] = {}
        self._cache_order: List[int] = []
        
        logger.info(f"Loaded video: {self.metadata}")
    
    def _validate_video(self) -> None:
        """Validate video file exists and has supported format."""
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video not found: {self.video_path}")
        
        suffix = self.video_path.suffix.lower()
        if suffix not in self.SUPPORTED_FORMATS:
            logger.warning(
                f"Format '{suffix}' may not be fully supported. "
                f"Supported: {self.SUPPORTED_FORMATS}"
            )
    
    def _extract_metadata(self) -> VideoMetadata:
        """Extract video metadata from capture."""
        # Get codec info
        fourcc = int(self._cap.get(cv2.CAP_PROP_FOURCC))
        codec = "".join([chr((fourcc >> 8 * i) & 0xFF) for i in range(4)])
        
        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = self._cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Handle edge cases
        if fps <= 0:
            fps = 30.0
            logger.warning(f"Invalid FPS detected, using default: {fps}")
        
        duration = total_frames / fps if fps > 0 else 0
        
        return VideoMetadata(
            path=str(self.video_path),
            filename=self.video_path.name,
            width=width,
            height=height,
            fps=fps,
            total_frames=total_frames,
            duration_seconds=duration,
            codec=codec
        )
    
    def _add_to_cache(self, frame_idx: int, frame: np.ndarray) -> None:
        """Add frame to cache with LRU eviction."""
        if frame_idx in self._frame_cache:
            # Move to end (most recently used)
            self._cache_order.remove(frame_idx)
            self._cache_order.append(frame_idx)
            return
        
        # Evict oldest if cache is full
        while len(self._cache_order) >= self._cache_size:
            oldest = self._cache_order.pop(0)
            del self._frame_cache[oldest]
        
        # Add new frame
        self._frame_cache[frame_idx] = frame.copy()
        self._cache_order.append(frame_idx)
    
    def _get_from_cache(self, frame_idx: int) -> Optional[np.ndarray]:
        """Get frame from cache if available."""
        if frame_idx in self._frame_cache:
            # Move to end (most recently used)
            self._cache_order.remove(frame_idx)
            self._cache_order.append(frame_idx)
            return self._frame_cache[frame_idx].copy()
        return None
    
    def get_frame(self, index: int, use_cache: bool = True) -> np.ndarray:
        """
        Get a single frame by index.
        
        Args:
            index: Frame index (0-based)
            use_cache: Whether to use frame cache (default: True)
            
        Returns:
            Frame as numpy array (BGR format)
            
        Raises:
            IndexError: If index is out of range
        """
        # Validate index
        if index < 0 or index >= self.metadata.total_frames:
            raise IndexError(
                f"Frame index {index} out of range [0, {self.metadata.total_frames - 1}]"
            )
        
        # Check cache first
        if use_cache:
            cached = self._get_from_cache(index)
            if cached is not None:
                return cached
        
        # Seek to frame
        self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame = self._cap.read()
        
        if not ret or frame is None:
            raise RuntimeError(f"Failed to read frame {index}")
        
        # Add to cache
        if use_cache:
            self._add_to_cache(index, frame)
        
        return frame
    
    def get_frame_rgb(self, index: int, use_cache: bool = True) -> np.ndarray:
        """
        Get a single frame in RGB format.
        
        Args:
            index: Frame index (0-based)
            use_cache: Whether to use frame cache (default: True)
            
        Returns:
            Frame as numpy array (RGB format)
        """
        frame_bgr = self.get_frame(index, use_cache)
        return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    def get_frame_range(
        self, 
        start: int, 
        end: int, 
        step: int = 1,
        use_cache: bool = True
    ) -> List[np.ndarray]:
        """
        Get a range of frames.
        
        Args:
            start: Start frame index (inclusive)
            end: End frame index (exclusive)
            step: Step between frames (default: 1)
            use_cache: Whether to use frame cache (default: True)
            
        Returns:
            List of frames as numpy arrays (BGR format)
        """
        frames = []
        for idx in range(start, min(end, self.metadata.total_frames), step):
            frames.append(self.get_frame(idx, use_cache))
        return frames
    
    def get_thumbnail(
        self, 
        index: int, 
        size: Tuple[int, int] = (160, 90)
    ) -> np.ndarray:
        """
        Get a thumbnail of a frame.
        
        Args:
            index: Frame index
            size: Thumbnail size as (width, height)
            
        Returns:
            Resized frame as numpy array (BGR format)
        """
        frame = self.get_frame(index)
        return cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    
    def get_thumbnail_rgb(
        self, 
        index: int, 
        size: Tuple[int, int] = (160, 90)
    ) -> np.ndarray:
        """
        Get a thumbnail in RGB format.
        
        Args:
            index: Frame index
            size: Thumbnail size as (width, height)
            
        Returns:
            Resized frame as numpy array (RGB format)
        """
        thumbnail = self.get_thumbnail(index, size)
        return cv2.cvtColor(thumbnail, cv2.COLOR_BGR2RGB)
    
    def generate_timeline_thumbnails(
        self, 
        num_thumbnails: int = 10,
        size: Tuple[int, int] = (160, 90)
    ) -> List[Tuple[int, np.ndarray]]:
        """
        Generate evenly spaced thumbnails for timeline display.
        
        Args:
            num_thumbnails: Number of thumbnails to generate
            size: Thumbnail size as (width, height)
            
        Returns:
            List of (frame_index, thumbnail) tuples
        """
        if num_thumbnails <= 0:
            return []
        
        step = max(1, self.metadata.total_frames // num_thumbnails)
        thumbnails = []
        
        for i in range(num_thumbnails):
            idx = min(i * step, self.metadata.total_frames - 1)
            thumb = self.get_thumbnail(idx, size)
            thumbnails.append((idx, thumb))
        
        return thumbnails
    
    def get_metadata(self) -> VideoMetadata:
        """Get video metadata."""
        return self.metadata
    
    def frame_to_timestamp(self, frame_idx: int) -> float:
        """
        Convert frame index to timestamp in seconds.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Timestamp in seconds
        """
        return frame_idx / self.metadata.fps
    
    def timestamp_to_frame(self, timestamp: float) -> int:
        """
        Convert timestamp to frame index.
        
        Args:
            timestamp: Time in seconds
            
        Returns:
            Frame index
        """
        frame_idx = int(timestamp * self.metadata.fps)
        return max(0, min(frame_idx, self.metadata.total_frames - 1))
    
    def format_timestamp(self, frame_idx: int) -> str:
        """
        Format frame index as MM:SS.mmm timestamp.
        
        Args:
            frame_idx: Frame index
            
        Returns:
            Formatted timestamp string
        """
        seconds = self.frame_to_timestamp(frame_idx)
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes:02d}:{secs:06.3f}"
    
    def clear_cache(self) -> None:
        """Clear the frame cache."""
        self._frame_cache.clear()
        self._cache_order.clear()
        logger.debug("Frame cache cleared")
    
    def preload_range(self, start: int, end: int) -> None:
        """
        Preload a range of frames into cache.
        
        Useful for smooth playback of a specific section.
        
        Args:
            start: Start frame index
            end: End frame index
        """
        for idx in range(start, min(end, self.metadata.total_frames)):
            if idx not in self._frame_cache:
                try:
                    self.get_frame(idx, use_cache=True)
                except RuntimeError:
                    logger.warning(f"Failed to preload frame {idx}")
    
    def __len__(self) -> int:
        """Return total number of frames."""
        return self.metadata.total_frames
    
    def __getitem__(self, index: int) -> np.ndarray:
        """Enable indexing: loader[100] returns frame 100."""
        return self.get_frame(index)
    
    def __iter__(self):
        """Enable iteration over all frames."""
        for idx in range(self.metadata.total_frames):
            yield self.get_frame(idx)
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - release resources."""
        self.release()
    
    def release(self) -> None:
        """Release video capture resources."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self.clear_cache()
        logger.debug(f"Released video: {self.video_path}")
    
    def __del__(self):
        """Destructor - ensure resources are released."""
        self.release()


# =============================================================================
# Convenience functions
# =============================================================================

def load_video(video_path: str, cache_size: int = 128) -> VideoLoader:
    """
    Convenience function to load a video.
    
    Args:
        video_path: Path to video file
        cache_size: Number of frames to cache
        
    Returns:
        VideoLoader instance
    """
    return VideoLoader(video_path, cache_size)


def extract_frames(
    video_path: str, 
    output_dir: str, 
    step: int = 1,
    format: str = "jpg"
) -> List[str]:
    """
    Extract frames from video and save to directory.
    
    Args:
        video_path: Path to video file
        output_dir: Output directory for frames
        step: Extract every N-th frame
        format: Output image format (jpg, png)
        
    Returns:
        List of saved frame paths
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    saved_paths = []
    
    with VideoLoader(video_path) as loader:
        for idx in range(0, loader.metadata.total_frames, step):
            frame = loader.get_frame(idx)
            frame_path = output_path / f"frame_{idx:06d}.{format}"
            cv2.imwrite(str(frame_path), frame)
            saved_paths.append(str(frame_path))
    
    return saved_paths


# =============================================================================
# Test / Demo
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Video Loader Test")
    parser.add_argument("video", type=str, help="Path to video file")
    parser.add_argument("--frame", type=int, default=0, help="Frame to extract")
    parser.add_argument("--output", type=str, default=None, help="Output path for frame")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    print("=" * 60)
    print("Video Loader Test")
    print("=" * 60)
    
    with VideoLoader(args.video) as loader:
        # Print metadata
        print(f"\n📹 {loader.metadata}")
        print(f"   Codec: {loader.metadata.codec}")
        print(f"   Duration: {loader.format_timestamp(loader.metadata.total_frames - 1)}")
        
        # Get specific frame
        print(f"\n⏳ Loading frame {args.frame}...")
        frame = loader.get_frame(args.frame)
        print(f"✅ Frame shape: {frame.shape}")
        print(f"   Timestamp: {loader.format_timestamp(args.frame)}")
        
        # Save if output specified
        if args.output:
            cv2.imwrite(args.output, frame)
            print(f"\n📁 Saved to: {args.output}")
        
        # Test cache
        print(f"\n⏳ Testing cache (re-reading frame {args.frame})...")
        frame2 = loader.get_frame(args.frame)
        print(f"✅ Cache hit - same frame retrieved")
        
        # Generate thumbnails
        print(f"\n⏳ Generating timeline thumbnails...")
        thumbnails = loader.generate_timeline_thumbnails(num_thumbnails=5)
        for idx, thumb in thumbnails:
            print(f"   Frame {idx}: {thumb.shape}")
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")
    print("=" * 60)
