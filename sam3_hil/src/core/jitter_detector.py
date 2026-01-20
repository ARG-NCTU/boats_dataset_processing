#!/usr/bin/env python3
"""
Jitter Detector Module
======================

Detects temporal inconsistencies (jitter) in object tracking across video frames.

This is a KEY CONTRIBUTION of the thesis:
- Monitor mask shape changes between consecutive frames
- Flag frames where tracking may have failed
- Trigger human review when jitter is detected

Jitter Detection Algorithm:
1. For each object, compute IoU between consecutive frame masks
2. If IoU drops below threshold (default 0.85), flag as jitter
3. Also detect sudden area changes (>15% change)

Usage:
    detector = JitterDetector(iou_threshold=0.85, area_change_threshold=0.15)
    
    # Analyze entire video
    jitter_results = detector.analyze_video(sam3_results)
    
    # Get frames needing review due to jitter
    jitter_frames = jitter_results.get_jitter_frames()
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class JitterEvent:
    """A single jitter event between two frames."""
    obj_id: int
    frame_from: int
    frame_to: int
    iou: float
    area_change: float  # Percentage change (can be negative)
    jitter_type: str    # "low_iou", "area_spike", "appearance", "disappearance"
    
    @property
    def severity(self) -> float:
        """Severity score (0-1, higher = more severe)."""
        if self.jitter_type == "disappearance":
            return 1.0
        elif self.jitter_type == "appearance":
            return 0.5
        elif self.jitter_type == "low_iou":
            return 1.0 - self.iou
        else:  # area_spike
            return min(1.0, abs(self.area_change))


@dataclass
class ObjectJitterAnalysis:
    """Jitter analysis for a single object across the video."""
    obj_id: int
    total_frames: int
    jitter_events: List[JitterEvent] = field(default_factory=list)
    frame_ious: Dict[int, float] = field(default_factory=dict)  # frame -> IoU with previous
    
    @property
    def jitter_count(self) -> int:
        return len(self.jitter_events)
    
    @property
    def jitter_rate(self) -> float:
        """Percentage of frames with jitter."""
        if self.total_frames <= 1:
            return 0.0
        return self.jitter_count / (self.total_frames - 1) * 100
    
    @property
    def avg_iou(self) -> float:
        """Average IoU between consecutive frames."""
        if not self.frame_ious:
            return 1.0
        return sum(self.frame_ious.values()) / len(self.frame_ious)
    
    @property
    def stability_score(self) -> float:
        """Overall stability score (0-1, higher = more stable)."""
        if self.total_frames <= 1:
            return 1.0
        # Combine avg IoU and jitter rate
        iou_factor = self.avg_iou
        jitter_factor = 1.0 - min(1.0, self.jitter_rate / 20)  # 20% jitter = 0 stability
        return (iou_factor + jitter_factor) / 2
    
    def get_jitter_frames(self) -> List[int]:
        """Get list of frames where jitter occurred."""
        return sorted(set(e.frame_to for e in self.jitter_events))


@dataclass
class VideoJitterAnalysis:
    """Complete jitter analysis for a video."""
    total_frames: int
    total_objects: int
    object_analyses: Dict[int, ObjectJitterAnalysis] = field(default_factory=dict)
    
    @property
    def total_jitter_events(self) -> int:
        return sum(a.jitter_count for a in self.object_analyses.values())
    
    @property
    def jitter_frame_count(self) -> int:
        """Number of frames with at least one jitter event."""
        jitter_frames = set()
        for analysis in self.object_analyses.values():
            jitter_frames.update(analysis.get_jitter_frames())
        return len(jitter_frames)
    
    @property
    def overall_stability(self) -> float:
        """Overall video stability score (0-1)."""
        if not self.object_analyses:
            return 1.0
        return sum(a.stability_score for a in self.object_analyses.values()) / len(self.object_analyses)
    
    def get_all_jitter_frames(self) -> List[int]:
        """Get sorted list of all frames with jitter."""
        jitter_frames = set()
        for analysis in self.object_analyses.values():
            jitter_frames.update(analysis.get_jitter_frames())
        return sorted(jitter_frames)
    
    def get_jitter_frames_by_severity(self, min_severity: float = 0.3) -> List[Tuple[int, float]]:
        """Get frames with jitter, sorted by severity."""
        frame_severity: Dict[int, float] = defaultdict(float)
        
        for analysis in self.object_analyses.values():
            for event in analysis.jitter_events:
                frame_severity[event.frame_to] = max(
                    frame_severity[event.frame_to],
                    event.severity
                )
        
        result = [(f, s) for f, s in frame_severity.items() if s >= min_severity]
        result.sort(key=lambda x: x[1], reverse=True)
        return result


# =============================================================================
# Jitter Detector
# =============================================================================

class JitterDetector:
    """
    Detects temporal jitter in object tracking.
    
    Jitter types:
    - low_iou: Mask shape changed significantly between frames
    - area_spike: Mask area changed by more than threshold
    - appearance: Object suddenly appeared (wasn't in previous frame)
    - disappearance: Object suddenly disappeared (was in previous frame)
    """
    
    def __init__(
        self,
        iou_threshold: float = 0.85,
        area_change_threshold: float = 0.15,
        min_mask_area: int = 100
    ):
        """
        Initialize Jitter Detector.
        
        Args:
            iou_threshold: Minimum IoU between consecutive frames (below = jitter)
            area_change_threshold: Maximum area change ratio (above = jitter)
            min_mask_area: Minimum mask area to consider (pixels)
        """
        self.iou_threshold = iou_threshold
        self.area_change_threshold = area_change_threshold
        self.min_mask_area = min_mask_area
        
        logger.info(
            f"JitterDetector initialized: IoU >= {iou_threshold}, "
            f"area change <= {area_change_threshold * 100}%"
        )
    
    def compute_iou(self, mask1: np.ndarray, mask2: np.ndarray) -> float:
        """Compute IoU between two binary masks."""
        if mask1 is None or mask2 is None:
            return 0.0
        
        # Ensure binary
        mask1 = (mask1 > 0).astype(np.uint8)
        mask2 = (mask2 > 0).astype(np.uint8)
        
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        
        if union == 0:
            return 1.0 if intersection == 0 else 0.0
        
        return intersection / union
    
    def compute_area_change(self, area1: int, area2: int) -> float:
        """Compute relative area change."""
        if area1 == 0:
            return 1.0 if area2 > 0 else 0.0
        return (area2 - area1) / area1
    
    def analyze_object(
        self,
        obj_id: int,
        masks_by_frame: Dict[int, np.ndarray]
    ) -> ObjectJitterAnalysis:
        """
        Analyze jitter for a single object.
        
        Args:
            obj_id: Object ID
            masks_by_frame: Dict mapping frame_index -> mask array
            
        Returns:
            ObjectJitterAnalysis with jitter events
        """
        if not masks_by_frame:
            return ObjectJitterAnalysis(obj_id=obj_id, total_frames=0)
        
        sorted_frames = sorted(masks_by_frame.keys())
        analysis = ObjectJitterAnalysis(
            obj_id=obj_id,
            total_frames=len(sorted_frames)
        )
        
        prev_frame = None
        prev_mask = None
        prev_area = 0
        
        for frame_idx in sorted_frames:
            mask = masks_by_frame[frame_idx]
            area = mask.sum() if mask is not None else 0
            
            if prev_frame is not None:
                # Check for appearance/disappearance
                if prev_mask is None or prev_area < self.min_mask_area:
                    if area >= self.min_mask_area:
                        # Object appeared
                        analysis.jitter_events.append(JitterEvent(
                            obj_id=obj_id,
                            frame_from=prev_frame,
                            frame_to=frame_idx,
                            iou=0.0,
                            area_change=1.0,
                            jitter_type="appearance"
                        ))
                elif mask is None or area < self.min_mask_area:
                    # Object disappeared
                    analysis.jitter_events.append(JitterEvent(
                        obj_id=obj_id,
                        frame_from=prev_frame,
                        frame_to=frame_idx,
                        iou=0.0,
                        area_change=-1.0,
                        jitter_type="disappearance"
                    ))
                else:
                    # Both frames have valid masks - compute IoU
                    iou = self.compute_iou(prev_mask, mask)
                    area_change = self.compute_area_change(prev_area, area)
                    
                    analysis.frame_ious[frame_idx] = iou
                    
                    # Check for jitter
                    if iou < self.iou_threshold:
                        analysis.jitter_events.append(JitterEvent(
                            obj_id=obj_id,
                            frame_from=prev_frame,
                            frame_to=frame_idx,
                            iou=iou,
                            area_change=area_change,
                            jitter_type="low_iou"
                        ))
                    elif abs(area_change) > self.area_change_threshold:
                        analysis.jitter_events.append(JitterEvent(
                            obj_id=obj_id,
                            frame_from=prev_frame,
                            frame_to=frame_idx,
                            iou=iou,
                            area_change=area_change,
                            jitter_type="area_spike"
                        ))
            
            prev_frame = frame_idx
            prev_mask = mask
            prev_area = area
        
        return analysis
    
    def analyze_video(
        self,
        sam3_results: Dict[int, 'FrameResult']
    ) -> VideoJitterAnalysis:
        """
        Analyze jitter for entire video.
        
        Args:
            sam3_results: Dict mapping frame_index -> FrameResult
            
        Returns:
            VideoJitterAnalysis with all jitter events
        """
        if not sam3_results:
            return VideoJitterAnalysis(total_frames=0, total_objects=0)
        
        # Group masks by object
        masks_by_object: Dict[int, Dict[int, np.ndarray]] = defaultdict(dict)
        all_obj_ids = set()
        
        for frame_idx, result in sam3_results.items():
            if result is None:
                continue
            for det in result.detections:
                masks_by_object[det.obj_id][frame_idx] = det.mask
                all_obj_ids.add(det.obj_id)
        
        # Analyze each object
        total_frames = max(sam3_results.keys()) + 1 if sam3_results else 0
        analysis = VideoJitterAnalysis(
            total_frames=total_frames,
            total_objects=len(all_obj_ids)
        )
        
        for obj_id, masks in masks_by_object.items():
            obj_analysis = self.analyze_object(obj_id, masks)
            analysis.object_analyses[obj_id] = obj_analysis
        
        logger.info(
            f"Jitter analysis complete: {analysis.total_jitter_events} events "
            f"in {analysis.jitter_frame_count} frames, "
            f"stability: {analysis.overall_stability:.1%}"
        )
        
        return analysis


# =============================================================================
# Utility Functions
# =============================================================================

def format_jitter_summary(analysis: VideoJitterAnalysis) -> str:
    """Format jitter analysis as human-readable summary."""
    lines = [
        "=== Jitter Analysis Summary ===",
        f"Total frames: {analysis.total_frames}",
        f"Total objects: {analysis.total_objects}",
        f"Overall stability: {analysis.overall_stability:.1%}",
        f"Jitter events: {analysis.total_jitter_events}",
        f"Frames with jitter: {analysis.jitter_frame_count}",
        "",
        "Per-object breakdown:"
    ]
    
    for obj_id, obj_analysis in sorted(analysis.object_analyses.items()):
        stability = obj_analysis.stability_score
        emoji = "🟢" if stability > 0.9 else "🟡" if stability > 0.7 else "🔴"
        lines.append(
            f"  Object {obj_id}: {emoji} stability={stability:.1%}, "
            f"jitter_events={obj_analysis.jitter_count}"
        )
    
    return "\n".join(lines)


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    # Simple test
    logging.basicConfig(level=logging.INFO)
    
    detector = JitterDetector(iou_threshold=0.85, area_change_threshold=0.15)
    
    # Create fake masks
    h, w = 100, 100
    
    # Object 0: stable tracking
    masks_obj0 = {}
    for i in range(10):
        mask = np.zeros((h, w), dtype=np.uint8)
        mask[20:50, 30+i:60+i] = 1  # Slowly moving rectangle
        masks_obj0[i] = mask
    
    # Object 1: jittery tracking
    masks_obj1 = {}
    for i in range(10):
        mask = np.zeros((h, w), dtype=np.uint8)
        if i == 5:
            # Sudden jump
            mask[60:80, 60:80] = 1
        else:
            mask[20:40, 20:40] = 1
        masks_obj1[i] = mask
    
    # Analyze
    obj0_analysis = detector.analyze_object(0, masks_obj0)
    obj1_analysis = detector.analyze_object(1, masks_obj1)
    
    print(f"Object 0: stability={obj0_analysis.stability_score:.2f}, jitter_events={obj0_analysis.jitter_count}")
    print(f"Object 1: stability={obj1_analysis.stability_score:.2f}, jitter_events={obj1_analysis.jitter_count}")
    
    for event in obj1_analysis.jitter_events:
        print(f"  Jitter at frame {event.frame_to}: {event.jitter_type}, IoU={event.iou:.2f}")
