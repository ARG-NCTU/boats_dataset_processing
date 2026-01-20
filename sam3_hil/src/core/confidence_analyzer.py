#!/usr/bin/env python3
"""
Confidence Analyzer Module
==========================

Core module for HIL-AA (Human-in-the-Loop Active Annotation) system.
Analyzes SAM3 confidence scores to categorize detections and identify
frames requiring human review.

This is the KEY INNOVATION of the thesis:
- Use SAM3's Presence Head confidence scores to guide human review
- Automatically categorize detections into HIGH/UNCERTAIN/LOW
- Reduce human annotation effort by 5-10x

Usage:
    analyzer = ConfidenceAnalyzer(high_threshold=0.80, low_threshold=0.50)
    
    # Analyze single detection
    category = analyzer.categorize(score=0.73)  # "UNCERTAIN"
    
    # Analyze frame results
    frame_analysis = analyzer.analyze_frame(frame_result)
    
    # Analyze entire video
    video_analysis = analyzer.analyze_video(all_results)
    
    # Get frames needing review
    review_frames = analyzer.get_review_frames(all_results)
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple
from collections import defaultdict

import numpy as np

# Import from sam3_engine (assumes same package)
try:
    from .sam3_engine import FrameResult, Detection
except ImportError:
    from sam3_engine import FrameResult, Detection

logger = logging.getLogger(__name__)


# =============================================================================
# Enums and Data Classes
# =============================================================================

class ConfidenceCategory(Enum):
    """Confidence level categories."""
    HIGH = "HIGH"           # Auto-accept, no human review needed
    UNCERTAIN = "UNCERTAIN" # Needs human confirmation
    LOW = "LOW"             # Likely false positive, needs checking
    
    @property
    def color_bgr(self) -> Tuple[int, int, int]:
        """Get BGR color for visualization."""
        colors = {
            ConfidenceCategory.HIGH: (0, 255, 0),       # Green
            ConfidenceCategory.UNCERTAIN: (0, 255, 255), # Yellow
            ConfidenceCategory.LOW: (0, 0, 255),        # Red
        }
        return colors[self]
    
    @property
    def emoji(self) -> str:
        """Get emoji for display."""
        emojis = {
            ConfidenceCategory.HIGH: "🟢",
            ConfidenceCategory.UNCERTAIN: "🟡",
            ConfidenceCategory.LOW: "🔴",
        }
        return emojis[self]


@dataclass
class DetectionAnalysis:
    """Analysis result for a single detection."""
    obj_id: int
    score: float
    category: ConfidenceCategory
    needs_review: bool
    
    def __str__(self) -> str:
        return f"{self.category.emoji} Obj {self.obj_id}: {self.score:.3f} [{self.category.value}]"


@dataclass
class FrameAnalysis:
    """Analysis result for a single frame."""
    frame_index: int
    total_objects: int
    high_count: int
    uncertain_count: int
    low_count: int
    detections: List[DetectionAnalysis] = field(default_factory=list)
    
    @property
    def needs_review(self) -> bool:
        """Frame needs review if any detection needs review."""
        return self.uncertain_count > 0 or self.low_count > 0
    
    @property
    def review_priority(self) -> int:
        """Higher priority = more urgent review needed."""
        # LOW detections are highest priority (potential false positives)
        # UNCERTAIN are medium priority
        return self.low_count * 10 + self.uncertain_count
    
    def __str__(self) -> str:
        status = "⚠️ REVIEW" if self.needs_review else "✅ AUTO"
        return (f"Frame {self.frame_index}: {self.total_objects} objects "
                f"[🟢{self.high_count} 🟡{self.uncertain_count} 🔴{self.low_count}] {status}")


@dataclass
class ObjectSummary:
    """Summary statistics for a single object across all frames."""
    obj_id: int
    frame_count: int
    first_frame: int
    last_frame: int
    scores: List[float] = field(default_factory=list)
    
    @property
    def avg_score(self) -> float:
        return sum(self.scores) / len(self.scores) if self.scores else 0.0
    
    @property
    def min_score(self) -> float:
        return min(self.scores) if self.scores else 0.0
    
    @property
    def max_score(self) -> float:
        return max(self.scores) if self.scores else 0.0
    
    @property
    def score_std(self) -> float:
        return float(np.std(self.scores)) if len(self.scores) > 1 else 0.0


@dataclass
class VideoAnalysis:
    """Complete analysis result for a video."""
    total_frames: int
    total_objects: int
    unique_objects: int
    
    # Category counts (across all frame-object pairs)
    high_count: int
    uncertain_count: int
    low_count: int
    
    # Frame-level stats
    frames_need_review: int
    frames_auto_accept: int
    
    # Per-object summaries
    object_summaries: Dict[int, ObjectSummary] = field(default_factory=dict)
    
    # Frame analyses
    frame_analyses: Dict[int, FrameAnalysis] = field(default_factory=dict)
    
    # Actual intervention tracking (updated by GUI)
    frames_actually_edited: Set[int] = field(default_factory=set)
    
    @property
    def potential_hir(self) -> float:
        """Potential HIR: Percentage of frames that MIGHT need review (based on confidence)."""
        if self.total_frames == 0:
            return 0.0
        return self.frames_need_review / self.total_frames * 100
    
    @property
    def actual_hir(self) -> float:
        """Actual HIR: Percentage of frames that user ACTUALLY edited."""
        if self.total_frames == 0:
            return 0.0
        return len(self.frames_actually_edited) / self.total_frames * 100
    
    @property
    def human_intervention_rate(self) -> float:
        """HIR: Use actual if available, otherwise potential."""
        if self.frames_actually_edited:
            return self.actual_hir
        return self.potential_hir
    
    @property
    def auto_accept_rate(self) -> float:
        """Percentage of detections auto-accepted."""
        total = self.high_count + self.uncertain_count + self.low_count
        if total == 0:
            return 0.0
        return self.high_count / total * 100
    
    def get_review_frames(self) -> List[int]:
        """Get sorted list of frame indices needing review."""
        return sorted([
            idx for idx, analysis in self.frame_analyses.items()
            if analysis.needs_review
        ])
    
    def get_priority_review_frames(self, top_n: int = 10) -> List[Tuple[int, int]]:
        """Get top N frames by review priority."""
        frames_with_priority = [
            (idx, analysis.review_priority)
            for idx, analysis in self.frame_analyses.items()
            if analysis.needs_review
        ]
        frames_with_priority.sort(key=lambda x: x[1], reverse=True)
        return frames_with_priority[:top_n]


# =============================================================================
# Main Analyzer Class
# =============================================================================

class ConfidenceAnalyzer:
    """
    Analyzes SAM3 confidence scores for HIL-AA annotation workflow.
    
    Key innovation: Uses confidence scores to minimize human effort
    by automatically accepting high-confidence detections and
    flagging uncertain ones for review.
    
    Attributes:
        high_threshold: Score >= this is HIGH (auto-accept)
        low_threshold: Score < this is LOW (likely false positive)
        Between thresholds is UNCERTAIN (needs review)
    """
    
    def __init__(
        self,
        high_threshold: float = 0.80,
        low_threshold: float = 0.50
    ):
        """
        Initialize analyzer with thresholds.
        
        Args:
            high_threshold: Scores >= this are HIGH confidence
            low_threshold: Scores < this are LOW confidence
        """
        if not 0 <= low_threshold < high_threshold <= 1:
            raise ValueError(
                f"Invalid thresholds: low={low_threshold}, high={high_threshold}. "
                f"Must satisfy: 0 <= low < high <= 1"
            )
        
        self.high_threshold = high_threshold
        self.low_threshold = low_threshold
        
        logger.info(
            f"ConfidenceAnalyzer initialized: "
            f"HIGH >= {high_threshold}, LOW < {low_threshold}"
        )
    
    def categorize(self, score: float) -> ConfidenceCategory:
        """
        Categorize a confidence score.
        
        Args:
            score: Confidence score (0-1)
            
        Returns:
            ConfidenceCategory (HIGH, UNCERTAIN, or LOW)
        """
        if score >= self.high_threshold:
            return ConfidenceCategory.HIGH
        elif score >= self.low_threshold:
            return ConfidenceCategory.UNCERTAIN
        else:
            return ConfidenceCategory.LOW
    
    def needs_review(self, score: float) -> bool:
        """Check if a score needs human review."""
        category = self.categorize(score)
        return category != ConfidenceCategory.HIGH
    
    def analyze_detection(self, detection: Detection) -> DetectionAnalysis:
        """
        Analyze a single detection.
        
        Args:
            detection: Detection object from SAM3
            
        Returns:
            DetectionAnalysis with category and review status
        """
        category = self.categorize(detection.score)
        return DetectionAnalysis(
            obj_id=detection.obj_id,
            score=detection.score,
            category=category,
            needs_review=(category != ConfidenceCategory.HIGH)
        )
    
    def analyze_frame(self, frame_result: FrameResult) -> FrameAnalysis:
        """
        Analyze all detections in a frame.
        
        Args:
            frame_result: FrameResult from SAM3
            
        Returns:
            FrameAnalysis with counts and detection analyses
        """
        detection_analyses = [
            self.analyze_detection(det) 
            for det in frame_result.detections
        ]
        
        high_count = sum(1 for d in detection_analyses 
                        if d.category == ConfidenceCategory.HIGH)
        uncertain_count = sum(1 for d in detection_analyses 
                             if d.category == ConfidenceCategory.UNCERTAIN)
        low_count = sum(1 for d in detection_analyses 
                       if d.category == ConfidenceCategory.LOW)
        
        return FrameAnalysis(
            frame_index=frame_result.frame_index,
            total_objects=len(detection_analyses),
            high_count=high_count,
            uncertain_count=uncertain_count,
            low_count=low_count,
            detections=detection_analyses
        )
    
    def analyze_video(
        self, 
        results: Dict[int, FrameResult]
    ) -> VideoAnalysis:
        """
        Analyze all frames in a video.
        
        Args:
            results: Dict mapping frame_index to FrameResult
            
        Returns:
            VideoAnalysis with complete statistics
        """
        frame_analyses: Dict[int, FrameAnalysis] = {}
        object_data: Dict[int, ObjectSummary] = {}
        
        total_high = 0
        total_uncertain = 0
        total_low = 0
        total_objects = 0
        
        for frame_idx, frame_result in results.items():
            # Analyze frame
            frame_analysis = self.analyze_frame(frame_result)
            frame_analyses[frame_idx] = frame_analysis
            
            # Update counts
            total_high += frame_analysis.high_count
            total_uncertain += frame_analysis.uncertain_count
            total_low += frame_analysis.low_count
            total_objects += frame_analysis.total_objects
            
            # Update per-object data
            for det in frame_result.detections:
                if det.obj_id not in object_data:
                    object_data[det.obj_id] = ObjectSummary(
                        obj_id=det.obj_id,
                        frame_count=0,
                        first_frame=frame_idx,
                        last_frame=frame_idx,
                        scores=[]
                    )
                
                obj_summary = object_data[det.obj_id]
                obj_summary.frame_count += 1
                obj_summary.last_frame = max(obj_summary.last_frame, frame_idx)
                obj_summary.scores.append(det.score)
        
        # Count frames needing review
        frames_need_review = sum(
            1 for fa in frame_analyses.values() if fa.needs_review
        )
        
        return VideoAnalysis(
            total_frames=len(results),
            total_objects=total_objects,
            unique_objects=len(object_data),
            high_count=total_high,
            uncertain_count=total_uncertain,
            low_count=total_low,
            frames_need_review=frames_need_review,
            frames_auto_accept=len(results) - frames_need_review,
            object_summaries=object_data,
            frame_analyses=frame_analyses
        )
    
    def get_review_frames(
        self, 
        results: Dict[int, FrameResult]
    ) -> List[int]:
        """
        Get list of frame indices that need human review.
        
        Args:
            results: Dict mapping frame_index to FrameResult
            
        Returns:
            Sorted list of frame indices needing review
        """
        review_frames = []
        
        for frame_idx, frame_result in results.items():
            for det in frame_result.detections:
                if self.needs_review(det.score):
                    review_frames.append(frame_idx)
                    break  # Only need one uncertain detection to flag frame
        
        return sorted(review_frames)
    
    def get_objects_by_category(
        self,
        results: Dict[int, FrameResult]
    ) -> Dict[ConfidenceCategory, Set[int]]:
        """
        Group object IDs by their average confidence category.
        
        Args:
            results: Dict mapping frame_index to FrameResult
            
        Returns:
            Dict mapping category to set of object IDs
        """
        # Collect scores per object
        object_scores: Dict[int, List[float]] = defaultdict(list)
        
        for frame_result in results.values():
            for det in frame_result.detections:
                object_scores[det.obj_id].append(det.score)
        
        # Categorize by average score
        categorized: Dict[ConfidenceCategory, Set[int]] = {
            ConfidenceCategory.HIGH: set(),
            ConfidenceCategory.UNCERTAIN: set(),
            ConfidenceCategory.LOW: set(),
        }
        
        for obj_id, scores in object_scores.items():
            avg_score = sum(scores) / len(scores)
            category = self.categorize(avg_score)
            categorized[category].add(obj_id)
        
        return categorized
    
    def generate_report(
        self, 
        video_analysis: VideoAnalysis,
        video_name: str = "video"
    ) -> str:
        """
        Generate a human-readable analysis report.
        
        Args:
            video_analysis: VideoAnalysis from analyze_video()
            video_name: Name of video for report header
            
        Returns:
            Formatted report string
        """
        lines = [
            "=" * 70,
            f"HIL-AA Confidence Analysis Report",
            f"Video: {video_name}",
            "=" * 70,
            "",
            "📊 SUMMARY",
            "-" * 40,
            f"Total Frames:      {video_analysis.total_frames}",
            f"Unique Objects:    {video_analysis.unique_objects}",
            f"Total Detections:  {video_analysis.total_objects}",
            "",
            "📈 CONFIDENCE DISTRIBUTION",
            "-" * 40,
            f"🟢 HIGH (auto):      {video_analysis.high_count:5d} "
            f"({video_analysis.auto_accept_rate:.1f}%)",
            f"🟡 UNCERTAIN:        {video_analysis.uncertain_count:5d}",
            f"🔴 LOW (check):      {video_analysis.low_count:5d}",
            "",
            "🎯 EFFICIENCY METRICS",
            "-" * 40,
            f"Frames Auto-Accept:    {video_analysis.frames_auto_accept:5d} "
            f"({100 - video_analysis.human_intervention_rate:.1f}%)",
            f"Frames Need Review:    {video_analysis.frames_need_review:5d} "
            f"({video_analysis.human_intervention_rate:.1f}%)",
            f"Human Intervention Rate (HIR): {video_analysis.human_intervention_rate:.1f}%",
            "",
            "📋 PER-OBJECT SUMMARY",
            "-" * 40,
        ]
        
        # Sort objects by average score (descending)
        sorted_objects = sorted(
            video_analysis.object_summaries.values(),
            key=lambda x: x.avg_score,
            reverse=True
        )
        
        for obj in sorted_objects:
            category = self.categorize(obj.avg_score)
            lines.append(
                f"{category.emoji} Object {obj.obj_id:2d}: "
                f"avg={obj.avg_score:.2f} "
                f"(min={obj.min_score:.2f}, max={obj.max_score:.2f}) "
                f"[{obj.frame_count} frames, "
                f"#{obj.first_frame}-#{obj.last_frame}]"
            )
        
        # Top priority frames
        lines.extend([
            "",
            "⚠️ TOP PRIORITY REVIEW FRAMES",
            "-" * 40,
        ])
        
        priority_frames = video_analysis.get_priority_review_frames(top_n=10)
        if priority_frames:
            for frame_idx, priority in priority_frames:
                fa = video_analysis.frame_analyses[frame_idx]
                lines.append(
                    f"   Frame {frame_idx:4d}: "
                    f"priority={priority} "
                    f"[🟢{fa.high_count} 🟡{fa.uncertain_count} 🔴{fa.low_count}]"
                )
        else:
            lines.append("   None - all frames auto-accepted! 🎉")
        
        lines.extend([
            "",
            "=" * 70,
            f"Thresholds: HIGH >= {self.high_threshold}, LOW < {self.low_threshold}",
            "=" * 70,
        ])
        
        return "\n".join(lines)
    
    def update_thresholds(
        self, 
        high_threshold: Optional[float] = None,
        low_threshold: Optional[float] = None
    ) -> None:
        """
        Update confidence thresholds.
        
        Args:
            high_threshold: New high threshold (optional)
            low_threshold: New low threshold (optional)
        """
        new_high = high_threshold if high_threshold is not None else self.high_threshold
        new_low = low_threshold if low_threshold is not None else self.low_threshold
        
        if not 0 <= new_low < new_high <= 1:
            raise ValueError(
                f"Invalid thresholds: low={new_low}, high={new_high}. "
                f"Must satisfy: 0 <= low < high <= 1"
            )
        
        self.high_threshold = new_high
        self.low_threshold = new_low
        
        logger.info(
            f"Thresholds updated: HIGH >= {new_high}, LOW < {new_low}"
        )


# =============================================================================
# Test / Demo
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Confidence Analyzer Test")
    parser.add_argument("--video", type=str, 
                        default="/app/data/video/taichung_port_last500f.mp4",
                        help="Video path")
    parser.add_argument("--prompt", type=str, default="boat",
                        help="Text prompt")
    parser.add_argument("--high-thresh", type=float, default=0.80,
                        help="High confidence threshold")
    parser.add_argument("--low-thresh", type=float, default=0.50,
                        help="Low confidence threshold")
    parser.add_argument("--mode", type=str, default="mock",
                        choices=["gpu", "mock"],
                        help="SAM3 engine mode")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 70)
    print("Confidence Analyzer Test")
    print("=" * 70)
    
    # Import SAM3Engine
    try:
        from sam3_engine import SAM3Engine
    except ImportError:
        from .sam3_engine import SAM3Engine
    
    # Create analyzer
    analyzer = ConfidenceAnalyzer(
        high_threshold=args.high_thresh,
        low_threshold=args.low_thresh
    )
    
    # Run SAM3
    print(f"\n🎬 Processing video: {args.video}")
    print(f"   Prompt: '{args.prompt}'")
    print(f"   Mode: {args.mode}")
    
    with SAM3Engine(mode=args.mode) as engine:
        session_id = engine.start_video_session(args.video)
        engine.add_prompt(session_id, 0, args.prompt)
        results = engine.propagate(session_id)
        engine.close_session(session_id)
    
    # Analyze results
    print(f"\n📊 Analyzing {len(results)} frames...")
    video_analysis = analyzer.analyze_video(results)
    
    # Generate and print report
    from pathlib import Path
    video_name = Path(args.video).name
    report = analyzer.generate_report(video_analysis, video_name)
    print("\n" + report)
    
    # Show review frames
    review_frames = video_analysis.get_review_frames()
    print(f"\n📋 Frames needing review: {len(review_frames)}")
    if len(review_frames) <= 20:
        print(f"   {review_frames}")
    else:
        print(f"   First 10: {review_frames[:10]}")
        print(f"   Last 10:  {review_frames[-10:]}")
    
    print("\n" + "=" * 70)
    print("✅ Test complete!")
    print("=" * 70)
