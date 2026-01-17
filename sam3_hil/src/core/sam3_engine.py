#!/usr/bin/env python3
"""
SAM3 Engine Module
==================

Wrapper for SAM3 model providing unified API for image and video inference.
Includes Mock Mode for GUI development without GPU.

Features:
- Unified API for image and video segmentation
- Mock mode for development without GPU
- Automatic resource management
- Confidence score extraction

Usage:
    # GPU mode (requires CUDA)
    engine = SAM3Engine(mode="gpu")
    
    # Mock mode (no GPU needed)
    engine = SAM3Engine(mode="mock")
    
    # Image inference
    results = engine.detect_image(image, prompt="boat")
    
    # Video inference
    session_id = engine.start_video_session(video_path)
    results = engine.add_prompt(session_id, frame_idx=0, prompt="boat")
    all_results = engine.propagate(session_id)
    engine.close_session(session_id)
"""

import logging
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class Detection:
    """Single object detection result."""
    obj_id: int
    mask: np.ndarray          # Binary mask (H, W)
    box: np.ndarray           # Bounding box [x, y, w, h]
    score: float              # Confidence score (0-1)
    
    @property
    def box_xyxy(self) -> np.ndarray:
        """Convert xywh to xyxy format."""
        x, y, w, h = self.box
        return np.array([x, y, x + w, y + h])


@dataclass
class FrameResult:
    """Results for a single frame."""
    frame_index: int
    detections: List[Detection] = field(default_factory=list)
    
    @property
    def num_objects(self) -> int:
        return len(self.detections)
    
    @property
    def masks(self) -> List[np.ndarray]:
        return [d.mask for d in self.detections]
    
    @property
    def boxes(self) -> List[np.ndarray]:
        return [d.box for d in self.detections]
    
    @property
    def scores(self) -> List[float]:
        return [d.score for d in self.detections]
    
    @property
    def obj_ids(self) -> List[int]:
        return [d.obj_id for d in self.detections]


@dataclass
class VideoSessionInfo:
    """Video session information."""
    session_id: str
    video_path: str
    total_frames: int
    width: int
    height: int
    fps: float


# =============================================================================
# Abstract Base Engine
# =============================================================================

class BaseSAM3Engine(ABC):
    """Abstract base class for SAM3 engines."""
    
    @abstractmethod
    def detect_image(
        self, 
        image: np.ndarray, 
        prompt: str
    ) -> FrameResult:
        """Run detection on a single image."""
        pass
    
    @abstractmethod
    def refine_mask(
        self,
        image: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
        mask_input: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Refine a mask using point prompts.
        
        Args:
            image: Input image (H, W, 3) BGR numpy array
            points: Point coordinates (N, 2) array of [x, y]
            labels: Point labels (N,) array of 1 (positive) or 0 (negative)
            mask_input: Optional previous mask (H, W) to refine
            
        Returns:
            Refined mask (H, W) boolean array
        """
        pass
    
    @abstractmethod
    def start_video_session(self, video_path: str) -> str:
        """Start a video session and return session ID."""
        pass
    
    @abstractmethod
    def add_prompt(
        self, 
        session_id: str, 
        frame_index: int, 
        prompt: str
    ) -> FrameResult:
        """Add text prompt to a frame."""
        pass
    
    @abstractmethod
    def propagate(self, session_id: str) -> Dict[int, FrameResult]:
        """Propagate masks through video."""
        pass
    
    @abstractmethod
    def propagate_mask(
        self,
        video_path: str,
        start_frame: int,
        mask: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
        obj_id: int = 0,
        progress_callback: Optional[callable] = None
    ) -> Dict[int, np.ndarray]:
        """
        Propagate a mask to following frames using point prompts.
        
        Args:
            video_path: Path to the video file
            start_frame: Frame index to start propagation from
            mask: Initial mask (H, W) for reference
            points: Point coordinates (N, 2) used to define the object
            labels: Point labels (N,) - 1 for positive, 0 for negative
            obj_id: Object ID to assign
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            Dictionary mapping frame_index -> mask (H, W numpy array)
        """
        pass
    
    @abstractmethod
    def close_session(self, session_id: str) -> None:
        """Close video session and free resources."""
        pass
    
    @abstractmethod
    def shutdown(self) -> None:
        """Shutdown engine and release all resources."""
        pass


# =============================================================================
# GPU Engine (Real SAM3)
# =============================================================================

class SAM3GPUEngine(BaseSAM3Engine):
    """
    Real SAM3 engine using GPU.
    
    Requires CUDA and the sam3 package to be installed.
    """
    
    def __init__(self):
        self._image_model = None
        self._image_processor = None
        self._video_predictor = None
        self._sessions: Dict[str, VideoSessionInfo] = {}
        
        logger.info("Initializing SAM3 GPU Engine...")
        self._check_cuda()
    
    def _check_cuda(self) -> None:
        """Check CUDA availability."""
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA not available. Use mode='mock' for development without GPU."
            )
        logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    def _load_image_model(self) -> None:
        """Lazy load image model with interactivity support for point prompts."""
        if self._image_model is None:
            logger.info("Loading SAM3 image model with interactivity support...")
            from sam3.model_builder import build_sam3_image_model
            from sam3.model.sam3_image_processor import Sam3Processor
            
            # Enable inst_interactivity for point prompt support
            # See: sam3_for_sam1_task_example.ipynb
            self._image_model = build_sam3_image_model(enable_inst_interactivity=True)
            self._image_processor = Sam3Processor(self._image_model)
            logger.info("SAM3 image model loaded!")
    
    def _load_video_predictor(self) -> None:
        """Lazy load video predictor."""
        if self._video_predictor is None:
            logger.info("Loading SAM3 video predictor...")
            from sam3.model_builder import build_sam3_video_predictor
            
            self._video_predictor = build_sam3_video_predictor()
            logger.info("SAM3 video predictor loaded!")
    
    def detect_image(
        self, 
        image: np.ndarray, 
        prompt: str
    ) -> FrameResult:
        """
        Run detection on a single image.
        
        Args:
            image: Input image (BGR or RGB numpy array)
            prompt: Text prompt for detection
            
        Returns:
            FrameResult with detections
        """
        from PIL import Image as PILImage
        
        self._load_image_model()
        
        # Convert BGR to RGB if needed (assume BGR from OpenCV)
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to PIL
        pil_image = PILImage.fromarray(image_rgb)
        
        # Run inference
        inference_state = self._image_processor.set_image(pil_image)
        output = self._image_processor.set_text_prompt(
            state=inference_state, 
            prompt=prompt
        )
        
        # Parse results
        detections = []
        masks = output.get("masks", [])
        boxes = output.get("boxes", [])
        scores = output.get("scores", [])
        
        import torch
        
        for i, (mask, box, score) in enumerate(zip(masks, boxes, scores)):
            # Convert tensors to numpy
            if torch.is_tensor(mask):
                mask = mask.cpu().numpy()
            if torch.is_tensor(box):
                box = box.cpu().numpy()
            if torch.is_tensor(score):
                score = float(score.cpu().numpy())
            
            # Ensure mask is 2D
            if mask.ndim == 3:
                mask = mask[0]
            
            # Convert box from xyxy to xywh
            x1, y1, x2, y2 = box
            box_xywh = np.array([x1, y1, x2 - x1, y2 - y1])
            
            detections.append(Detection(
                obj_id=i,
                mask=(mask > 0.5).astype(np.uint8),
                box=box_xywh,
                score=float(score)
            ))
        
        return FrameResult(frame_index=0, detections=detections)
    
    def refine_mask(
        self,
        image: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
        mask_input: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Refine a mask using point prompts.
        
        Uses SAM3's predict_inst with point prompts to generate
        a refined mask. Based on sam3_for_sam1_task_example.ipynb.
        
        Args:
            image: Input image (H, W, 3) BGR numpy array
            points: Point coordinates (N, 2) array of [x, y]
            labels: Point labels (N,) array of 1 (positive) or 0 (negative)
            mask_input: Optional previous mask logits (256, 256) to use as hint
            
        Returns:
            Refined mask (H, W) boolean array
        """
        from PIL import Image as PILImage
        
        self._load_image_model()
        
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
        
        # Convert to PIL
        pil_image = PILImage.fromarray(image_rgb)
        
        # Set image and get inference state
        inference_state = self._image_processor.set_image(pil_image)
        
        # Check if we have points
        if len(points) == 0:
            # Return original mask or empty mask
            if mask_input is not None:
                return mask_input.astype(bool)
            return np.zeros((image.shape[0], image.shape[1]), dtype=bool)
        
        # Prepare point prompts - SAM3 expects (N, 2) numpy arrays
        point_coords = points.astype(np.float32)  # (N, 2)
        point_labels = labels.astype(np.int32)    # (N,)
        
        # Run prediction with points using model.predict_inst
        try:
            # Use model.predict_inst for point prompts
            # API: model.predict_inst(inference_state, point_coords, point_labels, 
            #                         box=None, mask_input=None, multimask_output=True)
            masks, scores, logits = self._image_model.predict_inst(
                inference_state,
                point_coords=point_coords,
                point_labels=point_labels,
                mask_input=mask_input[None, :, :] if mask_input is not None else None,
                multimask_output=True,
            )
            
            # Select best mask by score
            if len(masks) > 0:
                best_idx = np.argmax(scores)
                best_mask = masks[best_idx]
                
                # Ensure 2D
                if best_mask.ndim == 3:
                    best_mask = best_mask[0]
                
                return (best_mask > 0.5).astype(bool)
            else:
                # No mask produced, return original or empty
                if mask_input is not None:
                    return mask_input.astype(bool)
                return np.zeros((image.shape[0], image.shape[1]), dtype=bool)
                
        except Exception as e:
            logger.error(f"Error in refine_mask: {e}")
            # Return original mask on error
            if mask_input is not None:
                return mask_input.astype(bool)
            return np.zeros((image.shape[0], image.shape[1]), dtype=bool)
    
    def propagate_mask(
        self,
        video_path: str,
        start_frame: int,
        mask: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
        obj_id: int = 0,
        progress_callback: Optional[callable] = None
    ) -> Dict[int, np.ndarray]:
        """
        Propagate a mask to following frames using SAM3 Tracker Predictor.
        
        Uses the SAM2-style API (Sam3TrackerPredictor) for proper object tracking.
        
        Args:
            video_path: Path to the video file
            start_frame: Frame index to start propagation from
            mask: Initial mask (H, W) for reference
            points: Point coordinates (N, 2) used to define the object
            labels: Point labels (N,) - 1 for positive, 0 for negative
            obj_id: Object ID to assign
            progress_callback: Optional callback(current, total) for progress
            
        Returns:
            Dictionary mapping frame_index -> mask (H, W numpy array)
        """
        import torch
        
        # Load the tracker model (SAM2-style API)
        self._load_tracker_model()
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Initialize inference state
        inference_state = self._tracker_predictor.init_state(video_path=video_path)
        
        try:
            # Clear any previous tracking
            self._tracker_predictor.clear_all_points_in_video(inference_state)
            
            # Convert points to relative coordinates (0-1 range)
            points_rel = []
            for x, y in points:
                points_rel.append([float(x) / width, float(y) / height])
            
            points_tensor = torch.tensor(points_rel, dtype=torch.float32)
            labels_tensor = torch.tensor(labels.astype(np.int32), dtype=torch.int32)
            
            # Add prompt at start frame
            _, out_obj_ids, low_res_masks, video_res_masks = self._tracker_predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=start_frame,
                obj_id=obj_id,
                points=points_tensor,
                labels=labels_tensor,
            )
            
            logger.info(f"Added points for object {obj_id} at frame {start_frame}")
            
            # Propagate through video
            results = {}
            frame_count = 0
            remaining_frames = total_frames - start_frame
            
            for frame_idx, out_obj_ids, low_res_masks, video_res_masks, obj_scores in self._tracker_predictor.propagate_in_video(
                inference_state,
                start_frame_idx=start_frame,
                max_frame_num_to_track=remaining_frames,
                reverse=False,
                propagate_preflight=True
            ):
                # Extract mask for our object
                for i, out_obj_id in enumerate(out_obj_ids):
                    if out_obj_id == obj_id:
                        result_mask = video_res_masks[i]
                        
                        # Convert to numpy
                        if torch.is_tensor(result_mask):
                            result_mask = result_mask.cpu().numpy()
                        
                        # Ensure correct shape (H, W)
                        if result_mask.ndim == 4:  # (1, 1, H, W)
                            result_mask = result_mask[0, 0]
                        elif result_mask.ndim == 3:  # (1, H, W)
                            result_mask = result_mask[0]
                        
                        results[frame_idx] = (result_mask > 0.0).astype(np.uint8)
                        break
                
                frame_count += 1
                if progress_callback:
                    progress_callback(frame_count, remaining_frames)
            
            logger.info(f"Propagated object {obj_id} to {len(results)} frames")
            return results
            
        finally:
            # Clear the tracking state for next use
            try:
                self._tracker_predictor.clear_all_points_in_video(inference_state)
            except Exception as e:
                logger.debug(f"Could not clear inference state: {e}")
    
    def _load_tracker_model(self) -> None:
        """Load the SAM3 tracker model (SAM2-style API)."""
        if not hasattr(self, '_tracker_predictor') or self._tracker_predictor is None:
            logger.info("Loading SAM3 tracker model...")
            from sam3.model_builder import build_sam3_video_model
            
            sam3_model = build_sam3_video_model()
            self._tracker_predictor = sam3_model.tracker
            self._tracker_predictor.backbone = sam3_model.detector.backbone
            logger.info("SAM3 tracker model loaded!")
    
    def start_video_session(self, video_path: str) -> str:
        """
        Start a video session.
        
        Args:
            video_path: Path to video file
            
        Returns:
            Session ID
        """
        self._load_video_predictor()
        
        response = self._video_predictor.handle_request(
            request=dict(
                type="start_session",
                resource_path=video_path,
            )
        )
        
        session_id = response["session_id"]
        
        # Get video info
        cap = cv2.VideoCapture(video_path)
        info = VideoSessionInfo(
            session_id=session_id,
            video_path=video_path,
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=cap.get(cv2.CAP_PROP_FPS)
        )
        cap.release()
        
        self._sessions[session_id] = info
        logger.info(f"Started session: {session_id}")
        
        return session_id
    
    def add_prompt(
        self, 
        session_id: str, 
        frame_index: int, 
        prompt: str
    ) -> FrameResult:
        """
        Add text prompt to a frame.
        
        Args:
            session_id: Session ID from start_video_session
            frame_index: Frame index to add prompt
            prompt: Text prompt
            
        Returns:
            FrameResult for the prompted frame
        """
        response = self._video_predictor.handle_request(
            request=dict(
                type="add_prompt",
                session_id=session_id,
                frame_index=frame_index,
                text=prompt,
            )
        )
        
        return self._parse_video_output(frame_index, response.get("outputs", {}))
    
    def propagate(self, session_id: str) -> Dict[int, FrameResult]:
        """
        Propagate masks through video.
        
        Args:
            session_id: Session ID
            
        Returns:
            Dict mapping frame_index to FrameResult
        """
        results = {}
        
        for response in self._video_predictor.handle_stream_request(
            request=dict(
                type="propagate_in_video",
                session_id=session_id,
            )
        ):
            frame_idx = response["frame_index"]
            outputs = response["outputs"]
            results[frame_idx] = self._parse_video_output(frame_idx, outputs)
        
        logger.info(f"Propagation complete: {len(results)} frames")
        return results
    
    def _parse_video_output(
        self, 
        frame_index: int, 
        outputs: Dict[str, Any]
    ) -> FrameResult:
        """Parse SAM3 video output into FrameResult."""
        import torch
        
        detections = []
        
        obj_ids = outputs.get("out_obj_ids", [])
        masks = outputs.get("out_binary_masks", [])
        boxes = outputs.get("out_boxes_xywh", [])
        scores = outputs.get("out_probs", [])
        
        # Convert tensors
        if torch.is_tensor(obj_ids):
            obj_ids = obj_ids.cpu().numpy()
        if torch.is_tensor(masks):
            masks = masks.cpu().numpy()
        if torch.is_tensor(boxes):
            boxes = boxes.cpu().numpy()
        if torch.is_tensor(scores):
            scores = scores.cpu().numpy()
        
        # Handle case where tensors have batch dimension
        if len(obj_ids) > 0:
            for i in range(len(obj_ids)):
                obj_id = int(obj_ids[i]) if hasattr(obj_ids[i], 'item') else int(obj_ids[i])
                
                mask = masks[i] if i < len(masks) else np.zeros((1, 1), dtype=np.uint8)
                if mask.ndim == 3:
                    mask = mask[0]
                
                box = boxes[i] if i < len(boxes) else np.zeros(4)
                score = float(scores[i]) if i < len(scores) else 0.0
                
                detections.append(Detection(
                    obj_id=obj_id,
                    mask=(mask > 0.5).astype(np.uint8),
                    box=np.array(box),
                    score=score
                ))
        
        return FrameResult(frame_index=frame_index, detections=detections)
    
    def reset_session(self, session_id: str) -> None:
        """Reset session state (clear prompts but keep video loaded)."""
        self._video_predictor.handle_request(
            request=dict(
                type="reset_session",
                session_id=session_id,
            )
        )
        logger.info(f"Reset session: {session_id}")
    
    def close_session(self, session_id: str) -> None:
        """Close video session and free resources."""
        if session_id in self._sessions:
            self._video_predictor.handle_request(
                request=dict(
                    type="close_session",
                    session_id=session_id,
                )
            )
            del self._sessions[session_id]
            logger.info(f"Closed session: {session_id}")
    
    def get_session_info(self, session_id: str) -> Optional[VideoSessionInfo]:
        """Get session information."""
        return self._sessions.get(session_id)
    
    def shutdown(self) -> None:
        """Shutdown engine and release all resources."""
        # Close all sessions
        for session_id in list(self._sessions.keys()):
            self.close_session(session_id)
        
        # Shutdown video predictor
        if self._video_predictor is not None:
            self._video_predictor.shutdown()
            self._video_predictor = None
        
        # Clear image model
        self._image_model = None
        self._image_processor = None
        
        logger.info("SAM3 GPU Engine shutdown complete")


# =============================================================================
# Mock Engine (For Development)
# =============================================================================

class SAM3MockEngine(BaseSAM3Engine):
    """
    Mock SAM3 engine for GUI development without GPU.
    
    Generates fake detection results with realistic structure.
    """
    
    def __init__(self, seed: int = 42):
        self._rng = np.random.default_rng(seed)
        self._sessions: Dict[str, VideoSessionInfo] = {}
        self._session_prompts: Dict[str, str] = {}
        logger.info("Initialized SAM3 Mock Engine")
    
    def detect_image(
        self, 
        image: np.ndarray, 
        prompt: str
    ) -> FrameResult:
        """Generate mock detections for an image."""
        h, w = image.shape[:2]
        
        # Generate 1-4 random detections
        num_detections = self._rng.integers(1, 5)
        detections = []
        
        for i in range(num_detections):
            # Random bounding box
            box_w = self._rng.integers(w // 8, w // 3)
            box_h = self._rng.integers(h // 8, h // 3)
            box_x = self._rng.integers(0, w - box_w)
            box_y = self._rng.integers(0, h - box_h)
            
            # Create elliptical mask within box
            mask = np.zeros((h, w), dtype=np.uint8)
            center = (box_x + box_w // 2, box_y + box_h // 2)
            axes = (box_w // 2, box_h // 2)
            cv2.ellipse(mask, center, axes, 0, 0, 360, 1, -1)
            
            # Random confidence score
            score = self._rng.uniform(0.3, 0.98)
            
            detections.append(Detection(
                obj_id=i,
                mask=mask,
                box=np.array([box_x, box_y, box_w, box_h]),
                score=score
            ))
        
        logger.debug(f"Mock image detection: {num_detections} objects")
        return FrameResult(frame_index=0, detections=detections)
    
    def refine_mask(
        self,
        image: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
        mask_input: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """
        Mock mask refinement using point prompts.
        
        For testing, this creates a simple mask based on the points:
        - Positive points expand a circular region
        - Negative points create holes
        """
        h, w = image.shape[:2]
        
        if mask_input is not None:
            result_mask = mask_input.astype(np.float32).copy()
        else:
            result_mask = np.zeros((h, w), dtype=np.float32)
        
        # Process each point
        for i, (point, label) in enumerate(zip(points, labels)):
            x, y = int(point[0]), int(point[1])
            radius = 30 + self._rng.integers(10, 40)  # Random radius
            
            # Create circular mask for this point
            yy, xx = np.ogrid[:h, :w]
            circle = ((xx - x) ** 2 + (yy - y) ** 2) <= radius ** 2
            
            if label == 1:  # Positive: add to mask
                result_mask = np.maximum(result_mask, circle.astype(np.float32))
            else:  # Negative: remove from mask
                result_mask[circle] = 0
        
        logger.debug(f"Mock refine_mask: {len(points)} points")
        return result_mask > 0.5
    
    def propagate_mask(
        self,
        video_path: str,
        start_frame: int,
        mask: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
        obj_id: int = 0,
        progress_callback: Optional[callable] = None
    ) -> Dict[int, np.ndarray]:
        """
        Mock propagate_mask - simulates tracking with gradual drift.
        
        For testing, this creates masks that gradually move to simulate
        object tracking (with some random drift).
        """
        # Get video info
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        results = {}
        
        # Find the center of the initial mask
        ys, xs = np.where(mask)
        if len(xs) > 0:
            center_x = int(xs.mean())
            center_y = int(ys.mean())
            mask_w = xs.max() - xs.min()
            mask_h = ys.max() - ys.min()
        else:
            center_x, center_y = width // 2, height // 2
            mask_w, mask_h = 100, 100
        
        remaining = total_frames - start_frame
        
        for i, frame_idx in enumerate(range(start_frame, total_frames)):
            # Simulate tracking with small random movement
            drift_x = self._rng.integers(-3, 4)
            drift_y = self._rng.integers(-3, 4)
            center_x = max(mask_w//2, min(width - mask_w//2, center_x + drift_x))
            center_y = max(mask_h//2, min(height - mask_h//2, center_y + drift_y))
            
            # Create mask at new position
            frame_mask = np.zeros((height, width), dtype=np.uint8)
            cv2.ellipse(
                frame_mask,
                (center_x, center_y),
                (mask_w // 2, mask_h // 2),
                0, 0, 360, 1, -1
            )
            
            results[frame_idx] = frame_mask
            
            if progress_callback:
                progress_callback(i + 1, remaining)
        
        logger.info(f"Mock propagated to {len(results)} frames")
        return results
    
    def start_video_session(self, video_path: str) -> str:
        """Start mock video session."""
        session_id = str(uuid.uuid4())
        
        # Get real video info
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
        
        info = VideoSessionInfo(
            session_id=session_id,
            video_path=video_path,
            total_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
            width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            fps=cap.get(cv2.CAP_PROP_FPS)
        )
        cap.release()
        
        self._sessions[session_id] = info
        logger.info(f"Mock session started: {session_id}")
        
        return session_id
    
    def add_prompt(
        self, 
        session_id: str, 
        frame_index: int, 
        prompt: str
    ) -> FrameResult:
        """Add mock prompt and generate initial detections."""
        if session_id not in self._sessions:
            raise ValueError(f"Unknown session: {session_id}")
        
        self._session_prompts[session_id] = prompt
        info = self._sessions[session_id]
        
        # Generate mock detections
        num_detections = self._rng.integers(1, 4)
        detections = []
        
        for i in range(num_detections):
            # Random box
            box_w = self._rng.integers(info.width // 10, info.width // 4)
            box_h = self._rng.integers(info.height // 10, info.height // 4)
            box_x = self._rng.integers(0, info.width - box_w)
            box_y = self._rng.integers(0, info.height - box_h)
            
            # Mock mask
            mask = np.zeros((info.height, info.width), dtype=np.uint8)
            cv2.rectangle(mask, (box_x, box_y), (box_x + box_w, box_y + box_h), 1, -1)
            
            detections.append(Detection(
                obj_id=i,
                mask=mask,
                box=np.array([box_x, box_y, box_w, box_h]),
                score=self._rng.uniform(0.5, 0.95)
            ))
        
        logger.debug(f"Mock prompt added: {num_detections} objects")
        return FrameResult(frame_index=frame_index, detections=detections)
    
    def propagate(self, session_id: str) -> Dict[int, FrameResult]:
        """Generate mock propagation results."""
        if session_id not in self._sessions:
            raise ValueError(f"Unknown session: {session_id}")
        
        info = self._sessions[session_id]
        results = {}
        
        # Generate base detections
        num_objects = self._rng.integers(1, 4)
        base_boxes = []
        
        for i in range(num_objects):
            box_w = self._rng.integers(info.width // 10, info.width // 4)
            box_h = self._rng.integers(info.height // 10, info.height // 4)
            box_x = self._rng.integers(0, info.width - box_w)
            box_y = self._rng.integers(0, info.height - box_h)
            base_boxes.append([box_x, box_y, box_w, box_h])
        
        # Propagate with slight movement
        for frame_idx in range(info.total_frames):
            detections = []
            
            for i, base_box in enumerate(base_boxes):
                # Add small random movement
                dx = self._rng.integers(-5, 6)
                dy = self._rng.integers(-3, 4)
                
                box_x = max(0, min(base_box[0] + dx, info.width - base_box[2]))
                box_y = max(0, min(base_box[1] + dy, info.height - base_box[3]))
                box_w, box_h = base_box[2], base_box[3]
                
                # Update base for next frame
                base_boxes[i][0] = box_x
                base_boxes[i][1] = box_y
                
                # Mock mask
                mask = np.zeros((info.height, info.width), dtype=np.uint8)
                cv2.ellipse(
                    mask, 
                    (box_x + box_w // 2, box_y + box_h // 2),
                    (box_w // 2, box_h // 2),
                    0, 0, 360, 1, -1
                )
                
                # Varying confidence
                score = self._rng.uniform(0.4, 0.95)
                
                detections.append(Detection(
                    obj_id=i,
                    mask=mask,
                    box=np.array([box_x, box_y, box_w, box_h]),
                    score=score
                ))
            
            results[frame_idx] = FrameResult(
                frame_index=frame_idx, 
                detections=detections
            )
        
        logger.info(f"Mock propagation complete: {len(results)} frames")
        return results
    
    def reset_session(self, session_id: str) -> None:
        """Reset mock session."""
        if session_id in self._session_prompts:
            del self._session_prompts[session_id]
        logger.debug(f"Mock session reset: {session_id}")
    
    def close_session(self, session_id: str) -> None:
        """Close mock session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
        if session_id in self._session_prompts:
            del self._session_prompts[session_id]
        logger.info(f"Mock session closed: {session_id}")
    
    def get_session_info(self, session_id: str) -> Optional[VideoSessionInfo]:
        """Get session information."""
        return self._sessions.get(session_id)
    
    def shutdown(self) -> None:
        """Shutdown mock engine."""
        self._sessions.clear()
        self._session_prompts.clear()
        logger.info("SAM3 Mock Engine shutdown complete")


# =============================================================================
# Main Engine Factory
# =============================================================================

class SAM3Engine:
    """
    SAM3 Engine factory and facade.
    
    Automatically selects GPU or Mock engine based on mode parameter.
    
    Usage:
        # Auto-detect (GPU if available, else Mock)
        engine = SAM3Engine()
        
        # Force GPU mode
        engine = SAM3Engine(mode="gpu")
        
        # Force Mock mode (for development)
        engine = SAM3Engine(mode="mock")
    """
    
    def __init__(self, mode: str = "auto"):
        """
        Initialize SAM3 Engine.
        
        Args:
            mode: "gpu", "mock", or "auto" (default)
        """
        self.mode = mode
        self._engine: BaseSAM3Engine = self._create_engine(mode)
    
    def _create_engine(self, mode: str) -> BaseSAM3Engine:
        """Create appropriate engine based on mode."""
        if mode == "mock":
            return SAM3MockEngine()
        
        if mode == "gpu":
            return SAM3GPUEngine()
        
        # Auto mode - try GPU, fallback to Mock
        if mode == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    logger.info("Auto mode: Using GPU engine")
                    return SAM3GPUEngine()
            except ImportError:
                pass
            
            logger.info("Auto mode: Using Mock engine (no GPU available)")
            return SAM3MockEngine()
        
        raise ValueError(f"Unknown mode: {mode}. Use 'gpu', 'mock', or 'auto'")
    
    @property
    def is_mock(self) -> bool:
        """Check if using mock engine."""
        return isinstance(self._engine, SAM3MockEngine)
    
    @property
    def is_gpu(self) -> bool:
        """Check if using GPU engine."""
        return isinstance(self._engine, SAM3GPUEngine)
    
    # Delegate all methods to internal engine
    def detect_image(self, image: np.ndarray, prompt: str) -> FrameResult:
        """Run detection on a single image."""
        return self._engine.detect_image(image, prompt)
    
    def refine_mask(
        self,
        image: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
        mask_input: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Refine a mask using point prompts."""
        return self._engine.refine_mask(image, points, labels, mask_input)
    
    def start_video_session(self, video_path: str) -> str:
        """Start a video session."""
        return self._engine.start_video_session(video_path)
    
    def add_prompt(
        self, 
        session_id: str, 
        frame_index: int, 
        prompt: str
    ) -> FrameResult:
        """Add text prompt to a frame."""
        return self._engine.add_prompt(session_id, frame_index, prompt)
    
    def propagate(self, session_id: str) -> Dict[int, FrameResult]:
        """Propagate masks through video."""
        return self._engine.propagate(session_id)
    
    def propagate_mask(
        self,
        video_path: str,
        start_frame: int,
        mask: np.ndarray,
        points: np.ndarray,
        labels: np.ndarray,
        obj_id: int = 0,
        progress_callback: Optional[callable] = None
    ) -> Dict[int, np.ndarray]:
        """
        Propagate a mask to following frames using point prompts.
        
        Uses SAM3 Tracker for proper object tracking.
        """
        return self._engine.propagate_mask(
            video_path=video_path,
            start_frame=start_frame,
            mask=mask,
            points=points,
            labels=labels,
            obj_id=obj_id,
            progress_callback=progress_callback
        )
    
    def reset_session(self, session_id: str) -> None:
        """Reset session state."""
        if hasattr(self._engine, 'reset_session'):
            self._engine.reset_session(session_id)
    
    def close_session(self, session_id: str) -> None:
        """Close video session."""
        self._engine.close_session(session_id)
    
    def get_session_info(self, session_id: str) -> Optional[VideoSessionInfo]:
        """Get session information."""
        if hasattr(self._engine, 'get_session_info'):
            return self._engine.get_session_info(session_id)
        return None
    
    def shutdown(self) -> None:
        """Shutdown engine."""
        self._engine.shutdown()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.shutdown()


# =============================================================================
# Visualization Helper
# =============================================================================

def visualize_frame_results(
    frame: np.ndarray,
    result: FrameResult,
    high_threshold: float = 0.85,
    low_threshold: float = 0.50
) -> np.ndarray:
    """
    Visualize detection results on a frame with object IDs and confidence scores.
    
    Color coding:
    - GREEN: HIGH confidence (>= high_threshold) - auto accept
    - YELLOW: UNCERTAIN confidence - needs review
    - RED: LOW confidence (< low_threshold) - likely false positive
    
    Args:
        frame: Input frame (BGR)
        result: FrameResult with detections
        high_threshold: Threshold for HIGH confidence
        low_threshold: Threshold for LOW confidence
        
    Returns:
        Annotated frame
    """
    output = frame.copy()
    overlay = frame.copy()
    
    for det in result.detections:
        # Determine color based on confidence
        if det.score >= high_threshold:
            color = (0, 255, 0)      # GREEN - HIGH
        elif det.score >= low_threshold:
            color = (0, 255, 255)    # YELLOW - UNCERTAIN
        else:
            color = (0, 0, 255)      # RED - LOW
        
        # Draw mask overlay
        mask = det.mask
        if mask.shape[:2] != frame.shape[:2]:
            mask = cv2.resize(mask.astype(np.float32), 
                            (frame.shape[1], frame.shape[0]))
        mask_bool = mask > 0.5
        overlay[mask_bool] = color
        
        # Get bounding box
        x, y, w, h = det.box.astype(int)
        
        # Draw bounding box (thinner line)
        cv2.rectangle(output, (x, y), (x + w, y + h), color, 1)
        
        # Calculate mask top-center for label placement
        mask_coords = np.where(mask_bool)
        if len(mask_coords[0]) > 0:
            top_y = int(np.min(mask_coords[0]))
            center_x = int(np.mean(mask_coords[1]))
        else:
            center_x = x + w // 2
            top_y = y
        
        # Draw compact label above mask
        label = f"{det.obj_id}:{det.score:.2f}"
        font_scale = 0.45
        thickness = 1
        (text_w, text_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
        )
        
        # Position label above mask
        label_x = center_x - text_w // 2
        label_y = max(top_y - 5, text_h + 5)
        
        # Background rectangle
        cv2.rectangle(output, 
                     (label_x - 2, label_y - text_h - 2), 
                     (label_x + text_w + 2, label_y + 2), 
                     color, -1)
        
        # Text
        cv2.putText(output, label, (label_x, label_y),
                   cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness)
    
    # Blend overlay
    result_frame = cv2.addWeighted(overlay, 0.3, output, 0.7, 0)
    
    # Add frame info (smaller)
    info = f"Frame {result.frame_index} | Objects: {result.num_objects}"
    cv2.putText(result_frame, info, (10, 20),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Add legend (smaller)
    cv2.putText(result_frame, "GREEN=AUTO YELLOW=REVIEW RED=CHECK", 
               (10, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    return result_frame


def save_visualization_video(
    video_path: str,
    results: Dict[int, FrameResult],
    output_path: str,
    high_threshold: float = 0.85,
    low_threshold: float = 0.50,
    max_frames: int = 0
) -> None:
    """
    Save visualization video with annotated detections.
    
    Args:
        video_path: Input video path
        results: Dict of frame_index -> FrameResult
        output_path: Output video path
        high_threshold: Threshold for HIGH confidence
        low_threshold: Threshold for LOW confidence
        max_frames: Maximum frames to process (0 = all)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if max_frames <= 0:
        max_frames = total_frames
    else:
        max_frames = min(max_frames, total_frames)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    print(f"⏳ Saving visualization to: {output_path}")
    
    frame_idx = 0
    while frame_idx < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_idx in results:
            frame = visualize_frame_results(
                frame, results[frame_idx],
                high_threshold, low_threshold
            )
        
        out.write(frame)
        frame_idx += 1
        
        if frame_idx % 50 == 0:
            print(f"   Processed {frame_idx}/{max_frames} frames...")
    
    cap.release()
    out.release()
    print(f"✅ Saved: {output_path}")


# =============================================================================
# Test / Demo
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SAM3 Engine Test")
    parser.add_argument("--mode", type=str, default="mock", 
                        choices=["gpu", "mock", "auto"],
                        help="Engine mode")
    parser.add_argument("--video", type=str, 
                        default="/app/data/video/taichung_port_last500f.mp4",
                        help="Video path for testing")
    parser.add_argument("--image", type=str, default=None,
                        help="Image path for testing")
    parser.add_argument("--prompt", type=str, default="boat, ship",
                        help="Text prompt")
    parser.add_argument("--output", type=str, default="/app/data/output",
                        help="Output directory")
    parser.add_argument("--high-thresh", type=float, default=0.85,
                        help="High confidence threshold")
    parser.add_argument("--low-thresh", type=float, default=0.50,
                        help="Low confidence threshold")
    parser.add_argument("--save-video", action="store_true",
                        help="Save visualization video")
    args = parser.parse_args()
    
    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    print("=" * 60)
    print(f"SAM3 Engine Test (mode={args.mode})")
    print("=" * 60)
    
    with SAM3Engine(mode=args.mode) as engine:
        print(f"\n📦 Engine type: {'Mock' if engine.is_mock else 'GPU'}")
        print(f"📊 Thresholds: HIGH >= {args.high_thresh}, LOW < {args.low_thresh}")
        
        # Test image detection
        if args.image:
            print(f"\n🖼️  Testing image detection...")
            image = cv2.imread(args.image)
            if image is None:
                print(f"❌ Cannot read image: {args.image}")
            else:
                result = engine.detect_image(image, args.prompt)
                print(f"✅ Found {result.num_objects} objects")
                for det in result.detections:
                    # Categorize
                    if det.score >= args.high_thresh:
                        status = "🟢 HIGH"
                    elif det.score >= args.low_thresh:
                        status = "🟡 UNCERTAIN"
                    else:
                        status = "🔴 LOW"
                    print(f"   Object {det.obj_id}: {det.score:.3f} {status}")
                
                # Save annotated image
                if args.save_video:
                    output_img = visualize_frame_results(
                        image, result, args.high_thresh, args.low_thresh
                    )
                    out_path = Path(args.output) / f"annotated_{Path(args.image).name}"
                    cv2.imwrite(str(out_path), output_img)
                    print(f"📁 Saved: {out_path}")
        
        # Test video session
        print(f"\n🎬 Testing video session...")
        print(f"   Video: {args.video}")
        
        try:
            session_id = engine.start_video_session(args.video)
            info = engine.get_session_info(session_id)
            print(f"✅ Session started: {session_id[:8]}...")
            if info:
                print(f"   Frames: {info.total_frames}, "
                      f"Size: {info.width}x{info.height}")
            
            # Add prompt
            print(f"\n⏳ Adding prompt: '{args.prompt}'...")
            result = engine.add_prompt(session_id, 0, args.prompt)
            print(f"✅ Found {result.num_objects} objects on frame 0")
            
            # Show categorized results
            high_count = sum(1 for d in result.detections if d.score >= args.high_thresh)
            uncertain_count = sum(1 for d in result.detections 
                                 if args.low_thresh <= d.score < args.high_thresh)
            low_count = sum(1 for d in result.detections if d.score < args.low_thresh)
            
            print(f"\n📊 Confidence Distribution:")
            print(f"   🟢 HIGH (auto):     {high_count}")
            print(f"   🟡 UNCERTAIN:       {uncertain_count}")
            print(f"   🔴 LOW (check):     {low_count}")
            
            for det in result.detections:
                if det.score >= args.high_thresh:
                    status = "🟢"
                elif det.score >= args.low_thresh:
                    status = "🟡"
                else:
                    status = "🔴"
                print(f"   {status} Object {det.obj_id}: {det.score:.3f}")
            
            # Propagate
            print(f"\n⏳ Propagating through video...")
            all_results = engine.propagate(session_id)
            print(f"✅ Got results for {len(all_results)} frames")
            
            # Collect all unique objects across all frames
            all_objects: Dict[int, List[float]] = {}  # obj_id -> list of scores
            for frame_idx, frame_result in all_results.items():
                for det in frame_result.detections:
                    if det.obj_id not in all_objects:
                        all_objects[det.obj_id] = []
                    all_objects[det.obj_id].append(det.score)
            
            # Print summary of all objects
            print(f"\n📊 All Objects Summary ({len(all_objects)} unique objects):")
            for obj_id in sorted(all_objects.keys()):
                scores = all_objects[obj_id]
                avg_score = sum(scores) / len(scores)
                min_score = min(scores)
                max_score = max(scores)
                frame_count = len(scores)
                
                # Categorize by average score
                if avg_score >= args.high_thresh:
                    status = "🟢"
                elif avg_score >= args.low_thresh:
                    status = "🟡"
                else:
                    status = "🔴"
                
                print(f"   {status} Object {obj_id}: avg={avg_score:.2f} "
                      f"(min={min_score:.2f}, max={max_score:.2f}) "
                      f"[{frame_count} frames]")
            
            # Sample some frames
            sample_frames = [0, len(all_results) // 2, len(all_results) - 1]
            print(f"\n📋 Sample Frames:")
            for idx in sample_frames:
                if idx in all_results:
                    r = all_results[idx]
                    scores_str = ", ".join([f"{s:.2f}" for s in r.scores])
                    print(f"   Frame {idx}: {r.num_objects} objects [{scores_str}]")
            
            # Save visualization video
            if args.save_video:
                Path(args.output).mkdir(parents=True, exist_ok=True)
                video_name = Path(args.video).stem
                prompt_clean = args.prompt.replace(" ", "_").replace(",", "")
                out_path = Path(args.output) / f"viz_{video_name}_{prompt_clean}.mp4"
                
                save_visualization_video(
                    args.video, all_results, str(out_path),
                    args.high_thresh, args.low_thresh
                )
            
            # Close session
            engine.close_session(session_id)
            print(f"\n✅ Session closed")
            
        except FileNotFoundError as e:
            print(f"❌ Video not found: {e}")
        except Exception as e:
            print(f"❌ Error: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("✅ Test complete!")
    print("=" * 60)
