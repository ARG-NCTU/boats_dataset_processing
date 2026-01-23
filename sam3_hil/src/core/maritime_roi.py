#!/usr/bin/env python3
"""
Maritime ROI Module
===================

Uses horizon detection to exclude non-maritime regions (sky/shore) from SAM3 detection.

Methods:
- 'traditional': Fast edge detection + Hough transform
- 'segformer': Accurate semantic segmentation  
- 'auto': Fallback strategy (Traditional first, SegFormer if failed)

Usage:
    roi = MaritimeROI(method='auto', segformer_model_path='models/Segformer/segformer_model')
    horizon = roi.detect_horizon(frame)
    sky_box = roi.get_sky_box_normalized_cxcywh(frame, horizon)  # For SAM3 negative box

Author: Sonic (Maritime Robotics Lab, NYCU)
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

import numpy as np
import cv2

warnings.filterwarnings("ignore", message="The value of the smallest subnormal")

logger = logging.getLogger(__name__)


@dataclass
class HorizonResult:
    """Result of horizon detection."""
    slope: float
    center: Tuple[int, int]  # (x, y)
    valid: bool
    method_used: str  # 'traditional', 'segformer', 'none'
    
    def to_dict(self) -> dict:
        return {
            'slope': self.slope,
            'center': self.center,
            'valid': self.valid,
            'method_used': self.method_used
        }


# =============================================================================
# Traditional Horizon Detector
# =============================================================================

class TraditionalHorizonDetector:
    """Fast horizon detection using edge detection + Hough transform."""
    
    def __init__(
        self,
        roi_ratio: List[float] = None,
        canny_th1: int = 25,
        canny_th2: int = 45,
        resize_factor: float = 0.6
    ):
        self.roi_ratio = roi_ratio if roi_ratio is not None else [0.3, 0.7, 0.3, 0.7]
        self.canny_th1 = canny_th1
        self.canny_th2 = canny_th2
        self.resize_factor = resize_factor
        
        # Check cv2.ximgproc availability
        if not hasattr(cv2, 'ximgproc') or not hasattr(cv2.ximgproc, 'createFastLineDetector'):
            logger.warning("cv2.ximgproc not available")
            self.fsd = None
        else:
            # 使用位置參數避免不同 OpenCV 版本的命名參數差異
            # createFastLineDetector(length_threshold, distance_threshold, canny_th1, canny_th2, ...)
            try:
                self.fsd = cv2.ximgproc.createFastLineDetector(
                    10,              # length_threshold
                    1.414,           # distance_threshold  
                    self.canny_th1,  # canny_th1
                    self.canny_th2   # canny_th2
                )
            except Exception as e:
                logger.warning(f"createFastLineDetector failed: {e}")
                self.fsd = None
        
        logger.info(f"TraditionalHorizonDetector initialized: roi={self.roi_ratio}")
    
    def detect(self, img: np.ndarray) -> HorizonResult:
        """Detect horizon in image."""
        h, w = img.shape[:2]
        
        if self.fsd is None:
            return HorizonResult(0.0, (w // 2, h // 2), False, 'traditional')
        
        # Extract ROI
        x0, x1, y0, y1 = self.roi_ratio
        roi_x = int(w * x0)
        roi_y = int(h * y0)
        roi_w = int(w * (x1 - x0))
        roi_h = int(h * (y1 - y0))
        
        roi_img = img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
        
        # Resize and use red channel
        if self.resize_factor < 1.0:
            proc_img = cv2.resize(roi_img[:, :, 2], None, 
                                  fx=self.resize_factor, fy=self.resize_factor)
        else:
            proc_img = roi_img[:, :, 2]
        
        # Detect line segments
        segments = self.fsd.detect(proc_img)
        
        if segments is None or len(segments) == 0:
            return HorizonResult(0.0, (w // 2, h // 2), False, 'traditional')
        
        # Filter near-horizontal lines
        segments = segments.reshape(-1, 4)
        valid_segments = []
        
        for seg in segments:
            xs, ys, xe, ye = seg
            dx = xe - xs
            if abs(dx) > 1e-6:
                slope = (ye - ys) / dx
                if abs(slope) < 0.58:  # ~30 degrees
                    valid_segments.append(seg)
        
        if len(valid_segments) == 0:
            return HorizonResult(0.0, (w // 2, h // 2), False, 'traditional')
        
        # Use longest segments
        valid_segments = np.array(valid_segments)
        lengths = np.sqrt((valid_segments[:, 2] - valid_segments[:, 0])**2 + 
                          (valid_segments[:, 3] - valid_segments[:, 1])**2)
        
        n_top = min(15, len(lengths))
        top_indices = np.argsort(lengths)[-n_top:]
        top_segments = valid_segments[top_indices]
        
        # Collect points
        points = []
        for seg in top_segments:
            xs, ys, xe, ye = seg
            if self.resize_factor < 1.0:
                xs, xe = xs / self.resize_factor, xe / self.resize_factor
                ys, ye = ys / self.resize_factor, ye / self.resize_factor
            points.append([xs, ys])
            points.append([xe, ye])
        
        points = np.array(points, dtype=np.float32)
        
        if len(points) < 4:
            return HorizonResult(0.0, (w // 2, h // 2), False, 'traditional')
        
        # Fit line
        vx, vy, x0_fit, y0_fit = cv2.fitLine(points.astype(np.int32), cv2.DIST_L2, 0, 0.01, 0.01)
        slope = float(vy / vx) if abs(vx) > 1e-6 else 0.0
        
        # Convert to full image coordinates
        center_x_roi = roi_w // 2
        center_y_roi = int(slope * (center_x_roi - x0_fit) + y0_fit)
        
        center_x = w // 2
        center_y = roi_y + center_y_roi
        center_y = max(0, min(h - 1, int(center_y)))
        
        return HorizonResult(slope, (center_x, center_y), True, 'traditional')


# =============================================================================
# SegFormer Horizon Detector
# =============================================================================

class SegFormerHorizonDetector:
    """Accurate horizon detection using SegFormer semantic segmentation."""
    
    def __init__(
        self,
        model_path: str,
        roi_ratio: List[float] = None,
        water_id: int = 1,
        window_ratio: float = 0.5
    ):
        self.model_path = model_path
        self.roi_ratio = roi_ratio if roi_ratio is not None else [0.3, 0.7, 0.3, 0.7]
        self.water_id = water_id
        self.window_ratio = window_ratio
        
        self.model = None
        self.processor = None
        self.device = None
        
        logger.info(f"SegFormerHorizonDetector initialized (lazy load)")
    
    def _load_model(self):
        """Lazy load the model."""
        if self.model is not None:
            return
        
        import torch
        from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = SegformerImageProcessor.from_pretrained(self.model_path)
        self.model = SegformerForSemanticSegmentation.from_pretrained(self.model_path)
        self.model = self.model.to(self.device)
        
        if self.device == "cuda":
            self.model = self.model.half()
        
        self.model.eval()
        logger.info(f"SegFormer model loaded on {self.device}")
    
    def detect(self, img: np.ndarray) -> HorizonResult:
        """Detect horizon using SegFormer."""
        h, w = img.shape[:2]
        
        try:
            self._load_model()
        except Exception as e:
            logger.error(f"SegFormer load failed: {e}")
            return HorizonResult(0.0, (w // 2, h // 2), False, 'segformer')
        
        import torch
        import torch.nn.functional as F
        from PIL import Image
        
        # Convert BGR to RGB PIL
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(img_rgb)
        
        # Inference
        inputs = self.processor(images=pil_img, return_tensors="pt")
        if self.model.dtype == torch.float16:
            inputs = {k: v.half().to(self.device) for k, v in inputs.items()}
        else:
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get mask
        logits = outputs.logits
        mask = torch.argmax(logits, dim=1)[0]
        
        mask = F.interpolate(
            mask.unsqueeze(0).unsqueeze(0).float(),
            size=(h, w), mode="nearest"
        )[0, 0].cpu().numpy().astype(np.uint8)
        
        return self._estimate_horizon_from_mask(mask, h, w)
    
    def _estimate_horizon_from_mask(self, mask: np.ndarray, h: int, w: int) -> HorizonResult:
        """Estimate horizon from segmentation mask."""
        water_mask = (mask == self.water_id).astype(np.uint8)
        
        x0, x1, y0, y1 = self.roi_ratio
        roi_x0 = int(w * x0)
        roi_x1 = int(w * x1)
        roi_y0 = int(h * y0)
        roi_y1 = int(h * y1)
        
        # Find horizon y for each x in ROI
        horizon_y = np.full(w, np.nan, dtype=np.float32)
        
        for x in range(roi_x0, roi_x1):
            col = water_mask[roi_y0:roi_y1, x]
            ys = np.where(col > 0)[0]
            if len(ys) > 0:
                horizon_y[x] = roi_y0 + ys[0]
        
        # Fit line in center window
        mid_x = (roi_x0 + roi_x1) // 2
        win_half = max(2, int((roi_x1 - roi_x0) * self.window_ratio * 0.5))
        x_range = np.arange(mid_x - win_half, mid_x + win_half + 1)
        x_range = x_range[(x_range >= roi_x0) & (x_range < roi_x1)]
        y_range = horizon_y[x_range]
        
        valid = ~np.isnan(y_range)
        if valid.sum() < 2:
            return HorizonResult(0.0, (w // 2, h // 2), False, 'segformer')
        
        x_fit = x_range[valid]
        y_fit = y_range[valid]
        
        A = np.vstack([x_fit, np.ones_like(x_fit)]).T
        m, b = np.linalg.lstsq(A, y_fit, rcond=None)[0]
        
        center_x = w // 2
        center_y = int(round(m * center_x + b))
        center_y = max(0, min(h - 1, center_y))
        
        return HorizonResult(float(m), (center_x, center_y), True, 'segformer')


# =============================================================================
# Maritime ROI (Main Interface)
# =============================================================================

class MaritimeROI:
    """
    Maritime ROI detector using horizon detection.
    
    Methods:
        - 'traditional': Fast (edge detection + Hough transform)
        - 'segformer': Accurate (semantic segmentation)
        - 'auto': Fallback (Traditional first, SegFormer if failed)
    """
    
    @staticmethod
    def _get_default_segformer_path() -> str:
        """
        智能選擇 SegFormer 模型路徑。
        
        優先順序：
        1. Docker 環境：/app/models/Segformer/segformer_model
        2. Host 環境：~/sam3_hil/models/Segformer/segformer_model
        """
        import os
        
        # Docker 環境
        docker_path = Path("/app/models/Segformer/segformer_model")
        if docker_path.exists() and (docker_path / "config.json").exists():
            return str(docker_path)
        
        # Host 環境
        host_path = Path.home() / "sam3_hil" / "models" / "Segformer" / "segformer_model"
        if host_path.exists() and (host_path / "config.json").exists():
            return str(host_path)
        
        # 預設返回 Docker 路徑（讓錯誤訊息更清楚）
        if Path("/app").exists():
            return str(docker_path)
        return str(host_path)
    
    def __init__(
        self,
        method: str = 'auto',
        segformer_model_path: str = None,
        roi_ratio: List[float] = None,
        sky_margin: float = 0.02
    ):
        """
        Initialize Maritime ROI.
        
        Args:
            method: 'traditional', 'segformer', or 'auto'
            segformer_model_path: Path to SegFormer model
            roi_ratio: [x0, x1, y0, y1] for detection region
            sky_margin: Extra margin below horizon (ratio of height)
        """
        self.method = method
        self.roi_ratio = roi_ratio if roi_ratio is not None else [0.3, 0.7, 0.3, 0.7]
        self.sky_margin = sky_margin
        
        # Initialize detectors
        self.traditional_detector = TraditionalHorizonDetector(roi_ratio=self.roi_ratio)
        
        self.segformer_detector = None
        if method in ['segformer', 'auto']:
            model_path = segformer_model_path or self._get_default_segformer_path()
            try:
                self.segformer_detector = SegFormerHorizonDetector(
                    model_path=model_path,
                    roi_ratio=self.roi_ratio
                )
            except Exception as e:
                logger.warning(f"SegFormer init failed: {e}")
        
        logger.info(f"MaritimeROI initialized: method={method}")
    
    def detect_horizon(self, frame: np.ndarray) -> HorizonResult:
        """Detect horizon in frame."""
        h, w = frame.shape[:2]
        
        if self.method == 'traditional':
            return self.traditional_detector.detect(frame)
        
        elif self.method == 'segformer':
            if self.segformer_detector is None:
                return HorizonResult(0.0, (w // 2, h // 2), False, 'none')
            return self.segformer_detector.detect(frame)
        
        elif self.method == 'auto':
            result = self.traditional_detector.detect(frame)
            
            if result.valid:
                return result
            
            if self.segformer_detector is not None:
                logger.info("Traditional failed, falling back to SegFormer")
                return self.segformer_detector.detect(frame)
            
            return result
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def get_sky_box_xyxy(self, frame: np.ndarray, horizon: HorizonResult) -> Optional[List[int]]:
        """Get sky region as [x1, y1, x2, y2] in absolute pixels."""
        if not horizon.valid:
            return None
        
        h, w = frame.shape[:2]
        slope = horizon.slope
        cx, cy = horizon.center
        
        # Calculate y at edges
        y_left = cy + slope * (0 - cx)
        y_right = cy + slope * (w - 1 - cx)
        
        # Sky box: from top to horizon + margin
        margin = int(h * self.sky_margin)
        y_max = int(max(y_left, y_right)) + margin
        y_max = max(0, min(h - 1, y_max))
        
        if y_max <= 0:
            return None
        
        return [0, 0, w, y_max]
    
    def get_sky_box_normalized_cxcywh(self, frame: np.ndarray, horizon: HorizonResult) -> Optional[List[float]]:
        """Get sky region as [cx, cy, w, h] normalized (0-1) for Sam3Processor."""
        xyxy = self.get_sky_box_xyxy(frame, horizon)
        if xyxy is None:
            return None
        
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = xyxy
        
        cx = (x1 + x2) / 2.0 / w
        cy = (y1 + y2) / 2.0 / h
        bw = (x2 - x1) / w
        bh = (y2 - y1) / h
        
        return [cx, cy, bw, bh]
    
    def get_sky_box_normalized_xyxy(self, frame: np.ndarray, horizon: HorizonResult) -> Optional[List[float]]:
        """Get sky region as [x1, y1, x2, y2] normalized (0-1) for Sam3TrackerPredictor."""
        xyxy = self.get_sky_box_xyxy(frame, horizon)
        if xyxy is None:
            return None
        
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = xyxy
        
        return [x1 / w, y1 / h, x2 / w, y2 / h]
    
    def visualize(self, frame: np.ndarray, horizon: HorizonResult) -> np.ndarray:
        """Visualize horizon and sky box on frame."""
        vis = frame.copy()
        h, w = frame.shape[:2]
        
        if not horizon.valid:
            cv2.putText(vis, "Horizon: FAILED", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            return vis
        
        slope = horizon.slope
        cx, cy = horizon.center
        
        # Draw horizon line
        y_left = int(cy + slope * (0 - cx))
        y_right = int(cy + slope * (w - 1 - cx))
        cv2.line(vis, (0, y_left), (w - 1, y_right), (0, 0, 255), 2, cv2.LINE_AA)
        cv2.circle(vis, (cx, cy), 5, (0, 255, 0), -1, cv2.LINE_AA)
        
        # Draw sky box (semi-transparent)
        sky_box = self.get_sky_box_xyxy(frame, horizon)
        if sky_box:
            x1, y1, x2, y2 = sky_box
            overlay = vis.copy()
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 200, 100), -1)
            cv2.addWeighted(overlay, 0.3, vis, 0.7, 0, vis)
        
        # Info text
        text = f"slope={slope:.4f}, center=({cx},{cy}), method={horizon.method_used}"
        cv2.putText(vis, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        return vis


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Could not read: {img_path}")
            sys.exit(1)
        
        roi = MaritimeROI(method='traditional')
        result = roi.detect_horizon(img)
        print(f"Result: {result}")
        
        vis = roi.visualize(img, result)
        cv2.imwrite("horizon_test.png", vis)
        print("Saved: horizon_test.png")
    else:
        print("Usage: python maritime_roi.py <image_path>")
