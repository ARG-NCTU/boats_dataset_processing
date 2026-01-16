#!/usr/bin/env python3
"""
HIL-AA Export Module
====================

Export annotation results to standard formats for training:
- COCO JSON: Standard object detection format (with train/val/test split)
- HuggingFace Parquet: For HuggingFace datasets training (with train/val/test split)
- Labelme JSON: Per-frame JSON + image for manual correction (no split)

Output Structure:
    output/dataset/
    ├── coco/
    │   ├── annotations/
    │   │   ├── instances_train2024.json
    │   │   ├── instances_val2024.json
    │   │   ├── instances_test2024.json
    │   │   └── classes.txt
    │   ├── train2024/
    │   ├── val2024/
    │   └── test2024/
    ├── parquet/
    │   ├── instances_train2024.parquet
    │   ├── instances_val2024.parquet
    │   └── instances_test2024.parquet
    └── json_image/
        ├── frame_000000.jpg
        ├── frame_000000.json
        └── ...

Author: Sonic (Maritime Robotics Lab, NYCU)
"""

import json
import logging
import random
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

import cv2
import numpy as np

# Optional: HuggingFace datasets support
try:
    from datasets import Dataset, Features, Value, Image, Sequence
    HAS_HF_DATASETS = True
except ImportError:
    HAS_HF_DATASETS = False

# Optional: PIL for image handling
try:
    from PIL import Image as PILImage
    HAS_PIL = True
except ImportError:
    HAS_PIL = False

# Import from our modules
from .sam3_engine import FrameResult, Detection

logger = logging.getLogger(__name__)

# Labelme version for compatibility
LABELME_VERSION = "5.4.1"


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ExportConfig:
    """Export configuration."""
    # Output settings
    output_dir: Path
    base_name: str  # e.g., "taichung_port" -> output_dir/taichung_port/
    
    # What to export
    include_rejected: bool = False      # Include rejected objects?
    include_hil_fields: bool = True     # Include HIL-AA specific fields in COCO?
    
    # Label settings
    label_name: str = "vessel"          # Label name for annotations
    
    # Frame sampling
    frame_step: int = 1                 # Export every N frames (1 = all frames)
    
    # Split settings (for COCO and Parquet only)
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    random_seed: int = 42
    
    # Video info (will be filled automatically)
    video_path: Optional[str] = None
    video_fps: float = 30.0
    video_width: int = 1920
    video_height: int = 1080
    
    # Category info (auto-generated from label_name if not provided)
    categories: Optional[List[Dict]] = None
    
    def __post_init__(self):
        self.output_dir = Path(self.output_dir)
        # Validate ratios
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        # Auto-generate categories if not provided
        if self.categories is None:
            self.categories = [{"id": 0, "name": self.label_name, "supercategory": "maritime"}]


@dataclass
class ExportStats:
    """Export statistics."""
    total_frames: int
    total_annotations: int
    train_images: int
    val_images: int
    test_images: int
    accepted_objects: int
    rejected_objects: int
    pending_objects: int
    export_time: str
    formats_exported: List[str]
    output_dir: str = ""


# =============================================================================
# Mask Utilities
# =============================================================================

def mask_to_polygon(mask: np.ndarray, simplify: bool = True) -> List[List[List[float]]]:
    """
    Convert binary mask to polygon(s).
    
    Args:
        mask: Binary mask (H, W) with values 0 or 1
        simplify: Whether to simplify polygon
        
    Returns:
        List of polygons, each polygon is [[x1,y1], [x2,y2], ...]
    """
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(
        mask_uint8, 
        cv2.RETR_EXTERNAL, 
        cv2.CHAIN_APPROX_SIMPLE
    )
    
    polygons = []
    for contour in contours:
        if simplify:
            epsilon = 0.002 * cv2.arcLength(contour, True)
            contour = cv2.approxPolyDP(contour, epsilon, True)
        
        if len(contour) < 3:
            continue
        
        # Convert to [[x1,y1], [x2,y2], ...] format
        polygon = [[float(pt[0][0]), float(pt[0][1])] for pt in contour]
        polygons.append(polygon)
    
    return polygons


def polygon_to_coco_format(polygons: List[List[List[float]]]) -> List[List[float]]:
    """
    Convert polygon list to COCO segmentation format.
    
    Args:
        polygons: List of [[x1,y1], [x2,y2], ...] polygons
        
    Returns:
        COCO format: [[x1,y1,x2,y2,...], ...]
    """
    coco_segs = []
    for polygon in polygons:
        # Flatten to [x1, y1, x2, y2, ...]
        flat = []
        for pt in polygon:
            flat.extend(pt)
        if len(flat) >= 6:  # At least 3 points
            coco_segs.append(flat)
    return coco_segs


def compute_area(mask: np.ndarray) -> float:
    """Compute mask area in pixels."""
    return float(np.sum(mask > 0.5))


def compute_bbox_from_mask(mask: np.ndarray) -> List[float]:
    """Compute bounding box from mask in COCO format [x, y, w, h]."""
    rows = np.any(mask > 0.5, axis=1)
    cols = np.any(mask > 0.5, axis=0)
    
    if not np.any(rows) or not np.any(cols):
        return [0.0, 0.0, 0.0, 0.0]
    
    y_min, y_max = np.where(rows)[0][[0, -1]]
    x_min, x_max = np.where(cols)[0][[0, -1]]
    
    return [float(x_min), float(y_min), float(x_max - x_min + 1), float(y_max - y_min + 1)]


# =============================================================================
# Frame Extraction
# =============================================================================

class FrameExtractor:
    """Extract frames from video to images."""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
    
    def extract_frames(
        self, 
        frame_indices: List[int],
        output_dir: Path,
        prefix: str = "frame"
    ) -> Dict[int, str]:
        """
        Extract specific frames from video.
        
        Returns:
            Dict mapping frame_idx -> image filename
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {self.video_path}")
        
        frame_to_filename = {}
        frame_set = set(frame_indices)
        
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx in frame_set:
                filename = f"{prefix}_{frame_idx:06d}.jpg"
                filepath = output_dir / filename
                cv2.imwrite(str(filepath), frame)
                frame_to_filename[frame_idx] = filename
            
            frame_idx += 1
        
        cap.release()
        logger.info(f"Extracted {len(frame_to_filename)} frames to {output_dir}")
        return frame_to_filename


# =============================================================================
# Dataset Splitter
# =============================================================================

def split_dataset(
    coco_data: Dict,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    random_seed: int = 42
) -> Tuple[Dict, Dict, Dict]:
    """
    Split COCO dataset into train/val/test.
    """
    random.seed(random_seed)
    
    images = coco_data["images"].copy()
    random.shuffle(images)
    
    n = len(images)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)
    
    train_images = images[:train_end]
    val_images = images[train_end:val_end]
    test_images = images[val_end:]
    
    train_ids = {img["id"] for img in train_images}
    val_ids = {img["id"] for img in val_images}
    test_ids = {img["id"] for img in test_images}
    
    train_annos = [a for a in coco_data["annotations"] if a["image_id"] in train_ids]
    val_annos = [a for a in coco_data["annotations"] if a["image_id"] in val_ids]
    test_annos = [a for a in coco_data["annotations"] if a["image_id"] in test_ids]
    
    def make_split(images, annotations):
        return {
            "info": coco_data.get("info", {}),
            "licenses": coco_data.get("licenses", []),
            "images": images,
            "annotations": annotations,
            "categories": coco_data["categories"]
        }
    
    return (
        make_split(train_images, train_annos),
        make_split(val_images, val_annos),
        make_split(test_images, test_annos)
    )


def copy_images_for_split(
    images: List[Dict],
    src_dir: Path,
    dst_dir: Path
):
    """Copy images for a specific split."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    for img in images:
        src = src_dir / img["file_name"]
        dst = dst_dir / img["file_name"]
        if src.exists():
            shutil.copy2(src, dst)


# =============================================================================
# Labelme JSON Exporter
# =============================================================================

class LabelmeExporter:
    """Export to Labelme JSON format (per-frame)."""
    
    def __init__(self, config: ExportConfig):
        self.config = config
    
    def export(
        self,
        results: Dict[int, FrameResult],
        object_status: Dict[int, str],
        output_dir: Path,
        frame_to_filename: Dict[int, str],
        object_labels: Optional[Dict[int, str]] = None
    ) -> int:
        """
        Export to Labelme JSON format.
        
        Creates one JSON file per frame with polygon annotations.
        
        Args:
            results: Detection results
            object_status: Review status per object
            output_dir: Output directory
            frame_to_filename: frame_idx -> filename mapping
            object_labels: obj_id -> label_name mapping
        
        Returns:
            Number of frames exported
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        exported_count = 0
        
        # Default category name
        default_label = "vessel"
        if self.config.categories:
            default_label = self.config.categories[0].get("name", "vessel")
        
        for frame_idx, frame_result in sorted(results.items()):
            if frame_idx not in frame_to_filename:
                continue
            
            filename = frame_to_filename[frame_idx]
            
            # Build shapes for this frame
            shapes = []
            for det in frame_result.detections:
                status = object_status.get(det.obj_id, "pending")
                if status == "rejected" and not self.config.include_rejected:
                    continue
                
                # Get polygons from mask
                polygons = mask_to_polygon(det.mask)
                
                # Get label name from object_labels
                if object_labels and det.obj_id in object_labels:
                    label = object_labels[det.obj_id]
                else:
                    label = default_label
                
                # Create shape for each polygon (in case of multiple disconnected regions)
                for polygon in polygons:
                    shape = {
                        "label": label,
                        "points": polygon,  # [[x1,y1], [x2,y2], ...]
                        "group_id": det.obj_id,  # Use obj_id as group_id
                        "description": f"score={det.score:.3f}, status={status}",
                        "shape_type": "polygon",
                        "flags": {}
                    }
                    shapes.append(shape)
            
            # Skip if no shapes
            if not shapes:
                continue
            
            # Build labelme JSON
            labelme_data = {
                "version": LABELME_VERSION,
                "flags": {},
                "shapes": shapes,
                "imagePath": filename,
                "imageData": None,  # Don't embed image data
                "imageHeight": self.config.video_height,
                "imageWidth": self.config.video_width
            }
            
            # Save JSON file
            json_filename = filename.rsplit('.', 1)[0] + '.json'
            json_path = output_dir / json_filename
            
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(labelme_data, f, indent=2, ensure_ascii=False)
            
            exported_count += 1
        
        logger.info(f"Exported {exported_count} Labelme JSON files to {output_dir}")
        return exported_count


# =============================================================================
# COCO Exporter
# =============================================================================

class COCOExporter:
    """Export annotations to COCO JSON format."""
    
    def __init__(self, config: ExportConfig):
        self.config = config
        
    def export(
        self,
        results: Dict[int, FrameResult],
        object_status: Dict[int, str],
        frame_to_filename: Dict[int, str],
        object_category_ids: Optional[Dict[int, int]] = None
    ) -> Dict:
        """
        Export to COCO JSON format.
        
        Args:
            results: Detection results
            object_status: Review status per object
            frame_to_filename: frame_idx -> filename mapping
            object_category_ids: obj_id -> category_id mapping
        
        Returns:
            COCO data dictionary (not yet split)
        """
        coco = {
            "info": self._build_info(),
            "licenses": [{"id": 1, "name": "Unknown", "url": ""}],
            "images": [],
            "annotations": [],
            "categories": self._build_categories()
        }
        
        annotation_id = 1
        
        for frame_idx, frame_result in sorted(results.items()):
            if frame_idx not in frame_to_filename:
                continue
            
            image_id = frame_idx + 1
            coco["images"].append({
                "id": image_id,
                "file_name": frame_to_filename[frame_idx],
                "width": self.config.video_width,
                "height": self.config.video_height
            })
            
            for det in frame_result.detections:
                status = object_status.get(det.obj_id, "pending")
                if status == "rejected" and not self.config.include_rejected:
                    continue
                
                # Get polygons and convert to COCO format
                polygons = mask_to_polygon(det.mask)
                segmentation = polygon_to_coco_format(polygons)
                
                # Get category_id from mapping, default to 0
                category_id = object_category_ids.get(det.obj_id, 0) if object_category_ids else 0
                
                annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [float(x) for x in det.box.tolist()],
                    "area": compute_area(det.mask),
                    "segmentation": segmentation,
                    "iscrowd": 0
                }
                
                # Add HIL-AA specific fields (optional)
                if self.config.include_hil_fields:
                    annotation["score"] = float(det.score)
                    annotation["review_status"] = status
                    annotation["obj_id"] = det.obj_id
                
                coco["annotations"].append(annotation)
                annotation_id += 1
        
        return coco
    
    def _build_info(self) -> Dict:
        return {
            "description": "HIL-AA Maritime Annotation",
            "version": "1.0",
            "year": datetime.now().year,
            "contributor": "HIL-AA System",
            "date_created": datetime.now().isoformat(),
            "video_source": self.config.video_path or "unknown"
        }
    
    def _build_categories(self) -> List[Dict]:
        if self.config.categories:
            return self.config.categories
        return [{"id": 0, "name": "vessel", "supercategory": "maritime"}]


# =============================================================================
# HuggingFace Parquet Exporter
# =============================================================================

class HuggingFaceExporter:
    """Export to HuggingFace datasets Parquet format."""
    
    def __init__(self, config: ExportConfig):
        self.config = config
        
        if not HAS_HF_DATASETS:
            raise ImportError(
                "HuggingFace datasets required for Parquet export.\n"
                "Install with: pip install datasets"
            )
        if not HAS_PIL:
            raise ImportError(
                "PIL required for Parquet export.\n"
                "Install with: pip install Pillow"
            )
    
    def export(
        self,
        coco_data: Dict,
        image_dir: Path,
        output_path: Path
    ) -> Path:
        """
        Convert COCO data to HuggingFace Parquet format.
        """
        image_id_to_info = {img["id"]: img for img in coco_data["images"]}
        
        annos_by_image = {}
        for anno in coco_data["annotations"]:
            img_id = anno["image_id"]
            if img_id not in annos_by_image:
                annos_by_image[img_id] = []
            annos_by_image[img_id].append(anno)
        
        records = []
        for img_id, img_info in sorted(image_id_to_info.items()):
            image_path = image_dir / img_info["file_name"]
            
            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue
            
            image = PILImage.open(image_path).convert("RGB")
            annos = annos_by_image.get(img_id, [])
            
            record = {
                "image_id": img_id,
                "image": image,
                "image_path": f"images/{img_info['file_name']}",
                "width": img_info["width"],
                "height": img_info["height"],
                "objects": {
                    "id": [a["id"] for a in annos],
                    "area": [float(a["area"]) for a in annos],
                    "bbox": [a["bbox"] for a in annos],
                    "category": [a["category_id"] for a in annos]
                }
            }
            records.append(record)
        
        if not records:
            logger.warning("No records to export")
            return output_path
        
        features = Features({
            'image_id': Value('int32'),
            "image": Image(),
            'image_path': Value('string'),
            'width': Value('int32'),
            'height': Value('int32'),
            'objects': Features({
                'id': Sequence(Value('int32')),
                'area': Sequence(Value('float32')),
                'bbox': Sequence(Sequence(Value('float32'), length=4)),
                'category': Sequence(Value('int32'))
            })
        })
        
        dataset = Dataset.from_list(records, features=features)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        dataset.to_parquet(str(output_path))
        
        logger.info(f"Exported HuggingFace Parquet: {output_path} ({len(records)} images)")
        return output_path


# =============================================================================
# Main Exporter Class
# =============================================================================

class AnnotationExporter:
    """
    Main exporter class that handles all formats.
    
    Usage:
        config = ExportConfig(
            output_dir="/app/data/output",
            base_name="dataset",
            video_path="/app/data/video/taichung.mp4",
            ...
        )
        exporter = AnnotationExporter(config)
        stats = exporter.export_all(results, object_status, formats=["coco", "parquet", "labelme"])
    """
    
    def __init__(self, config: ExportConfig):
        self.config = config
        self.config.output_dir = Path(config.output_dir)
    
    def export_all(
        self,
        results: Dict[int, FrameResult],
        object_status: Dict[int, str],
        video_analysis: Optional[Any] = None,
        formats: Optional[List[str]] = None,
        object_labels: Optional[Dict[int, str]] = None
    ) -> ExportStats:
        """
        Export to all requested formats.
        
        Args:
            results: Detection results
            object_status: Review status per object
            video_analysis: Optional analysis data
            formats: List of formats ("coco", "parquet", "labelme")
            object_labels: Dict mapping obj_id -> label_name
            
        Returns:
            ExportStats
        """
        if formats is None:
            formats = ["coco", "labelme"]
            if HAS_HF_DATASETS:
                formats.append("parquet")
        
        # Default object_labels: all objects get first category
        if object_labels is None:
            object_labels = {}
        
        # Build label_name -> category_id mapping
        label_to_cat_id = {}
        if self.config.categories:
            for cat in self.config.categories:
                label_to_cat_id[cat["name"]] = cat["id"]
        
        # Build obj_id -> category_id mapping
        object_category_ids = {}
        for obj_id, label_name in object_labels.items():
            object_category_ids[obj_id] = label_to_cat_id.get(label_name, 0)
        
        # Create directory structure
        base_dir = self.config.output_dir / self.config.base_name
        coco_dir = base_dir / "coco"
        parquet_dir = base_dir / "parquet"
        json_image_dir = base_dir / "json_image"
        
        exported_formats = []
        
        # Apply frame_step filtering
        frame_step = self.config.frame_step
        all_frame_indices = sorted(results.keys())
        
        if frame_step > 1:
            # Select every N-th frame
            filtered_indices = [idx for i, idx in enumerate(all_frame_indices) if i % frame_step == 0]
            filtered_results = {idx: results[idx] for idx in filtered_indices}
            logger.info(f"Frame sampling: {len(all_frame_indices)} -> {len(filtered_indices)} frames (every {frame_step} frames)")
        else:
            filtered_indices = all_frame_indices
            filtered_results = results
        
        # Step 1: Extract frames to json_image directory (used by all formats)
        logger.info("Step 1: Extracting frames from video...")
        
        if self.config.video_path:
            extractor = FrameExtractor(self.config.video_path)
            frame_to_filename = extractor.extract_frames(
                filtered_indices, 
                json_image_dir,
                prefix="frame"
            )
        else:
            # Mock filenames if no video
            json_image_dir.mkdir(parents=True, exist_ok=True)
            frame_to_filename = {idx: f"frame_{idx:06d}.jpg" for idx in filtered_indices}
        
        # Step 2: Export Labelme JSON (no split, all filtered frames)
        if "labelme" in formats:
            logger.info("Step 2: Exporting Labelme JSON files...")
            labelme_exporter = LabelmeExporter(self.config)
            labelme_exporter.export(
                filtered_results, 
                object_status, 
                json_image_dir,
                frame_to_filename,
                object_labels=object_labels  # Pass object_labels
            )
            exported_formats.append("labelme")
        
        # Step 3: Build COCO data
        logger.info("Step 3: Building COCO annotations...")
        coco_exporter = COCOExporter(self.config)
        coco_data = coco_exporter.export(
            filtered_results, 
            object_status, 
            frame_to_filename,
            object_category_ids=object_category_ids  # Pass category IDs
        )
        
        # Step 4: Split dataset (for COCO and Parquet)
        logger.info("Step 4: Splitting dataset into train/val/test...")
        train_data, val_data, test_data = split_dataset(
            coco_data,
            train_ratio=self.config.train_ratio,
            val_ratio=self.config.val_ratio,
            test_ratio=self.config.test_ratio,
            random_seed=self.config.random_seed
        )
        
        # Step 5: Save COCO JSON files with split images
        if "coco" in formats:
            logger.info("Step 5: Saving COCO JSON files...")
            annotations_dir = coco_dir / "annotations"
            annotations_dir.mkdir(parents=True, exist_ok=True)
            
            for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
                # Save JSON
                output_path = annotations_dir / f"instances_{split_name}2024.json"
                with open(output_path, 'w', encoding='utf-8') as f:
                    json.dump(split_data, f, indent=2, ensure_ascii=False)
                
                # Copy images to split directory
                split_image_dir = coco_dir / f"{split_name}2024"
                copy_images_for_split(split_data["images"], json_image_dir, split_image_dir)
            
            # Save classes.txt
            classes_path = annotations_dir / "classes.txt"
            with open(classes_path, 'w') as f:
                for cat in coco_data["categories"]:
                    f.write(f"{cat['name']}\n")
            
            exported_formats.append("coco")
        
        # Step 6: Export to HuggingFace Parquet
        if "parquet" in formats and HAS_HF_DATASETS:
            logger.info("Step 6: Exporting to HuggingFace Parquet...")
            parquet_dir.mkdir(parents=True, exist_ok=True)
            
            hf_exporter = HuggingFaceExporter(self.config)
            
            for split_name, split_data in [("train", train_data), ("val", val_data), ("test", test_data)]:
                # Use coco split images
                split_image_dir = coco_dir / f"{split_name}2024"
                if not split_image_dir.exists():
                    split_image_dir = json_image_dir
                
                output_path = parquet_dir / f"instances_{split_name}2024.parquet"
                try:
                    hf_exporter.export(split_data, split_image_dir, output_path)
                except Exception as e:
                    logger.error(f"Failed to export {split_name} parquet: {e}")
            
            exported_formats.append("parquet")
        
        # Calculate stats
        total_annotations = len(coco_data["annotations"])
        status_counts = {"accepted": 0, "rejected": 0, "pending": 0}
        for status in object_status.values():
            if status in status_counts:
                status_counts[status] += 1
        
        logger.info("Export complete!")
        
        return ExportStats(
            total_frames=len(filtered_results),  # Use filtered count
            total_annotations=total_annotations,
            train_images=len(train_data["images"]),
            val_images=len(val_data["images"]),
            test_images=len(test_data["images"]),
            accepted_objects=status_counts["accepted"],
            rejected_objects=status_counts["rejected"],
            pending_objects=status_counts["pending"],
            export_time=datetime.now().isoformat(),
            formats_exported=exported_formats,
            output_dir=str(base_dir)
        )


# =============================================================================
# CLI Test
# =============================================================================

def main():
    """Test export module with mock data."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Export Module")
    parser.add_argument("--output-dir", type=str, default="/app/data/output",
                        help="Output directory")
    parser.add_argument("--formats", type=str, nargs="+", 
                        default=["coco", "parquet", "labelme"],
                        help="Formats to export")
    args = parser.parse_args()
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    
    print("="*60)
    print("HIL-AA Export Module Test")
    print("="*60)
    
    print(f"\n✓ HuggingFace datasets: {'Available' if HAS_HF_DATASETS else 'Not available'}")
    print(f"✓ PIL: {'Available' if HAS_PIL else 'Not available'}")
    
    # Create mock data
    print("\nCreating mock data...")
    
    mock_results = {}
    for frame_idx in range(100):
        detections = []
        for obj_id in range(3):
            mask = np.zeros((480, 640), dtype=np.float32)
            cv2.circle(mask, (200 + obj_id * 150, 240), 50, 1.0, -1)
            
            det = Detection(
                obj_id=obj_id,
                score=0.7 + obj_id * 0.1 + random.random() * 0.1,
                mask=mask,
                box=np.array([150 + obj_id * 150, 190, 100, 100], dtype=np.float32)
            )
            detections.append(det)
        mock_results[frame_idx] = FrameResult(
            frame_idx=frame_idx,
            detections=detections
        )
    
    mock_status = {0: "accepted", 1: "pending", 2: "rejected"}
    
    print(f"  - Frames: {len(mock_results)}")
    print(f"  - Objects per frame: 3")
    print(f"  - Status: {mock_status}")
    
    # Create config
    config = ExportConfig(
        output_dir=Path(args.output_dir),
        base_name="test_dataset",
        video_path=None,  # No video for mock test
        video_width=640,
        video_height=480,
        video_fps=30.0,
        include_rejected=False,
        include_hil_fields=True,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )
    
    # Filter formats based on availability
    available_formats = ["coco", "labelme"]
    if HAS_HF_DATASETS:
        available_formats.append("parquet")
    formats = [f for f in args.formats if f in available_formats]
    
    print(f"\nExporting to {config.output_dir}...")
    print(f"Formats: {formats}")
    
    exporter = AnnotationExporter(config)
    stats = exporter.export_all(mock_results, mock_status, formats=formats)
    
    print(f"\n{'='*60}")
    print("Export Complete!")
    print(f"{'='*60}")
    print(f"Total Frames:      {stats.total_frames}")
    print(f"Total Annotations: {stats.total_annotations}")
    print(f"Train Images:      {stats.train_images}")
    print(f"Val Images:        {stats.val_images}")
    print(f"Test Images:       {stats.test_images}")
    print(f"Accepted Objects:  {stats.accepted_objects}")
    print(f"Rejected Objects:  {stats.rejected_objects}")
    print(f"Pending Objects:   {stats.pending_objects}")
    print(f"Formats Exported:  {stats.formats_exported}")
    print(f"Output Directory:  {stats.output_dir}")
    print(f"{'='*60}")
    
    # Show directory structure
    print("\nOutput Structure:")
    base = Path(stats.output_dir)
    if base.exists():
        for item in sorted(base.rglob("*")):
            rel = item.relative_to(base)
            depth = len(rel.parts) - 1
            if item.is_dir():
                print(f"  {'  '*depth}📁 {item.name}/")
            else:
                print(f"  {'  '*depth}📄 {item.name}")


if __name__ == "__main__":
    main()
