#!/usr/bin/env python3
"""
HIL-AA Annotation GUI - Main Window
===================================

Maritime Video Annotation Tool using SAM3 with confidence-based
Human-in-the-Loop active annotation.

Main Components:
- Video Display: Shows video frames with mask overlays
- Object Panel: Lists detected objects with confidence scores
- Timeline: Frame navigation with review status
- Controls: Play, pause, navigate, approve/reject
- Interactive Refinement: Point-based mask editing
- Object Management: Delete, merge, swap labels (NEW)

Author: Adam (Assistive Robotics Lab, NYCU)
"""

import sys
import logging
import time
import tempfile
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Set

import cv2
import numpy as np
import gc
import torch

from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QHBoxLayout,
    QSplitter,
    QSlider,
    QFileDialog,
    QMessageBox,
    QListWidget,
    QListWidgetItem,
    QProgressDialog,
    QGroupBox,
    QCheckBox,
    QComboBox,
    QSpinBox,
    QDoubleSpinBox,
    QFrame,
    QSizePolicy,
    QDialog,
    QDialogButtonBox,
    QScrollArea,
    QMenu,
    QInputDialog,
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize, QPoint
from PyQt6.QtGui import QImage, QPixmap, QAction, QKeySequence, QColor, QFont

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our modules
try:
    from core.video_loader import VideoLoader, ImageFolderLoader
    from core.sam3_engine import SAM3Engine, FrameResult, visualize_frame_results, Detection
    from core.confidence_analyzer import (
        ConfidenceAnalyzer, 
        ConfidenceCategory,
        VideoAnalysis
    )
    from core.exporter import AnnotationExporter, ExportConfig, ExportStats
    from gui.interactive_canvas import InteractiveCanvas, RefinementControlPanel, RefinementState
    from core.action_logger import ActionLogger, SessionAnalyzer, EfficiencyMetrics
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)

logger = logging.getLogger(__name__)

# =============================================================================
# GPU Memory Cleanup Utility
# =============================================================================

def clear_gpu_memory():
    """
    清理 GPU 記憶體，避免連續執行偵測時發生 OOM。
    
    調用時機：
    - 每次偵測開始前
    - Worker 執行緒開始時
    """
    # 清理 Python 垃圾收集
    gc.collect()
    
    # 清理 PyTorch CUDA 快取
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 記錄清理後的記憶體狀態
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        logger.info(f"GPU memory cleared: {allocated:.2f} GB allocated, {reserved:.2f} GB reserved")

# =============================================================================
# Worker Thread for SAM3 Processing
# =============================================================================

class SAM3Worker(QThread):
    """
    背景執行緒處理 SAM3 推理。
    
    為什麼需要執行緒？
    - SAM3 推理需要 1-2 分鐘
    - 如果在主執行緒執行，GUI 會凍結
    - 使用 QThread 讓 GUI 保持響應
    
    工作流程：
    1. 載入 SAM3 模型
    2. 啟動 video session
    3. 添加 prompt 進行初步偵測
    4. **暫停** - 發送 ready_to_propagate 信號，等待用戶確認
    5. 用戶確認後，執行 propagate
    
    信號 (Signals):
    - progress: 回報進度 (0-100)
    - ready_to_propagate: 準備好 propagate，等待確認 (num_objects)
    - finished: 完成時發出結果
    - error: 發生錯誤時發出
    - cancelled: 取消時發出
    """
    progress = pyqtSignal(int, str)           # (百分比, 訊息)
    ready_to_propagate = pyqtSignal(int)      # (偵測到的物件數量)
    finished = pyqtSignal(dict)               # 結果字典
    error = pyqtSignal(str)                   # 錯誤訊息
    cancelled = pyqtSignal()                  # 取消信號
    
    def __init__(self, video_path: str, prompt: str, mode: str = "gpu"):
        super().__init__()
        self.video_path = str(video_path)  # 確保是字串
        self.prompt = prompt
        self.mode = mode
        self._cancelled = False
        self._continue_event = threading.Event()
        self._engine = None
        self._session_id = None
    
    def cancel(self):
        """請求取消處理"""
        logger.info("SAM3Worker: Cancel requested")
        self._cancelled = True
        self._continue_event.set()  # 解除等待狀態
    
    def continue_propagation(self):
        """用戶確認後，繼續執行 propagate"""
        logger.info("SAM3Worker: User confirmed, continuing propagation")
        self._continue_event.set()
    
    def _check_cancelled(self) -> bool:
        """檢查是否被取消，如果是則清理資源"""
        if self._cancelled:
            logger.info("SAM3Worker: Cancellation detected, cleaning up...")
            self._cleanup()
            return True
        return False
    
    def _cleanup(self):
        """清理資源"""
        try:
            if self._session_id and self._engine:
                logger.info(f"SAM3Worker: Closing session {self._session_id}")
                self._engine.close_session(self._session_id)
                self._session_id = None
            if self._engine:
                logger.info("SAM3Worker: Shutting down engine")
                self._engine.shutdown()
                self._engine = None
        except Exception as e:
            logger.warning(f"SAM3Worker: Cleanup error (ignored): {e}")
        
        # 清理 GPU 記憶體
        try:
            clear_gpu_memory()
        except:
            pass
    
    def run(self):
        """執行緒主函數。"""
        try:
            # 檢查取消
            if self._check_cancelled():
                self.cancelled.emit()
                return
            
            # 清理 GPU 記憶體
            self.progress.emit(5, "Clearing GPU memory...")
            clear_gpu_memory()

            # 檢查取消
            if self._check_cancelled():
                self.cancelled.emit()
                return

            self.progress.emit(10, "Loading SAM3 model...")
            logger.info(f"Worker starting: video={self.video_path}, prompt={self.prompt}, mode={self.mode}")
            
            self._engine = SAM3Engine(mode=self.mode)
            
            # 檢查取消
            if self._check_cancelled():
                self.cancelled.emit()
                return
            
            self.progress.emit(30, "Starting video session...")
            self._session_id = self._engine.start_video_session(self.video_path)
            logger.info(f"Session started: {self._session_id}")
            
            # 檢查取消
            if self._check_cancelled():
                self.cancelled.emit()
                return
            
            self.progress.emit(40, f"Detecting objects (prompt: {self.prompt})...")
            try:
                result = self._engine.add_prompt(self._session_id, 0, self.prompt)
                # add_prompt 可能返回不同格式，嘗試解析
                if isinstance(result, tuple) and len(result) >= 2:
                    obj_ids = result[1]
                    num_objects = len(obj_ids) if obj_ids is not None else 0
                else:
                    num_objects = 0  # 無法確定，設為 0
            except Exception as e:
                logger.warning(f"Could not get object count from add_prompt: {e}")
                num_objects = 0
            logger.info(f"Initial detection: {num_objects} objects found")
            
            # 檢查取消
            if self._check_cancelled():
                self.cancelled.emit()
                return
            
            # ================================================================
            # 暫停點：等待用戶確認
            # ================================================================
            self.progress.emit(45, f"Found {num_objects} objects. Waiting for confirmation...")
            self.ready_to_propagate.emit(num_objects)
            
            # 等待用戶確認或取消
            logger.info("SAM3Worker: Waiting for user confirmation...")
            self._continue_event.wait()  # 阻塞直到 set()
            
            # 檢查是取消還是確認
            if self._cancelled:
                logger.info("SAM3Worker: User cancelled after detection")
                self._cleanup()
                self.cancelled.emit()
                return
            
            # ================================================================
            # 用戶確認，繼續 propagate
            # ================================================================
            # 注意：propagate() 是一個長時間操作，無法在中間中斷
            # 取消請求會在 propagate 完成後立即生效
            self.progress.emit(50, "Propagating masks (this may take a while)...")
            results = self._engine.propagate(self._session_id)
            logger.info(f"Propagation done: {len(results)} frames")
            
            # 檢查取消（在處理完成後）
            if self._check_cancelled():
                self.cancelled.emit()
                return
            
            self.progress.emit(90, "Closing session...")
            self._engine.close_session(self._session_id)
            self._session_id = None
            
            self._engine.shutdown()
            self._engine = None
            
            # 最後再檢查一次取消
            if self._cancelled:
                self.cancelled.emit()
                return
            
            self.progress.emit(100, "Done!")
            self.finished.emit({"results": results})
            
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}"
            logger.error(f"Worker error: {error_msg}")
            
            # 清理資源
            self._cleanup()
            
            # 如果是取消導致的錯誤，發送取消信號而不是錯誤
            if self._cancelled:
                self.cancelled.emit()
            else:
                self.error.emit(error_msg)


# =============================================================================
# Image Batch Worker (for Independent Image Processing)
# =============================================================================

class ImageBatchWorker(QThread):
    """
    背景執行緒處理批次獨立圖片。
    
    與 SAM3Worker 不同，這個 Worker：
    - 每張圖片獨立使用 detect_image()
    - 不使用 propagate（因為場景不連續）
    - 適用於不同場景的多張照片
    
    信號 (Signals):
    - progress: 回報進度 (current_idx, total, message)
    - image_result: 單張圖片處理完成 (frame_idx, FrameResult)
    - finished: 全部完成 (Dict[int, FrameResult])
    - error: 發生錯誤
    - cancelled: 取消時發出
    """
    progress = pyqtSignal(int, int, str)      # (當前索引, 總數, 訊息)
    image_result = pyqtSignal(int, object)    # (frame_idx, FrameResult) - 即時回報
    finished = pyqtSignal(dict)               # 全部結果
    error = pyqtSignal(str)                   # 錯誤訊息
    cancelled = pyqtSignal()                  # 取消信號
    
    def __init__(self, image_paths: list, prompt: str, mode: str = "gpu"):
        """
        初始化 ImageBatchWorker。
        
        Args:
            image_paths: 圖片路徑列表
            prompt: 文字提示
            mode: SAM3 模式 ("gpu" 或 "mock")
        """
        super().__init__()
        self.image_paths = image_paths
        self.prompt = prompt
        self.mode = mode
        self._cancelled = False
        self._engine = None
    
    def cancel(self):
        """請求取消處理"""
        logger.info("ImageBatchWorker: Cancel requested")
        self._cancelled = True
    
    def _cleanup(self):
        """清理資源"""
        try:
            if self._engine:
                self._engine.shutdown()
                self._engine = None
        except Exception as e:
            logger.warning(f"ImageBatchWorker: Cleanup error (ignored): {e}")
        
        try:
            clear_gpu_memory()
        except:
            pass
    
    def run(self):
        """執行緒主函數 - 逐張處理圖片"""
        results = {}
        total = len(self.image_paths)
        
        try:
            # 清理 GPU
            self.progress.emit(0, total, "Clearing GPU memory...")
            clear_gpu_memory()
            
            if self._cancelled:
                self.cancelled.emit()
                return
            
            # 載入 SAM3 模型
            self.progress.emit(0, total, "Loading SAM3 model...")
            logger.info(f"ImageBatchWorker starting: {total} images, prompt={self.prompt}")
            
            self._engine = SAM3Engine(mode=self.mode)
            
            if self._cancelled:
                self._cleanup()
                self.cancelled.emit()
                return
            
            # 逐張處理
            for idx, image_path in enumerate(self.image_paths):
                if self._cancelled:
                    self._cleanup()
                    self.cancelled.emit()
                    return
                
                filename = Path(image_path).name
                self.progress.emit(idx, total, f"Processing {filename} ({idx+1}/{total})...")
                
                try:
                    # 讀取圖片
                    image = cv2.imread(str(image_path))
                    if image is None:
                        logger.warning(f"Cannot read image: {image_path}")
                        continue
                    
                    # 使用 detect_image 進行獨立偵測
                    frame_result = self._engine.detect_image(image, self.prompt)
                    
                    # 更新 frame_index 為正確的索引
                    frame_result.frame_index = idx
                    
                    results[idx] = frame_result
                    
                    # 即時回報結果
                    self.image_result.emit(idx, frame_result)
                    
                    logger.info(f"Processed {filename}: {frame_result.num_objects} objects")
                    
                except Exception as e:
                    logger.error(f"Error processing {image_path}: {e}")
                    # 繼續處理下一張，不中斷整個流程
            
            # 完成
            self._cleanup()
            
            if self._cancelled:
                self.cancelled.emit()
                return
            
            self.progress.emit(total, total, "Done!")
            self.finished.emit({"results": results})
            
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}"
            logger.error(f"ImageBatchWorker error: {error_msg}")
            self._cleanup()
            
            if self._cancelled:
                self.cancelled.emit()
            else:
                self.error.emit(error_msg)


# =============================================================================
# Object Selection Dialog (for Merge/Swap operations)
# =============================================================================

class ObjectSelectionDialog(QDialog):
    """
    物件選擇對話框，用於 Merge 和 Swap 操作。
    """
    
    def __init__(
        self, 
        parent=None, 
        title: str = "Select Object",
        message: str = "Please select an object:",
        objects: List[Tuple[int, float]] = None,  # [(obj_id, score), ...]
        exclude_obj_id: int = None
    ):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(300)
        
        self.objects = objects or []
        self.exclude_obj_id = exclude_obj_id
        self.selected_obj_id = None
        
        self.setup_ui(message)
    
    def setup_ui(self, message: str):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        # 訊息
        msg_label = QLabel(message)
        msg_label.setWordWrap(True)
        layout.addWidget(msg_label)
        
        # 物件列表
        self.object_combo = QComboBox()
        for obj_id, score in self.objects:
            if obj_id != self.exclude_obj_id:
                self.object_combo.addItem(f"Object {obj_id} (score: {score:.2f})", obj_id)
        layout.addWidget(self.object_combo)
        
        # 按鈕
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def get_selected_obj_id(self) -> Optional[int]:
        """取得選中的物件 ID。"""
        if self.object_combo.count() > 0:
            return self.object_combo.currentData()
        return None


# =============================================================================
# Object List Item Widget
# =============================================================================

class ObjectListItem(QWidget):
    """
    物件列表中的單個項目。
    
    顯示：
    - 顏色標示（根據信心分數）
    - 物件 ID
    - 信心分數
    - 審閱狀態（待審閱/已接受/已拒絕）
    """
    
    status_changed = pyqtSignal(int, str)  # (obj_id, status)
    
    def __init__(
        self, 
        obj_id: int, 
        score: float, 
        category: ConfidenceCategory,
        parent=None
    ):
        super().__init__(parent)
        self.obj_id = obj_id
        self.score = score
        self.category = category
        self.status = "pending"  # pending, accepted, rejected
        
        self.setup_ui()
    
    def setup_ui(self):
        layout = QHBoxLayout()
        layout.setContentsMargins(5, 2, 5, 2)
        self.setLayout(layout)
        
        # 顏色指示器
        self.color_indicator = QLabel()
        self.color_indicator.setFixedSize(16, 16)
        self.update_color()
        layout.addWidget(self.color_indicator)
        
        # 物件資訊
        info_text = f"Obj {self.obj_id}: {self.score:.2f}"
        self.info_label = QLabel(info_text)
        self.info_label.setFont(QFont("Monospace", 10))
        layout.addWidget(self.info_label)
        
        layout.addStretch()
        
        # 狀態標籤
        self.status_label = QLabel("?")
        self.status_label.setToolTip("pending review")
        layout.addWidget(self.status_label)
        
        # 接受按鈕
        self.accept_btn = QPushButton("✓")
        self.accept_btn.setFixedSize(28, 28)
        self.accept_btn.setToolTip("accept")
        self.accept_btn.clicked.connect(self.accept)
        layout.addWidget(self.accept_btn)
        
        # 拒絕按鈕
        self.reject_btn = QPushButton("✗")
        self.reject_btn.setFixedSize(28, 28)
        self.reject_btn.setToolTip("reject")
        self.reject_btn.clicked.connect(self.reject)
        layout.addWidget(self.reject_btn)
    
    def update_color(self):
        """更新顏色指示器。"""
        colors = {
            ConfidenceCategory.HIGH: "#00ff00",      # 綠色
            ConfidenceCategory.UNCERTAIN: "#ffff00", # 黃色
            ConfidenceCategory.LOW: "#ff0000",       # 紅色
        }
        color = colors.get(self.category, "#888888")
        self.color_indicator.setStyleSheet(
            f"background-color: {color}; border-radius: 8px;"
        )
    
    def accept(self):
        """接受此物件。"""
        self.status = "accepted"
        self.status_label.setText("✓")
        self.status_label.setToolTip("accepted")
        self.accept_btn.setEnabled(False)
        self.reject_btn.setEnabled(True)
        self.status_changed.emit(self.obj_id, "accepted")
    
    def reject(self):
        """拒絕此物件。"""
        self.status = "rejected"
        self.status_label.setText("✗")
        self.status_label.setToolTip("rejected")
        self.accept_btn.setEnabled(True)
        self.reject_btn.setEnabled(False)
        self.status_changed.emit(self.obj_id, "rejected")
    
    def reset(self):
        """重設狀態。"""
        self.status = "pending"
        self.status_label.setText("?")
        self.status_label.setToolTip("pending review")
        self.accept_btn.setEnabled(True)
        self.reject_btn.setEnabled(True)


# =============================================================================
# Export Dialog
# =============================================================================

class ExportDialog(QDialog):
    """
    匯出設定對話框。
    
    讓使用者選擇：
    - 輸出目錄和 Dataset 名稱
    - 多個 Label 名稱
    - 為每個 Object 分配 Label
    - 截圖間隔（每 N 秒 1 幀）
    - 匯出格式（COCO, HuggingFace Parquet, Labelme JSON）
    - Train/Val/Test Split 比例
    """
    
    def __init__(
        self, 
        parent=None, 
        default_name: str = "dataset", 
        video_fps: float = 30.0,
        object_info: Optional[List[Dict]] = None
    ):
        """
        Args:
            parent: Parent widget
            default_name: Default dataset name
            video_fps: Video FPS for frame interval calculation
            object_info: List of dicts with keys: obj_id, avg_score, status
        """
        super().__init__(parent)
        self.setWindowTitle("Export Annotations")
        self.setMinimumWidth(600)
        self.setMinimumHeight(700)
        self.default_name = default_name
        self.video_fps = video_fps
        self.object_info = object_info or []
        
        # Label management
        self.labels = ["vessel"]  # Default labels
        self.object_label_combos: Dict[int, QComboBox] = {}  # obj_id -> ComboBox
        
        self.setup_ui()
    
    def setup_ui(self):
        layout = QVBoxLayout()
        self.setLayout(layout)
        
        from PyQt6.QtWidgets import QLineEdit, QFormLayout, QScrollArea
        
        # =====================================================================
        # Output Settings
        # =====================================================================
        output_group = QGroupBox("Output Settings")
        output_layout = QFormLayout()
        output_group.setLayout(output_layout)
        
        # Directory
        dir_widget = QWidget()
        dir_layout = QHBoxLayout()
        dir_layout.setContentsMargins(0, 0, 0, 0)
        dir_widget.setLayout(dir_layout)
        
        self.dir_input = QLineEdit("/app/data/output")
        dir_layout.addWidget(self.dir_input)
        
        browse_btn = QPushButton("Browse...")
        browse_btn.clicked.connect(self.browse_directory)
        dir_layout.addWidget(browse_btn)
        
        output_layout.addRow("Output Directory:", dir_widget)
        
        # Dataset Name
        self.name_input = QLineEdit(self.default_name)
        output_layout.addRow("Dataset Name:", self.name_input)
        
        # Preview
        self.path_preview = QLabel("")
        self.path_preview.setStyleSheet("color: gray; font-size: 10px;")
        self.update_path_preview()
        output_layout.addRow("", self.path_preview)
        
        self.dir_input.textChanged.connect(self.update_path_preview)
        self.name_input.textChanged.connect(self.update_path_preview)
        
        layout.addWidget(output_group)
        
        # =====================================================================
        # Label Settings (Multi-label support)
        # =====================================================================
        label_group = QGroupBox("Label Settings")
        label_layout = QVBoxLayout()
        label_group.setLayout(label_layout)
        
        # Add label row
        add_label_widget = QWidget()
        add_label_layout = QHBoxLayout()
        add_label_layout.setContentsMargins(0, 0, 0, 0)
        add_label_widget.setLayout(add_label_layout)
        
        self.new_label_input = QLineEdit()
        self.new_label_input.setPlaceholderText("Enter new label name...")
        add_label_layout.addWidget(self.new_label_input)
        
        add_label_btn = QPushButton("+ Add Label")
        add_label_btn.clicked.connect(self.add_label)
        add_label_layout.addWidget(add_label_btn)
        
        label_layout.addWidget(add_label_widget)
        
        # Current labels display
        self.labels_display = QWidget()
        self.labels_display_layout = QHBoxLayout()
        self.labels_display_layout.setContentsMargins(0, 0, 0, 0)
        self.labels_display.setLayout(self.labels_display_layout)
        self.update_labels_display()
        
        label_layout.addWidget(self.labels_display)
        
        layout.addWidget(label_group)
        
        # =====================================================================
        # Object → Label Assignment
        # =====================================================================
        if self.object_info:
            assign_group = QGroupBox(f"Object → Label Assignment ({len(self.object_info)} objects)")
            assign_layout = QVBoxLayout()
            assign_group.setLayout(assign_layout)
            
            # Scroll area for many objects
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setMaximumHeight(200)
            
            scroll_content = QWidget()
            scroll_layout = QVBoxLayout()
            scroll_content.setLayout(scroll_layout)
            
            # Sort by score descending
            sorted_objects = sorted(self.object_info, key=lambda x: x.get('avg_score', 0), reverse=True)
            
            for obj in sorted_objects:
                obj_id = obj['obj_id']
                score = obj.get('avg_score', 0)
                status = obj.get('status', 'pending')
                
                # Skip rejected objects
                if status == 'rejected':
                    continue
                
                row_widget = QWidget()
                row_layout = QHBoxLayout()
                row_layout.setContentsMargins(0, 2, 0, 2)
                row_widget.setLayout(row_layout)
                
                # Status indicator
                status_color = "#4CAF50" if status == "accepted" else "#FFC107"
                status_label = QLabel(f"●")
                status_label.setStyleSheet(f"color: {status_color}; font-size: 14px;")
                row_layout.addWidget(status_label)
                
                # Object info
                info_label = QLabel(f"Obj {obj_id}: {score:.2f} ({status})")
                info_label.setMinimumWidth(180)
                row_layout.addWidget(info_label)
                
                # Label combo
                combo = QComboBox()
                combo.addItems(self.labels)
                combo.setMinimumWidth(120)
                self.object_label_combos[obj_id] = combo
                row_layout.addWidget(combo)
                
                row_layout.addStretch()
                scroll_layout.addWidget(row_widget)
            
            scroll_layout.addStretch()
            scroll.setWidget(scroll_content)
            assign_layout.addWidget(scroll)
            
            # Quick assign buttons
            quick_widget = QWidget()
            quick_layout = QHBoxLayout()
            quick_layout.setContentsMargins(0, 5, 0, 0)
            quick_widget.setLayout(quick_layout)
            
            quick_layout.addWidget(QLabel("Quick assign all to:"))
            self.quick_assign_combo = QComboBox()
            self.quick_assign_combo.addItems(self.labels)
            quick_layout.addWidget(self.quick_assign_combo)
            
            quick_assign_btn = QPushButton("Apply to All")
            quick_assign_btn.clicked.connect(self.quick_assign_all)
            quick_layout.addWidget(quick_assign_btn)
            quick_layout.addStretch()
            
            assign_layout.addWidget(quick_widget)
            
            layout.addWidget(assign_group)
        
        # =====================================================================
        # Frame Interval
        # =====================================================================
        interval_group = QGroupBox("Frame Sampling")
        interval_layout = QHBoxLayout()
        interval_group.setLayout(interval_layout)
        
        interval_layout.addWidget(QLabel("Export every:"))
        
        self.interval_spin = QDoubleSpinBox()
        self.interval_spin.setRange(0.0, 60.0)
        self.interval_spin.setSingleStep(0.5)
        self.interval_spin.setValue(0.0)
        self.interval_spin.setDecimals(1)
        self.interval_spin.setSuffix(" sec")
        interval_layout.addWidget(self.interval_spin)
        
        self.interval_info = QLabel("")
        self.interval_info.setStyleSheet("color: gray;")
        interval_layout.addWidget(self.interval_info)
        
        interval_layout.addWidget(QLabel("(0 = all frames)"))
        interval_layout.addStretch()
        
        self.interval_spin.valueChanged.connect(self.update_interval_info)
        self.update_interval_info()
        
        layout.addWidget(interval_group)
        
        # =====================================================================
        # Export Formats
        # =====================================================================
        format_group = QGroupBox("Export Formats")
        format_layout = QVBoxLayout()
        format_group.setLayout(format_layout)
        
        self.labelme_checkbox = QCheckBox("Labelme JSON + Images (json_image/)")
        self.labelme_checkbox.setChecked(True)
        self.labelme_checkbox.setEnabled(False)
        format_layout.addWidget(self.labelme_checkbox)
        
        self.coco_checkbox = QCheckBox("COCO JSON with train/val/test split (coco/)")
        self.coco_checkbox.setChecked(True)
        format_layout.addWidget(self.coco_checkbox)
        
        self.parquet_checkbox = QCheckBox("HuggingFace Parquet with split (parquet/)")
        self.parquet_checkbox.setChecked(True)
        format_layout.addWidget(self.parquet_checkbox)
        
        layout.addWidget(format_group)
        
        # =====================================================================
        # Train/Val/Test Split
        # =====================================================================
        split_group = QGroupBox("Train/Val/Test Split")
        split_layout = QHBoxLayout()
        split_group.setLayout(split_layout)
        
        split_layout.addWidget(QLabel("Train:"))
        self.train_spin = QDoubleSpinBox()
        self.train_spin.setRange(0.0, 1.0)
        self.train_spin.setSingleStep(0.05)
        self.train_spin.setValue(0.8)
        self.train_spin.setDecimals(2)
        split_layout.addWidget(self.train_spin)
        
        split_layout.addWidget(QLabel("Val:"))
        self.val_spin = QDoubleSpinBox()
        self.val_spin.setRange(0.0, 1.0)
        self.val_spin.setSingleStep(0.05)
        self.val_spin.setValue(0.1)
        self.val_spin.setDecimals(2)
        split_layout.addWidget(self.val_spin)
        
        split_layout.addWidget(QLabel("Test:"))
        self.test_spin = QDoubleSpinBox()
        self.test_spin.setRange(0.0, 1.0)
        self.test_spin.setSingleStep(0.05)
        self.test_spin.setValue(0.1)
        self.test_spin.setDecimals(2)
        split_layout.addWidget(self.test_spin)
        
        self.split_warning = QLabel("✓")
        self.split_warning.setStyleSheet("color: green;")
        split_layout.addWidget(self.split_warning)
        
        self.train_spin.valueChanged.connect(self.validate_split)
        self.val_spin.valueChanged.connect(self.validate_split)
        self.test_spin.valueChanged.connect(self.validate_split)
        
        layout.addWidget(split_group)
        
        # =====================================================================
        # Options
        # =====================================================================
        options_group = QGroupBox("Options")
        options_layout = QHBoxLayout()
        options_group.setLayout(options_layout)
        
        self.rejected_checkbox = QCheckBox("Include rejected objects")
        self.rejected_checkbox.setChecked(False)
        options_layout.addWidget(self.rejected_checkbox)
        
        self.hil_fields_checkbox = QCheckBox("Include HIL-AA fields in COCO")
        self.hil_fields_checkbox.setChecked(True)
        options_layout.addWidget(self.hil_fields_checkbox)
        
        layout.addWidget(options_group)
        
        # =====================================================================
        # Buttons
        # =====================================================================
        button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | 
            QDialogButtonBox.StandardButton.Cancel
        )
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)
    
    def add_label(self):
        """新增 label"""
        new_label = self.new_label_input.text().strip()
        if not new_label:
            return
        if new_label in self.labels:
            QMessageBox.warning(self, "Warning", f"Label '{new_label}' already exists")
            return
        
        self.labels.append(new_label)
        self.new_label_input.clear()
        self.update_labels_display()
        self.update_object_combos()
        
        # Update quick assign combo
        if hasattr(self, 'quick_assign_combo'):
            self.quick_assign_combo.clear()
            self.quick_assign_combo.addItems(self.labels)
    
    def remove_label(self, label: str):
        """移除 label"""
        if len(self.labels) <= 1:
            QMessageBox.warning(self, "Warning", "Must have at least one label")
            return
        if label in self.labels:
            self.labels.remove(label)
            self.update_labels_display()
            self.update_object_combos()
            
            if hasattr(self, 'quick_assign_combo'):
                self.quick_assign_combo.clear()
                self.quick_assign_combo.addItems(self.labels)
    
    def update_labels_display(self):
        """更新 labels 顯示"""
        # Clear existing
        while self.labels_display_layout.count():
            item = self.labels_display_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        
        # Add label tags
        for label in self.labels:
            tag = QWidget()
            tag_layout = QHBoxLayout()
            tag_layout.setContentsMargins(5, 2, 5, 2)
            tag.setLayout(tag_layout)
            tag.setStyleSheet("background-color: #E3F2FD; border-radius: 3px;")
            
            tag_label = QLabel(label)
            tag_layout.addWidget(tag_label)
            
            remove_btn = QPushButton("✗")
            remove_btn.setFixedSize(20, 20)
            remove_btn.setStyleSheet("background: none; border: none; color: #666;")
            remove_btn.clicked.connect(lambda checked, l=label: self.remove_label(l))
            tag_layout.addWidget(remove_btn)
            
            self.labels_display_layout.addWidget(tag)
        
        self.labels_display_layout.addStretch()
    
    def update_object_combos(self):
        """更新所有 object 的 label combo"""
        for obj_id, combo in self.object_label_combos.items():
            current = combo.currentText()
            combo.clear()
            combo.addItems(self.labels)
            # Restore selection if still valid
            idx = combo.findText(current)
            if idx >= 0:
                combo.setCurrentIndex(idx)
    
    def quick_assign_all(self):
        """快速將所有 object 指定為同一個 label"""
        label = self.quick_assign_combo.currentText()
        for combo in self.object_label_combos.values():
            idx = combo.findText(label)
            if idx >= 0:
                combo.setCurrentIndex(idx)
    
    def update_path_preview(self):
        """更新路徑預覽"""
        output_dir = self.dir_input.text()
        name = self.name_input.text() or "dataset"
        self.path_preview.setText(f"→ {output_dir}/{name}/")
    
    def update_interval_info(self):
        """更新 frame interval 資訊"""
        interval = self.interval_spin.value()
        if interval <= 0:
            self.interval_info.setText("(all frames)")
        else:
            frame_step = max(1, int(interval * self.video_fps))
            self.interval_info.setText(f"(every {frame_step} frames)")
    
    def validate_split(self):
        """驗證 split ratio"""
        total = self.train_spin.value() + self.val_spin.value() + self.test_spin.value()
        if abs(total - 1.0) > 0.01:
            self.split_warning.setText(f"⚠ {total:.2f}")
            self.split_warning.setStyleSheet("color: red;")
        else:
            self.split_warning.setText("✓")
            self.split_warning.setStyleSheet("color: green;")
    
    def browse_directory(self):
        """選擇輸出目錄"""
        dir_path = QFileDialog.getExistingDirectory(
            self, "Select Output Directory", self.dir_input.text()
        )
        if dir_path:
            self.dir_input.setText(dir_path)
    
    # =========================================================================
    # Getters
    # =========================================================================
    
    def get_selected_formats(self) -> List[str]:
        formats = ["labelme"]
        if self.coco_checkbox.isChecked():
            formats.append("coco")
        if self.parquet_checkbox.isChecked():
            formats.append("parquet")
        return formats
    
    def get_output_dir(self) -> str:
        return self.dir_input.text()
    
    def get_dataset_name(self) -> str:
        return self.name_input.text() or "dataset"
    
    def get_labels(self) -> List[str]:
        """取得所有 labels"""
        return self.labels.copy()
    
    def get_object_labels(self) -> Dict[int, str]:
        """取得 object → label 對應 (obj_id -> label_name)"""
        return {obj_id: combo.currentText() for obj_id, combo in self.object_label_combos.items()}
    
    def get_frame_interval(self) -> float:
        return self.interval_spin.value()
    
    def get_include_rejected(self) -> bool:
        return self.rejected_checkbox.isChecked()
    
    def get_include_hil_fields(self) -> bool:
        return self.hil_fields_checkbox.isChecked()
    
    def get_split_ratios(self) -> Tuple[float, float, float]:
        return (self.train_spin.value(), self.val_spin.value(), self.test_spin.value())
    
    def is_valid(self) -> bool:
        total = self.train_spin.value() + self.val_spin.value() + self.test_spin.value()
        return abs(total - 1.0) < 0.01


# =============================================================================
# Main Window
# =============================================================================

class HILAAMainWindow(QMainWindow):
    """
    HIL-AA 標註工具主視窗。
    
    功能：
    1. 載入影片
    2. 執行 SAM3 偵測
    3. 顯示標註結果
    4. 讓使用者審閱 UNCERTAIN 物件
    5. Interactive Refinement（點擊修正 mask）
    6. Object Management（刪除、合併、交換標籤）- NEW
    7. 匯出標註結果
    """
    
    def __init__(self):
        super().__init__()
        
        # 狀態變數
        self.video_loader: Optional[VideoLoader] = None
        self.sam3_results: Dict[int, FrameResult] = {}
        self.analyzer = ConfidenceAnalyzer(high_threshold=0.80, low_threshold=0.50)
        self.video_analysis: Optional[VideoAnalysis] = None
        
        # Maritime ROI (海平線偵測)
        self.maritime_roi = None  # 延遲初始化
        self.horizon_result = None  # 儲存海平線偵測結果
        
        self.current_frame = 0
        self._last_logged_frame = -1  # 用於追蹤幀跳轉 logging
        self.is_playing = False
        self.show_masks = True
        self.show_boxes = True
        self.show_horizon = False  # 顯示海平線
        
        # 物件狀態（審閱結果）
        self.object_status: Dict[int, str] = {}  # obj_id -> status
        
        # Refinement 狀態
        self.refinement_active = False
        self.refinement_obj_id: Optional[int] = None
        self.add_object_mode = False  # True when adding new object instead of refining
        self.sam3_engine: Optional[SAM3Engine] = None  # Reuse for refinement
        
        # Action Logger (記錄使用者操作，計算 HIR/CPO/SPF)
        # 預設使用臨時目錄，open_video 時會重新設定為影片所在目錄的 logs 子目錄
        self.action_logger = ActionLogger(
            output_dir=tempfile.gettempdir(),
            format="jsonl",
            auto_flush=True
        )
        self._detection_start_time: Optional[float] = None  # 用於計算偵測時間
        
        # Detection worker 狀態
        self.worker: Optional[SAM3Worker] = None
        self.progress_dialog: Optional[QProgressDialog] = None
        self._is_image_folder_mode: bool = False  # 是否為圖片資料夾模式
        
        # 設定 UI
        self.setup_ui()
        self.setup_menu()
        self.setup_shortcuts()
        
        # 播放定時器
        self.play_timer = QTimer()
        self.play_timer.timeout.connect(self.next_frame)
        
        self.setWindowTitle("HIL-AA Maritime Annotation Tool")
        self.setGeometry(100, 50, 1400, 900)
        
        self.statusBar().showMessage("Ready - Please open a video file")
    
    def setup_ui(self):
        """建立使用者介面。"""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # 使用 QSplitter 讓使用者可以調整左右區域大小
        splitter = QSplitter(Qt.Orientation.Horizontal)
        main_layout.addWidget(splitter)
        
        # =====================================================================
        # 左側：影片顯示區
        # =====================================================================
        left_widget = QWidget()
        left_layout = QVBoxLayout()
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_widget.setLayout(left_layout)
        
        # 影片顯示 (使用 InteractiveCanvas 支援點擊修正)
        self.video_canvas = InteractiveCanvas()
        self.video_canvas.setMinimumSize(800, 450)
        self.video_canvas.setText("Please open a video file\n\nFile → Open Video (Ctrl+O)")
        self.video_canvas.point_added.connect(self.on_refinement_point_added)
        left_layout.addWidget(self.video_canvas, stretch=1)
        
        # Refinement 控制面板 (浮動在 canvas 上方)
        self.refinement_panel = RefinementControlPanel(self.video_canvas)
        self.refinement_panel.clear_clicked.connect(self.on_refinement_clear)
        self.refinement_panel.undo_clicked.connect(self.on_refinement_undo)
        self.refinement_panel.apply_clicked.connect(self.on_refinement_apply)
        self.refinement_panel.propagate_clicked.connect(self.on_refinement_propagate)
        self.refinement_panel.cancel_clicked.connect(self.on_refinement_cancel)
        self.refinement_panel.move(10, 10)  # 左上角
        
        # 為了向後相容，保留 video_label 別名
        self.video_label = self.video_canvas
        
        # 時間軸滑桿
        slider_layout = QHBoxLayout()
        
        self.frame_label = QLabel("0 / 0")
        self.frame_label.setFixedWidth(100)
        slider_layout.addWidget(self.frame_label)
        
        self.timeline_slider = QSlider(Qt.Orientation.Horizontal)
        self.timeline_slider.setMinimum(0)
        self.timeline_slider.setMaximum(100)
        self.timeline_slider.sliderMoved.connect(self.on_slider_moved)
        self.timeline_slider.setEnabled(False)
        slider_layout.addWidget(self.timeline_slider)
        
        left_layout.addLayout(slider_layout)
        
        # 控制按鈕
        control_layout = QHBoxLayout()
        
        self.open_btn = QPushButton("Open Video")
        self.open_btn.clicked.connect(self.smart_open)  # 根據模式智能開啟
        control_layout.addWidget(self.open_btn)
        
        self.detect_btn = QPushButton("Run Detection")
        self.detect_btn.clicked.connect(self.run_detection)
        self.detect_btn.setEnabled(False)
        control_layout.addWidget(self.detect_btn)
        
        control_layout.addSpacing(20)
        
        self.prev_btn = QPushButton("|<")
        self.prev_btn.setFixedWidth(40)
        self.prev_btn.clicked.connect(self.prev_frame)
        self.prev_btn.setEnabled(False)
        control_layout.addWidget(self.prev_btn)
        
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedWidth(40)
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        control_layout.addWidget(self.play_btn)
        
        self.next_btn = QPushButton(">|")
        self.next_btn.setFixedWidth(40)
        self.next_btn.clicked.connect(self.next_frame)
        self.next_btn.setEnabled(False)
        control_layout.addWidget(self.next_btn)
        
        control_layout.addStretch()
        
        # Display options
        self.mask_checkbox = QCheckBox("Show Mask")
        self.mask_checkbox.setChecked(True)
        self.mask_checkbox.stateChanged.connect(self.on_display_option_changed)
        control_layout.addWidget(self.mask_checkbox)
        
        self.box_checkbox = QCheckBox("Show Box")
        self.box_checkbox.setChecked(True)
        self.box_checkbox.stateChanged.connect(self.on_display_option_changed)
        control_layout.addWidget(self.box_checkbox)
        
        self.horizon_checkbox = QCheckBox("Show Horizon")
        self.horizon_checkbox.setChecked(False)
        self.horizon_checkbox.setToolTip("Display detected horizon line and sky region")
        self.horizon_checkbox.stateChanged.connect(self.on_display_option_changed)
        control_layout.addWidget(self.horizon_checkbox)
        
        left_layout.addLayout(control_layout)
        
        # Timeline 視覺化
        from gui.timeline_widget import TimelineWidget
        self.timeline_widget = TimelineWidget()
        self.timeline_widget.frame_selected.connect(self.seek_to_frame)
        self.timeline_widget.object_selected.connect(self.on_timeline_object_selected)
        self.timeline_widget.setVisible(False)  # 初始隱藏，有資料後再顯示
        left_layout.addWidget(self.timeline_widget)
        
        splitter.addWidget(left_widget)
        
        # =====================================================================
        # 右側：物件面板
        # =====================================================================
        right_widget = QWidget()
        right_widget.setMaximumWidth(350)
        right_layout = QVBoxLayout()
        right_widget.setLayout(right_layout)
        
        # 偵測設定
        settings_group = QGroupBox("Detection Settings")
        settings_layout = QVBoxLayout()
        settings_group.setLayout(settings_layout)
        
        # Prompt 輸入
        prompt_layout = QHBoxLayout()
        prompt_layout.addWidget(QLabel("Prompt:"))
        from PyQt6.QtWidgets import QLineEdit
        self.prompt_input = QLineEdit("boat, ship")
        prompt_layout.addWidget(self.prompt_input)
        settings_layout.addLayout(prompt_layout)
        
        # 閾值設定
        thresh_layout = QHBoxLayout()
        thresh_layout.addWidget(QLabel("HIGH ≥"))
        self.high_thresh_spin = QDoubleSpinBox()
        self.high_thresh_spin.setRange(0.0, 1.0)
        self.high_thresh_spin.setSingleStep(0.05)
        self.high_thresh_spin.setValue(0.80)
        self.high_thresh_spin.valueChanged.connect(self.on_threshold_changed)
        thresh_layout.addWidget(self.high_thresh_spin)
        
        thresh_layout.addWidget(QLabel("LOW <"))
        self.low_thresh_spin = QDoubleSpinBox()
        self.low_thresh_spin.setRange(0.0, 1.0)
        self.low_thresh_spin.setSingleStep(0.05)
        self.low_thresh_spin.setValue(0.50)
        self.low_thresh_spin.valueChanged.connect(self.on_threshold_changed)
        thresh_layout.addWidget(self.low_thresh_spin)
        settings_layout.addLayout(thresh_layout)
        
        # Engine mode
        mode_layout = QHBoxLayout()
        mode_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["gpu", "mock"])
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()
        settings_layout.addLayout(mode_layout)
        
        # Processing Mode (Video vs Images)
        proc_mode_layout = QHBoxLayout()
        proc_mode_layout.addWidget(QLabel("Input:"))
        self.processing_mode_combo = QComboBox()
        self.processing_mode_combo.addItems([
            "Video (Sequential Tracking)",
            "Images (Independent Detection)"
        ])
        self.processing_mode_combo.setToolTip(
            "Video: Load video file, use SAM3 tracking to propagate masks\n"
            "Images: Load image folder, process each image independently\n"
            "        (No cross-image tracking - suitable for different scenes)"
        )
        self.processing_mode_combo.currentIndexChanged.connect(self._on_processing_mode_changed)
        proc_mode_layout.addWidget(self.processing_mode_combo)
        settings_layout.addLayout(proc_mode_layout)
        
        # Maritime ROI (Horizon Detection)
        maritime_layout = QHBoxLayout()
        self.maritime_roi_checkbox = QCheckBox("Enable Maritime ROI")
        self.maritime_roi_checkbox.setToolTip(
            "Use horizon detection to exclude sky region from detection.\n"
            "This can improve accuracy for maritime scenes."
        )
        self.maritime_roi_checkbox.setChecked(False)
        self.maritime_roi_checkbox.stateChanged.connect(self._on_maritime_roi_changed)
        maritime_layout.addWidget(self.maritime_roi_checkbox)
        
        self.maritime_method_combo = QComboBox()
        self.maritime_method_combo.addItems(["Auto (Fallback)", "Traditional (Fast)", "SegFormer (Accurate)"])
        self.maritime_method_combo.setToolTip(
            "Auto: Try Traditional first, fall back to SegFormer if failed\n"
            "Traditional: Fast edge detection + Hough transform\n"
            "SegFormer: Accurate semantic segmentation (requires GPU)"
        )
        self.maritime_method_combo.setEnabled(False)
        maritime_layout.addWidget(self.maritime_method_combo)
        maritime_layout.addStretch()
        settings_layout.addLayout(maritime_layout)
        
        right_layout.addWidget(settings_group)
        
        # Analysis Results
        self.analysis_group = QGroupBox("Analysis Results")
        analysis_layout = QVBoxLayout()
        self.analysis_group.setLayout(analysis_layout)
        
        self.analysis_label = QLabel("Detection not yet run")
        self.analysis_label.setWordWrap(True)
        analysis_layout.addWidget(self.analysis_label)
        
        right_layout.addWidget(self.analysis_group)
        
        # Object List
        objects_group = QGroupBox("Detected Objects")
        objects_layout = QVBoxLayout()
        objects_group.setLayout(objects_layout)
        
        # 批次操作
        batch_layout = QHBoxLayout()
        self.accept_all_high_btn = QPushButton("✓ Accept all HIGH")
        self.accept_all_high_btn.clicked.connect(self.accept_all_high)
        self.accept_all_high_btn.setEnabled(False)
        batch_layout.addWidget(self.accept_all_high_btn)
        
        self.reset_all_btn = QPushButton("↺ Reset")
        self.reset_all_btn.clicked.connect(self.reset_all_objects)
        self.reset_all_btn.setEnabled(False)
        batch_layout.addWidget(self.reset_all_btn)
        objects_layout.addLayout(batch_layout)
        
        # Refine 按鈕（點擊修正選中的物件）
        refine_layout = QHBoxLayout()
        self.refine_btn = QPushButton("Refine Selected")
        self.refine_btn.setToolTip("Enter refinement mode: Left-click to include, Right-click to exclude")
        self.refine_btn.clicked.connect(self.start_refinement_for_selected)
        self.refine_btn.setEnabled(False)
        self.refine_btn.setStyleSheet("""
            QPushButton {
                background-color: #2196F3;
                color: white;
                padding: 5px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #1976D2;
            }
            QPushButton:disabled {
                background-color: #666;
                color: #999;
            }
        """)
        refine_layout.addWidget(self.refine_btn)
        
        # Add Object 按鈕（手動新增物件）
        self.add_object_btn = QPushButton("+ Add Object")
        self.add_object_btn.setToolTip("Add a new object by clicking on the image")
        self.add_object_btn.clicked.connect(self.start_add_object)
        self.add_object_btn.setEnabled(False)
        self.add_object_btn.setStyleSheet("""
            QPushButton {
                background-color: #FF9800;
                color: white;
                padding: 5px 10px;
                border-radius: 4px;
            }
            QPushButton:hover {
                background-color: #F57C00;
            }
            QPushButton:disabled {
                background-color: #666;
                color: #999;
            }
        """)
        refine_layout.addWidget(self.add_object_btn)
        
        refine_layout.addStretch()
        objects_layout.addLayout(refine_layout)
        
        # 物件列表 (啟用右鍵選單)
        self.object_list = QListWidget()
        self.object_list.setMinimumHeight(300)
        self.object_list.itemSelectionChanged.connect(self.on_object_selection_changed)
        self.object_list.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.object_list.customContextMenuRequested.connect(self.show_object_context_menu)
        objects_layout.addWidget(self.object_list)
        
        # 物件管理提示
        manage_hint = QLabel("💡 Right-click on object for more options")
        manage_hint.setStyleSheet("color: #888; font-size: 10px;")
        objects_layout.addWidget(manage_hint)
        
        right_layout.addWidget(objects_group)
        
        right_layout.addStretch()
        
        splitter.addWidget(right_widget)
        
        # 設定初始大小比例
        splitter.setSizes([1000, 350])
    
    def setup_menu(self):
        """建立選單。"""
        menu_bar = self.menuBar()
        
        # File 選單
        file_menu = menu_bar.addMenu("&File")
        
        open_action = QAction("&Open Video", self)
        open_action.setShortcut(QKeySequence("Ctrl+O"))
        open_action.triggered.connect(self.open_video)
        file_menu.addAction(open_action)
        
        open_folder_action = QAction("Open Image &Folder", self)
        open_folder_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        open_folder_action.triggered.connect(self.open_image_folder)
        file_menu.addAction(open_folder_action)
        
        file_menu.addSeparator()
        
        export_action = QAction("&Export Results", self)
        export_action.setShortcut(QKeySequence("Ctrl+E"))
        export_action.triggered.connect(self.export_results)
        file_menu.addAction(export_action)
        
        file_menu.addSeparator()
        
        exit_action = QAction("E&xit", self)
        exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)
        
        # View 選單
        view_menu = menu_bar.addMenu("&View")
        
        self.mask_action = QAction("Show &Masks", self, checkable=True)
        self.mask_action.setChecked(True)
        self.mask_action.triggered.connect(
            lambda: self.mask_checkbox.setChecked(self.mask_action.isChecked())
        )
        view_menu.addAction(self.mask_action)
        
        self.box_action = QAction("Show &Boxes", self, checkable=True)
        self.box_action.setChecked(True)
        self.box_action.triggered.connect(
            lambda: self.box_checkbox.setChecked(self.box_action.isChecked())
        )
        view_menu.addAction(self.box_action)
        
        # Help 選單
        help_menu = menu_bar.addMenu("&Help")
        
        about_action = QAction("&About", self)
        about_action.triggered.connect(self.show_about)
        help_menu.addAction(about_action)
    
    def setup_shortcuts(self):
        """設定快捷鍵。"""
        # 空白鍵：播放/暫停
        # 左右鍵：上一幀/下一幀（在 keyPressEvent 處理）
        pass
    
    # =========================================================================
    # Object Management Methods (NEW)
    # =========================================================================
    
    def show_object_context_menu(self, position: QPoint):
        """顯示物件右鍵選單。"""
        item = self.object_list.itemAt(position)
        if item is None:
            return
        
        widget = self.object_list.itemWidget(item)
        if widget is None:
            return
        
        obj_id = widget.property("obj_id")
        if obj_id is None:
            return
        
        # 建立右鍵選單
        menu = QMenu(self)
        
        # 刪除物件
        delete_action = menu.addAction("🗑️ Delete Object (All Frames)")
        delete_action.triggered.connect(lambda: self.delete_object(obj_id))
        
        delete_from_action = menu.addAction("🗑️ Delete From Current Frame")
        delete_from_action.triggered.connect(lambda: self.delete_object_from_frame(obj_id, self.current_frame))
        
        menu.addSeparator()
        
        # 合併物件
        merge_action = menu.addAction("🔗 Merge Into Another Object...")
        merge_action.triggered.connect(lambda: self.show_merge_dialog(obj_id))
        
        # 交換標籤
        swap_action = menu.addAction("🔄 Swap Label With...")
        swap_action.triggered.connect(lambda: self.show_swap_dialog(obj_id))
        
        menu.addSeparator()
        
        # 跳轉到物件首次出現的幀
        jump_action = menu.addAction("📍 Jump to First Appearance")
        jump_action.triggered.connect(lambda: self.jump_to_object_first_frame(obj_id))
        
        # 顯示選單
        menu.exec(self.object_list.mapToGlobal(position))
    
    def delete_object(self, obj_id: int):
        """
        完全刪除物件（從所有幀中移除）。
        
        Args:
            obj_id: 要刪除的物件 ID
        """
        # 確認操作
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete Object {obj_id} from ALL frames?\n\nThis action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        edited_frame = self.current_frame  # 記錄操作發生的幀
        
        # 從所有幀中移除該物件
        deleted_count = 0
        for frame_idx, frame_result in self.sam3_results.items():
            original_count = len(frame_result.detections)
            frame_result.detections = [
                d for d in frame_result.detections if d.obj_id != obj_id
            ]
            deleted_count += original_count - len(frame_result.detections)
        
        # 移除物件狀態
        if obj_id in self.object_status:
            del self.object_status[obj_id]
        
        # === ActionLogger: 記錄刪除物件 ===
        self.action_logger.log_delete_object(
            frame_idx=edited_frame,
            obj_id=obj_id,
            delete_type="all"
        )
        
        logger.info(f"Deleted object {obj_id}: removed {deleted_count} detections")
        
        # 更新 UI（先 reanalyze，再 track）
        self._reanalyze_with_preserved_edits()
        self._track_human_intervention(edited_frame)  # ← 添加這行！
        self.update_object_list()
        self.update_analysis_display()
        self.update_timeline()
        self.display_frame(self.current_frame)
        
        self.statusBar().showMessage(f"Deleted Object {obj_id} ({deleted_count} detections removed)")
    
    def delete_object_from_frame(self, obj_id: int, from_frame: int):
        """
        從指定幀開始刪除物件（保留之前的幀）。
        
        Args:
            obj_id: 要刪除的物件 ID
            from_frame: 從此幀開始刪除
        """
        if self.video_loader is None:
            return
        
        total_frames = self.video_loader.metadata.total_frames
        
        # 確認操作
        reply = QMessageBox.question(
            self, "Confirm Delete",
            f"Delete Object {obj_id} from frame {from_frame} to {total_frames - 1}?\n\n"
            f"Frames 0 to {from_frame - 1} will be preserved.\n"
            "This action cannot be undone.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # 從指定幀開始移除
        deleted_count = 0
        for frame_idx in range(from_frame, total_frames):
            if frame_idx in self.sam3_results:
                frame_result = self.sam3_results[frame_idx]
                original_count = len(frame_result.detections)
                frame_result.detections = [
                    d for d in frame_result.detections if d.obj_id != obj_id
                ]
                deleted_count += original_count - len(frame_result.detections)
        
        # === ActionLogger: 記錄刪除物件（從當前幀開始）===
        self.action_logger.log_delete_object(
            frame_idx=from_frame,
            obj_id=obj_id,
            delete_type="from_current"
        )
        
        logger.info(f"Deleted object {obj_id} from frame {from_frame}: removed {deleted_count} detections")
        
        # 更新 UI（先 reanalyze，再 track）
        self._reanalyze_with_preserved_edits()
        self._track_human_intervention(from_frame)  # ← 添加這行！
        self.update_object_list()
        self.update_analysis_display()
        self.update_timeline()
        self.display_frame(self.current_frame)
        
        self.statusBar().showMessage(
            f"Deleted Object {obj_id} from frame {from_frame} onwards ({deleted_count} detections removed)"
        )
    
    def show_merge_dialog(self, source_obj_id: int):
        """
        顯示合併物件對話框。
        
        Args:
            source_obj_id: 要合併的來源物件 ID（將被合併到目標物件）
        """
        # 收集所有物件資訊
        objects = self._get_all_object_info()
        
        if len(objects) < 2:
            QMessageBox.warning(self, "Warning", "Need at least 2 objects to merge")
            return
        
        dialog = ObjectSelectionDialog(
            self,
            title="Merge Object",
            message=f"Merge Object {source_obj_id} INTO which object?\n\n"
                    f"Object {source_obj_id} will be deleted and its detections "
                    f"will be assigned to the target object.",
            objects=objects,
            exclude_obj_id=source_obj_id
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            target_obj_id = dialog.get_selected_obj_id()
            if target_obj_id is not None:
                self.merge_objects(source_obj_id, target_obj_id)
    
    def merge_objects(self, source_obj_id: int, target_obj_id: int):
        """
        合併兩個物件：將 source 合併到 target。
        
        如果同一幀兩者都有 mask，保留 target 的。
        
        Args:
            source_obj_id: 來源物件 ID（將被刪除）
            target_obj_id: 目標物件 ID（保留）
        """
        # 先檢查當前幀是否兩者都有 mask
        current_frame_result = self.sam3_results.get(self.current_frame)
        if current_frame_result:
            has_source = any(d.obj_id == source_obj_id for d in current_frame_result.detections)
            has_target = any(d.obj_id == target_obj_id for d in current_frame_result.detections)
            
            if has_source and has_target:
                # 警告用戶
                reply = QMessageBox.warning(
                    self, "Warning: Overlapping Objects",
                    f"Both Object {source_obj_id} and Object {target_obj_id} exist in the current frame.\n\n"
                    f"If you merge, Object {source_obj_id}'s mask will be REMOVED in frames where both exist.\n\n"
                    f"Continue with merge?",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
                )
                if reply != QMessageBox.StandardButton.Yes:
                    return
        
        merged_count = 0
        discarded_count = 0  # 被丟棄的 mask 數量
        
        for frame_idx, frame_result in self.sam3_results.items():
            # 找到 source 和 target 的 detection
            source_det = None
            target_det = None
            source_idx = None
            
            for i, det in enumerate(frame_result.detections):
                if det.obj_id == source_obj_id:
                    source_det = det
                    source_idx = i
                elif det.obj_id == target_obj_id:
                    target_det = det
            
            if source_det is not None:
                if target_det is None:
                    # target 在這幀沒有 detection，將 source 改為 target
                    source_det.obj_id = target_obj_id
                    merged_count += 1
                else:
                    # 兩者都有，保留 target，移除 source
                    frame_result.detections.pop(source_idx)
                    discarded_count += 1
        
        # 移除 source 物件狀態
        if source_obj_id in self.object_status:
            del self.object_status[source_obj_id]
        
        edited_frame = self.current_frame  # 記錄操作發生的幀
        
        # === ActionLogger: 記錄合併物件 ===
        self.action_logger.log_merge_objects(
            frame_idx=edited_frame,
            source_obj_id=source_obj_id,
            target_obj_id=target_obj_id
        )
        
        logger.info(f"Merged object {source_obj_id} into {target_obj_id}: "
                   f"{merged_count} transferred, {discarded_count} discarded")
        
        # 更新 UI（先 reanalyze，再 track）
        self._reanalyze_with_preserved_edits()
        self._track_human_intervention(edited_frame)
        self.update_object_list()
        self.update_analysis_display()
        self.update_timeline()
        self.display_frame(self.current_frame)
        
        # 顯示更詳細的訊息
        msg = f"Merged Object {source_obj_id} into Object {target_obj_id}"
        if discarded_count > 0:
            msg += f" ({merged_count} transferred, {discarded_count} discarded due to overlap)"
        else:
            msg += f" ({merged_count} detections transferred)"
        self.statusBar().showMessage(msg)
    
    def show_swap_dialog(self, obj_id_a: int):
        """
        顯示交換標籤對話框。
        
        Args:
            obj_id_a: 第一個物件 ID
        """
        # 收集所有物件資訊
        objects = self._get_all_object_info()
        
        if len(objects) < 2:
            QMessageBox.warning(self, "Warning", "Need at least 2 objects to swap")
            return
        
        dialog = ObjectSelectionDialog(
            self,
            title="Swap Labels",
            message=f"Swap Object {obj_id_a} label with which object?\n\n"
                    f"This will exchange the object IDs in ALL frames.",
            objects=objects,
            exclude_obj_id=obj_id_a
        )
        
        if dialog.exec() == QDialog.DialogCode.Accepted:
            obj_id_b = dialog.get_selected_obj_id()
            if obj_id_b is not None:
                self.swap_object_labels(obj_id_a, obj_id_b)
    
    def swap_object_labels(self, obj_id_a: int, obj_id_b: int):
        """
        交換兩個物件的標籤（在所有幀中）。
        
        Args:
            obj_id_a: 第一個物件 ID
            obj_id_b: 第二個物件 ID
        """
        swap_count = 0
        temp_id = -9999  # 臨時 ID 避免衝突
        
        for frame_idx, frame_result in self.sam3_results.items():
            for det in frame_result.detections:
                if det.obj_id == obj_id_a:
                    det.obj_id = temp_id
                    swap_count += 1
            
            for det in frame_result.detections:
                if det.obj_id == obj_id_b:
                    det.obj_id = obj_id_a
            
            for det in frame_result.detections:
                if det.obj_id == temp_id:
                    det.obj_id = obj_id_b
        
        # 交換物件狀態
        status_a = self.object_status.get(obj_id_a, "pending")
        status_b = self.object_status.get(obj_id_b, "pending")
        self.object_status[obj_id_a] = status_b
        self.object_status[obj_id_b] = status_a
        
        edited_frame = self.current_frame  # 記錄操作發生的幀
        
        # === ActionLogger: 記錄交換標籤 ===
        self.action_logger.log_swap_labels(
            frame_idx=edited_frame,
            obj_a=obj_id_a,
            obj_b=obj_id_b
        )
        
        logger.info(f"Swapped labels: Object {obj_id_a} ↔ Object {obj_id_b} ({swap_count} detections affected)")
        
        # 更新 UI（先 reanalyze，再 track）
        self._reanalyze_with_preserved_edits()
        self._track_human_intervention(edited_frame)  # ← 添加這行！
        self.update_object_list()
        self.update_analysis_display()
        self.update_timeline()
        self.display_frame(self.current_frame)
        
        self.statusBar().showMessage(
            f"Swapped Object {obj_id_a} ↔ Object {obj_id_b}"
        )
    
    def jump_to_object_first_frame(self, obj_id: int):
        """跳轉到物件首次出現的幀。"""
        first_frame = None
        
        for frame_idx in sorted(self.sam3_results.keys()):
            frame_result = self.sam3_results[frame_idx]
            for det in frame_result.detections:
                if det.obj_id == obj_id:
                    first_frame = frame_idx
                    break
            if first_frame is not None:
                break
        
        if first_frame is not None:
            self.seek_to_frame(first_frame)
            self.statusBar().showMessage(f"Object {obj_id} first appears at frame {first_frame}")
        else:
            QMessageBox.information(self, "Info", f"Object {obj_id} not found in any frame")
    
    def _get_all_object_info(self) -> List[Tuple[int, float]]:
        """
        取得所有物件的資訊列表。
        
        Returns:
            List of (obj_id, avg_score) tuples
        """
        objects = []
        
        if self.video_analysis and self.video_analysis.object_summaries:
            for obj_id, summary in self.video_analysis.object_summaries.items():
                objects.append((obj_id, summary.avg_score))
        else:
            # Fallback: 從結果中收集
            obj_scores: Dict[int, List[float]] = {}
            for frame_result in self.sam3_results.values():
                for det in frame_result.detections:
                    if det.obj_id not in obj_scores:
                        obj_scores[det.obj_id] = []
                    obj_scores[det.obj_id].append(det.score)
            
            for obj_id, scores in obj_scores.items():
                avg_score = sum(scores) / len(scores) if scores else 0
                objects.append((obj_id, avg_score))
        
        return sorted(objects, key=lambda x: x[1], reverse=True)
    
    # =========================================================================
    # 影像處理與顯示
    # =========================================================================
    
    def cv2_to_qpixmap(self, cv_image: np.ndarray) -> QPixmap:
        """OpenCV 影像轉 QPixmap。"""
        if len(cv_image.shape) == 2:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_GRAY2RGB)
        else:
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
        
        height, width, channels = rgb_image.shape
        bytes_per_line = channels * width
        
        q_image = QImage(
            rgb_image.data,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888
        )
        
        return QPixmap.fromImage(q_image)
    
    def display_frame(self, frame_idx: int):
        """顯示指定幀。"""
        if self.video_loader is None:
            return
        
        # 取得原始幀
        frame = self.video_loader.get_frame(frame_idx)
        
        # 如果在 refinement 模式，使用 InteractiveCanvas 的方式顯示
        if self.refinement_active and frame_idx == self.current_frame:
            # 轉換 frame 為 QImage
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_frame.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format.Format_RGB888)
            
            # 設置 base image
            self.video_canvas.set_base_image(q_image)
            
            # 更新 UI
            self.current_frame = frame_idx
            total = self.video_loader.metadata.total_frames
            self.frame_label.setText(f"{frame_idx + 1} / {total}")
            self.timeline_slider.setValue(frame_idx)
            return
        
        # 如果有偵測結果，疊加視覺化
        if frame_idx in self.sam3_results and (self.show_masks or self.show_boxes):
            frame = self.visualize_frame(frame, self.sam3_results[frame_idx])
        
        # 繪製海平線（如果啟用）
        if self.show_horizon and self.horizon_result and self.horizon_result.valid:
            frame = self._draw_horizon_overlay(frame)
        
        # 轉換並顯示
        pixmap = self.cv2_to_qpixmap(frame)
        scaled_pixmap = pixmap.scaled(
            self.video_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.video_label.setPixmap(scaled_pixmap)
        
        # 更新 UI
        self.current_frame = frame_idx
        total = self.video_loader.metadata.total_frames
        self.frame_label.setText(f"{frame_idx + 1} / {total}")
        self.timeline_slider.setValue(frame_idx)
        
        # 更新 Timeline 當前幀指示器
        if hasattr(self, 'timeline_widget'):
            self.timeline_widget.set_current_frame(frame_idx)
    
    def visualize_frame(
        self, 
        frame: np.ndarray, 
        result: FrameResult
    ) -> np.ndarray:
        """疊加偵測結果視覺化。"""
        output = frame.copy()
        overlay = frame.copy()
        
        high_thresh = self.high_thresh_spin.value()
        low_thresh = self.low_thresh_spin.value()
        
        for det in result.detections:
            # 檢查物件狀態
            obj_status = self.object_status.get(det.obj_id, "pending")
            
            # 被拒絕的物件不顯示
            if obj_status == "rejected":
                continue
            
            # 決定顏色
            if det.score >= high_thresh:
                color = (0, 255, 0)  # 綠色
            elif det.score >= low_thresh:
                color = (0, 255, 255)  # 黃色
            else:
                color = (0, 0, 255)  # 紅色
            
            # 已接受的物件用藍色邊框標示
            if obj_status == "accepted":
                border_color = (255, 200, 0)  # 青色
            else:
                border_color = color
            
            # 繪製 mask
            if self.show_masks:
                mask = det.mask
                if mask.shape[:2] != frame.shape[:2]:
                    mask = cv2.resize(
                        mask.astype(np.float32),
                        (frame.shape[1], frame.shape[0])
                    )
                mask_bool = mask > 0.5
                overlay[mask_bool] = color
            
            # 繪製 bounding box
            if self.show_boxes:
                x, y, w, h = det.box.astype(int)
                cv2.rectangle(output, (x, y), (x + w, y + h), border_color, 2)
                
                # 標籤
                label = f"{det.obj_id}:{det.score:.2f}"
                
                # 計算 mask 上方位置
                if self.show_masks:
                    mask_coords = np.where(mask > 0.5)
                    if len(mask_coords[0]) > 0:
                        top_y = int(np.min(mask_coords[0]))
                        center_x = int(np.mean(mask_coords[1]))
                    else:
                        center_x = x + w // 2
                        top_y = y
                else:
                    center_x = x + w // 2
                    top_y = y
                
                font_scale = 0.45
                thickness = 1
                (text_w, text_h), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
                )
                
                label_x = center_x - text_w // 2
                label_y = max(top_y - 5, text_h + 5)
                
                cv2.rectangle(
                    output,
                    (label_x - 2, label_y - text_h - 2),
                    (label_x + text_w + 2, label_y + 2),
                    color, -1
                )
                cv2.putText(
                    output, label, (label_x, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), thickness
                )
        
        # 混合
        if self.show_masks:
            result_frame = cv2.addWeighted(overlay, 0.3, output, 0.7, 0)
        else:
            result_frame = output
        
        return result_frame
    
    # =========================================================================
    # 影片控制
    # =========================================================================
    
    def open_video(self):
        """開啟影片檔案。"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Video",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if not file_path:
            return
        
        try:
            # 結束之前的 session（如果有）
            if self.action_logger is not None and self.action_logger.session is not None:
                self.action_logger.end_session()
            
            # 釋放之前的 loader
            if self.video_loader is not None:
                self.video_loader.release()
            
            # 建立新的 loader
            self.video_loader = VideoLoader(file_path)
            
            # 重置圖片模式標誌
            self._is_image_folder_mode = False
            
            # 自動切換到 Video 模式
            self.processing_mode_combo.setCurrentIndex(0)
            
            # 清除之前的結果
            self.sam3_results = {}
            self.video_analysis = None
            self.object_status = {}
            self.object_list.clear()
            
            # 更新 UI
            total = self.video_loader.metadata.total_frames
            meta = self.video_loader.metadata
            self.timeline_slider.setMaximum(total - 1)
            self.timeline_slider.setValue(0)
            self.timeline_slider.setEnabled(True)
            
            self.prev_btn.setEnabled(True)
            self.play_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
            self.detect_btn.setEnabled(True)
            self.add_object_btn.setEnabled(True)  # 開啟影片後就可以手動新增物件
            
            # === ActionLogger: 創建/重新創建，使用影片所在目錄的 logs 子目錄 ===
            video_dir = Path(file_path).parent
            logs_dir = video_dir / "logs"
            self.action_logger = ActionLogger(
                output_dir=str(logs_dir),
                format="jsonl",
                auto_flush=True
            )
            
            # === ActionLogger: 開始新 session ===
            self.action_logger.start_session(
                video_path=file_path,
                total_frames=total,
                width=meta.width,
                height=meta.height,
                fps=meta.fps,
            )
            self._last_logged_frame = 0
            
            # 顯示第一幀
            self.display_frame(0)
            
            # 更新分析標籤
            self.analysis_label.setText(
                f"Video: {self.video_loader.metadata.filename}\n"
                f"Frames: {total}\n"
                f"FPS: {self.video_loader.metadata.fps:.1f}\n"
                f"Resolution: {self.video_loader.metadata.width}x"
                f"{self.video_loader.metadata.height}\n\n"
                "Please run detection"
            )
            
            self.statusBar().showMessage(f"Opened: {file_path} | Logs: {logs_dir}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot open video:\n{e}")
    
    def open_image_folder(self):
        """開啟圖片資料夾（獨立處理模式）。"""
        folder_path = QFileDialog.getExistingDirectory(
            self,
            "Select Image Folder",
            "",
            QFileDialog.Option.ShowDirsOnly
        )
        
        if not folder_path:
            return
        
        try:
            # 結束之前的 session（如果有）
            if self.action_logger is not None and self.action_logger.session is not None:
                self.action_logger.end_session()
            
            # 釋放之前的 loader
            if self.video_loader is not None:
                self.video_loader.release()
            
            # 建立 ImageFolderLoader
            self.video_loader = ImageFolderLoader(folder_path)
            
            # 標記為圖片模式
            self._is_image_folder_mode = True
            
            # 清除之前的結果
            self.sam3_results = {}
            self.video_analysis = None
            self.object_status = {}
            self.object_list.clear()
            
            # 更新 UI
            total = self.video_loader.metadata.total_frames
            meta = self.video_loader.metadata
            self.timeline_slider.setMaximum(total - 1)
            self.timeline_slider.setValue(0)
            self.timeline_slider.setEnabled(True)
            
            self.prev_btn.setEnabled(True)
            self.play_btn.setEnabled(False)  # 圖片模式不支援播放
            self.next_btn.setEnabled(True)
            self.detect_btn.setEnabled(True)
            self.add_object_btn.setEnabled(True)
            
            # 自動切換到 Images 模式
            self.processing_mode_combo.setCurrentIndex(1)
            
            # === ActionLogger: 創建，使用圖片資料夾的 logs 子目錄 ===
            logs_dir = Path(folder_path) / "logs"
            self.action_logger = ActionLogger(
                output_dir=str(logs_dir),
                format="jsonl",
                auto_flush=True
            )
            
            # === ActionLogger: 開始新 session ===
            self.action_logger.start_session(
                video_path=folder_path,  # 使用資料夾路徑
                total_frames=total,
                width=meta.width,
                height=meta.height,
                fps=1.0,  # 圖片沒有 fps
            )
            self._last_logged_frame = 0
            
            # 顯示第一張圖片
            self.display_frame(0)
            
            # 更新分析標籤
            self.analysis_label.setText(
                f"Folder: {self.video_loader.metadata.folder_name}\n"
                f"Images: {total}\n"
                f"Resolution: {meta.width}x{meta.height}\n\n"
                "Mode: Independent Detection\n"
                "(Each image processed separately)\n\n"
                "Please run detection"
            )
            
            self.statusBar().showMessage(
                f"Opened folder: {folder_path} ({total} images) | Logs: {logs_dir}"
            )
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot open image folder:\n{e}")
    
    def next_frame(self):
        """下一幀。"""
        if self.video_loader is None:
            return
        
        total = self.video_loader.metadata.total_frames
        if self.current_frame < total - 1:
            self.display_frame(self.current_frame + 1)
        else:
            self.stop_play()
    
    def prev_frame(self):
        """上一幀。"""
        if self.current_frame > 0:
            self.display_frame(self.current_frame - 1)
    
    def toggle_play(self):
        """切換播放/暫停。"""
        if self.is_playing:
            self.stop_play()
        else:
            self.start_play()
    
    def start_play(self):
        """開始播放。"""
        if self.video_loader is None:
            return
        
        self.is_playing = True
        self.play_btn.setText("||")
        interval = int(1000 / self.video_loader.metadata.fps)
        self.play_timer.start(interval)
    
    def stop_play(self):
        """停止播放。"""
        self.is_playing = False
        self.play_btn.setText("▶")
        self.play_timer.stop()
    
    def on_slider_moved(self, value: int):
        """滑桿移動。"""
        self.display_frame(value)
    
    def seek_to_frame(self, frame_idx: int):
        """跳轉到指定幀（供 Timeline 使用）。"""
        if self.video_loader is None:
            return
        
        from_frame = self.current_frame
        frame_idx = max(0, min(frame_idx, self.video_loader.metadata.total_frames - 1))
        
        # === ActionLogger: 記錄幀跳轉（只記錄重要跳轉，避免播放時大量 log）===
        if abs(frame_idx - from_frame) > 1:  # 只記錄跳躍超過 1 幀的情況
            self.action_logger.log_frame_navigation(from_frame=from_frame, to_frame=frame_idx)
        
        self.timeline_slider.setValue(frame_idx)
        self.display_frame(frame_idx)
    
    def on_timeline_object_selected(self, obj_id: int):
        """Timeline 上選擇了物件。"""
        # 在物件列表中選擇對應的物件
        for i in range(self.object_list.count()):
            item = self.object_list.item(i)
            if item.data(Qt.ItemDataRole.UserRole) == obj_id:
                self.object_list.setCurrentItem(item)
                break
        
        logger.info(f"Timeline selected object {obj_id}")
    
    def update_timeline(self):
        """更新 Timeline 顯示。"""
        if not hasattr(self, 'timeline_widget'):
            return
        
        if self.video_loader is None or not self.sam3_results:
            self.timeline_widget.setVisible(False)
            return
        
        # 取得 jitter frames
        jitter_frames = None
        if hasattr(self, 'jitter_analysis') and self.jitter_analysis:
            jitter_frames = self.jitter_analysis.get_all_jitter_frames()
        
        self.timeline_widget.set_data(
            sam3_results=self.sam3_results,
            total_frames=self.video_loader.metadata.total_frames,
            fps=self.video_loader.metadata.fps,
            object_status=getattr(self, 'object_status', None),
            jitter_frames=jitter_frames
        )
        self.timeline_widget.setVisible(True)
    
    # =========================================================================
    # SAM3 偵測
    # =========================================================================
    
    def run_detection(self):
        """執行 SAM3 偵測。"""
        if self.video_loader is None:
            return
            
        clear_gpu_memory()

        prompt = self.prompt_input.text().strip()
        if not prompt:
            QMessageBox.warning(self, "Warning", "Please enter a detection prompt")
            return
        
        mode = self.mode_combo.currentText()
        
        # 停止播放
        self.stop_play()
        
        # 判斷處理模式
        is_image_mode = (
            self.processing_mode_combo.currentIndex() == 1 or 
            getattr(self, '_is_image_folder_mode', False)
        )
        
        if is_image_mode:
            # === 圖片獨立處理模式 ===
            self._run_batch_detection(prompt, mode)
        else:
            # === 影片追蹤模式 ===
            self._run_video_detection(prompt, mode)
    
    def _run_video_detection(self, prompt: str, mode: str):
        """執行影片追蹤模式的偵測（原本的流程）。"""
        # Maritime ROI：偵測海平線
        if self.maritime_roi_checkbox.isChecked():
            self._run_horizon_detection()
        
        # 取得影片路徑（確保是字串）
        video_path = str(self.video_loader.video_path)
        logger.info(f"Starting video detection: video={video_path}, prompt={prompt}, mode={mode}")
        
        # === ActionLogger: 記錄偵測開始 ===
        self._detection_start_time = time.time()
        self.action_logger.log_detection_started(prompt=prompt, frame_idx=0)
        
        # 建立進度對話框
        self.progress_dialog = QProgressDialog(
            "Running SAM3 detection...", "Cancel", 0, 100, self
        )
        self.progress_dialog.setWindowTitle("Processing")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setAutoClose(False)  # 不要自動關閉
        self.progress_dialog.setAutoReset(False)  # 不要自動重設
        
        # 建立並啟動 worker 執行緒
        self.worker = SAM3Worker(video_path, prompt, mode)
        self.worker.progress.connect(self.on_detection_progress)
        self.worker.ready_to_propagate.connect(self.on_ready_to_propagate)
        self.worker.finished.connect(self.on_detection_finished)
        self.worker.error.connect(self.on_detection_error)
        self.worker.cancelled.connect(self.on_detection_cancelled)
        
        # 連接取消按鈕到 worker 的取消方法
        self.progress_dialog.canceled.connect(self._on_cancel_detection)
        
        self.detect_btn.setEnabled(False)
        self.worker.start()
    
    def _run_batch_detection(self, prompt: str, mode: str):
        """執行圖片獨立處理模式的偵測。"""
        # 取得圖片路徑列表
        if isinstance(self.video_loader, ImageFolderLoader):
            image_paths = self.video_loader.metadata.image_paths
        else:
            # 如果是 VideoLoader，需要先提取幀（這種情況較少見）
            QMessageBox.warning(
                self, "Warning", 
                "Please use 'Open Image Folder' for independent image processing,\n"
                "or switch to 'Video (Sequential Tracking)' mode."
            )
            return
        
        total = len(image_paths)
        logger.info(f"Starting batch detection: {total} images, prompt={prompt}, mode={mode}")
        
        # === ActionLogger: 記錄偵測開始 ===
        self._detection_start_time = time.time()
        self.action_logger.log_detection_started(prompt=prompt, frame_idx=0)
        
        # 建立進度對話框
        self.progress_dialog = QProgressDialog(
            f"Processing {total} images...", "Cancel", 0, total, self
        )
        self.progress_dialog.setWindowTitle("Batch Processing")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        self.progress_dialog.setAutoClose(False)
        self.progress_dialog.setAutoReset(False)
        
        # 建立並啟動 ImageBatchWorker
        self.worker = ImageBatchWorker(image_paths, prompt, mode)
        self.worker.progress.connect(self.on_batch_progress)
        self.worker.image_result.connect(self.on_batch_image_result)
        self.worker.finished.connect(self.on_batch_finished)
        self.worker.error.connect(self.on_batch_error)
        self.worker.cancelled.connect(self.on_batch_cancelled)
        
        # 連接取消按鈕
        self.progress_dialog.canceled.connect(self._on_cancel_detection)
        
        self.detect_btn.setEnabled(False)
        self.worker.start()
    
    # =========================================================================
    # Batch Processing Callbacks (for Independent Image Mode)
    # =========================================================================
    
    def on_batch_progress(self, current: int, total: int, message: str):
        """批次處理進度更新。"""
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.setValue(current)
            self.progress_dialog.setLabelText(message)
    
    def on_batch_image_result(self, frame_idx: int, frame_result):
        """單張圖片處理完成（即時回報）。"""
        # 儲存結果
        self.sam3_results[frame_idx] = frame_result
        
        # 如果是當前顯示的幀，立即更新顯示
        if frame_idx == self.current_frame:
            self.display_frame(frame_idx)
        
        logger.debug(f"Image {frame_idx} processed: {frame_result.num_objects} objects")
    
    def on_batch_finished(self, result: dict):
        """批次處理完成。"""
        # 清理對話框和 worker
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        if hasattr(self, 'worker') and self.worker:
            self.worker.wait(500)
            self.worker = None
        
        self.detect_btn.setEnabled(True)
        
        # 儲存結果（可能已經在 on_batch_image_result 中部分儲存）
        self.sam3_results = result.get("results", self.sam3_results)
        
        # 分析結果
        self.video_analysis = self.analyzer.analyze_video(self.sam3_results)
        
        # === ActionLogger: 記錄偵測完成 ===
        if self._detection_start_time:
            duration = time.time() - self._detection_start_time
        else:
            duration = 0.0
        self.action_logger.log_detection_finished(
            num_objects=self.video_analysis.unique_objects,
            duration_seconds=duration,
            num_frames=len(self.sam3_results)
        )
        self._detection_start_time = None
        
        # 注意：圖片模式不運行 Jitter Detection（因為沒有時序連續性）
        self.jitter_analysis = None
        
        # 更新物件列表
        self.update_object_list()
        
        # 更新分析顯示
        self.update_analysis_display()
        
        # 更新 Timeline
        self.update_timeline()
        
        # 重新顯示當前幀
        self.display_frame(self.current_frame)
        
        self.statusBar().showMessage(
            f"Batch detection completed: {self.video_analysis.unique_objects} objects "
            f"across {len(self.sam3_results)} images"
        )
    
    def on_batch_error(self, error_msg: str):
        """批次處理錯誤。"""
        # 清理對話框和 worker
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        if hasattr(self, 'worker') and self.worker:
            self.worker.wait(500)
            self.worker = None
        
        self.detect_btn.setEnabled(True)
        self._detection_start_time = None
        
        # 顯示錯誤
        display_msg = error_msg
        if len(error_msg) > 1000:
            display_msg = error_msg[:1000] + "\n\n... (see terminal for full error)"
        
        logger.error(f"Batch detection error:\n{error_msg}")
        QMessageBox.critical(self, "Batch Detection Error", f"Processing failed:\n\n{display_msg}")
    
    def on_batch_cancelled(self):
        """批次處理被取消。"""
        logger.info("Batch detection cancelled")
        
        # 清理
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        if hasattr(self, 'worker') and self.worker:
            self.worker.wait(1000)
            if self.worker.isRunning():
                self.worker.terminate()
                self.worker.wait(500)
            self.worker = None
        
        self.detect_btn.setEnabled(True)
        self._detection_start_time = None
        
        # 如果已經處理了一部分，保留那些結果
        if self.sam3_results:
            self.video_analysis = self.analyzer.analyze_video(self.sam3_results)
            self.update_object_list()
            self.update_analysis_display()
            self.statusBar().showMessage(
                f"Detection cancelled - {len(self.sam3_results)} images processed"
            )
        else:
            self.statusBar().showMessage("Detection cancelled")
    
    def _on_cancel_detection(self):
        """處理取消偵測請求"""
        logger.info("User requested detection cancellation")
        if hasattr(self, 'worker') and self.worker is not None:
            # 更新對話框顯示
            self.progress_dialog.setLabelText("Cancelling... please wait...")
            self.progress_dialog.setCancelButton(None)  # 隱藏取消按鈕，防止重複點擊
            
            # 請求取消
            self.worker.cancel()
    
    def on_detection_cancelled(self):
        """偵測被取消時的處理"""
        logger.info("Detection cancelled successfully")
        
        # 清理
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        if hasattr(self, 'worker') and self.worker:
            self.worker.wait(1000)  # 等待最多 1 秒
            if self.worker.isRunning():
                logger.warning("Worker still running after cancel, terminating...")
                self.worker.terminate()
                self.worker.wait(500)
            self.worker = None
        
        self.detect_btn.setEnabled(True)
        self._detection_start_time = None
        
        self.statusBar().showMessage("Detection cancelled")
    
    def on_ready_to_propagate(self, num_objects: int):
        """
        初步偵測完成，等待用戶確認是否繼續 propagate。
        
        這是一個關鍵暫停點，讓用戶可以在漫長的 propagate 開始前決定是否繼續。
        """
        logger.info(f"Ready to propagate: {num_objects} objects detected")
        
        # 暫時隱藏 progress dialog
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.hide()
        
        # 準備確認訊息
        if num_objects == 0:
            msg = (
                f"No objects detected with prompt: '{self.prompt_input.text()}'\n\n"
                f"Do you want to continue anyway?\n"
                f"(Propagation may still find objects in later frames)"
            )
        else:
            total_frames = self.video_loader.metadata.total_frames
            estimated_time = total_frames * 0.1  # 估算：每幀約 0.1 秒
            
            msg = (
                f"✅ Initial Detection Complete!\n\n"
                f"Objects found: {num_objects}\n"
                f"Total frames: {total_frames}\n"
                f"Estimated time: ~{estimated_time:.0f} seconds\n\n"
                f"⚠️ Warning: Once propagation starts, it cannot be interrupted.\n\n"
                f"Do you want to continue with propagation?"
            )
        
        # 顯示確認對話框
        reply = QMessageBox.question(
            self, 
            "Confirm Propagation",
            msg,
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes  # 預設選擇 Yes
        )
        
        if reply == QMessageBox.StandardButton.Yes:
            # 用戶確認，繼續 propagate
            logger.info("User confirmed propagation")
            
            # 重新顯示 progress dialog
            if hasattr(self, 'progress_dialog') and self.progress_dialog:
                self.progress_dialog.show()
                self.progress_dialog.setLabelText("Propagating masks (this may take a while)...")
                self.progress_dialog.setValue(50)
            
            # 通知 worker 繼續
            if hasattr(self, 'worker') and self.worker:
                self.worker.continue_propagation()
        else:
            # 用戶取消
            logger.info("User cancelled before propagation")
            
            # 通知 worker 取消
            if hasattr(self, 'worker') and self.worker:
                self.worker.cancel()
    
    def on_detection_progress(self, percent: int, message: str):
        """偵測進度更新。"""
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.setValue(percent)
            self.progress_dialog.setLabelText(message)
    
    def on_detection_finished(self, result: dict):
        """偵測完成。"""
        # 清理對話框和 worker
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        if hasattr(self, 'worker') and self.worker:
            self.worker.wait(500)
            self.worker = None
        
        self.detect_btn.setEnabled(True)
        
        self.sam3_results = result["results"]
        
        # Maritime ROI 後處理：過濾天空區域內的物件
        if self.maritime_roi_checkbox.isChecked() and self.horizon_result and self.horizon_result.valid:
            self._filter_sky_objects()
        
        # 分析結果
        self.video_analysis = self.analyzer.analyze_video(self.sam3_results)
        
        # === ActionLogger: 記錄偵測完成 ===
        if self._detection_start_time:
            duration = time.time() - self._detection_start_time
        else:
            duration = 0.0
        self.action_logger.log_detection_finished(
            num_objects=self.video_analysis.unique_objects,
            duration_seconds=duration,
            num_frames=len(self.sam3_results)
        )
        self._detection_start_time = None
        
        # 運行 Jitter Detection
        self._run_jitter_detection()
        
        # 更新物件列表
        self.update_object_list()
        
        # 更新分析顯示
        self.update_analysis_display()
        
        # 更新 Timeline
        self.update_timeline()
        
        # 重新顯示當前幀
        self.display_frame(self.current_frame)
        
        self.statusBar().showMessage(
            f"detection completed: {self.video_analysis.unique_objects} objects"
        )
    
    def _run_jitter_detection(self):
        """運行 Jitter Detection 分析時序穩定性。"""
        if not self.sam3_results:
            self.jitter_analysis = None
            return
        
        try:
            from core.jitter_detector import JitterDetector
            
            detector = JitterDetector(
                iou_threshold=0.85,
                area_change_threshold=0.15
            )
            self.jitter_analysis = detector.analyze_video(self.sam3_results)
            
            # 記錄結果
            ja = self.jitter_analysis
            logger.info(
                f"Jitter detection: {ja.total_jitter_events} events, "
                f"{ja.jitter_frame_count} frames, "
                f"stability: {ja.overall_stability:.1%}"
            )
            
            # 如果有 jitter，提示用戶
            if ja.jitter_frame_count > 0:
                jitter_frames = ja.get_all_jitter_frames()[:5]  # 前 5 個
                frames_str = ", ".join(str(f) for f in jitter_frames)
                if ja.jitter_frame_count > 5:
                    frames_str += f"... ({ja.jitter_frame_count} total)"
                logger.warning(f"Jitter detected at frames: {frames_str}")
                
        except Exception as e:
            logger.error(f"Jitter detection failed: {e}")
            self.jitter_analysis = None
    
    def _run_horizon_detection(self):
        """
        運行海平線偵測（Maritime ROI）。
        
        在第一幀上偵測海平線，用於排除天空區域。
        """
        if self.video_loader is None:
            self.horizon_result = None
            return
        
        try:
            from core.maritime_roi import MaritimeROI
            
            # 取得選擇的方法
            method = self._get_maritime_roi_method()
            
            # 初始化 MaritimeROI（如果需要）
            # MaritimeROI 會自動偵測 SegFormer 模型路徑（Docker 或 Host）
            if self.maritime_roi is None or getattr(self.maritime_roi, 'method', None) != method:
                self.maritime_roi = MaritimeROI(method=method)
            
            # 取得第一幀
            frame = self.video_loader.get_frame(0)
            if frame is None:
                logger.warning("Cannot read first frame for horizon detection")
                self.horizon_result = None
                return
            
            # 偵測海平線
            self.horizon_result = self.maritime_roi.detect_horizon(frame)
            
            if self.horizon_result.valid:
                logger.info(
                    f"Horizon detected: slope={self.horizon_result.slope:.4f}, "
                    f"center={self.horizon_result.center}, "
                    f"method={self.horizon_result.method_used}"
                )
                
                # 計算天空區域（供未來 SAM3 整合使用）
                sky_box = self.maritime_roi.get_sky_box_xyxy(frame, self.horizon_result)
                if sky_box:
                    logger.info(f"Sky box (negative region): {sky_box}")
            else:
                logger.warning("Horizon detection failed")
                
        except ImportError as e:
            logger.error(f"Maritime ROI module not found: {e}")
            self.horizon_result = None
        except Exception as e:
            logger.error(f"Horizon detection error: {e}")
            self.horizon_result = None
    
    def _draw_horizon_overlay(self, frame: np.ndarray) -> np.ndarray:
        """
        在畫面上繪製海平線和天空區域視覺化。
        
        - 黃色線：偵測到的海平線
        - 半透明紅色區域：天空區域（會被過濾的區域）
        - 顯示海平線資訊文字
        """
        if not self.horizon_result or not self.horizon_result.valid:
            return frame
        
        output = frame.copy()
        h, w = output.shape[:2]
        
        # 取得海平線參數
        slope = self.horizon_result.slope
        center_x, center_y = self.horizon_result.center
        
        # 計算海平線兩端點
        # y = slope * (x - center_x) + center_y
        x1, x2 = 0, w
        y1 = int(slope * (x1 - center_x) + center_y)
        y2 = int(slope * (x2 - center_x) + center_y)
        
        # 取得 sky box
        sky_box = None
        if self.maritime_roi:
            sky_box = self.maritime_roi.get_sky_box_xyxy(frame, self.horizon_result)
        
        # 繪製天空區域（半透明紅色）
        if sky_box:
            sky_x1, sky_y1, sky_x2, sky_y2 = sky_box
            overlay = output.copy()
            cv2.rectangle(overlay, (int(sky_x1), int(sky_y1)), (int(sky_x2), int(sky_y2)), 
                         (0, 0, 255), -1)  # 紅色填充
            cv2.addWeighted(overlay, 0.2, output, 0.8, 0, output)  # 20% 透明度
            
            # 繪製 sky box 邊框
            cv2.rectangle(output, (int(sky_x1), int(sky_y1)), (int(sky_x2), int(sky_y2)), 
                         (0, 0, 255), 2)  # 紅色邊框
        
        # 繪製海平線（黃色粗線）
        cv2.line(output, (x1, y1), (x2, y2), (0, 255, 255), 3)  # 黃色
        
        # 繪製海平線中心點
        cv2.circle(output, (int(center_x), int(center_y)), 8, (0, 255, 0), -1)  # 綠色圓點
        cv2.circle(output, (int(center_x), int(center_y)), 8, (0, 0, 0), 2)  # 黑色邊框
        
        # 顯示資訊文字
        info_lines = [
            f"Horizon: y={center_y}, slope={slope:.4f}",
            f"Method: {self.horizon_result.method_used}",
        ]
        if sky_box:
            info_lines.append(f"Sky region: y < {int(sky_box[3])}")
        
        # 繪製文字背景
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        thickness = 2
        padding = 10
        
        y_offset = 30
        for line in info_lines:
            (text_w, text_h), baseline = cv2.getTextSize(line, font, font_scale, thickness)
            cv2.rectangle(output, (10, y_offset - text_h - padding), 
                         (10 + text_w + padding * 2, y_offset + baseline + padding), 
                         (0, 0, 0), -1)
            cv2.putText(output, line, (10 + padding, y_offset), font, font_scale, 
                       (0, 255, 255), thickness)
            y_offset += text_h + baseline + padding * 2
        
        return output
    
    def _filter_sky_objects(self):
        """
        後處理過濾：移除 bounding box 中心點位於天空區域的物件。
        
        這個方法在 SAM3 偵測完成後執行，用於過濾掉可能是天空中
        誤偵測的物件（例如雲、飛機、海平線上的遠處船隻等）。
        
        改進：檢查物件「首次有效出現」的幀，而不是只看第一幀。
        這樣可以正確處理影片中途才出現的物件。
        """
        if not self.sam3_results or not self.horizon_result or not self.horizon_result.valid:
            return
        
        if self.video_loader is None:
            return
        
        # 取得第一幀來計算 sky box
        frame = self.video_loader.get_frame(0)
        if frame is None:
            return
        
        sky_box = self.maritime_roi.get_sky_box_xyxy(frame, self.horizon_result)
        if sky_box is None:
            return
        
        sky_x1, sky_y1, sky_x2, sky_y2 = sky_box
        
        # 收集所有物件的「首次有效出現」資訊
        # 有效出現 = bounding box 面積大於閾值
        MIN_BOX_AREA = 100  # 最小有效面積（像素²）
        
        object_first_appearance = {}  # obj_id -> (frame_idx, center_x, center_y)
        
        for frame_idx in sorted(self.sam3_results.keys()):
            frame_result = self.sam3_results[frame_idx]
            for detection in frame_result.detections:
                obj_id = detection.obj_id
                
                # 如果已經記錄過這個物件的首次出現，跳過
                if obj_id in object_first_appearance:
                    continue
                
                # 取得 bounding box (xywh 格式)
                x, y, w, h = detection.box
                box_area = w * h
                
                # 檢查是否為有效的 bounding box
                if box_area < MIN_BOX_AREA:
                    continue
                
                # 計算中心點
                center_x = x + w / 2
                center_y = y + h / 2
                
                object_first_appearance[obj_id] = (frame_idx, center_x, center_y)
        
        # 找出需要過濾的物件 ID
        objects_to_remove = set()
        
        for obj_id, (frame_idx, center_x, center_y) in object_first_appearance.items():
            # 檢查中心點是否在 sky box 內
            if sky_x1 <= center_x <= sky_x2 and sky_y1 <= center_y <= sky_y2:
                objects_to_remove.add(obj_id)
                logger.info(
                    f"Filtering object {obj_id}: "
                    f"first valid appearance at frame {frame_idx}, "
                    f"center=({center_x:.0f}, {center_y:.0f}) is in sky region (y < {sky_y2})"
                )
        
        if not objects_to_remove:
            logger.info("Maritime ROI: No objects filtered (all objects are below horizon)")
            return
        
        # 從所有幀中移除這些物件
        filtered_count = 0
        for frame_idx, frame_result in self.sam3_results.items():
            original_count = len(frame_result.detections)
            frame_result.detections = [
                d for d in frame_result.detections 
                if d.obj_id not in objects_to_remove
            ]
            filtered_count += original_count - len(frame_result.detections)
        
        logger.info(
            f"Maritime ROI filter: removed {len(objects_to_remove)} objects "
            f"({filtered_count} detections total) - sky region y < {sky_y2}"
        )
    
    def _reanalyze_with_preserved_edits(self):
        """
        重新分析 SAM3 結果，同時保留已編輯的幀記錄。
        
        解決問題：analyze_video() 會創建新的 VideoAnalysis 物件，
        導致 frames_actually_edited 被重置。
        """
        # 1. 保存現有的 edited frames
        preserved_edits = set()
        if self.video_analysis and self.video_analysis.frames_actually_edited:
            preserved_edits = self.video_analysis.frames_actually_edited.copy()
            logger.debug(f"Preserving {len(preserved_edits)} edited frames before reanalysis")
        
        # 2. 重新分析
        self.video_analysis = self.analyzer.analyze_video(self.sam3_results)
        
        # 3. 恢復 edited frames
        self.video_analysis.frames_actually_edited = preserved_edits
        
        # 4. 重新運行 Jitter Detection
        self._run_jitter_detection()
        
        logger.debug(f"Reanalysis complete, {len(preserved_edits)} edited frames restored")
    
    def on_detection_error(self, error_msg: str):
        """偵測錯誤。"""
        # 清理對話框和 worker
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        if hasattr(self, 'worker') and self.worker:
            self.worker.wait(500)
            self.worker = None
        
        self.detect_btn.setEnabled(True)
        self._detection_start_time = None
        
        # 顯示錯誤（如果太長就截斷）
        display_msg = error_msg
        if len(error_msg) > 1000:
            display_msg = error_msg[:1000] + "\n\n... (see terminal for full error)"
        
        logger.error(f"Detection error:\n{error_msg}")
        QMessageBox.critical(self, "Detection Error", f"SAM3 detection failed:\n\n{display_msg}")
    
    # =========================================================================
    # 物件列表管理
    # =========================================================================
    
    def update_object_list(self):
        """更新物件列表。"""
        self.object_list.clear()
        self.object_status = {}
        
        if self.video_analysis is None:
            return
        
        # 按平均分數排序
        sorted_objects = sorted(
            self.video_analysis.object_summaries.values(),
            key=lambda x: x.avg_score,
            reverse=True
        )
        
        for obj_summary in sorted_objects:
            category = self.analyzer.categorize(obj_summary.avg_score)
            
            # 建立列表項目
            item = QListWidgetItem()
            item_widget = ObjectListItem(
                obj_summary.obj_id,
                obj_summary.avg_score,
                category
            )
            item_widget.status_changed.connect(self.on_object_status_changed)
            
            # 設置 obj_id property 以便後續取得
            item_widget.setProperty("obj_id", obj_summary.obj_id)
            
            item.setSizeHint(item_widget.sizeHint())
            self.object_list.addItem(item)
            self.object_list.setItemWidget(item, item_widget)
            
            # 初始化狀態
            self.object_status[obj_summary.obj_id] = "pending"
        
        self.accept_all_high_btn.setEnabled(True)
        self.reset_all_btn.setEnabled(True)
    
    def update_analysis_display(self):
        """更新分析結果顯示。"""
        if self.video_analysis is None:
            return
        
        va = self.video_analysis
        
        # 計算實際 HIR
        edited_frames = len(va.frames_actually_edited) if va.frames_actually_edited else 0
        actual_hir = edited_frames / va.total_frames * 100 if va.total_frames > 0 else 0
        
        # Jitter 資訊（如果有）
        jitter_info = ""
        if hasattr(self, 'jitter_analysis') and self.jitter_analysis:
            ja = self.jitter_analysis
            jitter_info = f"\n\nStability: {ja.overall_stability:.1%}\nJitter Frames: {ja.jitter_frame_count}"
        
        text = (
            f"Unique Objects: {va.unique_objects}\n"
            f"Total Detections: {va.total_objects}\n\n"
            f"HIGH: {va.high_count} ({va.auto_accept_rate:.1f}%)\n"
            f"UNCERTAIN: {va.uncertain_count}\n"
            f"LOW: {va.low_count}\n\n"
            f"Potential Review: {va.frames_need_review} frames\n"
            f"Actually Edited: {edited_frames} frames\n"
            f"Actual HIR: {actual_hir:.1f}%"
            f"{jitter_info}"
        )
        self.analysis_label.setText(text)
    
    def on_object_status_changed(self, obj_id: int, status: str):
        """物件狀態改變。"""
        self.object_status[obj_id] = status
        
        # === ActionLogger: 記錄審核操作 ===
        if status == "accepted":
            self.action_logger.log_approve_object(
                frame_idx=self.current_frame,
                obj_id=obj_id
            )
        elif status == "rejected":
            self.action_logger.log_reject_object(
                frame_idx=self.current_frame,
                obj_id=obj_id
            )
        
        self.display_frame(self.current_frame)  # 重新顯示以更新視覺化
    
    def accept_all_high(self):
        """接受所有 HIGH 信心物件。"""
        for i in range(self.object_list.count()):
            item = self.object_list.item(i)
            widget = self.object_list.itemWidget(item)
            if widget.category == ConfidenceCategory.HIGH:
                widget.accept()
    
    def reset_all_objects(self):
        """重設所有物件狀態。"""
        for i in range(self.object_list.count()):
            item = self.object_list.item(i)
            widget = self.object_list.itemWidget(item)
            widget.reset()
            self.object_status[widget.obj_id] = "pending"
        
        self.display_frame(self.current_frame)
    
    # =========================================================================
    # 其他功能
    # =========================================================================
    
    def on_display_option_changed(self):
        """顯示選項改變。"""
        self.show_masks = self.mask_checkbox.isChecked()
        self.show_boxes = self.box_checkbox.isChecked()
        self.show_horizon = self.horizon_checkbox.isChecked()
        self.mask_action.setChecked(self.show_masks)
        self.box_action.setChecked(self.show_boxes)
        self.display_frame(self.current_frame)
    
    def on_threshold_changed(self):
        """閾值改變。"""
        high = self.high_thresh_spin.value()
        low = self.low_thresh_spin.value()
        
        if low >= high:
            return
        
        self.analyzer.update_thresholds(high, low)
        
        # 如果有結果，重新分析
        if self.sam3_results:
            self.video_analysis = self.analyzer.analyze_video(self.sam3_results)
            self.update_object_list()
            self.update_analysis_display()
            self.display_frame(self.current_frame)
    
    def _on_maritime_roi_changed(self, state):
        """Maritime ROI checkbox 狀態改變。"""
        enabled = (state == Qt.CheckState.Checked.value)
        self.maritime_method_combo.setEnabled(enabled)
        
        if enabled:
            logger.info("Maritime ROI enabled")
        else:
            logger.info("Maritime ROI disabled")
    
    def _get_maritime_roi_method(self) -> str:
        """取得選擇的 Maritime ROI 方法。"""
        text = self.maritime_method_combo.currentText()
        if "Auto" in text:
            return "auto"
        elif "Traditional" in text:
            return "traditional"
        elif "SegFormer" in text:
            return "segformer"
        return "auto"
    
    def export_results(self):
        """匯出標註結果。"""
        if not self.sam3_results:
            QMessageBox.warning(self, "Warning", "No results to export")
            return
        
        # 取得預設名稱（影片檔名）
        default_name = Path(self.video_loader.video_path).stem
        video_fps = self.video_loader.metadata.fps
        
        # 收集 object 資訊
        object_info = []
        if self.video_analysis and self.video_analysis.object_summaries:
            for obj_id, summary in self.video_analysis.object_summaries.items():
                object_info.append({
                    'obj_id': obj_id,
                    'avg_score': summary.avg_score,
                    'status': self.object_status.get(obj_id, 'pending')
                })
        else:
            # 從 results 中收集 unique object IDs
            all_obj_ids = set()
            obj_scores = {}
            for frame_result in self.sam3_results.values():
                for det in frame_result.detections:
                    all_obj_ids.add(det.obj_id)
                    if det.obj_id not in obj_scores:
                        obj_scores[det.obj_id] = []
                    obj_scores[det.obj_id].append(det.score)
            
            for obj_id in sorted(all_obj_ids):
                scores = obj_scores.get(obj_id, [0])
                object_info.append({
                    'obj_id': obj_id,
                    'avg_score': sum(scores) / len(scores),
                    'status': self.object_status.get(obj_id, 'pending')
                })
        
        # 建立 Export 對話框（傳入 object_info）
        dialog = ExportDialog(
            self, 
            default_name=default_name, 
            video_fps=video_fps,
            object_info=object_info
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        
        # 驗證設定
        if not dialog.is_valid():
            QMessageBox.warning(self, "Warning", "Train/Val/Test ratios must sum to 1.0")
            return
        
        # 取得匯出設定
        formats = dialog.get_selected_formats()
        output_dir = dialog.get_output_dir()
        dataset_name = dialog.get_dataset_name()
        labels = dialog.get_labels()
        object_labels = dialog.get_object_labels()  # obj_id -> label_name
        frame_interval = dialog.get_frame_interval()
        include_rejected = dialog.get_include_rejected()
        include_hil_fields = dialog.get_include_hil_fields()
        train_ratio, val_ratio, test_ratio = dialog.get_split_ratios()
        
        # 計算 frame step
        if frame_interval > 0:
            frame_step = max(1, int(frame_interval * video_fps))
        else:
            frame_step = 1  # Export all frames
        
        # 建立 categories（從 labels 生成）
        categories = [{"id": i, "name": label, "supercategory": "maritime"} for i, label in enumerate(labels)]
        
        # 建立 label_name -> category_id 的對應
        label_to_cat_id = {label: i for i, label in enumerate(labels)}
        
        # 建立 obj_id -> category_id 的對應
        object_category_ids = {obj_id: label_to_cat_id.get(label, 0) for obj_id, label in object_labels.items()}
        
        # 建立 ExportConfig
        config = ExportConfig(
            output_dir=Path(output_dir),
            base_name=dataset_name,
            video_path=str(self.video_loader.video_path),
            video_fps=video_fps,
            video_width=self.video_loader.metadata.width,
            video_height=self.video_loader.metadata.height,
            include_rejected=include_rejected,
            include_hil_fields=include_hil_fields,
            categories=categories,
            frame_step=frame_step,
            train_ratio=train_ratio,
            val_ratio=val_ratio,
            test_ratio=test_ratio
        )
        
        # 建立進度對話框
        progress = QProgressDialog("Exporting annotations...", None, 0, 0, self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.show()
        QApplication.processEvents()
        
        # 執行匯出
        try:
            exporter = AnnotationExporter(config)
            stats = exporter.export_all(
                self.sam3_results,
                self.object_status,
                self.video_analysis,
                formats=formats,
                object_labels=object_labels  # 傳入 object -> label 對應
            )
            
            progress.close()
            
            # === ActionLogger: 記錄匯出操作 ===
            self.action_logger.log_export(
                format=", ".join(stats.formats_exported),
                output_path=str(stats.output_dir),
                num_frames=stats.total_frames,
                num_objects=self.video_analysis.unique_objects if self.video_analysis else 0
            )
            
            # 顯示結果
            interval_info = f"(every {frame_step} frames)" if frame_step > 1 else "(all frames)"
            labels_str = ", ".join(labels)
            msg = (
                f"Export Complete!\n\n"
                f"Total Frames: {stats.total_frames} {interval_info}\n"
                f"Total Annotations: {stats.total_annotations}\n"
                f"Labels: {labels_str}\n\n"
                f"Dataset Split (COCO/Parquet):\n"
                f"  Train: {stats.train_images} images\n"
                f"  Val: {stats.val_images} images\n"
                f"  Test: {stats.test_images} images\n\n"
                f"Object Status:\n"
                f"  Accepted: {stats.accepted_objects}\n"
                f"  Rejected: {stats.rejected_objects}\n"
                f"  Pending: {stats.pending_objects}\n\n"
                f"Formats: {', '.join(stats.formats_exported)}\n\n"
                f"Output:\n"
                f"  {stats.output_dir}/\n"
            )
            
            QMessageBox.information(self, "Export Complete", msg)
            self.statusBar().showMessage(f"Exported to {stats.output_dir}")
            
        except Exception as e:
            progress.close()
            logger.error(f"Export error: {e}")
            import traceback
            traceback.print_exc()
            QMessageBox.critical(self, "Export Error", f"Failed to export:\n{e}")
    
    def show_about(self):
        """顯示關於對話框。"""
        QMessageBox.about(
            self,
            "About HIL-AA",
            "HIL-AA Maritime Annotation Tool\n\n"
            "Human-in-the-Loop Active Annotation\n"
            "for Maritime Video using SAM3\n\n"
            "Author: Sonic\n"
            "Assistive Robotics Group, NYCU\n\n"
            "Key Innovation:\n"
            "Use SAM3 confidence scores to minimize\n"
            "human annotation effort by 5-10x\n\n"
            "NEW: Object Management\n"
            "Right-click objects to delete, merge, or swap labels"
        )
    
    # =========================================================================
    # 事件處理
    # =========================================================================
    
    def keyPressEvent(self, event):
        """鍵盤事件處理。"""
        key = event.key()
        
        if key == Qt.Key.Key_Space:
            self.toggle_play()
        elif key == Qt.Key.Key_Left:
            self.prev_frame()
        elif key == Qt.Key.Key_Right:
            self.next_frame()
        elif key == Qt.Key.Key_Home:
            self.display_frame(0)
        elif key == Qt.Key.Key_End:
            if self.video_loader:
                self.display_frame(self.video_loader.metadata.total_frames - 1)
        elif key == Qt.Key.Key_Delete:
            # 刪除選中的物件
            self._delete_selected_object()
        else:
            super().keyPressEvent(event)
    
    def _delete_selected_object(self):
        """刪除選中的物件（快捷鍵）。"""
        selected_items = self.object_list.selectedItems()
        if not selected_items:
            return
        
        item = selected_items[0]
        widget = self.object_list.itemWidget(item)
        if widget:
            obj_id = widget.property("obj_id")
            if obj_id is not None:
                self.delete_object(obj_id)
    
    def resizeEvent(self, event):
        """視窗大小改變。"""
        super().resizeEvent(event)
        if self.video_loader:
            self.display_frame(self.current_frame)
    
    def closeEvent(self, event):
        """視窗關閉。"""
        self.stop_play()
        
        # 如果有正在運行的 worker，先取消它
        if hasattr(self, 'worker') and self.worker and self.worker.isRunning():
            logger.info("closeEvent: Cancelling running worker...")
            self.worker.cancel()
            self.worker.wait(2000)  # 等待最多 2 秒
            if self.worker.isRunning():
                logger.warning("closeEvent: Worker still running, terminating...")
                self.worker.terminate()
                self.worker.wait(500)
            self.worker = None
        
        # 關閉 progress dialog（如果有）
        if hasattr(self, 'progress_dialog') and self.progress_dialog:
            self.progress_dialog.close()
            self.progress_dialog = None
        
        # === ActionLogger: 結束 session 並顯示效率指標 ===
        if self.action_logger is not None and self.action_logger.session is not None:
            logs_path = self.action_logger.output_dir
            metrics = self.action_logger.end_session()
            if metrics and metrics.total_frames > 0:
                # 顯示效率指標摘要
                msg = (
                    f"Session Complete!\n\n"
                    f"=== Efficiency Metrics ===\n"
                    f"TEO: {metrics.total_edit_operations} edit operations "
                    f"(primary workload metric)\n"
                    f"EOR: {metrics.eor:.4f} edits/frame\n"
                    f"FCR: {metrics.fcr:.1f}% "
                    f"({metrics.edited_frame_count}/{metrics.total_frames} frames touched)\n"
                    f"CPO: {metrics.cpo:.2f} clicks/object "
                    f"({metrics.total_clicks} clicks, {metrics.total_objects} objects)\n"
                    f"SPF: {metrics.spf:.2f} seconds/frame "
                    f"({metrics.total_seconds:.1f}s total)\n\n"
                    f"Log saved to: {logs_path}/"
                )
                QMessageBox.information(self, "Session Summary", msg)
        
        if self.video_loader:
            self.video_loader.release()
        event.accept()
    
    # =========================================================================
    # Interactive Refinement Methods
    # =========================================================================
    
    def on_object_selection_changed(self):
        """當物件選擇改變時，更新 Refine 按鈕狀態。"""
        selected_items = self.object_list.selectedItems()
        self.refine_btn.setEnabled(len(selected_items) > 0 and not self.refinement_active)
    
    def start_refinement_for_selected(self):
        """開始對選中的物件進行 refinement。"""
        selected_items = self.object_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "Please select an object to refine")
            return
        
        # 取得選中物件的 ID
        item = selected_items[0]
        item_widget = self.object_list.itemWidget(item)
        if not item_widget:
            return
        
        # 從 widget 中取得 obj_id（存在 property 中）
        obj_id = item_widget.property("obj_id")
        if obj_id is None:
            return
        
        # 記住當前幀位置
        target_frame = self.current_frame
        
        # 取得該物件在當前幀的 mask
        frame_result = self.sam3_results.get(target_frame)
        if not frame_result:
            QMessageBox.warning(self, "Warning", "No detection result for current frame")
            return
        
        # 找到對應的 detection
        target_det = None
        for det in frame_result.detections:
            if det.obj_id == obj_id:
                target_det = det
                break
        
        if target_det is None:
            QMessageBox.warning(self, "Warning", f"Object {obj_id} not found in current frame")
            return
        
        # 停止播放（先停止，避免播放時改變 current_frame）
        self.stop_play()
        
        # 進入 refinement 模式
        self.refinement_active = True
        self.refinement_obj_id = obj_id
        
        # 設置 canvas 為 refinement 模式
        self.video_canvas.enter_refinement_mode(
            obj_id=obj_id,
            frame_idx=target_frame,
            mask=target_det.mask
        )
        
        # 顯示控制面板
        score = target_det.score
        self.refinement_panel.enter_refinement(obj_id, score)
        
        # 禁用其他控制
        self._set_controls_enabled(False)
        
        # 重新顯示當前幀（確保 canvas 顯示正確的幀）
        self.display_frame(target_frame)
        
        self.statusBar().showMessage(f"Refinement Mode: Object {obj_id} at Frame {target_frame} - Left click to include, Right click to exclude")
        logger.info(f"Started refinement for object {obj_id} at frame {target_frame}")
    
    def start_add_object(self):
        """開始手動新增物件模式。"""
        if self.video_loader is None:
            QMessageBox.warning(self, "Warning", "Please open a video first")
            return
        
        # 記住當前幀位置
        target_frame = self.current_frame
        
        # 取得當前幀圖像大小
        frame = self.video_loader.get_frame(target_frame)
        if frame is None:
            QMessageBox.warning(self, "Warning", "Cannot get current frame")
            return
        
        h, w = frame.shape[:2]
        
        # 停止播放（先停止，避免播放時改變 current_frame）
        self.stop_play()
        
        # 進入 add object 模式
        self.refinement_active = True
        self.add_object_mode = True
        self.refinement_obj_id = None
        
        # 設置 canvas 為 add object 模式
        self.video_canvas.enter_add_object_mode(
            frame_idx=target_frame,
            image_shape=(h, w)
        )
        
        # 顯示控制面板（add object 模式）
        self.refinement_panel.enter_add_object()
        
        # 禁用其他控制
        self._set_controls_enabled(False)
        
        # 重新顯示當前幀（關鍵！確保 canvas 顯示正確的幀）
        self.display_frame(target_frame)
        
        self.statusBar().showMessage(f"Add Object Mode (Frame {target_frame}): Left click to include, Right click to exclude")
        logger.info(f"Started add object mode at frame {target_frame}")
    
    def on_refinement_point_added(self, x: int, y: int, is_positive: bool):
        """處理 refinement 點擊。"""
        if not self.refinement_active:
            return
        
        # === ActionLogger: 記錄點擊 ===
        self.action_logger.log_click(
            frame_idx=self.current_frame,
            x=x,
            y=y,
            positive=is_positive,
            obj_id=self.refinement_obj_id  # 可能是 None（add object mode）
        )
        
        # 更新點數顯示
        if self.video_canvas.refinement_state:
            point_count = len(self.video_canvas.refinement_state.points)
            self.refinement_panel.set_point_count(point_count)
        
        # 執行 SAM3 refinement
        self._run_refinement()
    
    def _run_refinement(self):
        """執行 SAM3 refinement。"""
        if not self.video_canvas.refinement_state:
            return
        
        state = self.video_canvas.refinement_state
        
        # 取得當前幀圖像
        frame = self.video_loader.get_frame(self.current_frame)
        if frame is None:
            return
        
        # 取得 points 和 labels
        points, labels = state.get_sam_inputs()
        
        if len(points) == 0:
            # 沒有點，顯示原始 mask
            self.video_canvas.update_refined_mask(state.original_mask)
            return
        
        # 初始化 SAM3 engine（如果還沒有）
        if self.sam3_engine is None:
            try:
                self.sam3_engine = SAM3Engine(mode="auto")
            except Exception as e:
                logger.error(f"Failed to initialize SAM3 engine: {e}")
                # 使用 mock refinement
                self._run_mock_refinement(points, labels, state.original_mask)
                return
        
        # 執行 refinement（不傳 mask_input，讓 SAM3 純粹根據 point prompts 預測）
        # 注意：SAM3 的 mask_input 需要是 logits 格式，而我們只有 binary mask
        try:
            new_mask = self.sam3_engine.refine_mask(
                image=frame,
                points=points,
                labels=labels,
                mask_input=None  # 純粹使用 point prompts
            )
            
            # 更新顯示
            self.video_canvas.update_refined_mask(new_mask)
            
        except Exception as e:
            logger.error(f"Refinement error: {e}")
            # Fallback to mock
            self._run_mock_refinement(points, labels, state.original_mask)
    
    def _run_mock_refinement(self, points: np.ndarray, labels: np.ndarray, original_mask: np.ndarray):
        """Mock refinement for testing."""
        h, w = original_mask.shape
        result_mask = original_mask.astype(np.float32).copy()
        
        for point, label in zip(points, labels):
            x, y = int(point[0]), int(point[1])
            radius = 30
            
            yy, xx = np.ogrid[:h, :w]
            circle = ((xx - x) ** 2 + (yy - y) ** 2) <= radius ** 2
            
            if label == 1:
                result_mask = np.maximum(result_mask, circle.astype(np.float32))
            else:
                result_mask[circle] = 0
        
        self.video_canvas.update_refined_mask(result_mask > 0.5)
    
    def on_refinement_clear(self):
        """清除所有 refinement 點。"""
        self.video_canvas.clear_points()
        self.refinement_panel.set_point_count(0)
        self.display_frame(self.current_frame)  # 重新顯示原始 mask
    
    def on_refinement_undo(self):
        """撤銷上一個 refinement 點。"""
        self.video_canvas.undo_last_point()
        
        if self.video_canvas.refinement_state:
            point_count = len(self.video_canvas.refinement_state.points)
            self.refinement_panel.set_point_count(point_count)
        
        # 重新計算 mask
        self._run_refinement()
    
    def on_refinement_apply(self):
        """套用 refinement 結果或新增物件。"""
        if not self.refinement_active or not self.video_canvas.refinement_state:
            return
        
        state = self.video_canvas.refinement_state
        new_mask = state.current_mask
        
        # 檢查 mask 是否有效
        if new_mask is None or not np.any(new_mask):
            QMessageBox.warning(self, "Warning", "No valid mask to apply. Please add points first.")
            return
        
        edited_frame = self.current_frame
        
        if self.add_object_mode:
            # === Add New Object Mode ===
            new_obj_id = self._add_new_object(new_mask)
            
            # === ActionLogger: 記錄新增物件 ===
            if new_obj_id is not None:
                self.action_logger.log_add_object(
                    frame_idx=edited_frame,
                    obj_id=new_obj_id
                )
        else:
            # === Refinement Mode ===
            obj_id = state.object_id
            
            # 更新 sam3_results 中的 mask
            frame_result = self.sam3_results.get(self.current_frame)
            if frame_result:
                for det in frame_result.detections:
                    if det.obj_id == obj_id:
                        det.mask = new_mask.astype(np.uint8)
                        logger.info(f"Applied refined mask for object {obj_id}")
                        break
            
            # === ActionLogger: 記錄套用修正 ===
            self.action_logger.log_apply_refine(
                frame_idx=edited_frame,
                obj_id=obj_id
            )
            
            self.statusBar().showMessage(f"Refinement applied for object {obj_id}")
        
        # ====== 關鍵修復：正確的順序 ======
        # 1. 重新分析（保留已編輯的幀）
        self._reanalyze_with_preserved_edits()
        
        # 2. 追蹤人類介入（在 reanalyze 之後！）
        self._track_human_intervention(edited_frame)
        
        # 3. 更新 UI
        self.update_object_list()
        self.update_analysis_display()
        
        # 退出 refinement 模式
        self._exit_refinement_mode()
        
        # 重新顯示更新後的幀
        self.display_frame(self.current_frame)
    
    def _track_human_intervention(self, frame_idx: int):
        """
        追蹤人類介入的幀（用於計算實際 HIR）。
        
        注意：這個函數只負責記錄，不處理 UI 更新。
        UI 更新由調用者負責。
        """
        if self.video_analysis:
            self.video_analysis.frames_actually_edited.add(frame_idx)
            logger.info(f"Human intervention tracked at frame {frame_idx}, "
                       f"total edited: {len(self.video_analysis.frames_actually_edited)}")
        else:
            logger.warning(f"Cannot track intervention at frame {frame_idx}: video_analysis is None")
    
    def _add_new_object(self, mask: np.ndarray) -> Optional[int]:
        """
        新增一個新的物件到結果中。
        
        Returns:
            new_obj_id: 新建立的物件 ID，如果失敗則返回 None
        """
        # 計算新的 obj_id（找到最大的現有 ID + 1）
        max_obj_id = -1
        for frame_result in self.sam3_results.values():
            for det in frame_result.detections:
                max_obj_id = max(max_obj_id, det.obj_id)
        new_obj_id = max_obj_id + 1
        
        # 計算 bounding box
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            QMessageBox.warning(self, "Warning", "Empty mask, cannot add object.")
            return None
        
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        box = np.array([x_min, y_min, x_max - x_min, y_max - y_min])
        
        # 建立新的 Detection
        new_detection = Detection(
            obj_id=new_obj_id,
            mask=mask.astype(np.uint8),
            box=box,
            score=1.0  # 手動新增的給滿分
        )
        
        # 加入當前幀的結果
        if self.current_frame not in self.sam3_results:
            self.sam3_results[self.current_frame] = FrameResult(
                frame_index=self.current_frame,
                detections=[new_detection]
            )
        else:
            self.sam3_results[self.current_frame].detections.append(new_detection)
        
        # 設定新物件狀態為 accepted
        self.object_status[new_obj_id] = "accepted"
        
        logger.info(f"Added new object {new_obj_id} at frame {self.current_frame}")
        self.statusBar().showMessage(f"Added new object {new_obj_id}")
        
        return new_obj_id
        
        # 注意：不在這裡調用 analyze_video 和 UI 更新
        # 由調用者 on_refinement_apply 負責（避免順序問題）
    
    def on_refinement_propagate(self):
        """套用修改並傳播到後續所有幀。"""
        if not self.refinement_active or not self.video_canvas.refinement_state:
            return
        
        state = self.video_canvas.refinement_state
        new_mask = state.current_mask
        points, labels = state.get_sam_inputs()
        
        # 檢查是否有有效的 mask 和 points
        if new_mask is None or not np.any(new_mask):
            QMessageBox.warning(self, "Warning", "No valid mask to propagate. Please add points first.")
            return
        
        if len(points) == 0:
            QMessageBox.warning(self, "Warning", "No points added. Please click to define the object.")
            return
        
        # 確認操作
        total_frames = self.video_loader.metadata.total_frames
        remaining_frames = total_frames - self.current_frame - 1
        
        if remaining_frames <= 0:
            QMessageBox.information(self, "Info", "This is the last frame. Use 'Apply' instead.")
            return
        
        reply = QMessageBox.question(
            self, "Confirm Propagation",
            f"This will track the object from frame {self.current_frame} to frame {total_frames - 1} "
            f"({remaining_frames} frames).\n\n"
            f"This may take a while. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )
        
        if reply != QMessageBox.StandardButton.Yes:
            return
        
        # 執行傳播
        self._propagate_to_following_frames(new_mask, points, labels)
    
    def _propagate_to_following_frames(self, mask: np.ndarray, points: np.ndarray, labels: np.ndarray):
        """使用 SAM3 Video Predictor 傳播到後續幀。"""
        state = self.video_canvas.refinement_state
        start_frame = self.current_frame
        
        # 確定 obj_id
        if self.add_object_mode:
            # 新增物件：分配新 ID
            max_obj_id = -1
            for frame_result in self.sam3_results.values():
                for det in frame_result.detections:
                    max_obj_id = max(max_obj_id, det.obj_id)
            obj_id = max_obj_id + 1
        else:
            obj_id = state.object_id
        
        # 顯示進度對話框
        total_frames = self.video_loader.metadata.total_frames
        remaining = total_frames - start_frame
        
        progress = QProgressDialog(
            f"Propagating object {obj_id} to following frames...",
            "Cancel", 0, remaining, self
        )
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        progress.setMinimumDuration(0)
        progress.setValue(0)
        
        try:
            # 嘗試使用 SAM3 Video Predictor
            if self.sam3_engine is None:
                self.sam3_engine = SAM3Engine(mode="auto")
            
            # 檢查是否支援 video propagation
            if hasattr(self.sam3_engine, 'propagate_mask'):
                # 使用 SAM3 video predictor
                results = self.sam3_engine.propagate_mask(
                    video_path=str(self.video_loader.video_path),
                    start_frame=start_frame,
                    mask=mask,
                    points=points,
                    labels=labels,
                    obj_id=obj_id,
                    progress_callback=lambda i, n: progress.setValue(i) if not progress.wasCanceled() else None
                )
                
                if progress.wasCanceled():
                    self.statusBar().showMessage("Propagation cancelled")
                    return
                
                # 更新 sam3_results
                for frame_idx, frame_mask in results.items():
                    self._update_or_add_detection(frame_idx, obj_id, frame_mask)
            else:
                # Fallback: 簡易傳播（直接複製 mask）
                logger.warning("SAM3 video propagation not available, using simple copy")
                self._simple_propagate(obj_id, mask, start_frame, progress)
            
            progress.close()
            
            # 更新 object status
            if self.add_object_mode:
                self.object_status[obj_id] = "accepted"
            
            # === ActionLogger: 記錄傳播操作 ===
            end_frame = total_frames - 1
            if self.add_object_mode:
                # 新增物件 + 傳播
                self.action_logger.log_add_object(frame_idx=start_frame, obj_id=obj_id)
            self.action_logger.log_propagate(
                start_frame=start_frame,
                end_frame=end_frame,
                obj_id=obj_id
            )
            
            # ====== 關鍵修復：正確的順序 ======
            # 1. 重新分析（保留已編輯的幀）
            self._reanalyze_with_preserved_edits()
            
            # 2. 追蹤人類介入（在 reanalyze 之後！）
            self._track_human_intervention(start_frame)
            
            # 3. 更新 UI
            self.update_object_list()
            self.update_analysis_display()
            self.update_timeline()
            
            # 退出 refinement 模式
            self._exit_refinement_mode()
            self.display_frame(self.current_frame)
            
            self.statusBar().showMessage(
                f"Propagated object {obj_id} from frame {start_frame} to {total_frames - 1}"
            )
            
        except Exception as e:
            progress.close()
            logger.error(f"Propagation error: {e}")
            QMessageBox.warning(
                self, "Propagation Error",
                f"Failed to propagate: {e}\n\nFalling back to simple copy."
            )
            # Fallback
            self._simple_propagate(obj_id, mask, start_frame, None)
            
            # 更新 object status
            if self.add_object_mode:
                self.object_status[obj_id] = "accepted"
            
            # 同樣需要正確順序
            self._reanalyze_with_preserved_edits()
            self._track_human_intervention(start_frame)
            self.update_object_list()
            self._exit_refinement_mode()
            self.display_frame(self.current_frame)
    
    def _simple_propagate(self, obj_id: int, mask: np.ndarray, start_frame: int, progress: Optional[QProgressDialog]):
        """簡易傳播：將 mask 複製到後續所有幀（不追蹤）。"""
        total_frames = self.video_loader.metadata.total_frames
        
        for i, frame_idx in enumerate(range(start_frame, total_frames)):
            if progress and progress.wasCanceled():
                break
            if progress:
                progress.setValue(i)
            
            self._update_or_add_detection(frame_idx, obj_id, mask)
        
        # 注意：不在這裡調用 analyze_video 和 track_human_intervention
        # 由調用者 _propagate_to_following_frames 負責（避免重複調用和順序問題）
    
    def _update_or_add_detection(self, frame_idx: int, obj_id: int, mask: np.ndarray):
        """更新或新增特定幀的 detection。"""
        # 計算 bounding box
        ys, xs = np.where(mask)
        if len(xs) == 0 or len(ys) == 0:
            return  # Empty mask, skip
        
        x_min, x_max = xs.min(), xs.max()
        y_min, y_max = ys.min(), ys.max()
        box = np.array([x_min, y_min, x_max - x_min, y_max - y_min])
        
        new_detection = Detection(
            obj_id=obj_id,
            mask=mask.astype(np.uint8),
            box=box,
            score=1.0
        )
        
        if frame_idx not in self.sam3_results:
            self.sam3_results[frame_idx] = FrameResult(
                frame_index=frame_idx,
                detections=[new_detection]
            )
        else:
            # 檢查是否已存在該 obj_id
            found = False
            for i, det in enumerate(self.sam3_results[frame_idx].detections):
                if det.obj_id == obj_id:
                    self.sam3_results[frame_idx].detections[i] = new_detection
                    found = True
                    break
            if not found:
                self.sam3_results[frame_idx].detections.append(new_detection)
    
    def on_refinement_cancel(self):
        """取消 refinement。"""
        self._exit_refinement_mode()
        self.display_frame(self.current_frame)
        self.statusBar().showMessage("Refinement cancelled")
    
    def _exit_refinement_mode(self):
        """退出 refinement 或 add object 模式。"""
        self.refinement_active = False
        self.refinement_obj_id = None
        self.add_object_mode = False
        
        self.video_canvas.exit_refinement_mode()
        self.refinement_panel.exit_refinement()
        
        # 重新啟用控制
        self._set_controls_enabled(True)
    
    def _set_controls_enabled(self, enabled: bool):
        """啟用/禁用控制按鈕。"""
        self.prev_btn.setEnabled(enabled and self.video_loader is not None)
        self.next_btn.setEnabled(enabled and self.video_loader is not None)
        self.play_btn.setEnabled(enabled and self.video_loader is not None)
        self.timeline_slider.setEnabled(enabled and self.video_loader is not None)
        self.detect_btn.setEnabled(enabled and self.video_loader is not None)
        self.accept_all_high_btn.setEnabled(enabled and len(self.sam3_results) > 0)
        self.reset_all_btn.setEnabled(enabled and len(self.sam3_results) > 0)
        self.refine_btn.setEnabled(enabled and len(self.object_list.selectedItems()) > 0)
        # Add Object 按鈕在有影片時就可以用
        self.add_object_btn.setEnabled(enabled and self.video_loader is not None)


# =============================================================================
# 主程式
# =============================================================================

def main():
    # 設定 logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 建立應用程式
    app = QApplication(sys.argv)
    
    # 設定應用程式樣式
    app.setStyle("Fusion")
    
    # 建立主視窗
    window = HILAAMainWindow()
    window.show()
    
    # 執行
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
