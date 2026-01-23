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

Author: Sonic (Maritime Robotics Lab, NYCU)
"""

import sys
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

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
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QThread, QSize
from PyQt6.QtGui import QImage, QPixmap, QAction, QKeySequence, QColor, QFont

# 加入專案路徑
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import our modules
try:
    from core.video_loader import VideoLoader
    from core.sam3_engine import SAM3Engine, FrameResult, visualize_frame_results
    from core.confidence_analyzer import (
        ConfidenceAnalyzer, 
        ConfidenceCategory,
        VideoAnalysis
    )
    from core.exporter import AnnotationExporter, ExportConfig, ExportStats
    from gui.interactive_canvas import InteractiveCanvas, RefinementControlPanel, RefinementState
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)

logger = logging.getLogger(__name__)


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
    
    信號 (Signals):
    - progress: 回報進度 (0-100)
    - finished: 完成時發出結果
    - error: 發生錯誤時發出
    """
    progress = pyqtSignal(int, str)  # (百分比, 訊息)
    finished = pyqtSignal(dict)       # 結果字典
    error = pyqtSignal(str)           # 錯誤訊息
    
    def __init__(self, video_path: str, prompt: str, mode: str = "gpu"):
        super().__init__()
        self.video_path = str(video_path)  # 確保是字串
        self.prompt = prompt
        self.mode = mode
    
    def run(self):
        """執行緒主函數。"""
        engine = None
        session_id = None
        
        try:
            self.progress.emit(10, "Loading SAM3 model...")
            logger.info(f"Worker starting: video={self.video_path}, prompt={self.prompt}, mode={self.mode}")
            
            engine = SAM3Engine(mode=self.mode)
            
            self.progress.emit(30, "Starting video session...")
            session_id = engine.start_video_session(self.video_path)
            logger.info(f"Session started: {session_id}")
            
            self.progress.emit(40, f"Detecting objects (prompt: {self.prompt})...")
            engine.add_prompt(session_id, 0, self.prompt)
            logger.info("Prompt added")
            
            self.progress.emit(50, "Propagating masks...")
            results = engine.propagate(session_id)
            logger.info(f"Propagation done: {len(results)} frames")
            
            self.progress.emit(90, "Closing session...")
            engine.close_session(session_id)
            session_id = None
            
            engine.shutdown()
            engine = None
            
            self.progress.emit(100, "Done!")
            self.finished.emit({"results": results})
            
        except Exception as e:
            import traceback
            error_msg = f"{type(e).__name__}: {str(e)}\n\n{traceback.format_exc()}"
            logger.error(f"Worker error: {error_msg}")
            
            # 清理資源
            try:
                if session_id and engine:
                    engine.close_session(session_id)
                if engine:
                    engine.shutdown()
            except:
                pass
            
            self.error.emit(error_msg)


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
            
            remove_btn = QPushButton("×")
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
    6. 匯出標註結果
    """
    
    def __init__(self):
        super().__init__()
        
        # 狀態變數
        self.video_loader: Optional[VideoLoader] = None
        self.sam3_results: Dict[int, FrameResult] = {}
        self.analyzer = ConfidenceAnalyzer(high_threshold=0.80, low_threshold=0.50)
        self.video_analysis: Optional[VideoAnalysis] = None
        
        self.current_frame = 0
        self.is_playing = False
        self.show_masks = True
        self.show_boxes = True
        
        # 物件狀態（審閱結果）
        self.object_status: Dict[int, str] = {}  # obj_id -> status
        
        # Refinement 狀態
        self.refinement_active = False
        self.refinement_obj_id: Optional[int] = None
        self.add_object_mode = False  # True when adding new object instead of refining
        self.sam3_engine: Optional[SAM3Engine] = None  # Reuse for refinement
        
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
        self.open_btn.clicked.connect(self.open_video)
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
        self.refine_btn = QPushButton("🎯 Refine Selected")
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
        self.add_object_btn = QPushButton("➕ Add Object")
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
        
        # 物件列表
        self.object_list = QListWidget()
        self.object_list.setMinimumHeight(300)
        self.object_list.itemSelectionChanged.connect(self.on_object_selection_changed)
        objects_layout.addWidget(self.object_list)
        
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
            # 釋放之前的 loader
            if self.video_loader is not None:
                self.video_loader.release()
            
            # 建立新的 loader
            self.video_loader = VideoLoader(file_path)
            
            # 清除之前的結果
            self.sam3_results = {}
            self.video_analysis = None
            self.object_status = {}
            self.object_list.clear()
            
            # 更新 UI
            total = self.video_loader.metadata.total_frames
            self.timeline_slider.setMaximum(total - 1)
            self.timeline_slider.setValue(0)
            self.timeline_slider.setEnabled(True)
            
            self.prev_btn.setEnabled(True)
            self.play_btn.setEnabled(True)
            self.next_btn.setEnabled(True)
            self.detect_btn.setEnabled(True)
            self.add_object_btn.setEnabled(True)  # 開啟影片後就可以手動新增物件
            
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
            
            self.statusBar().showMessage(f"Opened: {file_path}")
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Cannot open video:\n{e}")
    
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
        
        frame_idx = max(0, min(frame_idx, self.video_loader.metadata.total_frames - 1))
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
        
        prompt = self.prompt_input.text().strip()
        if not prompt:
            QMessageBox.warning(self, "Warning", "Please enter a detection prompt")
            return
        
        mode = self.mode_combo.currentText()
        
        # 停止播放
        self.stop_play()
        
        # 取得影片路徑（確保是字串）
        video_path = str(self.video_loader.video_path)
        logger.info(f"Starting detection: video={video_path}, prompt={prompt}, mode={mode}")
        
        # 建立進度對話框
        self.progress_dialog = QProgressDialog(
            "Running SAM3 detection...", "Cancel", 0, 100, self
        )
        self.progress_dialog.setWindowTitle("Processing")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        
        # 建立並啟動 worker 執行緒
        self.worker = SAM3Worker(video_path, prompt, mode)
        self.worker.progress.connect(self.on_detection_progress)
        self.worker.finished.connect(self.on_detection_finished)
        self.worker.error.connect(self.on_detection_error)
        
        self.detect_btn.setEnabled(False)
        self.worker.start()
    
    def on_detection_progress(self, percent: int, message: str):
        """偵測進度更新。"""
        self.progress_dialog.setValue(percent)
        self.progress_dialog.setLabelText(message)
    
    def on_detection_finished(self, result: dict):
        """偵測完成。"""
        self.progress_dialog.close()
        self.detect_btn.setEnabled(True)
        
        self.sam3_results = result["results"]
        
        # 分析結果
        self.video_analysis = self.analyzer.analyze_video(self.sam3_results)
        
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
                iou_threshold=0.75,
                area_change_threshold=0.25
            )
            self.jitter_analysis = detector.analyze_video(self.sam3_results)
            
            # 記錄結果
            ja = self.jitter_analysis
            logger.info(
                f"Jitter detection: {ja.total_jitter_events} events, "
                f"{ja.jitter_frame_count} frames, "
                f"stability: {ja.overall_stability:.1%}"
            )
            # 在 _run_jitter_detection 中，logger.info 之後加上：
            for obj_id, obj_analysis in ja.object_analyses.items():
                logger.info(f"  Object {obj_id}: jitter_count={obj_analysis.jitter_count}, "
                            f"avg_iou={obj_analysis.avg_iou:.3f}, "
                            f"stability={obj_analysis.stability_score:.1%}")
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
        self.progress_dialog.close()
        self.detect_btn.setEnabled(True)
        
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
            jitter_info = f"\n\n📊 Stability: {ja.overall_stability:.1%}\nJitter Frames: {ja.jitter_frame_count}"
        
        text = (
            f"Unique Objects: {va.unique_objects}\n"
            f"Total Detections: {va.total_objects}\n\n"
            f"🟢 HIGH: {va.high_count} ({va.auto_accept_rate:.1f}%)\n"
            f"🟡 UNCERTAIN: {va.uncertain_count}\n"
            f"🔴 LOW: {va.low_count}\n\n"
            f"📋 Potential Review: {va.frames_need_review} frames\n"
            f"✏️ Actually Edited: {edited_frames} frames\n"
            f"📈 Actual HIR: {actual_hir:.1f}%"
            f"{jitter_info}"
        )
        self.analysis_label.setText(text)
    
    def on_object_status_changed(self, obj_id: int, status: str):
        """物件狀態改變。"""
        self.object_status[obj_id] = status
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
            "Maritime Robotics Lab, NYCU\n\n"
            "Key Innovation:\n"
            "Use SAM3 confidence scores to minimize\n"
            "human annotation effort by 5-10x"
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
        else:
            super().keyPressEvent(event)
    
    def resizeEvent(self, event):
        """視窗大小改變。"""
        super().resizeEvent(event)
        if self.video_loader:
            self.display_frame(self.current_frame)
    
    def closeEvent(self, event):
        """視窗關閉。"""
        self.stop_play()
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
        
        # 取得該物件在當前幀的 mask
        frame_result = self.sam3_results.get(self.current_frame)
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
        
        # 進入 refinement 模式
        self.refinement_active = True
        self.refinement_obj_id = obj_id
        
        # 設置 canvas 為 refinement 模式
        self.video_canvas.enter_refinement_mode(
            obj_id=obj_id,
            frame_idx=self.current_frame,
            mask=target_det.mask
        )
        
        # 顯示控制面板
        score = target_det.score
        self.refinement_panel.enter_refinement(obj_id, score)
        
        # 停止播放
        self.stop_play()
        
        # 禁用其他控制
        self._set_controls_enabled(False)
        
        self.statusBar().showMessage(f"Refinement Mode: Object {obj_id} - Left click to include, Right click to exclude")
        logger.info(f"Started refinement for object {obj_id}")
    
    def start_add_object(self):
        """開始手動新增物件模式。"""
        if self.video_loader is None:
            QMessageBox.warning(self, "Warning", "Please open a video first")
            return
        
        # 取得當前幀圖像大小
        frame = self.video_loader.get_frame(self.current_frame)
        if frame is None:
            QMessageBox.warning(self, "Warning", "Cannot get current frame")
            return
        
        h, w = frame.shape[:2]
        
        # 進入 add object 模式
        self.refinement_active = True
        self.add_object_mode = True
        self.refinement_obj_id = None
        
        # 設置 canvas 為 add object 模式
        self.video_canvas.enter_add_object_mode(
            frame_idx=self.current_frame,
            image_shape=(h, w)
        )
        
        # 顯示控制面板（add object 模式）
        self.refinement_panel.enter_add_object()
        
        # 停止播放
        self.stop_play()
        
        # 禁用其他控制
        self._set_controls_enabled(False)
        
        self.statusBar().showMessage("Add Object Mode: Left click to include, Right click to exclude")
        logger.info("Started add object mode")
    
    def on_refinement_point_added(self, x: int, y: int, is_positive: bool):
        """處理 refinement 點擊。"""
        if not self.refinement_active:
            return
        
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
            self._add_new_object(new_mask)
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
    
    def _add_new_object(self, mask: np.ndarray):
        """新增一個新的物件到結果中。"""
        from core.sam3_engine import Detection, FrameResult
        
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
            return
        
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
        from core.sam3_engine import Detection, FrameResult
        
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
        from core.sam3_engine import Detection, FrameResult
        
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
        from core.sam3_engine import Detection, FrameResult
        
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
