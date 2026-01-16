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
    5. 匯出標註結果
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
        
        # 影片顯示
        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(800, 450)
        self.video_label.setStyleSheet("background-color: #1a1a1a;")
        self.video_label.setText("Please open a video file\n\nFile → Open Video (Ctrl+O)")
        left_layout.addWidget(self.video_label, stretch=1)
        
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
        
        # 物件列表
        self.object_list = QListWidget()
        self.object_list.setMinimumHeight(300)
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
        
        # 更新物件列表
        self.update_object_list()
        
        # 更新分析顯示
        self.update_analysis_display()
        
        # 重新顯示當前幀
        self.display_frame(self.current_frame)
        
        self.statusBar().showMessage(
            f"detection completed: {self.video_analysis.unique_objects} objects"
        )
    
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
        text = (
            f"Unique Objects: {va.unique_objects}\n"
            f"Total Detections: {va.total_objects}\n\n"
            f"HIGH: {va.high_count} ({va.auto_accept_rate:.1f}%)\n"
            f"UNCERTAIN: {va.uncertain_count}\n"
            f"LOW: {va.low_count}\n\n"
            f"HIR: {va.human_intervention_rate:.1f}%\n"
            f"Frames Need Review: {va.frames_need_review}"
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
