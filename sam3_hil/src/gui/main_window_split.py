#!/usr/bin/env python3
"""
STAMP Annotation GUI - Main Window
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

Author: Adam (ARG LAB, NYCU)
"""

import sys
import logging
import time
import tempfile
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
    QScrollArea,
    QMenu,
    QInputDialog,
)
from PyQt6.QtCore import Qt, QTimer, QPoint
from PyQt6.QtGui import QImage, QPixmap, QAction, QKeySequence, QFont

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
    
    # Refactored modules (gui/components/)
    from gui.components import (
        SAM3Worker, ImageBatchWorker, clear_gpu_memory,
        ExportDialog, ObjectSelectionDialog,
        ObjectListItem,
        PlaybackMixin, ObjectManagerMixin, DetectionMixin, RefinementMixin,
    )
except ImportError as e:
    print(f"Import error: {e}")
    print("Make sure you're running from the correct directory")
    sys.exit(1)

logger = logging.getLogger(__name__)

# =============================================================================
# Main Window
# =============================================================================

class HILAAMainWindow(QMainWindow, PlaybackMixin, ObjectManagerMixin, DetectionMixin, RefinementMixin):
    """
    STAMP 標註工具主視窗。
    
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
            format="mcap",
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
        
        self.setWindowTitle("STAMP")
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
        self.prompt_input = QLineEdit("")
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
        self.accept_all_btn = QPushButton("✓ Accept all")
        self.accept_all_btn.clicked.connect(self.accept_all)
        self.accept_all_btn.setEnabled(False)
        batch_layout.addWidget(self.accept_all_btn)
        
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
        manage_hint = QLabel("Right-click on object for more options")
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
        
        # 判斷是否為 Independent 模式
        is_independent_mode = (self.processing_mode_combo.currentIndex() == 1)
        
        for det in result.detections:
            # 檢查物件狀態 - 根據模式使用不同的 key
            if is_independent_mode:
                status_key = f"F{result.frame_index}_{det.obj_id}"
            else:
                status_key = det.obj_id
            obj_status = self.object_status.get(status_key, "pending")
            
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
                format="mcap",
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
                format="mcap",
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
    
    # =========================================================================
    # SAM3 偵測
    # =========================================================================
    
    # =========================================================================
    # Batch Processing Callbacks (for Independent Image Mode)
    # =========================================================================
    
    # =========================================================================
    # 物件列表管理
    # =========================================================================
    
    def update_object_list(self):
        """更新物件列表。"""
        self.object_list.clear()
        
        # 判斷是否為 Independent Images 模式
        is_independent_mode = (self.processing_mode_combo.currentIndex() == 1)
        
        if is_independent_mode:
            # === Independent 模式：顯示所有幀的所有物件 ===
            self._update_object_list_independent()
        else:
            # === Video 模式：顯示跨幀聚合的物件 ===
            self._update_object_list_video()
        
        self.accept_all_btn.setEnabled(self.object_list.count() > 0)
        self.reset_all_btn.setEnabled(self.object_list.count() > 0)
    
    def _update_object_list_video(self):
        """Video 模式：顯示跨幀聚合的物件列表。"""
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
                category,
                frame_idx=None  # Video 模式不傳 frame_idx
            )
            item_widget.status_changed.connect(self.on_object_status_changed)
            
            # 設置 property
            item_widget.setProperty("obj_id", obj_summary.obj_id)
            
            item.setSizeHint(item_widget.sizeHint())
            self.object_list.addItem(item)
            self.object_list.setItemWidget(item, item_widget)
            
            # 初始化狀態（如果尚未設定）
            if obj_summary.obj_id not in self.object_status:
                self.object_status[obj_summary.obj_id] = "pending"
    
    def _update_object_list_independent(self):
        """Independent 模式：顯示所有幀的所有物件。"""
        if not self.sam3_results:
            return
        
        # 收集所有幀的所有物件
        all_detections = []  # [(frame_idx, obj_id, score), ...]
        
        for frame_idx, frame_result in self.sam3_results.items():
            for det in frame_result.detections:
                all_detections.append((frame_idx, det.obj_id, det.score))
        
        # 按幀索引排序，然後按分數排序
        all_detections.sort(key=lambda x: (x[0], -x[2]))
        
        for frame_idx, obj_id, score in all_detections:
            category = self.analyzer.categorize(score)
            composite_key = f"F{frame_idx}_{obj_id}"
            
            # 建立列表項目
            item = QListWidgetItem()
            item_widget = ObjectListItem(
                obj_id,
                score,
                category,
                frame_idx=frame_idx  # 傳遞幀索引
            )
            item_widget.status_changed.connect(self.on_object_status_changed)
            
            # 設置 property
            item_widget.setProperty("obj_id", obj_id)
            item_widget.setProperty("frame_idx", frame_idx)
            item_widget.setProperty("composite_key", composite_key)
            
            item.setSizeHint(item_widget.sizeHint())
            self.object_list.addItem(item)
            self.object_list.setItemWidget(item, item_widget)
            
            # 初始化狀態（如果尚未設定）
            if composite_key not in self.object_status:
                self.object_status[composite_key] = "pending"
    
    def update_analysis_display(self):
        """更新分析結果顯示。"""
        if self.video_analysis is None:
            return
        
        va = self.video_analysis
        
        jitter_info = ""
        if hasattr(self, 'jitter_analysis') and self.jitter_analysis:
            ja = self.jitter_analysis
            jitter_info = f"\nStability: {ja.overall_stability:.1%}\nJitter Frames: {ja.jitter_frame_count}"
        
        text = (
            f"Unique Objects: {va.unique_objects}\n"
            f"Total Detections: {va.total_objects}\n\n"
            f"HIGH: {va.high_count} ({va.auto_accept_rate:.1f}%)\n"
            f"UNCERTAIN: {va.uncertain_count}\n"
            f"LOW: {va.low_count}"
            f"{jitter_info}"
        )
        self.analysis_label.setText(text)
    
    def on_object_status_changed(self, obj_id: int, status: str, frame_idx: object = None):
        """物件狀態改變。"""
        # 判斷是否為 Independent 模式
        is_independent_mode = (self.processing_mode_combo.currentIndex() == 1)
        
        if is_independent_mode and frame_idx is not None:
            # Independent 模式：使用複合 key
            key = f"F{frame_idx}_{obj_id}"
        else:
            # Video 模式：使用 obj_id
            key = obj_id
        
        self.object_status[key] = status
        
        # === ActionLogger: 記錄審核操作 ===
        log_frame = frame_idx if frame_idx is not None else self.current_frame
        if status == "accepted":
            self.action_logger.log_approve_object(
                frame_idx=log_frame,
                obj_id=obj_id
            )
        elif status == "rejected":
            self.action_logger.log_reject_object(
                frame_idx=log_frame,
                obj_id=obj_id
            )
        
        self.display_frame(self.current_frame)  # 重新顯示以更新視覺化
    
    def accept_all(self):
        """接受所有尚未審核的物件（pending 狀態）。"""
        for i in range(self.object_list.count()):
            item = self.object_list.item(i)
            widget = self.object_list.itemWidget(item)
            if widget and widget.status == "pending":  # 只接受 pending 的
                widget.accept()
    
    def reset_all_objects(self):
        """重設所有物件狀態。"""
        is_independent_mode = (self.processing_mode_combo.currentIndex() == 1)
        
        for i in range(self.object_list.count()):
            item = self.object_list.item(i)
            widget = self.object_list.itemWidget(item)
            if widget:
                widget.reset()
                
                # 根據模式使用正確的 key
                if is_independent_mode and widget.frame_idx is not None:
                    key = f"F{widget.frame_idx}_{widget.obj_id}"
                else:
                    key = widget.obj_id
                self.object_status[key] = "pending"
        
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
    
    def _on_processing_mode_changed(self, index):
        """Processing Mode 改變時的處理。"""
        if index == 0:
            # Video mode
            logger.info("Processing mode: Video (連續影片，使用追蹤)")
            self.open_btn.setText("Open Video")
            self.statusBar().showMessage("Mode: Video")
        elif index == 1:
            # Sequential Images mode
            logger.info("Processing mode: Sequential Images (連續照片，使用追蹤)")
            self.open_btn.setText("Open Folder")
            self.statusBar().showMessage("Mode: Sequential Images - 點擊 Open Folder 選擇圖片資料夾")
        else:
            # Independent Images mode
            logger.info("Processing mode: Independent Images (獨立照片，分別處理)")
            self.open_btn.setText("Open Folder")
            self.statusBar().showMessage("Mode: Independent Images - 點擊 Open Folder 選擇圖片資料夾")

    def smart_open(self):
        """根據當前模式智能開啟檔案或資料夾。"""
        mode_index = self.processing_mode_combo.currentIndex()
        
        if mode_index == 0:
            # Video mode - 開啟影片
            self.open_video()
        else:
            # Sequential or Independent Images - 開啟資料夾
            self.open_image_folder()


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
            "About STAMP",
            "STAMP Maritime Annotation Tool\n\n"
            "Human-in-the-Loop Active Annotation\n"
            "for Maritime Video using SAM3\n\n"
            "Author: Adam Shih\n"
            "Assistive Robotics Group, NYCU"
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
