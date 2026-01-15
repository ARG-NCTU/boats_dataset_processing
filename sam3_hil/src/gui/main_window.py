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
from typing import Dict, List, Optional

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
        self.video_path = video_path
        self.prompt = prompt
        self.mode = mode
    
    def run(self):
        """執行緒主函數。"""
        try:
            self.progress.emit(10, "loading SAM3 model...")
            
            engine = SAM3Engine(mode=self.mode)
            
            self.progress.emit(30, "starting video session...")
            session_id = engine.start_video_session(self.video_path)
            
            self.progress.emit(40, f"detecting objects (prompt: {self.prompt})...")
            engine.add_prompt(session_id, 0, self.prompt)
            
            self.progress.emit(50, "propagating masks...")
            results = engine.propagate(session_id)
            
            self.progress.emit(90, "closing session...")
            engine.close_session(session_id)
            engine.shutdown()
            
            self.progress.emit(100, "done!")
            self.finished.emit({"results": results})
            
        except Exception as e:
            self.error.emit(str(e))


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
        self.status_label = QLabel("⏳")
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
        self.status_label.setText("⏳")
        self.status_label.setToolTip("pending review")
        self.accept_btn.setEnabled(True)
        self.reject_btn.setEnabled(True)


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
        
        self.prev_btn = QPushButton("<<")
        self.prev_btn.setFixedWidth(40)
        self.prev_btn.clicked.connect(self.prev_frame)
        self.prev_btn.setEnabled(False)
        control_layout.addWidget(self.prev_btn)
        
        self.play_btn = QPushButton("▶")
        self.play_btn.setFixedWidth(40)
        self.play_btn.clicked.connect(self.toggle_play)
        self.play_btn.setEnabled(False)
        control_layout.addWidget(self.play_btn)
        
        self.next_btn = QPushButton(">>")
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
        self.accept_all_high_btn = QPushButton("✓ accept all HIGH")
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
        self.play_btn.setText("⏸")
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
        
        # 建立進度對話框
        self.progress_dialog = QProgressDialog(
            "Running SAM3 detection...", "Cancel", 0, 100, self
        )
        self.progress_dialog.setWindowTitle("Processing")
        self.progress_dialog.setWindowModality(Qt.WindowModality.WindowModal)
        self.progress_dialog.setMinimumDuration(0)
        
        # 建立並啟動 worker 執行緒
        self.worker = SAM3Worker(
            self.video_loader.video_path,
            prompt,
            mode
        )
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
        QMessageBox.critical(self, "Detection Error", f"SAM3 detection failed:\n{error_msg}")
    
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
        
        # TODO: 實作匯出功能
        QMessageBox.information(
            self, "Export", 
            "Export functionality is not yet implemented\n\n"
            "Planned supported formats:\n"
            "- COCO JSON\n"
            "- YOLO TXT"
        )
    
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
