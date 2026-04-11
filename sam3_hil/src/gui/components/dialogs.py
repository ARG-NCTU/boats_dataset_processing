#!/usr/bin/env python3
"""
STAMP - Dialog Windows
=======================

Dialog windows for the STAMP annotation tool.

Classes:
- ObjectSelectionDialog: Select an object for merge/swap operations
- ExportDialog: Configure and execute annotation export
"""

from typing import Dict, List, Optional, Tuple

from PyQt6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QGroupBox,
    QCheckBox,
    QDoubleSpinBox,
    QSpinBox,
    QScrollArea,
    QWidget,
    QFileDialog,
    QMessageBox,
    QLineEdit,
    QFormLayout,
)


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
        
        self.hil_fields_checkbox = QCheckBox("Include STAMP fields in COCO")
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
