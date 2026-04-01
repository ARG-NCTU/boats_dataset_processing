#!/usr/bin/env python3
"""
STAMP - Object List Item Widget
=================================

Custom widget for displaying a single detected object
in the object list panel.

Shows:
- Color indicator (based on confidence category)
- Object ID and confidence score
- Review status (pending/accepted/rejected)
- Accept/reject buttons
"""

from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel, QPushButton
from PyQt6.QtCore import pyqtSignal
from PyQt6.QtGui import QFont

from core.confidence_analyzer import ConfidenceCategory


class ObjectListItem(QWidget):
    """
    物件列表中的單個項目。
    
    顯示：
    - 顏色標示（根據信心分數）
    - 物件 ID
    - 信心分數
    - 審閱狀態（待審閱/已接受/已拒絕）
    """
    
    status_changed = pyqtSignal(int, str, object)  # (obj_id, status, frame_idx) - frame_idx 可能是 None
    
    def __init__(
        self, 
        obj_id: int, 
        score: float, 
        category: ConfidenceCategory,
        frame_idx: int = None,  # Independent 模式用
        parent=None
    ):
        super().__init__(parent)
        self.obj_id = obj_id
        self.score = score
        self.category = category
        self.frame_idx = frame_idx
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
        
        # 物件資訊 - Independent 模式顯示幀號
        if self.frame_idx is not None:
            info_text = f"[F{self.frame_idx}] Det {self.obj_id}: {self.score:.2f}"
        else:
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
        self.status_changed.emit(self.obj_id, "accepted", self.frame_idx)
    
    def reject(self):
        """拒絕此物件。"""
        self.status = "rejected"
        self.status_label.setText("✗")
        self.status_label.setToolTip("rejected")
        self.accept_btn.setEnabled(True)
        self.reject_btn.setEnabled(False)
        self.status_changed.emit(self.obj_id, "rejected", self.frame_idx)
    
    def reset(self):
        """重設狀態。"""
        self.status = "pending"
        self.status_label.setText("?")
        self.status_label.setToolTip("pending review")
        self.accept_btn.setEnabled(True)
        self.reject_btn.setEnabled(True)
