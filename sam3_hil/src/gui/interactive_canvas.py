"""
Interactive Canvas for SAM3 Point-Based Refinement.

Features:
- Left click: Add positive point (include region)
- Right click: Add negative point (exclude region)
- Real-time mask update preview
- Undo/Clear/Apply/Cancel controls

Author: Adam
Date: 2025-01
"""

import logging
from typing import List, Tuple, Optional, Callable
from dataclasses import dataclass, field

import numpy as np
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, 
    QLabel, QFrame, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QSize
from PyQt6.QtGui import (
    QImage, QPixmap, QPainter, QColor, QPen, QBrush,
    QMouseEvent, QPaintEvent, QResizeEvent, QWheelEvent
)

from gui.canvas_viewport import CanvasViewportTransform

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class RefinementPoint:
    """A point for SAM3 refinement."""
    x: int  # Image coordinates
    y: int  # Image coordinates
    is_positive: bool  # True = include, False = exclude
    
    def to_tuple(self) -> Tuple[int, int]:
        return (self.x, self.y)


@dataclass
class RefinementState:
    """Current state of refinement session."""
    object_id: int
    frame_idx: int
    original_mask: Optional[np.ndarray] = None
    current_mask: Optional[np.ndarray] = None
    points: List[RefinementPoint] = field(default_factory=list)
    
    def clear_points(self):
        self.points = []
        self.current_mask = self.original_mask.copy() if self.original_mask is not None else None
    
    def undo_last_point(self):
        if self.points:
            self.points.pop()
    
    def add_point(self, x: int, y: int, is_positive: bool):
        self.points.append(RefinementPoint(x, y, is_positive))
    
    def get_sam_inputs(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get points and labels in SAM3 format.
        
        Returns:
            points: np.ndarray of shape (N, 2) with [x, y] coordinates
            labels: np.ndarray of shape (N,) with 1 for positive, 0 for negative
        """
        if not self.points:
            return np.array([]).reshape(0, 2), np.array([])
        
        points = np.array([[p.x, p.y] for p in self.points])
        labels = np.array([1 if p.is_positive else 0 for p in self.points])
        return points, labels


# =============================================================================
# Interactive Canvas Widget
# =============================================================================

class InteractiveCanvas(QLabel):
    """
    A canvas that supports mouse interaction for SAM3 refinement.
    
    Signals:
        point_added: Emitted when a point is added (x, y, is_positive)
        refinement_applied: Emitted when refinement is applied
        refinement_cancelled: Emitted when refinement is cancelled
    """
    
    point_added = pyqtSignal(int, int, bool)  # x, y, is_positive
    refinement_applied = pyqtSignal()
    refinement_cancelled = pyqtSignal()
    zoom_changed = pyqtSignal(float)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # State
        self.refinement_mode = False
        self.refinement_state: Optional[RefinementState] = None
        
        # Display
        self.base_image: Optional[QImage] = None
        self.mask_overlay: Optional[np.ndarray] = None
        self.display_scale = 1.0
        self.image_offset = QPoint(0, 0)
        self.zoom_factor = 1.0
        self.min_zoom = 1.0
        self.max_zoom = 8.0
        self.zoom_step = 1.25
        self.pan_offset = QPoint(0, 0)
        self._is_panning = False
        self._last_pan_pos = QPoint(0, 0)
        
        # Settings
        self.positive_color = QColor(0, 255, 0, 180)  # Green
        self.negative_color = QColor(255, 0, 0, 180)  # Red
        self.point_radius = 8
        self.mask_color = QColor(0, 120, 255, 100)  # Blue with transparency
        
        # Setup
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(640, 360)
        self.setStyleSheet("background-color: #1a1a1a;")
        self.setMouseTracking(True)
        
        # Cursor
        self.setCursor(Qt.CursorShape.ArrowCursor)
    
    def set_base_image(self, image: QImage):
        """Set the base image to display."""
        self.base_image = image
        self._update_display()
    
    def set_mask_overlay(self, mask: Optional[np.ndarray]):
        """Set the mask overlay (H, W boolean array)."""
        self.mask_overlay = mask
        self._update_display()

    def set_display_image(self, image: QImage, mask: Optional[np.ndarray] = None):
        """Set the displayed image and optional mask overlay in one repaint."""
        self.base_image = image
        self.mask_overlay = mask
        self._update_display()

    def fit_to_view(self):
        """Reset zoom and pan to show the full image."""
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self._update_display()
        self.zoom_changed.emit(self.zoom_factor)

    def zoom_in(self):
        """Zoom in around the canvas center."""
        self._set_zoom(self.zoom_factor * self.zoom_step)

    def zoom_out(self):
        """Zoom out around the canvas center."""
        self._set_zoom(self.zoom_factor / self.zoom_step)

    def _set_zoom(self, zoom_factor: float, anchor: Optional[QPoint] = None):
        """Set zoom while keeping the anchor point visually stable."""
        if self.base_image is None:
            return

        old_zoom = self.zoom_factor
        new_zoom = max(self.min_zoom, min(self.max_zoom, zoom_factor))
        if abs(new_zoom - old_zoom) < 0.001:
            return

        if anchor is None:
            anchor = QPoint(self.width() // 2, self.height() // 2)

        image_x, image_y = self._screen_to_image_coords(anchor)
        self.zoom_factor = new_zoom
        self._update_display()

        new_screen = self._image_to_screen_coords(image_x, image_y)
        self.pan_offset += anchor - new_screen
        if self.zoom_factor <= self.min_zoom:
            self.pan_offset = QPoint(0, 0)

        self._update_display()
        self.zoom_changed.emit(self.zoom_factor)

    def _build_viewport(self) -> Optional[CanvasViewportTransform]:
        if self.base_image is None:
            return None

        return CanvasViewportTransform(
            widget_width=max(1, self.width()),
            widget_height=max(1, self.height()),
            image_width=max(1, self.base_image.width()),
            image_height=max(1, self.base_image.height()),
            zoom_factor=self.zoom_factor,
            pan_x=self.pan_offset.x(),
            pan_y=self.pan_offset.y(),
        )
    
    def enter_refinement_mode(self, obj_id: int, frame_idx: int, mask: np.ndarray):
        """
        Enter refinement mode for a specific object.
        
        Args:
            obj_id: Object ID being refined
            frame_idx: Current frame index
            mask: Original mask (H, W boolean array)
        """
        self.refinement_mode = True
        self.refinement_state = RefinementState(
            object_id=obj_id,
            frame_idx=frame_idx,
            original_mask=mask.copy(),
            current_mask=mask.copy()
        )
        self.mask_overlay = mask
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._update_display()
        logger.info(f"Entered refinement mode for object {obj_id}")
    
    def enter_add_object_mode(self, frame_idx: int, image_shape: Tuple[int, int]):
        """
        Enter add object mode to create a new object.
        
        Args:
            frame_idx: Current frame index
            image_shape: (height, width) of the image
        """
        self.refinement_mode = True
        # Use -1 as placeholder obj_id for new object
        empty_mask = np.zeros(image_shape, dtype=bool)
        self.refinement_state = RefinementState(
            object_id=-1,  # Will be assigned later
            frame_idx=frame_idx,
            original_mask=empty_mask,
            current_mask=empty_mask
        )
        self.mask_overlay = None  # No mask to show initially
        self.setCursor(Qt.CursorShape.CrossCursor)
        self._update_display()
        logger.info("Entered add object mode")
    
    def exit_refinement_mode(self):
        """Exit refinement mode."""
        self.refinement_mode = False
        self.refinement_state = None
        self.setCursor(Qt.CursorShape.ArrowCursor)
        self._update_display()
        logger.info("Exited refinement mode")
    
    def update_refined_mask(self, new_mask: np.ndarray):
        """Update the displayed mask after refinement."""
        if self.refinement_state:
            self.refinement_state.current_mask = new_mask
            self.mask_overlay = new_mask
            self._update_display()
    
    def clear_points(self):
        """Clear all refinement points."""
        if self.refinement_state:
            self.refinement_state.clear_points()
            self.mask_overlay = self.refinement_state.original_mask
            self._update_display()
    
    def undo_last_point(self):
        """Undo the last added point."""
        if self.refinement_state and self.refinement_state.points:
            self.refinement_state.undo_last_point()
            # Mask will be updated by the callback from main window
    
    def _screen_to_image_coords(self, screen_pos: QPoint) -> Tuple[int, int]:
        """Convert screen coordinates to image coordinates."""
        if self.base_image is None:
            return (0, 0)

        viewport = self._build_viewport()
        if viewport is None:
            return (0, 0)

        return viewport.screen_to_image(screen_pos.x(), screen_pos.y())
    
    def _image_to_screen_coords(self, img_x: int, img_y: int) -> QPoint:
        """Convert image coordinates to screen coordinates."""
        viewport = self._build_viewport()
        if viewport is None:
            return QPoint(0, 0)

        screen_x, screen_y = viewport.image_to_screen(img_x, img_y)
        return QPoint(screen_x, screen_y)
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse press events."""
        if event.button() == Qt.MouseButton.MiddleButton and self.base_image is not None:
            self._is_panning = True
            self._last_pan_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            return

        if not self.refinement_mode or self.base_image is None:
            super().mousePressEvent(event)
            return
        
        # Get image coordinates
        img_x, img_y = self._screen_to_image_coords(event.pos())
        
        # Determine point type based on button
        if event.button() == Qt.MouseButton.LeftButton:
            is_positive = True
        elif event.button() == Qt.MouseButton.RightButton:
            is_positive = False
        else:
            return
        
        # Add point to state
        if self.refinement_state:
            self.refinement_state.add_point(img_x, img_y, is_positive)
        
        # Emit signal for main window to process
        self.point_added.emit(img_x, img_y, is_positive)
        
        # Update display
        self._update_display()
        
        logger.debug(f"Point added: ({img_x}, {img_y}), positive={is_positive}")

    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle middle-button panning."""
        if self._is_panning:
            delta = event.pos() - self._last_pan_pos
            self.pan_offset += delta
            self._last_pan_pos = event.pos()
            self._update_display()
            return

        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        """Stop panning when the middle mouse button is released."""
        if event.button() == Qt.MouseButton.MiddleButton and self._is_panning:
            self._is_panning = False
            self.setCursor(Qt.CursorShape.CrossCursor if self.refinement_mode else Qt.CursorShape.ArrowCursor)
            return

        super().mouseReleaseEvent(event)

    def wheelEvent(self, event: QWheelEvent):
        """Use Ctrl + mouse wheel for zooming without interfering with normal scroll."""
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            if event.angleDelta().y() > 0:
                self._set_zoom(self.zoom_factor * self.zoom_step, event.position().toPoint())
            else:
                self._set_zoom(self.zoom_factor / self.zoom_step, event.position().toPoint())
            event.accept()
            return

        super().wheelEvent(event)
    
    def _update_display(self):
        """Update the displayed image with overlays."""
        if self.base_image is None:
            self.setText("No image")
            return
        
        widget_size = self.size()
        viewport = self._build_viewport()
        if viewport is None:
            return

        self.display_scale = viewport.display_scale
        scaled_width, scaled_height = viewport.scaled_size
        offset_x, offset_y = viewport.image_offset
        self.image_offset = QPoint(offset_x, offset_y)
        
        # Create display image
        display = QPixmap(widget_size)
        display.fill(QColor("#1a1a1a"))
        
        painter = QPainter(display)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Draw base image
        scaled_image = self.base_image.scaled(
            scaled_width, scaled_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        painter.drawImage(self.image_offset, scaled_image)
        
        # Draw mask overlay
        if self.mask_overlay is not None:
            self._draw_mask_overlay(painter, scaled_width, scaled_height)
        
        # Draw refinement points
        if self.refinement_mode and self.refinement_state:
            self._draw_refinement_points(painter)
        
        painter.end()
        self.setPixmap(display)
    
    def _draw_mask_overlay(self, painter: QPainter, scaled_width: int, scaled_height: int):
        """Draw the mask overlay."""
        if self.mask_overlay is None:
            return
        
        # Create mask image
        h, w = self.mask_overlay.shape
        mask_image = QImage(w, h, QImage.Format.Format_ARGB32)
        mask_image.fill(Qt.GlobalColor.transparent)
        
        # Fill mask areas
        for y in range(h):
            for x in range(w):
                if self.mask_overlay[y, x]:
                    mask_image.setPixelColor(x, y, self.mask_color)
        
        # Scale and draw
        scaled_mask = mask_image.scaled(
            scaled_width, scaled_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        painter.drawImage(self.image_offset, scaled_mask)
    
    def _draw_refinement_points(self, painter: QPainter):
        """Draw the refinement points."""
        if not self.refinement_state:
            return
        
        for point in self.refinement_state.points:
            screen_pos = self._image_to_screen_coords(point.x, point.y)
            
            # Choose color based on point type
            color = self.positive_color if point.is_positive else self.negative_color
            
            # Draw outer circle
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            painter.setBrush(QBrush(color))
            painter.drawEllipse(
                screen_pos, 
                self.point_radius, 
                self.point_radius
            )
            
            # Draw inner symbol (+ for positive, - for negative)
            painter.setPen(QPen(Qt.GlobalColor.white, 2))
            if point.is_positive:
                # Draw +
                painter.drawLine(
                    screen_pos.x() - 4, screen_pos.y(),
                    screen_pos.x() + 4, screen_pos.y()
                )
                painter.drawLine(
                    screen_pos.x(), screen_pos.y() - 4,
                    screen_pos.x(), screen_pos.y() + 4
                )
            else:
                # Draw -
                painter.drawLine(
                    screen_pos.x() - 4, screen_pos.y(),
                    screen_pos.x() + 4, screen_pos.y()
                )
    
    def resizeEvent(self, event: QResizeEvent):
        """Handle resize events."""
        super().resizeEvent(event)
        self._update_display()


# =============================================================================
# Refinement Control Panel
# =============================================================================

class RefinementControlPanel(QFrame):
    """
    Control panel for refinement mode and add object mode.
    
    Provides buttons for:
    - Clear all points
    - Undo last point
    - Apply refinement / Add object
    - Propagate to following frames
    - Cancel
    """
    
    clear_clicked = pyqtSignal()
    undo_clicked = pyqtSignal()
    apply_clicked = pyqtSignal()
    propagate_clicked = pyqtSignal()  # NEW: Propagate to following frames
    cancel_clicked = pyqtSignal()
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self.mode = "refinement"  # "refinement" or "add_object"
        
        self.setFrameStyle(QFrame.Shape.StyledPanel | QFrame.Shadow.Raised)
        self.setStyleSheet("""
            RefinementControlPanel {
                background-color: #2d2d2d;
                border: 2px solid #4CAF50;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton {
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton#applyBtn {
                background-color: #4CAF50;
                color: white;
            }
            QPushButton#applyBtn:hover {
                background-color: #45a049;
            }
            QPushButton#cancelBtn {
                background-color: #f44336;
                color: white;
            }
            QPushButton#cancelBtn:hover {
                background-color: #da190b;
            }
            QPushButton#clearBtn, QPushButton#undoBtn {
                background-color: #555;
                color: white;
            }
            QPushButton#clearBtn:hover, QPushButton#undoBtn:hover {
                background-color: #666;
            }
        """)
        
        self._setup_ui()
        self.hide()  # Hidden by default
    
    def _setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(10, 10, 10, 10)
        self.setLayout(layout)
        
        # Title
        self.title_label = QLabel("Refinement Mode")
        self.title_label.setStyleSheet("color: #4CAF50; font-size: 14px; font-weight: bold;")
        self.title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.title_label)
        
        # Object info
        self.object_label = QLabel("Object: -")
        self.object_label.setStyleSheet("color: white;")
        self.object_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.object_label)
        
        # Instructions
        self.instructions = QLabel(
            "Left click: Include region (+)\n"
            "Right click: Exclude region (-)"
        )
        self.instructions.setStyleSheet("color: #aaa; font-size: 11px;")
        self.instructions.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.instructions)
        
        # Point counter
        self.point_counter = QLabel("Points: 0")
        self.point_counter.setStyleSheet("color: white;")
        self.point_counter.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.point_counter)
        
        layout.addSpacing(10)
        
        # Buttons row 1: Clear / Undo
        row1 = QHBoxLayout()
        
        self.clear_btn = QPushButton("Clear")
        self.clear_btn.setObjectName("clearBtn")
        self.clear_btn.clicked.connect(self.clear_clicked.emit)
        row1.addWidget(self.clear_btn)
        
        self.undo_btn = QPushButton("Undo")
        self.undo_btn.setObjectName("undoBtn")
        self.undo_btn.clicked.connect(self.undo_clicked.emit)
        row1.addWidget(self.undo_btn)
        
        layout.addLayout(row1)
        
        # Buttons row 2: Apply (current frame only)
        row2 = QHBoxLayout()
        
        self.apply_btn = QPushButton("✓ Apply (This Frame)")
        self.apply_btn.setObjectName("applyBtn")
        self.apply_btn.setToolTip("Apply changes to current frame only")
        self.apply_btn.clicked.connect(self.apply_clicked.emit)
        row2.addWidget(self.apply_btn)
        
        layout.addLayout(row2)
        
        # Buttons row 3: Propagate (to following frames)
        row3 = QHBoxLayout()
        
        self.propagate_btn = QPushButton("Propagate →")
        self.propagate_btn.setObjectName("propagateBtn")
        self.propagate_btn.setToolTip("Apply and track to all following frames using SAM3 Video")
        self.propagate_btn.clicked.connect(self.propagate_clicked.emit)
        self.propagate_btn.setStyleSheet("""
            QPushButton#propagateBtn {
                background-color: #9C27B0;
                color: white;
            }
            QPushButton#propagateBtn:hover {
                background-color: #7B1FA2;
            }
        """)
        row3.addWidget(self.propagate_btn)
        
        layout.addLayout(row3)
        
        # Buttons row 4: Cancel
        row4 = QHBoxLayout()
        
        self.cancel_btn = QPushButton("✗ Cancel")
        self.cancel_btn.setObjectName("cancelBtn")
        self.cancel_btn.clicked.connect(self.cancel_clicked.emit)
        row4.addWidget(self.cancel_btn)
        
        layout.addLayout(row4)
    
    def set_object_info(self, obj_id: int, score: float):
        """Update the object info display."""
        self.object_label.setText(f"Object: {obj_id} (score: {score:.2f})")
    
    def set_point_count(self, count: int):
        """Update the point counter."""
        self.point_counter.setText(f"Points: {count}")
    
    def enter_refinement(self, obj_id: int, score: float = 0.0):
        """Show the panel for refinement mode."""
        self.mode = "refinement"
        self.title_label.setText("Refinement Mode")
        self.title_label.setStyleSheet("color: #4CAF50; font-size: 14px; font-weight: bold;")
        self.setStyleSheet(self.styleSheet().replace("#2196F3", "#4CAF50"))
        self.set_object_info(obj_id, score)
        self.set_point_count(0)
        self.apply_btn.setText("✓ Apply")
        self.show()
    
    def enter_add_object(self):
        """Show the panel for add object mode."""
        self.mode = "add_object"
        self.title_label.setText("+ Add New Object")
        self.title_label.setStyleSheet("color: #2196F3; font-size: 14px; font-weight: bold;")
        self.object_label.setText("Click to define new object")
        self.set_point_count(0)
        self.apply_btn.setText("✓ Add Object")
        self.setStyleSheet("""
            RefinementControlPanel {
                background-color: #2d2d2d;
                border: 2px solid #2196F3;
                border-radius: 8px;
                padding: 10px;
            }
            QPushButton {
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton#applyBtn {
                background-color: #2196F3;
                color: white;
            }
            QPushButton#applyBtn:hover {
                background-color: #1976D2;
            }
            QPushButton#cancelBtn {
                background-color: #f44336;
                color: white;
            }
            QPushButton#cancelBtn:hover {
                background-color: #da190b;
            }
            QPushButton#clearBtn, QPushButton#undoBtn {
                background-color: #555;
                color: white;
            }
            QPushButton#clearBtn:hover, QPushButton#undoBtn:hover {
                background-color: #666;
            }
        """)
        self.show()
    
    def exit_refinement(self):
        """Hide the panel."""
        self.hide()

    def set_propagate_visible(self, visible: bool):
        """Show or hide the propagate button (for Images mode)."""
        self.propagate_btn.setVisible(visible)
