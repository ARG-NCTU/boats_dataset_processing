"""
Timeline Widget for HIL-AA

Visualizes object tracking confidence across video frames.

Features:
- Color-coded confidence bars per object
- Current frame indicator
- Click to seek
- Hover for details
- Zoom and pan support
"""

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt, QRect, QPoint, pyqtSignal, QSize
from PyQt6.QtGui import (
    QPainter, QColor, QPen, QBrush, QFont, QFontMetrics,
    QMouseEvent, QWheelEvent, QPaintEvent, QResizeEvent
)
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, 
    QScrollArea, QToolTip, QSizePolicy
)

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ObjectTrack:
    """Tracking data for a single object."""
    obj_id: int
    label: str
    frames: Dict[int, float]  # frame_index -> confidence score
    status: str = "pending"   # pending, accepted, rejected
    color: Optional[QColor] = None


# =============================================================================
# Color Utilities
# =============================================================================

def confidence_to_color(confidence: float, alpha: int = 200) -> QColor:
    """Convert confidence score to color.
    
    GREEN (high) -> YELLOW (medium) -> RED (low)
    """
    if confidence >= 0.8:
        # High: Green
        return QColor(76, 175, 80, alpha)
    elif confidence >= 0.5:
        # Medium: Yellow to Orange gradient
        t = (confidence - 0.5) / 0.3  # 0 to 1
        r = int(255 - t * 30)  # 255 -> 225
        g = int(152 + t * 48)  # 152 -> 200
        return QColor(r, g, 0, alpha)
    else:
        # Low: Red
        return QColor(244, 67, 54, alpha)


def get_object_color(obj_id: int, alpha: int = 255) -> QColor:
    """Get consistent color for object ID."""
    colors = [
        (66, 133, 244),   # Blue
        (234, 67, 53),    # Red
        (251, 188, 5),    # Yellow
        (52, 168, 83),    # Green
        (255, 109, 0),    # Orange
        (156, 39, 176),   # Purple
        (0, 188, 212),    # Cyan
        (255, 87, 34),    # Deep Orange
        (63, 81, 181),    # Indigo
        (139, 195, 74),   # Light Green
    ]
    r, g, b = colors[obj_id % len(colors)]
    return QColor(r, g, b, alpha)


# =============================================================================
# Timeline Canvas
# =============================================================================

class TimelineCanvas(QWidget):
    """
    Canvas that draws the timeline visualization.
    
    Signals:
        frame_selected(int): Emitted when user clicks on a frame
        object_selected(int): Emitted when user clicks on an object track
    """
    
    frame_selected = pyqtSignal(int)
    object_selected = pyqtSignal(int)
    
    # Layout constants
    HEADER_HEIGHT = 30
    TRACK_HEIGHT = 24
    TRACK_PADDING = 4
    LABEL_WIDTH = 100
    RULER_HEIGHT = 20
    MIN_FRAME_WIDTH = 2
    MAX_FRAME_WIDTH = 20
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        # Data
        self.tracks: List[ObjectTrack] = []
        self.total_frames: int = 0
        self.current_frame: int = 0
        self.fps: float = 30.0
        
        # View state
        self.zoom_level: float = 1.0  # pixels per frame
        self.scroll_offset: int = 0
        self.hovered_frame: Optional[int] = -1
        self.hovered_track: Optional[int] = -1
        self.selected_track: Optional[int] = None
        
        # Appearance
        self.setMinimumHeight(100)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.setMouseTracking(True)
        
        # Calculate initial zoom
        self._auto_zoom()
    
    def set_data(
        self,
        tracks: List[ObjectTrack],
        total_frames: int,
        fps: float = 30.0
    ):
        """Set timeline data."""
        self.tracks = tracks
        self.total_frames = total_frames
        self.fps = fps
        self._auto_zoom()
        self._update_size()
        self.update()
    
    def set_current_frame(self, frame: int):
        """Update current frame indicator."""
        self.current_frame = frame
        self.update()
    
    def _auto_zoom(self):
        """Calculate zoom level to fit timeline in view."""
        if self.total_frames <= 0:
            self.zoom_level = self.MIN_FRAME_WIDTH
            return
        
        available_width = self.width() - self.LABEL_WIDTH - 20
        if available_width <= 0:
            available_width = 800 - self.LABEL_WIDTH
        
        self.zoom_level = max(
            self.MIN_FRAME_WIDTH,
            min(self.MAX_FRAME_WIDTH, available_width / self.total_frames)
        )
    
    def _update_size(self):
        """Update widget size based on content."""
        height = (
            self.HEADER_HEIGHT + 
            self.RULER_HEIGHT +
            len(self.tracks) * (self.TRACK_HEIGHT + self.TRACK_PADDING) +
            self.TRACK_PADDING
        )
        height = max(height, 100)
        
        width = int(self.LABEL_WIDTH + self.total_frames * self.zoom_level + 20)
        
        self.setMinimumHeight(height)
        self.setMinimumWidth(width)
    
    def _frame_to_x(self, frame: int) -> int:
        """Convert frame index to x coordinate."""
        return int(self.LABEL_WIDTH + frame * self.zoom_level - self.scroll_offset)
    
    def _x_to_frame(self, x: int) -> int:
        """Convert x coordinate to frame index."""
        frame = int((x - self.LABEL_WIDTH + self.scroll_offset) / self.zoom_level)
        return max(0, min(self.total_frames - 1, frame))
    
    def _track_at_y(self, y: int) -> int:
        """Get track index at y coordinate, or -1 if none."""
        y_offset = y - self.HEADER_HEIGHT - self.RULER_HEIGHT
        if y_offset < 0:
            return -1
        
        track_idx = int(y_offset / (self.TRACK_HEIGHT + self.TRACK_PADDING))
        if 0 <= track_idx < len(self.tracks):
            return track_idx
        return -1
    
    def paintEvent(self, event: QPaintEvent):
        """Draw the timeline."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        
        # Background
        painter.fillRect(self.rect(), QColor(250, 250, 250))
        
        if self.total_frames <= 0:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No video loaded")
            return
        
        # Draw components
        self._draw_ruler(painter)
        self._draw_tracks(painter)
        self._draw_current_frame(painter)
        self._draw_hover_info(painter)
    
    def _draw_ruler(self, painter: QPainter):
        """Draw time ruler at top."""
        ruler_rect = QRect(
            self.LABEL_WIDTH, self.HEADER_HEIGHT,
            self.width() - self.LABEL_WIDTH, self.RULER_HEIGHT
        )
        
        # Background
        painter.fillRect(ruler_rect, QColor(240, 240, 240))
        
        # Calculate tick interval
        if self.zoom_level >= 10:
            tick_interval = 10
        elif self.zoom_level >= 5:
            tick_interval = 30
        elif self.zoom_level >= 2:
            tick_interval = 60
        else:
            tick_interval = 150
        
        # Draw ticks and labels
        painter.setPen(QPen(QColor(100, 100, 100), 1))
        font = QFont("Arial", 8)
        painter.setFont(font)
        
        for frame in range(0, self.total_frames, tick_interval):
            x = self._frame_to_x(frame)
            if x < self.LABEL_WIDTH or x > self.width():
                continue
            
            # Tick mark
            painter.drawLine(x, self.HEADER_HEIGHT + self.RULER_HEIGHT - 5,
                           x, self.HEADER_HEIGHT + self.RULER_HEIGHT)
            
            # Time label
            time_sec = frame / self.fps
            if time_sec >= 60:
                label = f"{int(time_sec // 60)}:{int(time_sec % 60):02d}"
            else:
                label = f"{time_sec:.1f}s"
            
            painter.drawText(x - 20, self.HEADER_HEIGHT + 5,
                           40, self.RULER_HEIGHT - 5,
                           Qt.AlignmentFlag.AlignCenter, label)
    
    def _draw_tracks(self, painter: QPainter):
        """Draw object tracks."""
        y_start = self.HEADER_HEIGHT + self.RULER_HEIGHT + self.TRACK_PADDING
        
        for idx, track in enumerate(self.tracks):
            track_y = y_start + idx * (self.TRACK_HEIGHT + self.TRACK_PADDING)
            
            # Track background
            bg_color = QColor(255, 255, 255) if idx != self.selected_track else QColor(230, 240, 255)
            painter.fillRect(
                0, track_y,
                self.width(), self.TRACK_HEIGHT,
                bg_color
            )
            
            # Object label
            label_rect = QRect(5, track_y, self.LABEL_WIDTH - 10, self.TRACK_HEIGHT)
            
            # Status indicator
            status_color = {
                "accepted": QColor(76, 175, 80),
                "rejected": QColor(244, 67, 54),
                "pending": QColor(158, 158, 158)
            }.get(track.status, QColor(158, 158, 158))
            
            painter.setBrush(QBrush(status_color))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(8, track_y + 8, 8, 8)
            
            # Label text
            painter.setPen(QPen(QColor(33, 33, 33)))
            font = QFont("Arial", 9)
            painter.setFont(font)
            
            label_text = f"Obj {track.obj_id}"
            if track.label:
                label_text = track.label[:10]
            painter.drawText(
                20, track_y, self.LABEL_WIDTH - 25, self.TRACK_HEIGHT,
                Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
                label_text
            )
            
            # Draw confidence bars
            self._draw_track_bars(painter, track, track_y)
    
    def _draw_track_bars(self, painter: QPainter, track: ObjectTrack, y: int):
        """Draw confidence bars for a track."""
        if not track.frames:
            return
        
        bar_height = self.TRACK_HEIGHT - 4
        y_bar = y + 2
        
        # Group consecutive frames for efficiency
        sorted_frames = sorted(track.frames.keys())
        
        for frame_idx in sorted_frames:
            x = self._frame_to_x(frame_idx)
            if x < self.LABEL_WIDTH - self.zoom_level or x > self.width():
                continue
            
            confidence = track.frames[frame_idx]
            color = confidence_to_color(confidence)
            
            # Draw bar
            bar_width = max(1, int(self.zoom_level))
            painter.fillRect(x, y_bar, bar_width, bar_height, color)
    
    def _draw_current_frame(self, painter: QPainter):
        """Draw current frame indicator."""
        x = self._frame_to_x(self.current_frame)
        
        if x < self.LABEL_WIDTH or x > self.width():
            return
        
        # Vertical line
        pen = QPen(QColor(33, 33, 33), 2)
        painter.setPen(pen)
        painter.drawLine(
            x, self.HEADER_HEIGHT,
            x, self.height()
        )
        
        # Frame number label
        painter.setBrush(QBrush(QColor(33, 33, 33)))
        label = f"F{self.current_frame}"
        font = QFont("Arial", 8, QFont.Weight.Bold)
        painter.setFont(font)
        
        fm = QFontMetrics(font)
        label_width = fm.horizontalAdvance(label) + 8
        
        # Draw label background
        label_rect = QRect(x - label_width // 2, 5, label_width, 18)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(label_rect, 3, 3)
        
        # Draw label text
        painter.setPen(QPen(QColor(255, 255, 255)))
        painter.drawText(label_rect, Qt.AlignmentFlag.AlignCenter, label)
    
    def _draw_hover_info(self, painter: QPainter):
        """Draw hover highlight."""
        if self.hovered_frame < 0 or self.hovered_track < 0:
            return
        
        # Highlight hovered frame
        x = self._frame_to_x(self.hovered_frame)
        
        pen = QPen(QColor(100, 100, 100, 100), 1, Qt.PenStyle.DashLine)
        painter.setPen(pen)
        painter.drawLine(x, self.HEADER_HEIGHT, x, self.height())
    
    def mousePressEvent(self, event: QMouseEvent):
        """Handle mouse click."""
        if event.button() == Qt.MouseButton.LeftButton:
            x = int(event.position().x())
            y = int(event.position().y())
            
            if x >= self.LABEL_WIDTH:
                # Click on timeline area - seek to frame
                frame = self._x_to_frame(x)
                self.frame_selected.emit(frame)
            
            # Check for track selection
            track_idx = self._track_at_y(y)
            if track_idx >= 0:
                self.selected_track = track_idx
                self.object_selected.emit(self.tracks[track_idx].obj_id)
                self.update()
    
    def mouseMoveEvent(self, event: QMouseEvent):
        """Handle mouse move for hover effects."""
        x = int(event.position().x())
        y = int(event.position().y())
        
        old_frame = self.hovered_frame
        old_track = self.hovered_track
        
        if x >= self.LABEL_WIDTH:
            self.hovered_frame = self._x_to_frame(x)
        else:
            self.hovered_frame = -1
        
        self.hovered_track = self._track_at_y(y)
        
        # Show tooltip
        if self.hovered_frame >= 0 and self.hovered_track >= 0:
            track = self.tracks[self.hovered_track]
            if self.hovered_frame in track.frames:
                conf = track.frames[self.hovered_frame]
                time_sec = self.hovered_frame / self.fps
                tooltip = f"Frame {self.hovered_frame} ({time_sec:.2f}s)\nObj {track.obj_id}: {conf:.1%}"
                QToolTip.showText(event.globalPosition().toPoint(), tooltip)
        
        if old_frame != self.hovered_frame or old_track != self.hovered_track:
            self.update()
    
    def leaveEvent(self, event):
        """Handle mouse leave."""
        self.hovered_frame = -1
        self.hovered_track = -1
        self.update()
    
    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zoom."""
        delta = event.angleDelta().y()
        
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
            # Zoom
            zoom_factor = 1.2 if delta > 0 else 0.8
            new_zoom = self.zoom_level * zoom_factor
            self.zoom_level = max(self.MIN_FRAME_WIDTH, 
                                 min(self.MAX_FRAME_WIDTH, new_zoom))
            self._update_size()
            self.update()
        else:
            # Scroll (let parent handle)
            event.ignore()
    
    def resizeEvent(self, event: QResizeEvent):
        """Handle resize."""
        super().resizeEvent(event)
        if self.total_frames > 0:
            self._auto_zoom()
            self._update_size()


# =============================================================================
# Timeline Widget (Container)
# =============================================================================

class TimelineWidget(QWidget):
    """
    Complete timeline widget with scroll area and controls.
    
    Signals:
        frame_selected(int): Emitted when user selects a frame
        object_selected(int): Emitted when user selects an object
    """
    
    frame_selected = pyqtSignal(int)
    object_selected = pyqtSignal(int)
    
    def __init__(self, parent=None):
        super().__init__(parent)
        
        self._setup_ui()
    
    def _setup_ui(self):
        """Setup the UI."""
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        
        # Header
        header = QWidget()
        header.setStyleSheet("background-color: #f5f5f5; border-bottom: 1px solid #ddd;")
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(10, 5, 10, 5)
        
        title = QLabel("📊 Timeline")
        title.setStyleSheet("font-weight: bold; font-size: 12px;")
        header_layout.addWidget(title)
        
        self.info_label = QLabel("")
        self.info_label.setStyleSheet("color: #666; font-size: 11px;")
        header_layout.addWidget(self.info_label)
        
        header_layout.addStretch()
        
        # Legend
        legend = QWidget()
        legend_layout = QHBoxLayout(legend)
        legend_layout.setContentsMargins(0, 0, 0, 0)
        legend_layout.setSpacing(10)
        
        for label, color in [("High", "#4CAF50"), ("Medium", "#FFC107"), ("Low", "#F44336")]:
            dot = QLabel(f"● {label}")
            dot.setStyleSheet(f"color: {color}; font-size: 10px;")
            legend_layout.addWidget(dot)
        
        header_layout.addWidget(legend)
        layout.addWidget(header)
        
        # Scroll area with canvas
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.scroll_area.setStyleSheet("QScrollArea { border: none; }")
        
        self.canvas = TimelineCanvas()
        self.canvas.frame_selected.connect(self.frame_selected.emit)
        self.canvas.object_selected.connect(self.object_selected.emit)
        
        self.scroll_area.setWidget(self.canvas)
        layout.addWidget(self.scroll_area)
        
        # Set size
        self.setMinimumHeight(120)
        self.setMaximumHeight(300)
    
    def set_data(
        self,
        sam3_results: Dict[int, 'FrameResult'],
        total_frames: int,
        fps: float = 30.0,
        object_status: Optional[Dict[int, str]] = None
    ):
        """
        Set timeline data from SAM3 results.
        
        Args:
            sam3_results: Dict mapping frame_index -> FrameResult
            total_frames: Total number of frames in video
            fps: Video frame rate
            object_status: Optional dict mapping obj_id -> status
        """
        # Build tracks from results
        track_data: Dict[int, Dict[int, float]] = {}  # obj_id -> {frame -> confidence}
        
        for frame_idx, result in sam3_results.items():
            if result is None:
                continue
            for det in result.detections:
                if det.obj_id not in track_data:
                    track_data[det.obj_id] = {}
                track_data[det.obj_id][frame_idx] = det.score
        
        # Create track objects
        tracks = []
        for obj_id in sorted(track_data.keys()):
            status = "pending"
            if object_status and obj_id in object_status:
                status = object_status[obj_id]
            
            tracks.append(ObjectTrack(
                obj_id=obj_id,
                label=f"Object {obj_id}",
                frames=track_data[obj_id],
                status=status,
                color=get_object_color(obj_id)
            ))
        
        # Update canvas
        self.canvas.set_data(tracks, total_frames, fps)
        
        # Update info label
        num_objects = len(tracks)
        num_annotations = sum(len(t.frames) for t in tracks)
        self.info_label.setText(f"{num_objects} objects, {num_annotations} annotations")
    
    def set_current_frame(self, frame: int):
        """Update current frame indicator."""
        self.canvas.set_current_frame(frame)
        
        # Auto-scroll to keep current frame visible
        x = self.canvas._frame_to_x(frame)
        viewport_width = self.scroll_area.viewport().width()
        
        if x < self.canvas.LABEL_WIDTH + 50 or x > viewport_width - 50:
            # Scroll to center frame
            scroll_x = max(0, x - viewport_width // 2)
            self.scroll_area.horizontalScrollBar().setValue(scroll_x)
    
    def select_object(self, obj_id: int):
        """Select an object track."""
        for idx, track in enumerate(self.canvas.tracks):
            if track.obj_id == obj_id:
                self.canvas.selected_track = idx
                self.canvas.update()
                break
    
    def clear(self):
        """Clear timeline data."""
        self.canvas.set_data([], 0)
        self.info_label.setText("")


# =============================================================================
# Standalone Test
# =============================================================================

if __name__ == "__main__":
    import sys
    from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QSlider
    
    # Mock FrameResult for testing
    from dataclasses import dataclass
    
    @dataclass
    class Detection:
        obj_id: int
        score: float
        mask: np.ndarray = None
        box: np.ndarray = None
    
    @dataclass
    class FrameResult:
        frame_index: int
        detections: List[Detection]
    
    # Create test data
    def create_test_data(num_frames: int = 500):
        results = {}
        
        for frame in range(num_frames):
            detections = []
            
            # Object 0: appears throughout, varying confidence
            conf = 0.6 + 0.3 * np.sin(frame / 50)
            detections.append(Detection(obj_id=0, score=conf))
            
            # Object 1: appears in middle section
            if 100 <= frame <= 400:
                conf = 0.8 + 0.1 * np.sin(frame / 30)
                detections.append(Detection(obj_id=1, score=conf))
            
            # Object 2: intermittent with low confidence
            if frame % 3 == 0 and 200 <= frame <= 350:
                conf = 0.3 + 0.2 * np.random.random()
                detections.append(Detection(obj_id=2, score=conf))
            
            results[frame] = FrameResult(frame_index=frame, detections=detections)
        
        return results
    
    # Run test
    app = QApplication(sys.argv)
    
    window = QMainWindow()
    window.setWindowTitle("Timeline Widget Test")
    window.setGeometry(100, 100, 1200, 400)
    
    central = QWidget()
    layout = QVBoxLayout(central)
    
    # Timeline
    timeline = TimelineWidget()
    layout.addWidget(timeline)
    
    # Frame slider for testing
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setRange(0, 499)
    slider.valueChanged.connect(timeline.set_current_frame)
    layout.addWidget(slider)
    
    # Load test data
    test_data = create_test_data(500)
    timeline.set_data(
        test_data, 
        total_frames=500, 
        fps=30.0,
        object_status={0: "accepted", 1: "pending", 2: "rejected"}
    )
    
    # Connect signals
    timeline.frame_selected.connect(lambda f: slider.setValue(f))
    timeline.frame_selected.connect(lambda f: print(f"Selected frame: {f}"))
    timeline.object_selected.connect(lambda o: print(f"Selected object: {o}"))
    
    window.setCentralWidget(central)
    window.show()
    
    sys.exit(app.exec())
