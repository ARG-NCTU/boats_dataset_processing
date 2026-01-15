"""
GUI modules for HIL-AA system.

PyQt6-based interface components:
- main_window: Main application window
- video_canvas: Video display with click interaction
- timeline_widget: Confidence-colored timeline
- control_panel: Text input and controls

Tutorials:
- tutorial_01_basics.py: PyQt6 fundamentals
- tutorial_02_image_display.py: Image/video display
"""

from .main_window import HILAAMainWindow

__all__ = [
    "HILAAMainWindow",
]
