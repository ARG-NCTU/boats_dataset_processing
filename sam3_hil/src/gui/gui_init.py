"""
GUI modules for HIL-AA system.

PyQt6-based interface components:
- main_window_with_action_logger: Main application window (with ActionLogger integration)
- interactive_canvas: Video display with click interaction
- timeline_widget: Confidence-colored timeline
"""

from .main_window_with_action_logger import MainWindow

__all__ = [
    "MainWindow",
]
