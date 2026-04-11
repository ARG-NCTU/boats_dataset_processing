"""
STAMP Annotation System
=================================

Efficiency-Driven Semi-Automated Data Engine:
Leveraging SAM 3 Presence Confidence for Video Annotation

Author: Adam Shih @ NYCU ARG Lab
"""

__version__ = "0.1.0"
__author__ = "Adam Shih @ NYCU ARG Lab"

from .config import config, ensure_directories

__all__ = ["config", "ensure_directories"]