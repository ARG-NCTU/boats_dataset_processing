"""
HIL-AA Maritime Annotation System
=================================

Efficiency-Driven Semi-Automated Data Engine:
Leveraging SAM 3 Presence Confidence for Maritime Video Annotation

Author: Sonic @ NYCU Maritime Robotics Lab
"""

__version__ = "0.1.0"
__author__ = "Adam"

from .config import config, ensure_directories

__all__ = ["config", "ensure_directories"]