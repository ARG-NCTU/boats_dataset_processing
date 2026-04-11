"""
Bridge module - re-export from configs/config.py

This allows `from src.config import ...` to work
while keeping the actual config in configs/config.py
"""

import sys
from pathlib import Path

# Add project root to path so we can import from configs/
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from configs.config import (
    config,
    ensure_directories,
    print_config,
    # Re-export commonly used items
    PROJECT_ROOT,
    DATA_DIR,
    MODELS_DIR,
    OUTPUT_DIR,
    HIGH_THRESHOLD,
    LOW_THRESHOLD,
    JITTER_THRESHOLD,
    # Classes (if needed elsewhere)
    AppConfig,
    ConfidenceConfig,
    TemporalConfig,
    SAM3Config,
    MaritimeConfig,
    GUIConfig,
    ExportConfig,
    LoggingConfig,
)

__all__ = [
    "config",
    "ensure_directories",
    "print_config",
    "PROJECT_ROOT",
    "DATA_DIR",
    "MODELS_DIR",
    "OUTPUT_DIR",
    "HIGH_THRESHOLD",
    "LOW_THRESHOLD",
    "JITTER_THRESHOLD",
    "AppConfig",
    "ConfidenceConfig",
    "TemporalConfig",
    "SAM3Config",
    "MaritimeConfig",
    "GUIConfig",
    "ExportConfig",
    "LoggingConfig",
]
