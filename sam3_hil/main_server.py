#!/usr/bin/env python3
"""
STAMP - SAM Tracking Annotation with Minimal Processing
========================================================

Main program entry point.

Startup flow:
1. Show startup dialog (choose local/remote mode)
2. Initialize engine based on mode (SAM3Engine or StampAPIClient)
3. Launch main window

Usage:
    # Normal startup (shows dialog)
    python main.py
    
    # Force local mode
    python main.py --mode local
    
    # Force server mode
    python main.py --mode server --server-url http://192.168.1.100:8000
    
    # Reset settings (dialog will show next time)
    python main.py --reset-settings
"""

import sys
import argparse
from pathlib import Path

# 確保可以 import 專案模組
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from PyQt6.QtWidgets import QApplication, QMessageBox
from PyQt6.QtCore import Qt
from loguru import logger

# 設定 logger
logger.remove()
logger.add(
    sys.stderr,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    level="INFO"
)


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="STAMP - SAM Tracking Annotation with Minimal Processing"
    )
    
    parser.add_argument(
        "--mode",
        choices=["local", "server", "auto"],
        default="auto",
        help="Execution mode: local (local GPU), server (remote), auto (show dialog)"
    )
    
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="Remote server URL (only used in server mode)"
    )
    
    parser.add_argument(
        "--reset-settings",
        action="store_true",
        help="Reset saved settings"
    )
    
    parser.add_argument(
        "--skip-dialog",
        action="store_true",
        help="Skip dialog if remembered settings are available"
    )
    
    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()
    
    # 建立 QApplication
    app = QApplication(sys.argv)
    app.setApplicationName("STAMP")
    app.setOrganizationName("NYCU-ARG")
    
    # 重設設定
    if args.reset_settings:
        from src.gui.startup_dialog import StartupDialog
        StartupDialog.clear_saved_settings()
        logger.info("Settings cleared. Restart to see the dialog.")
        return 0
    
    # 決定模式
    if args.mode == "auto":
        # 顯示對話框
        from src.gui.startup_dialog import StartupDialog, ExecutionMode
        
        dialog = StartupDialog(skip_if_remembered=args.skip_dialog)
        
        if dialog.exec() != 1:  # QDialog.DialogCode.Accepted = 1
            logger.info("User cancelled startup dialog")
            return 0
        
        config = dialog.get_config()
        mode = config.mode
        server_url = config.server_url
        
    elif args.mode == "local":
        from src.gui.startup_dialog import ExecutionMode
        mode = ExecutionMode.LOCAL
        server_url = None
        
    else:  # server
        from src.gui.startup_dialog import ExecutionMode
        mode = ExecutionMode.SERVER
        server_url = args.server_url or "http://localhost:8000"
    
    logger.info(f"Starting STAMP in {mode.value} mode")
    
    # 初始化 Engine
    engine = None
    
    if mode.value == "local":
        # 本地模式：載入 SAM3Engine
        try:
            logger.info("Loading SAM3 Engine (this may take a while)...")
            from src.core.sam3_engine import SAM3Engine
            engine = SAM3Engine(mode="auto")
            logger.info("SAM3 Engine loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load SAM3 Engine: {e}")
            QMessageBox.critical(
                None,
                "Error",
                f"Failed to load SAM3 Engine:\n{str(e)}\n\nPlease check:\n1. CUDA is installed\n2. GPU memory is sufficient\n3. Model weights are downloaded"
            )
            return 1
    else:
        # 遠端模式：建立 API Client
        try:
            logger.info(f"Connecting to server: {server_url}")
            from src.api_client import StampAPIClient
            engine = StampAPIClient(server_url)
            
            # 測試連線
            if not engine.check_connection():
                raise ConnectionError("Cannot connect to server")
            
            logger.info("Connected to server successfully")
        except Exception as e:
            logger.error(f"Failed to connect to server: {e}")
            QMessageBox.critical(
                None,
                "Connection Error",
                f"Failed to connect to server:\n{server_url}\n\nError: {str(e)}\n\nPlease make sure the server is running."
            )
            return 1
    
    # 啟動主視窗
    try:
        from src.gui.main_window_server import HILAAMainWindow
        
        window = HILAAMainWindow(
            engine=engine,
            mode=mode.value,
            server_url=server_url if mode.value == "server" else None,
        )
        window.show()
        
        logger.info("STAMP is ready!")
        
        return app.exec()
        
    except Exception as e:
        logger.error(f"Failed to start main window: {e}")
        import traceback
        traceback.print_exc()
        QMessageBox.critical(
            None,
            "Startup Error",
            f"Failed to start main window:\n{str(e)}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
