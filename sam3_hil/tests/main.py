#!/usr/bin/env python3
"""
STAMP - SAM Tracking Annotation with Minimal Processing
========================================================

主程式入口點。

啟動流程：
1. 顯示啟動對話框（選擇本地/遠端模式）
2. 根據模式初始化 Engine（SAM3Engine 或 StampAPIClient）
3. 啟動主視窗

用法：
    # 正常啟動（會顯示對話框）
    python main.py
    
    # 強制本地模式
    python main.py --mode local
    
    # 強制遠端模式
    python main.py --mode server --server-url http://192.168.1.100:8000
    
    # 重設設定（下次會顯示對話框）
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
    """解析命令列參數"""
    parser = argparse.ArgumentParser(
        description="STAMP - SAM Tracking Annotation with Minimal Processing"
    )
    
    parser.add_argument(
        "--mode",
        choices=["local", "server", "auto"],
        default="auto",
        help="執行模式：local（本地GPU）、server（遠端）、auto（顯示對話框）"
    )
    
    parser.add_argument(
        "--server-url",
        type=str,
        default=None,
        help="遠端伺服器網址（只在 server 模式有效）"
    )
    
    parser.add_argument(
        "--reset-settings",
        action="store_true",
        help="重設儲存的設定"
    )
    
    parser.add_argument(
        "--skip-dialog",
        action="store_true",
        help="如果有記住的設定，跳過對話框"
    )
    
    return parser.parse_args()


def main():
    """主程式"""
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
                "錯誤",
                f"無法載入 SAM3 Engine:\n{str(e)}\n\n請確認：\n1. 是否有安裝 CUDA\n2. 是否有足夠的 GPU 記憶體\n3. 是否已下載模型權重"
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
                "連線錯誤",
                f"無法連線到伺服器:\n{server_url}\n\n錯誤: {str(e)}\n\n請確認伺服器是否已啟動。"
            )
            return 1
    
    # 啟動主視窗
    try:
        from src.gui.main_window_with_action_logger import HILAAMainWindow
        
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
            "啟動錯誤",
            f"無法啟動主視窗:\n{str(e)}"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
