"""
Startup Dialog
==============

啟動對話框，讓用戶選擇執行模式（本地 SAM3 或遠端 Server）。

用法：
    from src.gui.startup_dialog import StartupDialog, StartupConfig
    
    dialog = StartupDialog()
    if dialog.exec() == QDialog.DialogCode.Accepted:
        config = dialog.get_config()
        print(f"Mode: {config.mode}")
        print(f"Server URL: {config.server_url}")
    else:
        print("User cancelled")

設定會自動儲存，下次啟動時帶入。
"""

import sys
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from PyQt6.QtCore import Qt, QSettings, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QApplication,
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QRadioButton,
    QLineEdit,
    QLabel,
    QPushButton,
    QCheckBox,
    QMessageBox,
    QButtonGroup,
)
from PyQt6.QtGui import QFont, QIcon, QCloseEvent
from loguru import logger


# =============================================================================
# 資料類別
# =============================================================================

class ExecutionMode(Enum):
    """執行模式"""
    LOCAL = "local"
    SERVER = "server"


@dataclass
class StartupConfig:
    """啟動設定"""
    mode: ExecutionMode
    server_url: str = "http://localhost:8000"
    remember: bool = True
    
    def to_dict(self) -> dict:
        return {
            "mode": self.mode.value,
            "server_url": self.server_url,
            "remember": self.remember,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "StartupConfig":
        return cls(
            mode=ExecutionMode(data.get("mode", "local")),
            server_url=data.get("server_url", "http://localhost:8000"),
            remember=data.get("remember", True),
        )


# =============================================================================
# 連線測試 Worker
# =============================================================================

class ConnectionTestWorker(QThread):
    """背景測試連線"""
    
    finished = pyqtSignal(bool, str)  # (success, message)
    
    def __init__(self, server_url: str):
        super().__init__()
        self.server_url = server_url
    
    def run(self):
        try:
            if self.isInterruptionRequested():
                return

            # 動態導入避免循環依賴
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent.parent))
            
            from src.api_client import StampAPIClient
            
            client = StampAPIClient(self.server_url, timeout=10)

            if self.isInterruptionRequested():
                return
            
            # 檢查連線
            if not client.check_connection():
                self.finished.emit(False, "Cannot connect to server")
                return

            if self.isInterruptionRequested():
                return
            
            # 取得狀態
            status = client.get_server_status()
            
            sam3_status = status.get("sam3_engine", "unknown")
            gpu_info = status.get("gpu", {})
            
            if sam3_status == "loaded":
                if gpu_info:
                    gpu_name = gpu_info.get("device", "Unknown GPU")
                    vram = gpu_info.get("memory_allocated_gb", 0)
                    message = f"Connected ✓\nSAM3 loaded\n{gpu_name}\nVRAM: {vram:.1f} GB"
                else:
                    message = "Connected ✓\nSAM3 loaded"
            else:
                message = "Connected, but SAM3 is not loaded"
            
            self.finished.emit(True, message)
            
        except Exception as e:
            logger.error(f"Connection test failed: {e}")
            self.finished.emit(False, f"Connection failed: {str(e)}")


# =============================================================================
# GPU 檢測 Worker
# =============================================================================

class GPUCheckWorker(QThread):
    """背景檢測本地 GPU"""
    
    finished = pyqtSignal(bool, str)  # (has_gpu, message)
    
    def run(self):
        try:
            if self.isInterruptionRequested():
                return

            import torch
            
            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                self.finished.emit(True, f"{device_name}\nVRAM: {total_memory:.1f} GB")
            else:
                self.finished.emit(False, "No NVIDIA GPU detected")
                
        except ImportError:
            self.finished.emit(False, "PyTorch is not installed")
        except Exception as e:
            self.finished.emit(False, f"Detection failed: {str(e)}")


# =============================================================================
# 啟動對話框
# =============================================================================

class StartupDialog(QDialog):
    """
    STAMP 啟動對話框
    
    讓用戶選擇執行模式：
    - 本地模式：使用本機 GPU 執行 SAM3
    - 遠端模式：連接遠端 Server
    """
    
    # 設定 key
    SETTINGS_ORG = "NYCU-ARG"
    SETTINGS_APP = "STAMP"
    
    def __init__(self, parent=None, skip_if_remembered: bool = True):
        """
        初始化對話框
        
        Args:
            parent: 父視窗
            skip_if_remembered: 如果有記住的設定，是否跳過對話框
        """
        super().__init__(parent)
        
        self._config: Optional[StartupConfig] = None
        self._connection_worker: Optional[ConnectionTestWorker] = None
        self._gpu_worker: Optional[GPUCheckWorker] = None
        
        # 讀取儲存的設定
        self._load_settings()
        
        # 如果有記住的設定且 skip_if_remembered，直接接受
        if skip_if_remembered and self._saved_config and self._saved_config.remember:
            self._config = self._saved_config
            # 不顯示對話框，直接標記為接受
            self._should_skip = True
        else:
            self._should_skip = False
            self._init_ui()
    
    def exec(self) -> int:
        """執行對話框"""
        if self._should_skip:
            return QDialog.DialogCode.Accepted
        return super().exec()

    def _stop_thread(self, thread: Optional[QThread], name: str) -> None:
        """安全停止 QThread。"""
        if not thread:
            return

        if thread.isRunning():
            logger.info(f"Stopping {name} thread...")
            thread.requestInterruption()
            thread.quit()
            if not thread.wait(1000):
                logger.warning(f"{name} thread did not stop in time, terminating...")
                thread.terminate()
                thread.wait(500)

        thread.deleteLater()

    def _cleanup_workers(self) -> None:
        """清理所有背景 worker。"""
        self._stop_thread(self._gpu_worker, "GPUCheckWorker")
        self._gpu_worker = None

        self._stop_thread(self._connection_worker, "ConnectionTestWorker")
        self._connection_worker = None
    
    def _init_ui(self):
        """Initialize UI."""
        self.setWindowTitle("STAMP Startup Settings")
        self.setMinimumWidth(450)
        self.setModal(True)
        
        # 主佈局
        layout = QVBoxLayout(self)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # 標題
        title_label = QLabel("STAMP Startup Settings")
        title_font = QFont()
        title_font.setPointSize(14)
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)
        
        # 模式選擇
        mode_group = QGroupBox("Execution Mode")
        mode_layout = QVBoxLayout(mode_group)
        
        self._mode_button_group = QButtonGroup(self)
        
        # 本地模式
        self._local_radio = QRadioButton("Local mode (requires NVIDIA GPU, recommended 12GB+ VRAM)")
        self._mode_button_group.addButton(self._local_radio, 0)
        mode_layout.addWidget(self._local_radio)
        
        # 本地模式狀態
        self._local_status = QLabel("Detecting...")
        self._local_status.setStyleSheet("color: gray; margin-left: 20px;")
        mode_layout.addWidget(self._local_status)
        
        # 遠端模式
        self._server_radio = QRadioButton("Remote server")
        self._mode_button_group.addButton(self._server_radio, 1)
        mode_layout.addWidget(self._server_radio)
        
        layout.addWidget(mode_group)
        
        # Server 設定
        self._server_group = QGroupBox("Server Settings")
        server_layout = QVBoxLayout(self._server_group)
        
        # URL 輸入
        url_layout = QHBoxLayout()
        url_label = QLabel("URL:")
        self._url_input = QLineEdit()
        self._url_input.setPlaceholderText("http://192.168.1.100:8000")
        url_layout.addWidget(url_label)
        url_layout.addWidget(self._url_input)
        server_layout.addLayout(url_layout)
        
        # 測試連線按鈕
        test_layout = QHBoxLayout()
        self._test_button = QPushButton("Test Connection")
        self._test_button.clicked.connect(self._on_test_connection)
        test_layout.addWidget(self._test_button)
        test_layout.addStretch()
        server_layout.addLayout(test_layout)
        
        # 連線狀態
        self._server_status = QLabel("Not tested yet")
        self._server_status.setStyleSheet("color: gray;")
        self._server_status.setWordWrap(True)
        server_layout.addWidget(self._server_status)
        
        layout.addWidget(self._server_group)
        
        # 記住設定
        self._remember_check = QCheckBox("Remember settings for next startup")
        self._remember_check.setChecked(True)
        layout.addWidget(self._remember_check)
        
        # 按鈕
        button_layout = QHBoxLayout()
        button_layout.addStretch()
        
        self._cancel_button = QPushButton("Cancel")
        self._cancel_button.clicked.connect(self.reject)
        button_layout.addWidget(self._cancel_button)
        
        self._start_button = QPushButton("Start")
        self._start_button.setDefault(True)
        self._start_button.clicked.connect(self._on_start)
        button_layout.addWidget(self._start_button)
        
        layout.addLayout(button_layout)
        
        # 連接信號
        self._local_radio.toggled.connect(self._on_mode_changed)
        self._server_radio.toggled.connect(self._on_mode_changed)
        
        # 套用儲存的設定
        self._apply_saved_settings()
        
        # 開始檢測 GPU
        self._check_gpu()
    
    def _load_settings(self):
        """讀取儲存的設定"""
        settings = QSettings(self.SETTINGS_ORG, self.SETTINGS_APP)
        
        if settings.contains("startup/mode"):
            try:
                self._saved_config = StartupConfig(
                    mode=ExecutionMode(settings.value("startup/mode", "local")),
                    server_url=settings.value("startup/server_url", "http://localhost:8000"),
                    remember=settings.value("startup/remember", True, type=bool),
                )
            except Exception as e:
                logger.warning(f"Failed to load settings: {e}")
                self._saved_config = None
        else:
            self._saved_config = None
    
    def _save_settings(self, config: StartupConfig):
        """儲存設定"""
        settings = QSettings(self.SETTINGS_ORG, self.SETTINGS_APP)
        
        if config.remember:
            settings.setValue("startup/mode", config.mode.value)
            settings.setValue("startup/server_url", config.server_url)
            settings.setValue("startup/remember", config.remember)
        else:
            # 不記住的話，清除設定
            settings.remove("startup/mode")
            settings.remove("startup/server_url")
            settings.remove("startup/remember")
    
    def _apply_saved_settings(self):
        """套用儲存的設定到 UI"""
        if self._saved_config:
            if self._saved_config.mode == ExecutionMode.LOCAL:
                self._local_radio.setChecked(True)
            else:
                self._server_radio.setChecked(True)
            
            self._url_input.setText(self._saved_config.server_url)
            self._remember_check.setChecked(self._saved_config.remember)
        else:
            # 預設：遠端模式（因為大多數用戶沒有足夠 VRAM）
            self._server_radio.setChecked(True)
            self._url_input.setText("http://localhost:8000")
        
        self._on_mode_changed()
    
    def _on_mode_changed(self):
        """模式變更時更新 UI"""
        is_server = self._server_radio.isChecked()
        self._server_group.setEnabled(is_server)
        
        # 如果是本地模式，檢查 GPU
        if self._local_radio.isChecked():
            self._check_gpu()
    
    def _check_gpu(self):
        """Check local GPU."""
        self._local_status.setText("Detecting...")
        self._local_status.setStyleSheet("color: gray; margin-left: 20px;")

        # 避免重複啟動 GPU worker
        self._stop_thread(self._gpu_worker, "GPUCheckWorker")
        self._gpu_worker = GPUCheckWorker()
        self._gpu_worker.finished.connect(self._on_gpu_check_finished)
        self._gpu_worker.finished.connect(self._gpu_worker.deleteLater)
        self._gpu_worker.start()
    
    def _on_gpu_check_finished(self, has_gpu: bool, message: str):
        """GPU 檢測完成"""
        sender = self.sender()
        if sender is self._gpu_worker:
            self._gpu_worker = None

        if has_gpu:
            self._local_status.setText(f"✓ {message}")
            self._local_status.setStyleSheet("color: green; margin-left: 20px;")
            self._local_radio.setEnabled(True)
        else:
            self._local_status.setText(f"✗ {message}")
            self._local_status.setStyleSheet("color: red; margin-left: 20px;")
            # 不禁用，讓用戶自己決定
            # self._local_radio.setEnabled(False)

        self._gpu_worker = None
        if worker is not None:
            worker.deleteLater()
    
    def _on_test_connection(self):
        """測試連線"""
        url = self._url_input.text().strip()
        
        if not url:
            self._server_status.setText("✗ Please enter server URL")
            self._server_status.setStyleSheet("color: red;")
            return
        
        # 確保有 http:// 前綴
        if not url.startswith("http://") and not url.startswith("https://"):
            url = "http://" + url
            self._url_input.setText(url)
        
        self._test_button.setEnabled(False)
        self._test_button.setText("Testing...")
        self._server_status.setText("Connecting...")
        self._server_status.setStyleSheet("color: gray;")

        # 避免重複啟動連線測試 worker
        self._stop_thread(self._connection_worker, "ConnectionTestWorker")
        self._connection_worker = ConnectionTestWorker(url)
        self._connection_worker.finished.connect(self._on_connection_test_finished)
        self._connection_worker.finished.connect(self._connection_worker.deleteLater)
        self._connection_worker.start()
    
    def _on_connection_test_finished(self, success: bool, message: str):
        """連線測試完成"""
        sender = self.sender()
        if sender is self._connection_worker:
            self._connection_worker = None

        self._test_button.setEnabled(True)
        self._test_button.setText("Test Connection")
        
        if success:
            self._server_status.setText(f"✓ {message}")
            self._server_status.setStyleSheet("color: green;")
        else:
            self._server_status.setText(f"✗ {message}")
            self._server_status.setStyleSheet("color: red;")

        self._connection_worker = None
        if worker is not None:
            worker.deleteLater()
    
    def _on_start(self):
        """按下啟動按鈕"""
        # 驗證
        if self._server_radio.isChecked():
            url = self._url_input.text().strip()
            if not url:
                QMessageBox.warning(self, "Error", "Please enter server URL")
                return
            
            # 確保有 http:// 前綴
            if not url.startswith("http://") and not url.startswith("https://"):
                url = "http://" + url
        else:
            url = ""
        
        # 建立設定
        self._config = StartupConfig(
            mode=ExecutionMode.LOCAL if self._local_radio.isChecked() else ExecutionMode.SERVER,
            server_url=url if self._server_radio.isChecked() else "http://localhost:8000",
            remember=self._remember_check.isChecked(),
        )
        
        # 儲存設定
        self._save_settings(self._config)

        # 避免 dialog 關閉時 thread 仍在執行
        self._cleanup_workers()

        # 接受對話框
        self.accept()

    def accept(self):
        """覆寫 accept，確保 worker 已被清理。"""
        self._cleanup_workers()
        super().accept()

    def reject(self):
        """覆寫 reject，確保 worker 已被清理。"""
        self._cleanup_workers()
        super().reject()

    def closeEvent(self, event: QCloseEvent):
        """視窗關閉時確保 worker 已被清理。"""
        self._cleanup_workers()
        super().closeEvent(event)
    
    def get_config(self) -> Optional[StartupConfig]:
        """取得設定"""
        return self._config
    
    @classmethod
    def clear_saved_settings(cls):
        """清除儲存的設定（用於重設）"""
        settings = QSettings(cls.SETTINGS_ORG, cls.SETTINGS_APP)
        settings.remove("startup/mode")
        settings.remove("startup/server_url")
        settings.remove("startup/remember")
        logger.info("Startup settings cleared")


# =============================================================================
# 測試
# =============================================================================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # 清除設定（測試用）
    # StartupDialog.clear_saved_settings()
    
    dialog = StartupDialog(skip_if_remembered=False)  # 測試時不跳過
    
    if dialog.exec() == QDialog.DialogCode.Accepted:
        config = dialog.get_config()
        print("=" * 40)
        print("Startup Settings")
        print("=" * 40)
        print(f"Mode: {config.mode.value}")
        print(f"Server URL: {config.server_url}")
        print(f"Remember settings: {config.remember}")
    else:
        print("User cancelled")
    
    sys.exit(0)
