#!/usr/bin/env python3
"""
HIL-AA Maritime Annotation System
=================================

Main entry point for the application.

Usage:
    python main.py              # Launch GUI
    python main.py --mock       # Launch with mock SAM 3 (for development)
    python main.py --config     # Print configuration
    python main.py --version    # Print version

Author: Sonic @ NYCU Maritime Robotics Lab
"""

import sys
from pathlib import Path

import typer
from loguru import logger

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src import __version__, ensure_directories
from src.config import config, print_config

app = typer.Typer(
    name="hil-aa",
    help="HIL-AA Maritime Annotation System",
    add_completion=False,
)


def setup_logging() -> None:
    """Configure loguru logging."""
    log_config = config.logging
    
    # Remove default handler
    logger.remove()
    
    # Console handler
    logger.add(
        sys.stderr,
        level=log_config.level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan> - <level>{message}</level>",
    )
    
    # File handler (if enabled)
    if log_config.log_to_file:
        log_config.log_dir.mkdir(parents=True, exist_ok=True)
        logger.add(
            log_config.log_dir / "hil_aa_{time:YYYY-MM-DD}.log",
            level="DEBUG",
            rotation="1 day",
            retention="7 days",
        )


@app.command()
def main(
    mock: bool = typer.Option(
        False,
        "--mock", "-m",
        help="Run with mock SAM 3 predictions (for GUI development without GPU)"
    ),
    show_config: bool = typer.Option(
        False,
        "--config", "-c",
        help="Print configuration and exit"
    ),
    version: bool = typer.Option(
        False,
        "--version", "-v",
        help="Print version and exit"
    ),
    video: str = typer.Option(
        None,
        "--video", "-i",
        help="Video file to open on startup"
    ),
) -> None:
    """
    Launch the HIL-AA Maritime Annotation System.
    
    This system uses SAM 3's semantic confidence scores to minimize
    human annotation effort through intelligent active learning.
    """
    # Handle simple flags
    if version:
        typer.echo(f"HIL-AA Maritime Annotation System v{__version__}")
        raise typer.Exit()
    
    if show_config:
        print_config()
        raise typer.Exit()
    
    # Setup
    setup_logging()
    ensure_directories()
    
    # Override mock mode if specified
    if mock:
        config.sam3.mock_mode = True
        logger.info("Running in MOCK mode (no real SAM 3 inference)")
    
    logger.info(f"Starting HIL-AA v{__version__}")
    logger.info(f"Device: {config.sam3.device}")
    logger.info(f"Mock mode: {config.sam3.mock_mode}")
    
    # Launch GUI
    try:
        from PyQt6.QtWidgets import QApplication
        
        # Check if GUI modules exist
        try:
            from src.gui.main_window import MainWindow
        except ImportError:
            logger.warning("GUI modules not yet implemented. Creating placeholder...")
            
            # Placeholder until GUI is built
            from PyQt6.QtWidgets import QMainWindow, QLabel, QVBoxLayout, QWidget
            from PyQt6.QtCore import Qt
            
            class MainWindow(QMainWindow):
                def __init__(self):
                    super().__init__()
                    self.setWindowTitle(config.gui.window_title)
                    self.setGeometry(100, 100, config.gui.window_width, config.gui.window_height)
                    
                    central = QWidget()
                    layout = QVBoxLayout(central)
                    
                    label = QLabel(
                        "🚢 HIL-AA Maritime Annotation System\n\n"
                        f"Version: {__version__}\n"
                        f"Mode: {'MOCK' if config.sam3.mock_mode else 'PRODUCTION'}\n\n"
                        "GUI modules coming in Week 3!\n\n"
                        "Press Ctrl+Q to quit."
                    )
                    label.setAlignment(Qt.AlignmentFlag.AlignCenter)
                    label.setStyleSheet("font-size: 18px; padding: 50px;")
                    layout.addWidget(label)
                    
                    self.setCentralWidget(central)
        
        # Create and run application
        qt_app = QApplication(sys.argv)
        qt_app.setApplicationName("HIL-AA")
        qt_app.setApplicationVersion(__version__)
        
        window = MainWindow()
        
        # Open video if specified
        if video and hasattr(window, 'open_video'):
            window.open_video(video)
        
        window.show()
        
        logger.info("GUI launched successfully")
        sys.exit(qt_app.exec())
        
    except ImportError as e:
        logger.error(f"Failed to import PyQt6: {e}")
        logger.error("Please ensure PyQt6 is installed: pip install PyQt6")
        raise typer.Exit(1)
    except Exception as e:
        logger.exception(f"Application error: {e}")
        raise typer.Exit(1)


@app.command()
def test_x11() -> None:
    """Test X11 display connection (for Docker debugging)."""
    import os
    
    display = os.environ.get("DISPLAY", "not set")
    typer.echo(f"DISPLAY={display}")
    
    try:
        from PyQt6.QtWidgets import QApplication, QMessageBox
        
        app = QApplication(sys.argv)
        
        msg = QMessageBox()
        msg.setWindowTitle("X11 Test")
        msg.setText("✅ X11 connection successful!\n\nPyQt6 can display windows.")
        msg.setIcon(QMessageBox.Icon.Information)
        msg.exec()
        
        typer.echo("✅ X11 test passed!")
        
    except Exception as e:
        typer.echo(f"❌ X11 test failed: {e}", err=True)
        raise typer.Exit(1)


@app.command()
def test_gpu() -> None:
    """Test CUDA/GPU availability."""
    try:
        import torch
        
        typer.echo(f"PyTorch version: {torch.__version__}")
        typer.echo(f"CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            typer.echo(f"CUDA version: {torch.version.cuda}")
            typer.echo(f"GPU count: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                typer.echo(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            typer.echo("✅ GPU test passed!")
        else:
            typer.echo("⚠️ Running in CPU mode")
            
    except Exception as e:
        typer.echo(f"❌ GPU test failed: {e}", err=True)
        raise typer.Exit(1)


if __name__ == "__main__":
    app()