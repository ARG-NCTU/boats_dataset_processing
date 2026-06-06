# -*- mode: python ; coding: utf-8 -*-
# =============================================================================
# STAMP Client PyInstaller Spec File v2
# =============================================================================
#
# Remote-server client packaging profile.
#
# This spec intentionally packages only the lightweight PyQt6 client. GPU
# inference, SAM3, PyTorch, server routes, model weights, data, and old local-GPU
# GUI entry points are excluded.
#
# Build:
#   pyinstaller stamp_client_v2.spec --noconfirm
#
# Output:
#   dist/STAMP_Client/STAMP_Client
#

from pathlib import Path


PROJECT_ROOT = Path(SPECPATH).resolve()
APP_NAME = "STAMP_Client"


CLIENT_HIDDEN_IMPORTS = [
    # PyQt6
    "PyQt6",
    "PyQt6.QtCore",
    "PyQt6.QtGui",
    "PyQt6.QtWidgets",
    "PyQt6.sip",

    # Config
    "pydantic",
    "pydantic.fields",
    "pydantic_settings",
    "configs",
    "configs.config",

    # Data/image stack
    "numpy",
    "numpy.core._methods",
    "numpy.lib.format",
    "PIL",
    "PIL.Image",
    "cv2",

    # Mask / export stack
    "pycocotools",
    "pycocotools.mask",
    "pycocotools.coco",
    "pycocotools._mask",
    "pandas",
    "pandas.core",
    "pandas.core.frame",
    "pandas.core.series",
    "pandas.core.arrays",
    "pandas._libs",
    "pandas._libs.lib",
    "pandas.io",
    "pyarrow",
    "pyarrow.lib",
    "pyarrow.parquet",
    "datasets",
    "datasets.arrow_dataset",
    "datasets.features",

    # Network/logging
    "requests",
    "urllib3",
    "websocket",
    "websocket._core",
    "loguru",
    "mcap",

    # src namespace imports used by main_server.py and dynamic GUI imports.
    "src",
    "src.config",
    "src.api_client",
    "src.gui",
    "src.gui.startup_dialog",
    "src.gui.main_window_server",
    "src.gui.export_paths",
    "src.gui.canvas_viewport",
    "src.gui.interactive_canvas",
    "src.gui.timeline_widget",
    "src.gui.server_workers",
    "src.gui.server_workers.server_worker",
    "src.core",
    "src.core.action_logger",
    "src.core.confidence_analyzer",
    "src.core.exporter",
    "src.core.jitter_detector",
    "src.core.maritime_roi",
    "src.core.refinement_utils",
    "src.core.video_loader",
    "src.utils",

    # Top-level imports used after src/ is injected into sys.path.
    "gui",
    "gui.startup_dialog",
    "gui.main_window_server",
    "gui.export_paths",
    "gui.canvas_viewport",
    "gui.interactive_canvas",
    "gui.timeline_widget",
    "gui.server_workers",
    "gui.server_workers.server_worker",
    "core",
    "core.action_logger",
    "core.confidence_analyzer",
    "core.exporter",
    "core.jitter_detector",
    "core.maritime_roi",
    "core.refinement_utils",
    "core.video_loader",
]


CLIENT_EXCLUDES = [
    # Server-only code
    "server",
    "server.main",
    "server.task_manager",
    "server.routes",
    "fastapi",
    "uvicorn",

    # SAM3/local-GPU inference stays server-side.
    "src.core.sam3_engine",
    "core.sam3_engine",
    "sam3",
    "third_party",
    "torch",
    "torchvision",
    "torchaudio",

    # SegFormer model download/inference is not part of the remote client bundle.
    "transformers",
    "src.download_segformer_model",
    "download_segformer_model",

    # Old/local GUI variants and extracted local-GPU components.
    "src.gui.main_window",
    "src.gui.main_window_split",
    "src.gui.main_window_with_action_logger",
    "src.gui.main_window_with_maritime_roi",
    "src.gui.test",
    "src.gui.components",
    "src.gui.components.workers",
    "src.gui.components.detection_mixin",
    "src.gui.components.dialogs",
    "src.gui.components.object_list_item",
    "src.gui.components.object_manager_mixin",
    "src.gui.components.playback_mixin",
    "src.gui.components.refinement_mixin",
    "gui.main_window",
    "gui.main_window_split",
    "gui.main_window_with_action_logger",
    "gui.main_window_with_maritime_roi",
    "gui.test",
    "gui.components",
    "gui.components.workers",
    "gui.components.detection_mixin",
    "gui.components.dialogs",
    "gui.components.object_list_item",
    "gui.components.object_manager_mixin",
    "gui.components.playback_mixin",
    "gui.components.refinement_mixin",

    # Demo/development entry points.
    "src.demo",
    "src.demo_fixed",
    "demo",
    "demo_fixed",
    "main",
    "run_server",
    "test_api",
    "tests",

    # Large optional scientific/dev stacks not required by the client.
    "tensorflow",
    "keras",
    "matplotlib",
    "scipy",
    "sklearn",
    "IPython",
    "jupyter",
    "PyQt5",
]


a = Analysis(
    ["main_server.py"],
    pathex=[
        str(PROJECT_ROOT),
        str(PROJECT_ROOT / "src"),
    ],
    binaries=[],
    datas=[
        (str(PROJECT_ROOT / "configs" / "config.py"), "configs"),
    ],
    hiddenimports=CLIENT_HIDDEN_IMPORTS,
    hookspath=[str(PROJECT_ROOT / "hooks")],
    hooksconfig={},
    runtime_hooks=[],
    excludes=CLIENT_EXCLUDES,
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=None,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=None)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name=APP_NAME,
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name=APP_NAME,
)
