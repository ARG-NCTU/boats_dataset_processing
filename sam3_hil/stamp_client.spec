# -*- mode: python ; coding: utf-8 -*-
# =============================================================================
# STAMP Client PyInstaller Spec File
# =============================================================================
#
# 放置位置：sam3_hil/stamp_client.spec
#
# 安裝依賴：
#   pip install pyinstaller PyQt6 numpy pillow opencv-python requests \
#               loguru websocket-client cython pycocotools pydantic-settings
#
# 打包：
#   pyinstaller stamp_client.spec
#
# 輸出：
#   dist/STAMP_Client/STAMP_Client
#

from pathlib import Path

PROJECT_ROOT = Path(SPECPATH).resolve()


a = Analysis(
    ['main_server.py'],
    
    pathex=[
        str(PROJECT_ROOT),
        str(PROJECT_ROOT / 'src'),
    ],
    
    binaries=[],
    
    # 資料檔案（非 .py）
    datas=[
        ('configs', 'configs'),
    ],
    
    hiddenimports=[
        # ----- PyQt6 -----
        'PyQt6',
        'PyQt6.QtCore',
        'PyQt6.QtGui',
        'PyQt6.QtWidgets',
        'PyQt6.sip',
        
        # ----- pydantic（configs/config.py 用）-----
        'pydantic',
        'pydantic.fields',
        'pydantic_settings',
        
        # ----- 資料處理 -----
        'numpy',
        'numpy.core._methods',
        'numpy.lib.format',
        
        # ----- 圖像處理 -----
        'PIL',
        'PIL.Image',
        'cv2',
        
        # ----- pycocotools -----
        'pycocotools',
        'pycocotools.mask',
	    'pycocotools.coco',
        'pycocotools._mask',

        # ----- parquet -----
        'pyarrow',
        'pyarrow.parquet',
        'pyarrow.lib',
        'datasets',
        'datasets.arrow_dataset',
    	'datasets.features',
	    'pandas',
        'pandas.core',
        'pandas.core.frame',
        'pandas.core.series',
        'pandas.core.arrays',
        'pandas._libs',
        'pandas._libs.lib',
        'pandas.io',
        
        # ----- 網路 -----
        'requests',
        'urllib3',
        'websocket',
        'websocket._core',
        
        # ----- 日誌 -----
        'loguru',
	    'mcap',
        
        # ----- configs/ -----
        'configs',
        'configs.config',
        
        # ----- src/ -----
        'src',
        'src.config',
        'src.api_client',
        
        # ----- src/gui/ -----
        'src.gui',
        'src.gui.startup_dialog',
        'src.gui.main_window_server',
        'src.gui.main_window_with_action_logger',
        'src.gui.interactive_canvas',
        'src.gui.timeline_widget',
        
        # ----- src/gui/components/ -----
        'src.gui.components',
        'src.gui.components.detection_mixin',
        'src.gui.components.dialogs',
        'src.gui.components.object_list_item',
        'src.gui.components.object_manager_mixin',
        'src.gui.components.playback_mixin',
        'src.gui.components.refinement_mixin',
        'src.gui.components.workers',
        
        # ----- src/gui/server_workers/ -----
        'src.gui.server_workers',
        'src.gui.server_workers.server_worker',
        
        # ----- src/core/（不含 sam3_engine）-----
        'src.core',
        'src.core.action_logger',
        'src.core.confidence_analyzer',
        'src.core.exporter',
        'src.core.jitter_detector',
        'src.core.maritime_roi',
        'src.core.video_loader',
        
        # ----- src/utils/ -----
        'src.utils',
    ] ,
    
    hookspath=['./hooks'],
    hooksconfig={},
    runtime_hooks=[],
    
    excludes=[
        # Server 端
        'server',
        
        # SAM3（在 Server 跑）
        'src.core.sam3_engine',
        'sam3',
        'third_party',
        
        # PyTorch
        'torch',
        'torchvision',
        'torchaudio',
        
        # 其他
        'tensorflow',
        'keras',
        'matplotlib',
        'scipy',
        'sklearn',
        'IPython',
        'jupyter',
    ],
    
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
    name='STAMP_Client',
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
    name='STAMP_Client',
)
