# -*- mode: python ; coding: utf-8 -*-
# =============================================================================
# STAMP Client PyInstaller Spec File v2
# =============================================================================
#
# Conservative variant of stamp_client.spec.
# It keeps the original packaging shape and only adds modules that were created
# after the original spec, plus explicit helper imports needed by refinement.
#
# Build:
#   pyinstaller stamp_client_v2.spec --noconfirm
#
# Output:
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
        
        # ----- pydantic / configs -----
        'pydantic',
        'pydantic.fields',
        'pydantic_settings',
        
        # ----- data stack -----
        'numpy',
        'numpy.core._methods',
        'numpy.lib.format',
        
        # ----- image stack -----
        'PIL',
        'PIL.Image',
        'cv2',
        
        # ----- pycocotools -----
        'pycocotools',
        'pycocotools.mask',
        'pycocotools.coco',
        'pycocotools._mask',

        # ----- parquet / dataframe export -----
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
        
        # ----- network -----
        'requests',
        'urllib3',
        'websocket',
        'websocket._core',
        
        # ----- logging -----
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
        'src.gui.export_paths',
        'src.gui.canvas_viewport',
        'src.gui.interactive_canvas',
        'src.gui.timeline_widget',
        
        # These are imported as top-level gui.* after src/ is placed on sys.path.
        'gui.export_paths',
        'gui.canvas_viewport',
        
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
        
        # ----- src/core/ -----
        'src.core',
        'src.core.action_logger',
        'src.core.confidence_analyzer',
        'src.core.exporter',
        'src.core.jitter_detector',
        'src.core.maritime_roi',
        'src.core.sam3_engine',
        'src.core.video_loader',
        
        # main_window_server imports this helper through top-level core.*.
        'core.sam3_engine',
        
        # ----- src/utils/ -----
        'src.utils',
    ],
    
    hookspath=['./hooks'],
    hooksconfig={},
    runtime_hooks=[],
    
    excludes=[
        # Server side is still not part of the desktop client.
        'server',
        
        # Heavy optional stacks unrelated to the client.
        'tensorflow',
        'keras',
        'matplotlib',
        'scipy',
        'sklearn',
        'IPython',
        'jupyter',
        'PyQt5',
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
