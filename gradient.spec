# -*- mode: python ; coding: utf-8 -*-

import os

block_cipher = None

# Create a temporary entry point script
entry_script_content = """
from gradient.cli import run
if __name__ == '__main__':
    run()
"""

with open('gradient_entry.py', 'w') as f:
    f.write(entry_script_content)

a = Analysis(
    ['gradient_entry.py'],
    pathex=[],
    binaries=[],
    datas=[
        ('gradient/_vendor', 'gradient/_vendor'),
    ],
    hiddenimports=[
        'gradient.cli',
        'gradient.cli.cli', 
        'gradient.cli.services',
        'gradient.cli.interfaces',
        'gradient.sdk',
        'gradient.sdk.decorator',
        'gradient.runtime',
        'gradient.runtime.manager',
        'gradient.runtime.context',
        'gradient.runtime.tracker',
        'typer',
        'click',
        'yaml',
        'fastapi',
        'uvicorn',
        'pydantic',
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name='gradient',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)

# Clean up temporary file
import os
if os.path.exists('gradient_entry.py'):
    os.remove('gradient_entry.py')
