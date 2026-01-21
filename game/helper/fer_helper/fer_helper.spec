# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['fer_helper.py'],
    pathex=[],
    binaries=[],
    datas=[('models/facial_expression_recognition_mobilefacenet_2022july.onnx', 'models'), ('/Users/zhengzhu/Developer/renpy-8.5.0-sdk/the_question/game/helper/fer_helper/fer-env/lib/python3.11/site-packages/cv2/data/haarcascade_frontalface_default.xml', 'cv2/data')],
    hiddenimports=[],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='fer_helper',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='fer_helper',
)
app = BUNDLE(
    coll,
    name='fer_helper.app',
    icon=None,
    bundle_identifier=None,
)
