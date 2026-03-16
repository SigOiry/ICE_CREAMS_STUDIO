# -*- mode: python ; coding: utf-8 -*-


a = Analysis(
    ['ice_creams_ui.py'],
    pathex=[],
    binaries=[],
    datas=[('models', 'models'), ('about', 'about'), ('Data\\Input\\Validation', 'Data\\Input\\Validation'), ('icons', 'icons'), ('C:\\Users\\Simon\\miniconda3\\envs\\ICE_CREAMS\\Library\\share\\gdal', 'gdal_data'), ('C:\\Users\\Simon\\miniconda3\\envs\\ICE_CREAMS\\Library\\share\\proj', 'proj_data'), ('C:\\Users\\Simon\\miniconda3\\envs\\ICE_CREAMS\\Library\\lib\\gdalplugins', 'gdalplugins')],
    hiddenimports=['rasterio.sample', 'rasterio._io', 'rasterio._base', 'rasterio.serde', 'rasterio.crs', 'rasterio.vrt', 'pyogrio', 'pyogrio._geometry', 'fiona', 'xarray', 'rioxarray', 'dask', 'dask.array', 'geopandas', 'fastai.tabular.all', 'openpyxl', 'pandas._libs.window.aggregations', 'pandas._libs.groupby'],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)
splash = Splash(
    'C:\\Users\\Simon\\AppData\\Local\\Temp\\ice_creams_splash_clean.png',
    binaries=a.binaries,
    datas=a.datas,
    text_pos=None,
    text_size=12,
    minify_script=True,
    always_on_top=True,
)

exe = EXE(
    pyz,
    a.scripts,
    splash,
    [],
    exclude_binaries=True,
    name='ICE_CREAMS_Studio',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    version='C:\\Users\\Simon\\AppData\\Local\\Temp\\bbacfa80-6cb2-4bca-b729-2bda8ca0569d',
    icon=['C:\\Users\\Simon\\AppData\\Local\\Temp\\ice_creams_icon_square.png'],
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    splash.binaries,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='ICE_CREAMS_Studio',
)
