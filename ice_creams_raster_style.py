"""QGIS-compatible class styling for ICE CREAMS raster outputs."""

from __future__ import annotations

import ctypes
import ctypes.util
import html
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Mapping, Sequence

import rasterio


# Colors are adapted from QGIS_color_scale/ICE_CREAMS.qml.  The scientific
# names match the class terminology used throughout the application.
ICE_CREAMS_CLASS_STYLE: dict[int, tuple[str, str]] = {
    1: ("Bare Sediment", "#3e3d1b"),
    2: ("Sand", "#3e3d1b"),
    3: ("Chlorophyceae", "#99ff13"),
    4: ("Magnoliopsida", "#09861a"),
    5: ("Bacillariophyceae", "#ff9e36"),
    6: ("Phaeophyceae", "#a40205"),
    7: ("Florideophyceae", "#ff0004"),
    8: ("Water", "#3d2fff"),
}

_FALLBACK_COLORS = tuple(color for _, color in ICE_CREAMS_CLASS_STYLE.values()) + (
    "#fff708",
    "#8a8a8a",
    "#00bcd4",
    "#9c27b0",
)


def class_style_for_labels(labels: Sequence[object]) -> dict[int, tuple[str, str]]:
    """Return a stable one-based class style for an arbitrary model vocabulary."""
    normalized = [str(label) for label in labels]
    if len(normalized) == len(ICE_CREAMS_CLASS_STYLE):
        try:
            numeric_labels = [int(label) for label in normalized]
        except ValueError:
            numeric_labels = []
        if numeric_labels == list(ICE_CREAMS_CLASS_STYLE):
            return dict(ICE_CREAMS_CLASS_STYLE)
    return {
        index: (label, _FALLBACK_COLORS[(index - 1) % len(_FALLBACK_COLORS)])
        for index, label in enumerate(normalized, start=1)
    }


def _hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    color_value = hex_color.strip().lstrip("#")
    if len(color_value) != 6:
        raise ValueError(f"Expected #RRGGBB color, got: {hex_color}")
    return (
        int(color_value[0:2], 16),
        int(color_value[2:4], 16),
        int(color_value[4:6], 16),
        alpha,
    )


def _gdal_library_candidates() -> list[str]:
    roots = [
        Path(rasterio.__file__).resolve().parent.parent,
        Path(sys.prefix) / "Library" / "bin",
        Path(sys.executable).resolve().parent,
    ]
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        roots.insert(0, Path(frozen_root))

    patterns = (
        "rasterio.libs/gdal-*.dll",
        "rasterio.libs/libgdal-*.so*",
        "rasterio.libs/libgdal*.dylib",
        "gdal*.dll",
        "libgdal*.so*",
        "libgdal*.dylib",
    )
    candidates: list[str] = []
    for root in roots:
        for pattern in patterns:
            candidates.extend(str(path) for path in sorted(root.glob(pattern)))
    system_library = ctypes.util.find_library("gdal")
    if system_library:
        candidates.append(system_library)
    return list(dict.fromkeys(candidates))


@lru_cache(maxsize=1)
def _gdal_rat_api() -> ctypes.CDLL:
    api = None
    load_errors: list[str] = []
    for candidate in _gdal_library_candidates():
        try:
            api = ctypes.CDLL(candidate)
            if hasattr(api, "GDALCreateRasterAttributeTable"):
                break
        except OSError as exc:
            load_errors.append(f"{candidate}: {exc}")
            api = None
    if api is None or not hasattr(api, "GDALCreateRasterAttributeTable"):
        detail = "; ".join(load_errors[-2:])
        raise RuntimeError(
            "The bundled GDAL library does not expose raster attribute table support"
            + (f" ({detail})" if detail else "")
        )

    void_pointer = ctypes.c_void_p
    char_pointer = ctypes.c_char_p
    integer = ctypes.c_int
    signatures = {
        "GDALAllRegister": ([], None),
        "GDALOpen": ([char_pointer, integer], void_pointer),
        "GDALGetRasterBand": ([void_pointer, integer], void_pointer),
        "GDALClose": ([void_pointer], None),
        "GDALCreateRasterAttributeTable": ([], void_pointer),
        "GDALDestroyRasterAttributeTable": ([void_pointer], None),
        "GDALRATCreateColumn": ([void_pointer, char_pointer, integer, integer], integer),
        "GDALRATSetRowCount": ([void_pointer, integer], None),
        "GDALRATSetValueAsInt": ([void_pointer, integer, integer, integer], None),
        "GDALRATSetValueAsString": ([void_pointer, integer, integer, char_pointer], None),
        "GDALRATSetTableType": ([void_pointer, integer], integer),
        "GDALSetDefaultRAT": ([void_pointer, void_pointer], integer),
        "GDALGetDefaultRAT": ([void_pointer], void_pointer),
        "GDALRATGetColumnCount": ([void_pointer], integer),
        "GDALRATGetRowCount": ([void_pointer], integer),
        "GDALRATGetNameOfCol": ([void_pointer, integer], char_pointer),
        "GDALRATGetValueAsString": ([void_pointer, integer, integer], char_pointer),
        "CPLGetLastErrorMsg": ([], char_pointer),
        "CPLGetConfigOption": ([char_pointer, char_pointer], char_pointer),
        "CPLSetConfigOption": ([char_pointer, char_pointer], None),
    }
    for name, (argument_types, result_type) in signatures.items():
        function = getattr(api, name)
        function.argtypes = argument_types
        function.restype = result_type
    api.GDALAllRegister()
    return api


def _gdal_error(api: ctypes.CDLL) -> str:
    message = api.CPLGetLastErrorMsg()
    return message.decode("utf-8", errors="replace") if message else "unknown GDAL error"


def embed_raster_attribute_table(
    raster_path: str | os.PathLike[str],
    class_style: Mapping[int, tuple[str, str]],
    *,
    band_index: int = 1,
    nodata_value: int = -1,
) -> None:
    """Attach exact class values, labels, and RGBA colors through a GDAL RAT.

    GDAL 3.12+ stores the table inside a GeoTIFF. Older runtimes use GDAL's
    standard ``.aux.xml`` companion, which QGIS discovers automatically.
    """
    api = _gdal_rat_api()
    config_key = b"GTIFF_WRITE_RAT_TO_PAM"
    previous_config_pointer = api.CPLGetConfigOption(config_key, None)
    previous_config = bytes(previous_config_pointer) if previous_config_pointer else None
    api.CPLSetConfigOption(config_key, b"NO")
    dataset = None
    rat = None
    try:
        dataset = api.GDALOpen(os.fsencode(os.fspath(raster_path)), 1)  # GA_Update
        if not dataset:
            raise RuntimeError(f"Could not open raster for styling: {_gdal_error(api)}")
        band = api.GDALGetRasterBand(dataset, band_index)
        if not band:
            raise RuntimeError(f"Raster band {band_index} does not exist")

        rat = api.GDALCreateRasterAttributeTable()
        if not rat:
            raise RuntimeError("Could not allocate a GDAL raster attribute table")
        # GDALRATFieldType: Integer=0, String=2. GDALRATFieldUsage:
        # Name=2, MinMax=5, Red=6, Green=7, Blue=8, Alpha=9.
        columns = (
            ("Value", 0, 5),
            ("Class", 2, 2),
            ("Red", 0, 6),
            ("Green", 0, 7),
            ("Blue", 0, 8),
            ("Alpha", 0, 9),
        )
        for name, field_type, usage in columns:
            if api.GDALRATCreateColumn(rat, name.encode("utf-8"), field_type, usage) != 0:
                raise RuntimeError(f"Could not create RAT column {name}: {_gdal_error(api)}")

        rows = [
            (nodata_value, "NoData", (0, 0, 0, 0)),
            (0, "NoData", (0, 0, 0, 0)),
        ]
        rows.extend(
            (class_id, class_name, _hex_to_rgba(hex_color))
            for class_id, (class_name, hex_color) in sorted(class_style.items())
        )
        api.GDALRATSetTableType(rat, 0)  # GRTT_THEMATIC
        api.GDALRATSetRowCount(rat, len(rows))
        for row_index, (class_id, class_name, rgba) in enumerate(rows):
            api.GDALRATSetValueAsInt(rat, row_index, 0, int(class_id))
            api.GDALRATSetValueAsString(
                rat, row_index, 1, str(class_name).encode("utf-8")
            )
            for column_index, component in enumerate(rgba, start=2):
                api.GDALRATSetValueAsInt(
                    rat, row_index, column_index, int(component)
                )
        if api.GDALSetDefaultRAT(band, rat) != 0:
            raise RuntimeError(f"Could not embed raster class table: {_gdal_error(api)}")
    finally:
        if rat:
            api.GDALDestroyRasterAttributeTable(rat)
        if dataset:
            api.GDALClose(dataset)
        api.CPLSetConfigOption(config_key, previous_config)


def read_raster_attribute_table(
    raster_path: str | os.PathLike[str], *, band_index: int = 1
) -> list[dict[str, str]]:
    """Read an embedded RAT; intended for validation and diagnostics."""
    api = _gdal_rat_api()
    dataset = api.GDALOpen(os.fsencode(os.fspath(raster_path)), 0)  # GA_ReadOnly
    if not dataset:
        raise RuntimeError(f"Could not open raster: {_gdal_error(api)}")
    try:
        band = api.GDALGetRasterBand(dataset, band_index)
        rat = api.GDALGetDefaultRAT(band) if band else None
        if not rat:
            return []
        column_names = []
        for column_index in range(api.GDALRATGetColumnCount(rat)):
            value = api.GDALRATGetNameOfCol(rat, column_index)
            column_names.append(value.decode("utf-8", errors="replace"))
        rows = []
        for row_index in range(api.GDALRATGetRowCount(rat)):
            row = {}
            for column_index, column_name in enumerate(column_names):
                value = api.GDALRATGetValueAsString(rat, row_index, column_index)
                row[column_name] = value.decode("utf-8", errors="replace") if value else ""
            rows.append(row)
        return rows
    finally:
        api.GDALClose(dataset)


def qml_sidecar_path(raster_path: str | os.PathLike[str]) -> Path:
    return Path(raster_path).with_suffix(".qml")


def write_qgis_style_sidecar(
    raster_path: str | os.PathLike[str],
    class_style: Mapping[int, tuple[str, str]],
    *,
    nodata_value: int = -1,
) -> Path:
    """Write a same-basename QML fallback for QGIS versions predating RAT support."""
    entries = [
        f'          <item alpha="0" value="{nodata_value}" label="NoData" color="#000000"/>',
        '          <item alpha="0" value="0" label="NoData" color="#000000"/>',
    ]
    for class_id, (class_name, hex_color) in sorted(class_style.items()):
        entries.append(
            "          <item alpha=\"255\" value=\"{}\" label=\"{}\" color=\"{}\"/>".format(
                class_id, html.escape(str(class_name), quote=True), html.escape(hex_color, quote=True)
            )
        )
    minimum = min(nodata_value, 0)
    maximum = max(class_style) if class_style else 0
    qml = f"""<!DOCTYPE qgis PUBLIC 'http://mrcc.com/qgis.dtd' 'SYSTEM'>
<qgis version="3.28.1-Firenze" styleCategories="Symbology|Rendering">
  <pipe>
    <provider>
      <resampling enabled="false" maxOversampling="2" zoomedOutResamplingMethod="nearestNeighbour" zoomedInResamplingMethod="nearestNeighbour"/>
    </provider>
    <rasterrenderer band="1" classificationMin="{minimum}" alphaBand="-1" opacity="1" classificationMax="{maximum}" nodataColor="" type="singlebandpseudocolor">
      <rasterTransparency/>
      <rastershader>
        <colorrampshader colorRampType="EXACT" minimumValue="{minimum}" maximumValue="{maximum}" classificationMode="1" labelPrecision="0" clip="1">
{os.linesep.join(entries)}
        </colorrampshader>
      </rastershader>
    </rasterrenderer>
    <brightnesscontrast brightness="0" contrast="0" gamma="1"/>
    <huesaturation saturation="0" colorizeStrength="100" colorizeRed="255" colorizeGreen="128" colorizeBlue="128" grayscaleMode="0" colorizeOn="0" invertColors="0"/>
    <rasterresampler maxOversampling="2"/>
    <resamplingStage>resamplingFilter</resamplingStage>
  </pipe>
  <blendMode>0</blendMode>
</qgis>
"""
    sidecar_path = qml_sidecar_path(raster_path)
    sidecar_path.write_text(qml, encoding="utf-8")
    return sidecar_path


def apply_classification_style(
    raster_path: str | os.PathLike[str],
    class_style: Mapping[int, tuple[str, str]],
    *,
    band_descriptions: Sequence[str] = (),
    band_index: int = 1,
    nodata_value: int = -1,
) -> Path:
    """Attach class metadata and create an older-QGIS QML fallback."""
    category_names = ["NoData"] + [
        class_style[class_id][0] for class_id in sorted(class_style)
    ]
    tags = {
        "CLASS_SCHEMA": "ICE_CREAMS_Out_Class_v2",
        "CLASS_COUNT": str(len(class_style)),
        "CATEGORY_NAMES": "|".join(category_names),
    }
    for class_id, (class_name, hex_color) in sorted(class_style.items()):
        tags[f"CLASS_{class_id}"] = str(class_name)
        tags[f"CLASS_{class_id}_COLOR"] = hex_color

    with rasterio.open(raster_path, "r+") as output:
        if output.count < band_index:
            raise ValueError(f"Raster band {band_index} does not exist")
        output.nodata = nodata_value
        output.update_tags(band_index, **tags)
        for index, description in enumerate(band_descriptions, start=1):
            if index <= output.count:
                output.set_band_description(index, description)

    embed_raster_attribute_table(
        raster_path, class_style, band_index=band_index, nodata_value=nodata_value
    )
    return write_qgis_style_sidecar(
        raster_path, class_style, nodata_value=nodata_value
    )
