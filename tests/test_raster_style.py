from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from ice_creams_raster_style import (
    ICE_CREAMS_CLASS_STYLE,
    apply_classification_style,
    class_style_for_labels,
    read_raster_attribute_table,
)


def test_classification_style_is_attached_to_multiband_float_geotiff(tmp_path):
    raster_path = tmp_path / "classified.tif"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=4,
        height=3,
        count=4,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(0, 3, 1, 1),
        tiled=False,
    ) as output:
        output.write(np.ones((4, 3, 4), dtype=np.float32))

    sidecar = apply_classification_style(
        raster_path,
        ICE_CREAMS_CLASS_STYLE,
        band_descriptions=("Out_Class", "Class_Probs", "Seagrass_Cover", "NDVI"),
    )

    with rasterio.open(raster_path) as styled:
        assert styled.nodata == -1
        assert styled.descriptions == (
            "Out_Class",
            "Class_Probs",
            "Seagrass_Cover",
            "NDVI",
        )
        assert styled.tags(1)["CLASS_SCHEMA"] == "ICE_CREAMS_Out_Class_v2"
        assert styled.tags(1)["CLASS_3"] == "Chlorophyceae"
        assert styled.tags(1)["CLASS_3_COLOR"] == "#99ff13"

    rows = read_raster_attribute_table(raster_path)
    assert [int(row["Value"]) for row in rows] == [-1, 0, 1, 2, 3, 4, 5, 6, 7, 8]
    assert rows[4] == {
        "Value": "3",
        "Class": "Chlorophyceae",
        "Red": "153",
        "Green": "255",
        "Blue": "19",
        "Alpha": "255",
    }
    assert rows[-1]["Class"] == "Water"
    embedded = b"GDALRasterAttributeTable" in raster_path.read_bytes()
    auxiliary_table = Path(f"{raster_path}.aux.xml").exists()
    assert embedded or auxiliary_table

    assert sidecar == raster_path.with_suffix(".qml")
    qml = sidecar.read_text(encoding="utf-8")
    assert 'colorRampType="EXACT"' in qml
    assert 'value="3" label="Chlorophyceae" color="#99ff13"' in qml
    assert 'value="8" label="Water" color="#3d2fff"' in qml


def test_generic_class_style_preserves_model_labels():
    style = class_style_for_labels(["Bare Sand", "Vegetation"])
    assert style[1][0] == "Bare Sand"
    assert style[2][0] == "Vegetation"
    assert style[1][1] == ICE_CREAMS_CLASS_STYLE[1][1]


def test_numeric_eight_class_vocab_uses_canonical_ice_creams_legend():
    style = class_style_for_labels([str(value) for value in range(1, 9)])
    assert style == ICE_CREAMS_CLASS_STYLE
