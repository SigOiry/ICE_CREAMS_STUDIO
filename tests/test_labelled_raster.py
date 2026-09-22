from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point, box

from ice_creams_feature_modes import FEATURE_MODE_HIGH_SPATIAL_ACCURACY
from ice_creams_labelled_raster import labelled_raster_dataframe, labelled_point_raster_dataframe


def _fixture(tmp_path, *, labels=(1, "water"), descriptions=False, nodata=False):
    raster_path = tmp_path / "image.tif"
    polygon_path = tmp_path / "labels.geojson"
    with rasterio.open(
        raster_path,
        "w",
        driver="GTiff",
        width=4,
        height=2,
        count=4,
        dtype="float32",
        crs="EPSG:3857",
        transform=from_origin(0, 2, 1, 1),
        nodata=-9999,
    ) as dst:
        for index in range(1, 5):
            values = np.full((2, 4), index / 10, dtype="float32")
            if nodata:
                values[0, 0] = -9999
            dst.write(values, index)
            if descriptions:
                dst.set_band_description(index, ("B02", "B03", "B04", "B08")[index - 1])
    gpd.GeoDataFrame(
        {"class_id": list(labels)},
        geometry=[box(0, 0, 2, 2), box(2, 0, 4, 2)],
        crs="EPSG:3857",
    ).to_crs("EPSG:4326").to_file(polygon_path, driver="GeoJSON")
    return raster_path, polygon_path


def test_extract_numeric_and_character_labels_with_crs_and_reflectance_scale(tmp_path):
    raster, polygons = _fixture(tmp_path, descriptions=True, nodata=True)
    frame = labelled_raster_dataframe(
        str(raster), str(polygons), "class_id",
        feature_mode=FEATURE_MODE_HIGH_SPATIAL_ACCURACY,
    )
    assert len(frame) == 7
    assert set(frame["class_id"]) == {"1", "water"}
    assert frame["Reflectance_B02"].eq(1000).all()
    assert frame["NDVI"].notna().all()
    assert frame["Pixel_Row"].between(0, 1).all()


def test_missing_class_column_has_clear_error(tmp_path):
    raster, polygons = _fixture(tmp_path)
    with pytest.raises(ValueError, match="Class column 'missing'"):
        labelled_raster_dataframe(
            str(raster), str(polygons), "missing",
            feature_mode=FEATURE_MODE_HIGH_SPATIAL_ACCURACY,
        )


def test_conflicting_polygons_rejected(tmp_path):
    raster, polygons = _fixture(tmp_path)
    gpd.GeoDataFrame(
        {"class_id": [1, 2]},
        geometry=[box(0, 0, 3, 2), box(2, 0, 4, 2)],
        crs="EPSG:3857",
    ).to_file(polygons, driver="GeoJSON")
    with pytest.raises(ValueError, match="different classes"):
        labelled_raster_dataframe(
            str(raster), str(polygons), "class_id",
            feature_mode=FEATURE_MODE_HIGH_SPATIAL_ACCURACY,
        )


def test_band_descriptions_override_file_order(tmp_path):
    raster, polygons = _fixture(tmp_path)
    with rasterio.open(raster, "r+") as dst:
        for index, name in enumerate(("B08", "B04", "B03", "B02"), start=1):
            dst.set_band_description(index, name)
    frame = labelled_raster_dataframe(
        str(raster), str(polygons), "class_id",
        feature_mode=FEATURE_MODE_HIGH_SPATIAL_ACCURACY,
    )
    assert frame["Reflectance_B02"].eq(4000).all()
    assert frame["Reflectance_B08"].eq(1000).all()


def test_validation_points_sample_sentinel_raster_and_skip_nodata(tmp_path):
    raster, _ = _fixture(tmp_path, descriptions=True, nodata=True)
    points = tmp_path / "points.shp"
    gpd.GeoDataFrame(
        {"class_id": ["missing", "water"]},
        geometry=[Point(0.5, 1.5), Point(1.5, 1.5)], crs="EPSG:3857",
    ).to_file(points)
    frame = labelled_point_raster_dataframe(
        str(raster), str(points), "class_id",
        feature_mode=FEATURE_MODE_HIGH_SPATIAL_ACCURACY,
    )
    assert list(frame["class_id"]) == ["water"]
    assert frame["Reflectance_B02"].iloc[0] == pytest.approx(1000)
    assert frame["NDVI"].notna().all()
