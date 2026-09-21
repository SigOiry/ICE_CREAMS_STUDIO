"""Extract labelled Sentinel-2 pixels from a multiband raster and polygons."""

from __future__ import annotations

import re
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import from_bounds, transform as window_transform

from ice_creams_feature_modes import (
    FEATURE_MODE_HIGH_SPECTRAL_COMPLEXITY,
    RAW_BANDS_BY_MODE,
    SPECTRAL_RAW_BANDS,
    normalize_feature_mode,
    prepare_feature_dataframe,
    raw_column_name,
)

RASTER_SUFFIXES = {".tif", ".tiff"}
POLYGON_SUFFIXES = {".shp", ".gpkg", ".geojson", ".json"}


def _band_name(description: str | None) -> str | None:
    if not description:
        return None
    match = re.search(r"(?<![A-Z0-9])B(0[1-9]|1[0-2]|8A)(?![A-Z0-9])", description.upper())
    return f"B{match.group(1)}" if match else None


def _band_indices(raster: rasterio.io.DatasetReader, required: tuple[str, ...]) -> list[int]:
    names = [_band_name(value) for value in raster.descriptions]
    if any(names):
        if len(set(name for name in names if name)) != len([name for name in names if name]):
            raise ValueError("Raster band descriptions contain duplicate Sentinel-2 band names.")
        missing = [name for name in required if name not in names]
        if missing:
            raise ValueError(f"Raster band descriptions are missing required bands: {', '.join(missing)}")
        return [names.index(name) + 1 for name in required]
    if raster.count == len(SPECTRAL_RAW_BANDS):
        return [SPECTRAL_RAW_BANDS.index(name) + 1 for name in required]
    if raster.count == len(required):
        return list(range(1, raster.count + 1))
    raise ValueError(
        "Raster bands cannot be identified. Add Sentinel-2 band descriptions "
        f"or supply {len(SPECTRAL_RAW_BANDS)} bands in order {', '.join(SPECTRAL_RAW_BANDS)} "
        f"(or {len(required)} bands in mode order {', '.join(required)})."
    )


def labelled_raster_dataframe(
    raster_path: str,
    polygon_path: str,
    label_column: str,
    *,
    feature_mode: str,
) -> pd.DataFrame:
    """Return one row per labelled valid pixel with raw and derived model features.

    A pixel is included when its centre lies inside a polygon. Overlapping
    polygons with different labels are rejected rather than silently overwritten.
    """
    image = Path(raster_path).expanduser()
    polygons = Path(polygon_path).expanduser()
    if not image.is_file() or image.suffix.lower() not in RASTER_SUFFIXES:
        raise ValueError(f"Select an existing .tif or .tiff raster: {raster_path}")
    if not polygons.is_file() or polygons.suffix.lower() not in POLYGON_SUFFIXES:
        raise ValueError(f"Select an existing polygon file (.shp, .gpkg, .geojson): {polygon_path}")
    label_name = str(label_column).strip()
    if not label_name:
        raise ValueError("A polygon class column is required.")
    mode = normalize_feature_mode(feature_mode)
    required = RAW_BANDS_BY_MODE[mode]

    vector = gpd.read_file(polygons)
    if label_name not in vector.columns:
        raise ValueError(f"Class column '{label_name}' was not found in {polygons.name}.")
    if vector.empty:
        raise ValueError("The polygon file contains no features.")
    if vector.crs is None:
        raise ValueError("The polygon file has no CRS. Assign its true CRS before use.")
    labels = vector[label_name].astype("string").str.strip()
    invalid_labels = labels.isna() | labels.eq("")
    if invalid_labels.any():
        raise ValueError(f"{int(invalid_labels.sum())} polygon(s) have a missing class in '{label_name}'.")
    vector = vector.copy()
    vector[label_name] = labels.astype(str)
    if vector.geometry.isna().any() or vector.geometry.is_empty.any():
        raise ValueError("The polygon file contains empty geometries.")
    if not vector.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all():
        raise ValueError("All labelled geometries must be polygons or multipolygons.")
    if not vector.geometry.is_valid.all():
        raise ValueError("The polygon file contains invalid geometries. Repair them before use.")

    with rasterio.open(image) as raster:
        if raster.crs is None:
            raise ValueError("The raster has no CRS. Assign its true CRS before use.")
        indices = _band_indices(raster, required)
        vector = vector.to_crs(raster.crs)
        bounds = vector.total_bounds
        left = max(float(bounds[0]), raster.bounds.left)
        bottom = max(float(bounds[1]), raster.bounds.bottom)
        right = min(float(bounds[2]), raster.bounds.right)
        top = min(float(bounds[3]), raster.bounds.top)
        if left >= right or bottom >= top:
            raise ValueError("The labelled polygons do not overlap the raster.")
        window = from_bounds(left, bottom, right, top, raster.transform).round_offsets().round_lengths()
        window = window.intersection(rasterio.windows.Window(0, 0, raster.width, raster.height))
        transform = window_transform(window, raster.transform)
        shape = (int(window.height), int(window.width))
        assigned = np.zeros(shape, dtype=np.int32)
        classes: list[str] = []
        for label in vector[label_name].unique():
            geometries = vector.loc[vector[label_name] == label, "geometry"]
            class_mask = rasterize(
                ((geometry, 1) for geometry in geometries),
                out_shape=shape,
                transform=transform,
                dtype="uint8",
            ).astype(bool)
            if np.any(class_mask & (assigned != 0)):
                raise ValueError("Polygons with different classes cover the same raster pixel.")
            classes.append(str(label))
            assigned[class_mask] = len(classes)
        if not assigned.any():
            raise ValueError("No raster pixel centres fall inside the labelled polygons.")
        data = raster.read(indices, window=window, masked=True).astype(np.float32)
        values = np.ma.filled(data, np.nan)
        valid = (assigned > 0) & np.isfinite(values).all(axis=0)
        if not valid.any():
            raise ValueError("No labelled raster pixels have valid values in every required band.")
        rows, cols = np.nonzero(valid)
        pixel_values = values[:, rows, cols]
        # Apply uses 0-10000 reflectance; normalized floating-point rasters need the same scale.
        if np.issubdtype(np.dtype(raster.dtypes[0]), np.floating) and np.nanmax(pixel_values) <= 1.5:
            pixel_values = pixel_values * 10000.0
        xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
        frame = pd.DataFrame({
            label_name: [classes[code - 1] for code in assigned[rows, cols]],
            "Pixel_Row": rows + int(window.row_off),
            "Pixel_Col": cols + int(window.col_off),
            "Pixel_X": xs,
            "Pixel_Y": ys,
        })
        for band, band_values in zip(required, pixel_values):
            frame[raw_column_name(band)] = band_values
    features = prepare_feature_dataframe(
        frame,
        feature_mode=mode,
        context="Labelled raster",
        rebuild_standardised=True,
        rebuild_indices=True,
    )
    for name in features.columns:
        frame[name] = features[name]
    return frame
