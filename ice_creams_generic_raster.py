"""Band-agnostic raster features and inference for user-trained models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, bounds as window_bounds, from_bounds, transform as window_transform
from shapely.geometry import box


def raster_schema(raster: rasterio.io.DatasetReader) -> dict[str, Any]:
    """Capture the exact band contract used for later validation and apply."""
    if raster.count < 1:
        raise ValueError("A generic raster must have at least one band.")
    descriptions = [str(name).strip() if name else "" for name in raster.descriptions]
    if any(descriptions) and (not all(descriptions) or len(set(descriptions)) != len(descriptions)):
        raise ValueError("Raster band descriptions must be complete and unique, or all blank.")
    return {
        "band_count": raster.count,
        "band_descriptions": descriptions if all(descriptions) else [],
        "feature_names": [f"Band_{index}" for index in range(1, raster.count + 1)],
    }


def matching_band_indices(raster: rasterio.io.DatasetReader, schema: dict[str, Any]) -> list[int]:
    expected_count = int(schema["band_count"])
    expected_names = list(schema.get("band_descriptions") or [])
    if raster.count != expected_count:
        raise ValueError(f"Model expects {expected_count} raster bands; input has {raster.count}.")
    if expected_names:
        actual = [str(name).strip() if name else "" for name in raster.descriptions]
        if len(set(actual)) != len(actual) or set(actual) != set(expected_names):
            raise ValueError("Input raster band descriptions do not match the model's training bands.")
        return [actual.index(name) + 1 for name in expected_names]
    return list(range(1, expected_count + 1))


def prepare_generic_features(frame: pd.DataFrame, schema: dict[str, Any], *, standardized: bool = False) -> pd.DataFrame:
    names = list(schema["feature_names"])
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing raster features: {', '.join(missing)}")
    values = frame.loc[:, names].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Raster features contain missing or nonfinite values.")
    if not standardized:
        return values
    row_min = values.min(axis=1)
    row_range = values.max(axis=1) - row_min
    standardized_values = values.sub(row_min, axis=0).div(row_range.where(row_range.gt(0), 1), axis=0)
    standardized_values.columns = [f"{name}_Standardized" for name in names]
    return pd.concat([values, standardized_values], axis=1)


def labelled_generic_raster_dataframe(
    raster_path: str,
    polygon_path: str,
    label_column: str,
    expected_schema: dict[str, Any] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Extract polygon-labelled pixels, preserving input band values as-is."""
    image = Path(raster_path).expanduser()
    polygons = Path(polygon_path).expanduser()
    if not image.is_file() or image.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError(f"Select an existing multiband GeoTIFF: {raster_path}")
    if not polygons.is_file() or polygons.suffix.lower() not in {".shp", ".gpkg", ".geojson", ".json"}:
        raise ValueError(f"Select an existing polygon file: {polygon_path}")
    vector = gpd.read_file(polygons)
    if label_column not in vector.columns:
        raise ValueError(f"Class column '{label_column}' was not found in {polygons.name}.")
    if vector.empty or vector.crs is None:
        raise ValueError("Labelled polygons must contain features and a CRS.")
    labels = vector[label_column].astype("string").str.strip()
    if labels.isna().any() or labels.eq("").any():
        raise ValueError(f"Polygon class column '{label_column}' contains missing values.")
    vector = vector.copy()
    vector[label_column] = labels.astype(str)
    if vector.geometry.isna().any() or vector.geometry.is_empty.any():
        raise ValueError("Labelled polygons contain empty geometries.")
    if not vector.geometry.geom_type.isin(["Polygon", "MultiPolygon"]).all() or not vector.geometry.is_valid.all():
        raise ValueError("Labelled geometries must be valid polygons or multipolygons.")

    with rasterio.open(image) as raster:
        if raster.crs is None:
            raise ValueError("The raster has no CRS.")
        schema = expected_schema or raster_schema(raster)
        band_indices = matching_band_indices(raster, schema)
        vector = vector.to_crs(raster.crs)
        frames: list[pd.DataFrame] = []
        labels_order = list(vector[label_column].unique())
        # Process tiles so distant polygons do not require a huge in-memory window.
        left, bottom, right, top = vector.total_bounds
        crop = from_bounds(
            max(left, raster.bounds.left), max(bottom, raster.bounds.bottom),
            min(right, raster.bounds.right), min(top, raster.bounds.top), raster.transform,
        ) if left < raster.bounds.right and right > raster.bounds.left and bottom < raster.bounds.top and top > raster.bounds.bottom else None
        if crop is None:
            raise ValueError("The labelled polygons do not overlap the raster.")
        col_start = max(0, int(np.floor(crop.col_off)))
        row_start = max(0, int(np.floor(crop.row_off)))
        col_stop = min(raster.width, int(np.ceil(crop.col_off + crop.width)))
        row_stop = min(raster.height, int(np.ceil(crop.row_off + crop.height)))
        for row_start_tile in range(row_start, row_stop, 512):
            for col_start_tile in range(col_start, col_stop, 512):
                window = Window(col_start_tile, row_start_tile,
                                min(512, col_stop - col_start_tile), min(512, row_stop - row_start_tile))
                window_box = box(*window_bounds(window, raster.transform))
                subset_indices = vector.sindex.query(window_box, predicate="intersects")
                if not len(subset_indices):
                    continue
                subset = vector.iloc[subset_indices]
                shape = (int(window.height), int(window.width))
                transform = window_transform(window, raster.transform)
                assigned = np.zeros(shape, dtype=np.int32)
                for class_index, label in enumerate(labels_order, start=1):
                    geometries = subset.loc[subset[label_column] == label, "geometry"]
                    if geometries.empty:
                        continue
                    mask = rasterize(((geometry, 1) for geometry in geometries),
                                     out_shape=shape, transform=transform, dtype="uint8").astype(bool)
                    if np.any(mask & (assigned != 0)):
                        raise ValueError("Polygons with different classes cover the same raster pixel.")
                    assigned[mask] = class_index
                if not assigned.any():
                    continue
                values = np.ma.filled(raster.read(band_indices, window=window, masked=True).astype(np.float32), np.nan)
                valid = (assigned > 0) & np.isfinite(values).all(axis=0)
                if not valid.any():
                    continue
                rows, cols = np.nonzero(valid)
                xs, ys = rasterio.transform.xy(transform, rows, cols, offset="center")
                frame = pd.DataFrame({
                    label_column: [labels_order[code - 1] for code in assigned[rows, cols]],
                    "Pixel_Row": rows + row_start_tile,
                    "Pixel_Col": cols + col_start_tile,
                    "Pixel_X": xs,
                    "Pixel_Y": ys,
                })
                for name, band_values in zip(schema["feature_names"], values[:, rows, cols]):
                    frame[name] = band_values
                frames.append(frame)
    if not frames:
        raise ValueError("No valid labelled pixel centres were found in the raster.")
    return pd.concat(frames, ignore_index=True), schema


def classify_generic_raster(
    input_path: str,
    output_path: str,
    learner: Any,
    model_metadata: dict[str, Any],
    *,
    mask_polygon_path: str | None = None,
    status_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> str:
    """Apply a trained generic model tile by tile to a compatible GeoTIFF."""
    from ice_creams_model_families import predict_model_probabilities

    schema = model_metadata.get("raster_schema") or {}
    if not schema:
        raise ValueError("The selected model has no generic raster band schema.")
    source = Path(input_path)
    if source.suffix.lower() not in {".tif", ".tiff"}:
        raise ValueError("Generic raster models can only be applied to GeoTIFF images.")
    with rasterio.open(source) as raster:
        indices = matching_band_indices(raster, schema)
        mask_geometries = None
        if mask_polygon_path:
            mask_vector = gpd.read_file(mask_polygon_path)
            if mask_vector.empty or mask_vector.crs is None or raster.crs is None:
                raise ValueError("The apply mask and raster must have CRS information and valid features.")
            mask_geometries = list(mask_vector.to_crs(raster.crs).geometry)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        profile = raster.profile.copy()
        profile.update(driver="GTiff", count=2, dtype="float32", nodata=-1, compress="deflate")
        vocab = [str(value) for value in learner.dls.vocab]
        windows = list(raster.block_windows(1))
        with rasterio.open(output, "w", **profile) as dst:
            dst.set_band_description(1, "Predicted_Class_ID")
            dst.set_band_description(2, "Predicted_Confidence")
            dst.update_tags(CLASS_LABELS=json.dumps({index + 1: name for index, name in enumerate(vocab)}))
            for position, (_, window) in enumerate(windows, start=1):
                values = np.ma.filled(raster.read(indices, window=window, masked=True).astype(np.float32), np.nan)
                valid = np.isfinite(values).all(axis=0)
                if mask_geometries is not None:
                    mask = rasterize(((geometry, 1) for geometry in mask_geometries),
                                     out_shape=valid.shape,
                                     transform=window_transform(window, raster.transform),
                                     dtype="uint8").astype(bool)
                    valid &= mask
                classes = np.full(valid.shape, -1, dtype=np.float32)
                confidence = np.full(valid.shape, -1, dtype=np.float32)
                if valid.any():
                    rows, cols = np.nonzero(valid)
                    frame = pd.DataFrame({name: band_values for name, band_values
                                          in zip(schema["feature_names"], values[:, rows, cols])})
                    frame = prepare_generic_features(
                        frame, schema,
                        standardized=bool(model_metadata.get("sequence_use_standardized_reflectance")),
                    )
                    probabilities = predict_model_probabilities(
                        learner, frame, model_metadata, batch_size=4096
                    )
                    classes[rows, cols] = probabilities.argmax(dim=1).cpu().numpy() + 1
                    confidence[rows, cols] = probabilities.max(dim=1).values.cpu().numpy()
                dst.write(classes, 1, window=window)
                dst.write(confidence, 2, window=window)
                if progress_callback:
                    progress_callback(position / len(windows))
                if status_callback and (position == len(windows) or position % 20 == 0):
                    status_callback(f"Classified {position}/{len(windows)} raster tiles")
    return str(output)
