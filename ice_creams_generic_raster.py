"""Band-agnostic raster features and inference for user-trained models."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import geopandas as gpd
import numpy as np
import pandas as pd
import rasterio
from rasterio.features import rasterize
from rasterio.windows import Window, bounds as window_bounds, from_bounds, transform as window_transform
from shapely.geometry import box

from ice_creams_sensors import index_band_names


def polygon_attribute_columns(path: str) -> list[str]:
    """Read polygon attribute names without loading the full training layer."""
    vector = gpd.read_file(path, rows=1)
    if vector.geometry.name not in vector.columns:
        raise ValueError("The selected file has no geometry column.")
    columns = [str(name) for name in vector.columns if name != vector.geometry.name]
    if not columns:
        raise ValueError("The polygon file has no attribute columns.")
    return columns


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
    csv_columns = list(schema.get("csv_band_columns") or [])
    if any(name not in frame.columns for name in names) and len(csv_columns) == len(names):
        if all(column in frame.columns for column in csv_columns):
            frame = frame.rename(columns=dict(zip(csv_columns, names)))
    missing = [name for name in names if name not in frame.columns]
    if missing:
        raise ValueError(f"Missing raster features: {', '.join(missing)}")
    values = frame.loc[:, names].apply(pd.to_numeric, errors="coerce")
    if not np.isfinite(values.to_numpy()).all():
        raise ValueError("Raster features contain missing or nonfinite values.")
    index_bands = schema.get("index_bands") or {}
    indices = pd.DataFrame(index=values.index)
    nir_name = index_bands.get("nir")
    for index_name, other_name, reverse in (
        ("NDVI", index_bands.get("red"), False),
        ("NDWI", index_bands.get("green"), True),
    ):
        if nir_name and other_name:
            nir = values[nir_name]
            other = values[other_name]
            denominator = nir + other
            numerator = other - nir if reverse else nir - other
            indices[index_name] = numerator.div(denominator.where(denominator.ne(0), 1)).where(
                denominator.ne(0), 0.0
            )
    if not standardized:
        return pd.concat([values, indices], axis=1)
    row_min = values.min(axis=1)
    row_range = values.max(axis=1) - row_min
    standardized_values = values.sub(row_min, axis=0).div(row_range.where(row_range.gt(0), 1), axis=0)
    standardized_values.columns = [f"{name}_Standardized" for name in names]
    standardized_indices = indices.rename(columns={name: f"{name}_Standardized" for name in indices})
    return pd.concat([values, indices, standardized_values, standardized_indices], axis=1)


def configure_raster_schema_sensor(schema: dict[str, Any], sensor: dict[str, Any]) -> dict[str, Any]:
    """Bind generic raster bands and derived indices to the selected sensor."""
    bands = sensor.get("bands") or {}
    names = list(schema["feature_names"])
    if len(bands) != len(names):
        raise ValueError(
            f"Sensor '{sensor.get('name', '')}' defines {len(bands)} bands, "
            f"but the training raster has {len(names)} bands."
        )
    descriptions = list(schema.get("band_descriptions") or [])
    ordered_wavelengths = (
        [bands[description] for description in descriptions]
        if len(descriptions) == len(names) and set(descriptions) == set(bands)
        else list(bands.values())
    )
    matched = index_band_names({name: wavelength for name, wavelength in zip(names, ordered_wavelengths)})
    result = dict(schema)
    result["sensor_name"] = sensor["name"]
    result["band_wavelengths_nm"] = ordered_wavelengths
    result["index_bands"] = matched
    return result


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
        mask_vector = None
        mask_index = None
        if mask_polygon_path:
            mask_vector = gpd.read_file(mask_polygon_path)
            if mask_vector.empty or mask_vector.crs is None or raster.crs is None:
                raise ValueError("The apply mask and raster must have CRS information and valid features.")
            mask_vector = mask_vector.to_crs(raster.crs)
            mask_index = mask_vector.sindex
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        profile = raster.profile.copy()
        profile.pop("photometric", None)
        profile.pop("nbits", None)
        profile.update(
            driver="GTiff", count=2, dtype="float32", nodata=-1,
            compress="deflate", tiled=True, blockxsize=256, blockysize=256,
        )
        vocab = [str(value) for value in learner.dls.vocab]
        # Raster storage blocks can be only a few rows tall. Running fastai on
        # each such block repeatedly rebuilds its test data loader. Use larger,
        # bounded inference windows regardless of the TIFF's internal layout.
        tile_size = min(512, max(128, int(math.sqrt(4_000_000 / max(raster.count, 1)))))
        total_windows = math.ceil(raster.width / tile_size) * math.ceil(raster.height / tile_size)
        completed_windows = 0
        last_status_time = time.monotonic()
        if status_callback:
            status_callback(f"Classifying raster in {total_windows} inference windows")
        with rasterio.open(output, "w", **profile) as dst:
            dst.set_band_description(1, "Predicted_Class_ID")
            dst.set_band_description(2, "Predicted_Confidence")
            dst.update_tags(CLASS_LABELS=json.dumps({index + 1: name for index, name in enumerate(vocab)}))
            for row_off in range(0, raster.height, tile_size):
                for col_off in range(0, raster.width, tile_size):
                    window = Window(
                        col_off, row_off,
                        min(tile_size, raster.width - col_off),
                        min(tile_size, raster.height - row_off),
                    )
                    shape = (int(window.height), int(window.width))
                    classes = np.full(shape, -1, dtype=np.float32)
                    confidence = np.full(shape, -1, dtype=np.float32)
                    mask = None
                    if mask_vector is not None:
                        matches = mask_index.query(
                            box(*window_bounds(window, raster.transform)), predicate="intersects"
                        )
                        if len(matches):
                            mask = rasterize(
                                ((geometry, 1) for geometry in mask_vector.iloc[matches].geometry),
                                out_shape=shape,
                                transform=window_transform(window, raster.transform),
                                dtype="uint8",
                            ).astype(bool)
                        else:
                            mask = np.zeros(shape, dtype=bool)
                    if mask is None or mask.any():
                        values = np.ma.filled(
                            raster.read(indices, window=window, masked=True).astype(np.float32), np.nan
                        )
                        valid = np.isfinite(values).all(axis=0)
                        if mask is not None:
                            valid &= mask
                        if valid.any():
                            rows, cols = np.nonzero(valid)
                            frame = pd.DataFrame({
                                name: band_values
                                for name, band_values in zip(schema["feature_names"], values[:, rows, cols])
                            })
                            frame = prepare_generic_features(
                                frame, schema,
                                standardized=bool(model_metadata.get("sequence_use_standardized_reflectance")),
                            )
                            probabilities = predict_model_probabilities(
                                learner, frame, model_metadata, batch_size=65536
                            )
                            classes[rows, cols] = probabilities.argmax(dim=1).cpu().numpy() + 1
                            confidence[rows, cols] = probabilities.max(dim=1).values.cpu().numpy()
                    dst.write(classes, 1, window=window)
                    dst.write(confidence, 2, window=window)
                    completed_windows += 1
                    if progress_callback:
                        progress_callback(completed_windows / total_windows)
                    now = time.monotonic()
                    if status_callback and (completed_windows == total_windows or now - last_status_time >= 5):
                        status_callback(f"Classified {completed_windows}/{total_windows} raster windows")
                        last_status_time = now
    return str(output)
