#!/usr/bin/env python3
# coding: utf-8
"""
Apply ICE CREAMS to masked Sentinel-2 or 12-band TIFF imagery.

Takes a trained Neural Network model (saved with pickle) and runs this over a
Sentinel-2 SAFE scene, a zipped SAFE scene, or a multi-band GeoTIFF using xarray.

Author: Bede Ffinian Rowe Davies 
Date: 2023-03-30 edited 2025-09-08

"""
import argparse
import gc
import glob
import os
import re
import shutil
import sys
import tempfile
import time
import zipfile
from contextlib import contextmanager, nullcontext
from datetime import datetime
from pathlib import Path
from typing import Callable

import dask
import numpy
import pandas


def _prepend_env_path(path_value: str) -> None:
    """Prepend a directory to PATH once if it exists."""
    if not os.path.isdir(path_value):
        return

    existing_parts = os.environ.get("PATH", "").split(os.pathsep)
    normalized_existing = {os.path.normcase(part) for part in existing_parts if part}
    if os.path.normcase(path_value) not in normalized_existing:
        os.environ["PATH"] = os.pathsep.join([path_value, *existing_parts]) if existing_parts else path_value


def _existing_first(paths: list[str]) -> str | None:
    """Return the first existing directory path from a candidate list."""
    for candidate_path in paths:
        if candidate_path and os.path.isdir(candidate_path):
            return candidate_path
    return None


def _env_path_missing_or_invalid(variable_name: str) -> bool:
    """Return True when env var is unset or does not point to an existing directory."""
    env_value = os.environ.get(variable_name, "")
    if not env_value:
        return True
    return not any(os.path.isdir(path_part) for path_part in env_value.split(os.pathsep) if path_part)


def _runtime_roots() -> list[str]:
    """Build a list of likely runtime roots for source and frozen execution."""
    roots: list[str] = []

    meipass_root = getattr(sys, "_MEIPASS", None)
    if isinstance(meipass_root, str):
        roots.extend(
            [
                meipass_root,
                os.path.join(meipass_root, "_internal"),
            ]
        )

    executable_dir = os.path.dirname(os.path.abspath(sys.executable))
    module_dir = os.path.dirname(os.path.abspath(__file__))
    roots.extend(
        [
            executable_dir,
            os.path.join(executable_dir, "_internal"),
            module_dir,
            os.path.join(module_dir, "_internal"),
            sys.prefix,
        ]
    )

    unique_roots: list[str] = []
    seen_roots: set[str] = set()
    for root in roots:
        normalized_root = os.path.normcase(os.path.abspath(root))
        if normalized_root not in seen_roots:
            seen_roots.add(normalized_root)
            unique_roots.append(root)
    return unique_roots


def _configure_gdal_data() -> None:
    """
    Set GDAL/PROJ runtime paths for Conda and frozen (PyInstaller) builds.

    The packaged app bundles GDAL and PROJ under `_internal/gdal_data` and
    `_internal/proj_data`. Conda installs usually expose these under
    `Library/share/gdal` and `Library/share/proj`.
    """
    runtime_roots = _runtime_roots()

    path_candidates = []
    for root in runtime_roots:
        path_candidates.extend(
            [
                os.path.join(root, "Library", "bin"),
                os.path.join(root, "bin"),
                root,
            ]
        )
    for path_candidate in path_candidates:
        _prepend_env_path(path_candidate)

    if _env_path_missing_or_invalid("GDAL_DATA"):
        gdal_data_candidates: list[str] = []
        for root in runtime_roots:
            gdal_data_candidates.extend(
                [
                    os.path.join(root, "gdal_data"),
                    os.path.join(root, "share", "gdal"),
                    os.path.join(root, "Library", "share", "gdal"),
                ]
            )
        gdal_data_path = _existing_first(gdal_data_candidates)
        if gdal_data_path:
            os.environ["GDAL_DATA"] = gdal_data_path

    if _env_path_missing_or_invalid("PROJ_LIB"):
        proj_data_candidates: list[str] = []
        for root in runtime_roots:
            proj_data_candidates.extend(
                [
                    os.path.join(root, "proj_data"),
                    os.path.join(root, "proj"),
                    os.path.join(root, "share", "proj"),
                    os.path.join(root, "Library", "share", "proj"),
                ]
            )
        proj_data_path = _existing_first(proj_data_candidates)
        if proj_data_path:
            os.environ["PROJ_LIB"] = proj_data_path

    if _env_path_missing_or_invalid("GDAL_DRIVER_PATH"):
        gdal_plugin_candidates: list[str] = []
        for root in runtime_roots:
            gdal_plugin_candidates.extend(
                [
                    os.path.join(root, "gdalplugins"),
                    os.path.join(root, "lib", "gdalplugins"),
                    os.path.join(root, "Library", "lib", "gdalplugins"),
                ]
            )
        gdal_plugin_path = _existing_first(gdal_plugin_candidates)
        if gdal_plugin_path:
            os.environ["GDAL_DRIVER_PATH"] = gdal_plugin_path


_configure_gdal_data()

import geopandas
import xarray
import rasterio
import rioxarray
from dask.diagnostics import ProgressBar
from fastai.tabular.all import load_learner

from ice_creams_feature_modes import (
    FEATURE_MODE_HIGH_SPECTRAL_COMPLEXITY,
    RAW_BANDS_BY_MODE,
    SPECTRAL_RAW_BANDS,
    prepare_feature_dataframe,
    raw_column_name,
)
from ice_creams_model_families import (
    extract_model_metadata,
    predict_model_probabilities,
    prepare_sequence_feature_dataframe,
    spectral_cnn_sequence_input_label,
)

## Update .pkl file as new version becomes available.

DEFAULT_FASTAI_MODEL = os.path.join(
    os.path.dirname(__file__), "ICECREAMS_V1_1.pkl"
)

CLASSES_NUMBER_ID_DICT = {
    1: "Bare Sediment",
    2: "Sand",
    3: "Chlorophyceae",
    4: "Magnoliopsida",
    5: "Bacillariophyceae",
    6: "Phaeophyceae",
    7: "Florideophyceae",
    8: "Water",
}

OUT_CLASS_QGIS_STYLE: dict[int, tuple[str, str]] = {
    1: ("Bare Sediment", "#3e3d1b"),
    2: ("Sand", "#3e3d1b"),
    3: ("Chlorophyceae", "#99ff13"),
    4: ("Magnoliopsida", "#09861a"),
    5: ("Bacillariophyceae", "#ff9e36"),
    6: ("Phaeophyceae", "#a40205"),
    7: ("Florideophyceae", "#ff0004"),
    8: ("Water", "#3d2fff"),
}

DEFAULT_DASK_WORKERS = max(1, os.cpu_count() or 1)
DEFAULT_CLASSIFICATION_BATCH_SIZE = 8192
MASK_EXTENT_BUFFER_DISTANCE = 60.0
CLASS4_TO_CLASS5_NDVI_THRESHOLD = 0.25
MODEL_CLASS_INDEX_4 = 3
MODEL_CLASS_INDEX_5 = 4
_MODEL_CACHE: dict[str, tuple[float, object]] = {}
_INFERENCE_DEVICE: tuple | None = None  # (device, label_str) — populated on first use
S2_RAW_BAND_FILE_PATTERNS: dict[str, str] = {
    "B01": "*B01_60m.jp2",
    "B02": "*B02_10m.jp2",
    "B03": "*B03_10m.jp2",
    "B04": "*B04_10m.jp2",
    "B05": "*B05_20m.jp2",
    "B06": "*B06_20m.jp2",
    "B07": "*B07_20m.jp2",
    "B08": "*B08_10m.jp2",
    "B8A": "*B8A_20m.jp2",
    "B09": "*B09_60m.jp2",
    "B11": "*B11_20m.jp2",
    "B12": "*B12_20m.jp2",
}
S2_NATIVE_10M_BANDS = {"B02", "B03", "B04", "B08"}
MULTIBAND_TIFF_SUFFIXES = {".tif", ".tiff"}
TIFF_RAW_BAND_ORDER = SPECTRAL_RAW_BANDS


# Substrings (lower-case) that identify integrated or software-renderer adapters.
# Discrete GPUs are preferred; these are skipped when a better option exists.
_INTEGRATED_GPU_MARKERS = (
    "intel uhd",
    "intel iris",
    "intel hd graphics",
    "intel(r) uhd",
    "intel(r) iris",
    "intel(r) hd",
    "amd radeon(tm) graphics",   # AMD integrated (Ryzen APU, no model number)
    "amd radeon(tm) vega",       # older AMD APU integrated
    "microsoft basic render",    # Windows software renderer
    "microsoft warp",            # Windows software rasterizer
)


def _best_directml_index() -> int:
    """Return the DirectML adapter index that is most likely a discrete GPU.

    Windows enumerates adapters in DXGI order — the integrated GPU is almost
    always index 0.  We score each adapter by whether its name matches known
    integrated-GPU patterns and return the highest-scoring index, falling back
    to 0 when no name information is available.
    """
    import torch_directml

    count = torch_directml.device_count()
    if count <= 1:
        return 0

    has_name = hasattr(torch_directml, "device_name")
    best_score, best_idx = -1, 0
    for i in range(count):
        name = (torch_directml.device_name(i) if has_name else "").lower()
        is_integrated = any(marker in name for marker in _INTEGRATED_GPU_MARKERS)
        score = 0 if is_integrated else 1
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx


def _detect_inference_device() -> tuple:
    """Detect the best available compute device for PyTorch inference.

    Priority: NVIDIA CUDA → AMD/Intel DirectML → CPU.
    For CUDA, the GPU with the most VRAM is chosen (avoids picking an
    integrated or low-end secondary card on multi-GPU machines).
    For DirectML, integrated adapters are skipped in favour of discrete ones.
    Result is cached for the lifetime of the process so detection runs only once.
    Returns (device, human-readable label).
    """
    global _INFERENCE_DEVICE
    if _INFERENCE_DEVICE is not None:
        return _INFERENCE_DEVICE

    import torch

    if torch.cuda.is_available():
        # Pick the CUDA device with the most dedicated VRAM.
        best_idx = max(
            range(torch.cuda.device_count()),
            key=lambda i: torch.cuda.get_device_properties(i).total_memory,
        )
        device = torch.device(f"cuda:{best_idx}")
        label = f"CUDA ({torch.cuda.get_device_name(best_idx)})"
        _INFERENCE_DEVICE = (device, label)
        return _INFERENCE_DEVICE

    try:
        import torch_directml  # optional: pip install torch-directml
        if torch_directml.device_count() > 0:
            best_idx = _best_directml_index()
            device = torch_directml.device(best_idx)
            device_name = (
                torch_directml.device_name(best_idx)
                if hasattr(torch_directml, "device_name")
                else "AMD/Intel GPU"
            )
            label = f"DirectML ({device_name})"
            _INFERENCE_DEVICE = (device, label)
            return _INFERENCE_DEVICE
    except Exception:
        pass

    n_threads = os.cpu_count() or 1
    torch.set_num_threads(n_threads)
    device = torch.device("cpu")
    label = f"CPU ({n_threads} thread{'s' if n_threads != 1 else ''})"
    _INFERENCE_DEVICE = (device, label)
    return _INFERENCE_DEVICE


def _optimal_batch_size(device) -> int:
    """Return an inference batch size suited to the compute device.

    GPU devices benefit from very large batches to maximise kernel utilisation.
    CPU inference still benefits from larger batches vs the 8 192 default by
    reducing DataLoader iteration overhead.
    """
    try:
        device_type = device.type.lower()
    except AttributeError:
        device_type = str(device).lower()
    # "cuda" = NVIDIA, "privateuseone" = torch-directml (AMD/Intel)
    if "cuda" in device_type or "privateuseone" in device_type:
        return 131072
    return 65536
_S2_SCENE_ID_PATTERN = re.compile(
    r"^S2[A-Z]_MSIL[12][AC]?_\d{8}T\d{6}_N\d{4}_R\d{3}_T[0-9A-Z]{5}_\d{8}T\d{6}$",
    re.IGNORECASE,
)
_ZIP_SCENE_CACHE: dict[str, tuple[int, float, list[str], bool]] = {}
_SCENE_BATCH_CACHE_TTL_SECONDS = 30.0
_SCENE_BATCH_CACHE: dict[str, tuple[float, tuple[str, int, int], dict[str, object]]] = {}
_SCENE_STATUS_FOLDERS = {
    "done": "Done",
    "failed": "Failed",
}
_SCENE_STATUS_FOLDER_NAMES = {
    folder_name.upper() for folder_name in _SCENE_STATUS_FOLDERS.values()
}


def _console_log(message: str, enabled: bool = True) -> None:
    """Print to console only when verbose output is enabled."""
    if enabled:
        print(message, flush=True)


def _emit_status(status_callback: Callable[[str], None] | None, message: str) -> None:
    """Safely emit a status update for CLI or UI callers."""
    if status_callback is not None:
        status_callback(message)


def _emit_progress(progress_callback: Callable[[float], None] | None, value: float) -> None:
    """Safely emit bounded progress values between 0 and 1."""
    if progress_callback is not None:
        progress_callback(max(0.0, min(1.0, value)))


def _format_bounds(bounds: tuple[float, float, float, float]) -> str:
    """Return a compact string representation of raster/vector bounds."""
    xmin, ymin, xmax, ymax = bounds
    return f"xmin={xmin:.3f}, ymin={ymin:.3f}, xmax={xmax:.3f}, ymax={ymax:.3f}"


def _bounds_match(
    left_bounds: tuple[float, float, float, float],
    right_bounds: tuple[float, float, float, float],
    tolerance: float,
) -> bool:
    """Return True when all four bounds match within the provided tolerance."""
    return all(
        numpy.isclose(left_value, right_value, atol=tolerance, rtol=0.0)
        for left_value, right_value in zip(left_bounds, right_bounds)
    )


def _bounds_overlap(
    left_bounds: tuple[float, float, float, float],
    right_bounds: tuple[float, float, float, float],
    tolerance: float,
) -> bool:
    """Return True when two bounding boxes overlap within the provided tolerance."""
    left_xmin, left_ymin, left_xmax, left_ymax = left_bounds
    right_xmin, right_ymin, right_xmax, right_ymax = right_bounds
    return (
        left_xmin <= right_xmax + tolerance
        and left_xmax >= right_xmin - tolerance
        and left_ymin <= right_ymax + tolerance
        and left_ymax >= right_ymin - tolerance
    )


def _bounds_within(
    inner_bounds: tuple[float, float, float, float],
    outer_bounds: tuple[float, float, float, float],
    tolerance: float,
) -> bool:
    """Return True when one bounding box is fully contained within another."""
    inner_xmin, inner_ymin, inner_xmax, inner_ymax = inner_bounds
    outer_xmin, outer_ymin, outer_xmax, outer_ymax = outer_bounds
    return (
        inner_xmin >= outer_xmin - tolerance
        and inner_ymin >= outer_ymin - tolerance
        and inner_xmax <= outer_xmax + tolerance
        and inner_ymax <= outer_ymax + tolerance
    )


def _bounds_intersection(
    left_bounds: tuple[float, float, float, float],
    right_bounds: tuple[float, float, float, float],
) -> tuple[float, float, float, float] | None:
    """Return the intersection of two bounding boxes, or None when they do not overlap."""
    intersection = (
        max(left_bounds[0], right_bounds[0]),
        max(left_bounds[1], right_bounds[1]),
        min(left_bounds[2], right_bounds[2]),
        min(left_bounds[3], right_bounds[3]),
    )
    if intersection[0] >= intersection[2] or intersection[1] >= intersection[3]:
        return None
    return intersection


def _expand_bounds(
    bounds: tuple[float, float, float, float],
    pad_x: float,
    pad_y: float,
) -> tuple[float, float, float, float]:
    """Expand bounding box symmetrically by the provided padding."""
    xmin, ymin, xmax, ymax = bounds
    return (xmin - pad_x, ymin - pad_y, xmax + pad_x, ymax + pad_y)


def _clip_xarray_to_bounds(
    data: xarray.Dataset | xarray.DataArray,
    clip_bounds: tuple[float, float, float, float],
) -> xarray.Dataset | xarray.DataArray:
    """Clip an xarray object to map bounds using coordinate selection."""
    xmin, ymin, xmax, ymax = clip_bounds
    x_values = data.coords["x"].values
    y_values = data.coords["y"].values

    if x_values[0] <= x_values[-1]:
        clipped = data.sel(x=slice(xmin, xmax))
    else:
        clipped = data.sel(x=slice(xmax, xmin))

    if y_values[0] <= y_values[-1]:
        clipped = clipped.sel(y=slice(ymin, ymax))
    else:
        clipped = clipped.sel(y=slice(ymax, ymin))

    return clipped


def _align_mask_vector_to_scene(
    mask_vector: geopandas.GeoDataFrame,
    mask_vector_file: str,
    scene_data: xarray.Dataset | xarray.DataArray,
    verbose_console: bool = True,
) -> geopandas.GeoDataFrame:
    """
    Reproject a mask vector to the scene CRS, then ensure it fits the scene.

    Masks are allowed to cover only part of the image. They must overlap the
    scene extent, but they may extend beyond the image boundary. In that case,
    only the overlapping area contributes to the rasterized mask. Extent
    comparisons allow up to one output pixel of tolerance to absorb small
    floating-point differences introduced by reprojection.
    """
    if mask_vector.crs is None:
        raise ValueError(f"Mask file {mask_vector_file} has no CRS defined.")

    scene_crs = scene_data.rio.crs
    if scene_crs is None:
        raise ValueError("The input Sentinel-2 scene has no CRS defined.")

    mask_vector = mask_vector.loc[mask_vector.geometry.notna()].copy()
    if mask_vector.empty:
        raise ValueError(f"Mask file {mask_vector_file} does not contain any geometries.")

    if mask_vector.crs != scene_crs:
        original_crs = mask_vector.crs
        _console_log(
            f"Reprojecting mask from {original_crs} to {scene_crs}",
            verbose_console,
        )
        mask_vector = mask_vector.to_crs(scene_crs)

    mask_bounds = tuple(float(value) for value in mask_vector.total_bounds)
    scene_bounds = tuple(float(value) for value in scene_data.rio.bounds(recalc=True))
    scene_resolution = scene_data.rio.resolution()
    extent_tolerance = max(
        abs(float(scene_resolution[0])),
        abs(float(scene_resolution[1])),
    )
    if not numpy.isfinite(extent_tolerance) or extent_tolerance <= 0:
        extent_tolerance = 1e-6

    if not _bounds_overlap(mask_bounds, scene_bounds, tolerance=extent_tolerance):
        raise ValueError(
            "Mask extent does not overlap the Sentinel-2 scene extent after CRS alignment. "
            f"Mask extent: {_format_bounds(mask_bounds)}. "
            f"Scene extent: {_format_bounds(scene_bounds)}. "
            f"Allowed tolerance: {extent_tolerance:.3f} map units."
        )

    if not _bounds_within(mask_bounds, scene_bounds, tolerance=extent_tolerance):
        _console_log(
            "Mask extent extends outside the Sentinel-2 scene extent after CRS alignment. "
            "Continuing with the overlapping area inside the image extent.",
            verbose_console,
        )

    return mask_vector


def _close_xarray_resources(resources: list[object]) -> None:
    """Close xarray-backed resources once, ignoring already-released handles."""
    seen_resource_ids: set[int] = set()
    for resource in resources:
        if resource is None:
            continue
        resource_id = id(resource)
        if resource_id in seen_resource_ids:
            continue
        seen_resource_ids.add(resource_id)
        close_method = getattr(resource, "close", None)
        if callable(close_method):
            try:
                close_method()
            except Exception:
                continue


def _open_s2_band(
    raster_path: str,
    clip_bounds: tuple[float, float, float, float] | None = None,
    cleanup_resources: list[object] | None = None,
) -> xarray.DataArray:
    """Open one Sentinel-2 band and optionally crop it to buffered mask bounds."""
    band = rioxarray.open_rasterio(raster_path, chunks={"x": 512, "y": 512})
    if cleanup_resources is not None:
        cleanup_resources.append(band)
    if clip_bounds is None:
        return band

    clipped_band = _clip_xarray_to_bounds(band, clip_bounds)
    if cleanup_resources is not None:
        cleanup_resources.append(clipped_band)
    if clipped_band.sizes.get("x", 0) == 0 or clipped_band.sizes.get("y", 0) == 0:
        raise ValueError(
            "Buffered mask extent produced an empty raster crop. "
            f"Requested crop extent: {_format_bounds(clip_bounds)}."
        )
    return clipped_band


def _open_multiband_raster(
    raster_path: str,
    clip_bounds: tuple[float, float, float, float] | None = None,
    cleanup_resources: list[object] | None = None,
) -> xarray.DataArray:
    """Open one multi-band raster and optionally crop it to buffered mask bounds."""
    raster = rioxarray.open_rasterio(raster_path, chunks={"x": 512, "y": 512}, masked=True)
    if cleanup_resources is not None:
        cleanup_resources.append(raster)
    if clip_bounds is None:
        return raster

    clipped_raster = _clip_xarray_to_bounds(raster, clip_bounds)
    if cleanup_resources is not None:
        cleanup_resources.append(clipped_raster)
    if clipped_raster.sizes.get("x", 0) == 0 or clipped_raster.sizes.get("y", 0) == 0:
        raise ValueError(
            "Buffered mask extent produced an empty raster crop. "
            f"Requested crop extent: {_format_bounds(clip_bounds)}."
        )
    return clipped_raster


def _hex_to_rgba(hex_color: str, alpha: int = 255) -> tuple[int, int, int, int]:
    """Convert #RRGGBB color to an RGBA tuple."""
    color_value = hex_color.strip().lstrip("#")
    if len(color_value) != 6:
        raise ValueError(f"Expected #RRGGBB color, got: {hex_color}")
    red = int(color_value[0:2], 16)
    green = int(color_value[2:4], 16)
    blue = int(color_value[4:6], 16)
    return (red, green, blue, alpha)


def _apply_out_class_qgis_style(
    output_gtiff: str,
    status_callback: Callable[[str], None] | None = None,
) -> None:
    """
    Attach class colors and labels to the Out_Class band for QGIS rendering.

    This updates only band 1, which is Out_Class in non-debug exports.
    """
    band_index = 1
    category_names = ["NoData"] + [
        OUT_CLASS_QGIS_STYLE[class_id][0]
        for class_id in sorted(OUT_CLASS_QGIS_STYLE)
    ]
    colormap = {0: (0, 0, 0, 0)}
    for class_id, (_, hex_color) in OUT_CLASS_QGIS_STYLE.items():
        colormap[class_id] = _hex_to_rgba(hex_color)

    style_tags = {
        "CLASS_SCHEMA": "ICE_CREAMS_Out_Class_v1",
        "CLASS_COUNT": str(len(OUT_CLASS_QGIS_STYLE)),
        "CATEGORY_NAMES": "|".join(category_names),
    }
    for class_id, (class_name, hex_color) in OUT_CLASS_QGIS_STYLE.items():
        style_tags[f"CLASS_{class_id}"] = class_name
        style_tags[f"CLASS_{class_id}_COLOR"] = hex_color

    try:
        with rasterio.open(output_gtiff, "r+", IGNORE_COG_LAYOUT_BREAK="YES") as out_raster:
            if out_raster.count < band_index:
                return
            out_raster.set_band_description(band_index, "Out_Class")
            out_raster.write_colormap(band_index, colormap)
            out_raster.update_tags(band_index, **style_tags)
            if out_raster.count >= 2:
                out_raster.set_band_description(2, "Class_Probs")
            if out_raster.count >= 3:
                out_raster.set_band_description(3, "Seagrass_Cover")
            if out_raster.count >= 4:
                out_raster.set_band_description(4, "NDVI")
        _emit_status(status_callback, "Applied QGIS class colors and labels to Out_Class band")
    except Exception as exc:  # noqa: BLE001 - styling should not fail the main workflow.
        _emit_status(
            status_callback,
            f"Warning: could not apply Out_Class QGIS style metadata ({exc})",
        )


def _cleanup_temp_dir(
    temp_dir: str,
    status_callback: Callable[[str], None] | None = None,
) -> None:
    """Delete one extracted temp scene before the next batch item starts."""
    cleanup_path = os.path.abspath(temp_dir)
    for attempt in range(6):
        try:
            shutil.rmtree(cleanup_path)
            return
        except FileNotFoundError:
            return
        except OSError:
            if attempt == 5:
                _emit_status(
                    status_callback,
                    f"Windows is still holding a file lock; keeping extracted scene at {cleanup_path}",
                )
                return
            time.sleep(0.35 * (attempt + 1))


def _load_cached_learner(
    saved_model: str,
    status_callback: Callable[[str], None] | None = None,
):
    """Load a FastAI learner, place it on the best available device, and cache it."""
    import torch

    model_path = os.path.abspath(saved_model)
    model_mtime = os.path.getmtime(model_path)
    cached_entry = _MODEL_CACHE.get(model_path)
    if cached_entry and cached_entry[0] == model_mtime:
        return cached_entry[1]

    # Always load to CPU first so the file works regardless of the device it
    # was originally saved on (avoids CUDA/CPU mismatch errors on load).
    learner = load_learner(model_path, cpu=True)

    device, device_label = _detect_inference_device()
    _emit_status(status_callback, f"Inference device: {device_label}")

    try:
        learner.dls.device = device
        learner.model = learner.model.to(device)
    except Exception as exc:
        _emit_status(
            status_callback,
            f"Warning: could not move model to {device_label} ({exc}); falling back to CPU",
        )
        device = torch.device("cpu")
        learner.dls.device = device
        learner.model = learner.model.to(device)

    # JIT-compile the model for additional throughput where supported.
    # torch.compile() is lazy: the actual C++ kernel compilation happens at the
    # first forward pass (inside get_preds), not here.  On Windows the Inductor
    # backend requires MSVC's cl.exe; if it is absent the deferred error would
    # surface mid-inference and kill the run.  We check for cl.exe first so the
    # skip happens silently before any work is done.
    _compile_supported = sys.platform != "win32" or bool(shutil.which("cl"))
    if _compile_supported:
        try:
            learner.model = torch.compile(learner.model)
            _emit_status(status_callback, "Model JIT-compiled with torch.compile()")
        except Exception:
            pass

    _MODEL_CACHE[model_path] = (model_mtime, learner)
    return learner


def _locate_safe_directory(search_root: str) -> str:
    """Find the first valid .SAFE directory inside a folder tree."""
    root_path = Path(search_root)
    if _is_safe_directory_path(root_path):
        return str(root_path)

    safe_directories = sorted(
        path for path in root_path.rglob("*") if _is_safe_directory_path(path)
    )
    if safe_directories:
        return str(safe_directories[0])

    manifest_directories = sorted(
        manifest.parent
        for manifest in root_path.rglob("manifest.safe")
        if (manifest.parent / "GRANULE").exists()
    )
    if manifest_directories:
        return str(manifest_directories[0])

    raise FileNotFoundError(
        f"No Sentinel-2 .SAFE directory was found in {search_root}"
    )


def _looks_like_sentinel_scene_id(scene_id: str) -> bool:
    """Return True when the scene identifier matches Sentinel-2 naming."""
    return bool(_S2_SCENE_ID_PATTERN.match(scene_id.strip()))


def _strip_scene_suffixes(scene_name: str) -> str:
    """Remove supported archive and SAFE suffixes from a scene name."""
    normalized_name = scene_name.strip()
    if normalized_name.lower().endswith(".zip"):
        normalized_name = normalized_name[:-4]
    if normalized_name.lower().endswith(".tiff"):
        normalized_name = normalized_name[:-5]
    elif normalized_name.lower().endswith(".tif"):
        normalized_name = normalized_name[:-4]
    if normalized_name.upper().endswith(".SAFE"):
        normalized_name = normalized_name[:-5]
    return normalized_name


def _scene_id_from_filename_hint(path_value: Path) -> str | None:
    """Infer a Sentinel-2 scene identifier from the filename when possible."""
    hinted_scene_id = _strip_scene_suffixes(path_value.name)
    if not hinted_scene_id:
        return None
    if not _looks_like_sentinel_scene_id(hinted_scene_id):
        return None
    return hinted_scene_id


def _scene_path_cache_key(path_value: Path) -> str:
    """Return a stable cache key for a filesystem path."""
    try:
        return str(path_value.resolve())
    except OSError:
        return str(path_value.absolute())


def _scene_batch_cache_signature(input_path: Path) -> tuple[str, int, int]:
    """Build a lightweight signature for short-lived scene discovery caching."""
    path_stat = input_path.stat()
    path_kind = "dir" if input_path.is_dir() else "file"
    return (path_kind, int(path_stat.st_mtime_ns), int(getattr(path_stat, "st_size", 0)))


def _get_cached_scene_batch_info(input_path: Path) -> dict[str, object] | None:
    """Return a recent cached discovery result for the provided input path."""
    cache_key = _scene_path_cache_key(input_path)
    cached_entry = _SCENE_BATCH_CACHE.get(cache_key)
    if not cached_entry:
        return None

    cached_at, cached_signature, cached_batch_info = cached_entry
    if (time.monotonic() - cached_at) > _SCENE_BATCH_CACHE_TTL_SECONDS:
        _SCENE_BATCH_CACHE.pop(cache_key, None)
        return None

    try:
        current_signature = _scene_batch_cache_signature(input_path)
    except OSError:
        _SCENE_BATCH_CACHE.pop(cache_key, None)
        return None

    if cached_signature != current_signature:
        _SCENE_BATCH_CACHE.pop(cache_key, None)
        return None

    return cached_batch_info


def _store_scene_batch_info(input_path: Path, batch_info: dict[str, object]) -> dict[str, object]:
    """Store a discovery result in the short-lived batch cache and return it."""
    try:
        cache_signature = _scene_batch_cache_signature(input_path)
    except OSError:
        return batch_info

    _SCENE_BATCH_CACHE[_scene_path_cache_key(input_path)] = (
        time.monotonic(),
        cache_signature,
        batch_info,
    )
    return batch_info


def _extract_zip_safe_scene_ids(
    zip_path: Path,
    *,
    allow_filename_hint: bool = False,
) -> list[str]:
    """
    Return Sentinel-2 scene identifiers found inside a zip archive.

    The archive is considered valid when at least one *.SAFE folder contains
    both manifest.safe and a GRANULE directory.
    """
    try:
        zip_stat = zip_path.stat()
    except OSError:
        return []

    cache_key = _scene_path_cache_key(zip_path)
    cached_entry = _ZIP_SCENE_CACHE.get(cache_key)
    if (
        cached_entry
        and cached_entry[0] == zip_stat.st_size
        and cached_entry[1] == zip_stat.st_mtime
    ):
        if cached_entry[3] or allow_filename_hint:
            return list(cached_entry[2])

    if allow_filename_hint:
        hinted_scene_id = _scene_id_from_filename_hint(zip_path)
        if hinted_scene_id:
            _ZIP_SCENE_CACHE[cache_key] = (
                zip_stat.st_size,
                zip_stat.st_mtime,
                [hinted_scene_id],
                False,
            )
            return [hinted_scene_id]

    manifest_safe_dirs: set[str] = set()
    granule_safe_dirs: set[str] = set()
    try:
        with zipfile.ZipFile(zip_path) as archive:
            for member_name in archive.namelist():
                normalized_name = member_name.replace("\\", "/").strip("/")
                if not normalized_name:
                    continue
                parts = normalized_name.split("/")
                safe_index = next(
                    (index for index, part in enumerate(parts) if part.upper().endswith(".SAFE")),
                    None,
                )
                if safe_index is None:
                    continue
                safe_dir = "/".join(parts[: safe_index + 1])
                remainder = "/".join(parts[safe_index + 1 :]).upper()
                if remainder == "MANIFEST.SAFE":
                    manifest_safe_dirs.add(safe_dir)
                if remainder.startswith("GRANULE/"):
                    granule_safe_dirs.add(safe_dir)
    except (zipfile.BadZipFile, OSError, PermissionError):
        _ZIP_SCENE_CACHE[cache_key] = (zip_stat.st_size, zip_stat.st_mtime, [], True)
        return []

    valid_safe_dirs = sorted(manifest_safe_dirs & granule_safe_dirs)
    scene_ids = []
    for safe_dir in valid_safe_dirs:
        safe_name = Path(safe_dir).name
        scene_id = safe_name[:-5] if safe_name.upper().endswith(".SAFE") else safe_name
        if scene_id:
            scene_ids.append(scene_id)

    preferred_ids = [scene_id for scene_id in scene_ids if _looks_like_sentinel_scene_id(scene_id)]
    resolved_ids = sorted(set(preferred_ids or scene_ids))
    _ZIP_SCENE_CACHE[cache_key] = (zip_stat.st_size, zip_stat.st_mtime, resolved_ids, True)
    return list(resolved_ids)


def _is_safe_directory_path(path_value: Path) -> bool:
    """Return True when the path points to a Sentinel-2 .SAFE directory."""
    if not (path_value.is_dir() and path_value.name.upper().endswith(".SAFE")):
        return False
    return (path_value / "manifest.safe").is_file() and (path_value / "GRANULE").is_dir()


def _is_zip_scene_path(path_value: Path, *, allow_filename_hint: bool = False) -> bool:
    """Return True when the path points to a zip containing a Sentinel-2 SAFE scene."""
    if not (path_value.is_file() and path_value.suffix.lower() == ".zip"):
        return False
    return bool(_extract_zip_safe_scene_ids(path_value, allow_filename_hint=allow_filename_hint))


def _is_multiband_tiff_path(path_value: Path) -> bool:
    """Return True when the path points to a 12-band TIFF input image."""
    if not (path_value.is_file() and path_value.suffix.lower() in MULTIBAND_TIFF_SUFFIXES):
        return False
    try:
        with rasterio.open(path_value) as raster:
            return int(raster.count) == len(TIFF_RAW_BAND_ORDER)
    except Exception:
        return False


def _derive_scene_id(scene_path: Path) -> str:
    """Extract a format-agnostic scene identifier from a SAFE/ZIP path."""
    scene_name = scene_path.name
    if scene_name.lower().endswith(".zip"):
        zip_scene_ids = _extract_zip_safe_scene_ids(scene_path, allow_filename_hint=True)
        if zip_scene_ids:
            return zip_scene_ids[0]
    return _strip_scene_suffixes(scene_name)


def _extract_scene_acquisition_datetime(scene_id: str) -> str | None:
    """Extract acquisition timestamp from a Sentinel-2 scene identifier."""
    match = re.search(r"(\d{8}T\d{6})", scene_id)
    if not match:
        return None
    try:
        acquisition_dt = datetime.strptime(match.group(1), "%Y%m%dT%H%M%S")
    except ValueError:
        return None
    return acquisition_dt.strftime("%Y-%m-%d %H:%M:%S")


def _scene_format(scene_path: Path) -> str:
    """Return normalized scene format label for UI/backend reporting."""
    if scene_path.suffix.lower() == ".zip":
        return "ZIP"
    if _is_multiband_tiff_path(scene_path):
        return "TIFF"
    return "SAFE" if _is_safe_directory_path(scene_path) else "ZIP"


def _build_scene_record(
    scene_path: Path,
    *,
    scene_id: str | None = None,
    format_label: str | None = None,
) -> dict[str, str | None]:
    """Build normalized metadata for one discovered scene input."""
    resolved_scene_id = scene_id or _derive_scene_id(scene_path)
    acquisition_datetime = _extract_scene_acquisition_datetime(resolved_scene_id)
    acquisition_date = acquisition_datetime.split(" ")[0] if acquisition_datetime else None
    return {
        "path": str(scene_path),
        "scene_id": resolved_scene_id,
        "format": format_label or _scene_format(scene_path),
        "acquisition_datetime": acquisition_datetime,
        "acquisition_date": acquisition_date,
    }


def _select_preferred_scene(candidates: list[dict[str, str | None]]) -> dict[str, str | None]:
    """Select the preferred candidate when the same scene exists as SAFE and ZIP."""
    format_priority = {"SAFE": 0, "TIFF": 1, "ZIP": 2}
    return sorted(
        candidates,
        key=lambda item: (
            format_priority.get(str(item["format"]), 99),
            str(item["path"]).lower(),
        ),
    )[0]


def discover_scene_batch_info(input_scene_path: str) -> dict[str, object]:
    """
    Discover scene inputs and return metadata, including duplicate resolution.

    Duplicate scenes are grouped by scene identifier. When multiple formats are
    available, SAFE is preferred over TIFF, and TIFF is preferred over ZIP.
    """
    input_path = Path(input_scene_path)
    cached_batch_info = _get_cached_scene_batch_info(input_path)
    if cached_batch_info is not None:
        return cached_batch_info

    if _is_safe_directory_path(input_path) or _is_zip_scene_path(input_path) or _is_multiband_tiff_path(input_path):
        selected = [_build_scene_record(input_path)]
        return _store_scene_batch_info(input_path, {
            "input_path": str(input_path),
            "raw_count": 1,
            "unique_count": 1,
            "skipped_count": 0,
            "ignored_count": 0,
            "ignored_inputs": [],
            "selected": selected,
            "duplicate_count": 0,
            "duplicate_groups": [],
            "format_counts": {
                "SAFE": int(selected[0]["format"] == "SAFE"),
                "TIFF": int(selected[0]["format"] == "TIFF"),
                "ZIP": int(selected[0]["format"] == "ZIP"),
            },
            "acquisition_dates": [selected[0]["acquisition_date"]] if selected[0]["acquisition_date"] else [],
            "selection_rule": "Prefer .SAFE over .tif/.tiff over .zip when duplicate scenes are found.",
        })

    if input_path.is_dir() and input_path.name.upper().endswith(".SAFE"):
        raise FileNotFoundError(
            "Selected .SAFE folder is not a valid Sentinel-2 scene (missing manifest.safe or GRANULE)."
        )

    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        raise FileNotFoundError(
            "Selected .zip does not contain a valid Sentinel-2 .SAFE scene."
        )

    if input_path.is_file() and input_path.suffix.lower() in MULTIBAND_TIFF_SUFFIXES:
        raise FileNotFoundError(
            "Selected TIFF input must contain exactly 12 spectral bands."
        )

    if not input_path.is_dir():
        raise FileNotFoundError(
            "Input scene must be a .SAFE directory, a .zip archive, a 12-band .tif/.tiff image, or a folder containing them."
        )

    discovered_records: list[dict[str, str | None]] = []
    ignored_candidates: list[str] = []
    for root, dirnames, filenames in os.walk(input_path):
        dirnames[:] = [
            dirname for dirname in dirnames if dirname.upper() not in _SCENE_STATUS_FOLDER_NAMES
        ]
        safe_dir_paths: list[Path] = []
        for dirname in list(dirnames):
            if dirname.upper().endswith(".SAFE"):
                safe_dir_paths.append(Path(root) / dirname)
                dirnames.remove(dirname)

        for safe_dir_path in safe_dir_paths:
            if _is_safe_directory_path(safe_dir_path):
                discovered_records.append(
                    _build_scene_record(
                        safe_dir_path,
                        scene_id=_strip_scene_suffixes(safe_dir_path.name),
                        format_label="SAFE",
                    )
                )
            else:
                ignored_candidates.append(str(safe_dir_path))

        for filename in filenames:
            file_path = Path(root) / filename
            if filename.lower().endswith(".zip"):
                if not _scene_id_from_filename_hint(file_path):
                    ignored_candidates.append(str(file_path))
                    continue
                zip_scene_ids = _extract_zip_safe_scene_ids(file_path, allow_filename_hint=True)
                if zip_scene_ids:
                    discovered_records.append(
                        _build_scene_record(
                            file_path,
                            scene_id=zip_scene_ids[0],
                            format_label="ZIP",
                        )
                    )
                else:
                    ignored_candidates.append(str(file_path))
                continue

            if file_path.suffix.lower() in MULTIBAND_TIFF_SUFFIXES:
                if _is_multiband_tiff_path(file_path):
                    discovered_records.append(
                        _build_scene_record(
                            file_path,
                            scene_id=_strip_scene_suffixes(file_path.name),
                            format_label="TIFF",
                        )
                    )
                else:
                    ignored_candidates.append(str(file_path))

    if not discovered_records:
        if ignored_candidates:
            raise FileNotFoundError(
                f"No valid apply inputs were found in {input_scene_path}. "
                f"Ignored {len(ignored_candidates)} unsupported .SAFE/.zip/.tif/.tiff candidate(s)."
            )
        raise FileNotFoundError(
            f"No .SAFE folders, .zip archives, or 12-band .tif/.tiff files were found in {input_scene_path}"
        )

    grouped_records: dict[str, list[dict[str, str | None]]] = {}
    for record in discovered_records:
        grouped_records.setdefault(str(record["scene_id"]).upper(), []).append(record)

    selected_records: list[dict[str, str | None]] = []
    duplicate_groups: list[dict[str, object]] = []
    for scene_group_key in sorted(grouped_records.keys()):
        candidates = grouped_records[scene_group_key]
        chosen = _select_preferred_scene(candidates)
        selected_records.append(chosen)
        skipped = [record for record in candidates if record is not chosen]
        if skipped:
            duplicate_groups.append(
                {
                    "scene_id": chosen["scene_id"],
                    "kept": chosen,
                    "skipped": sorted(skipped, key=lambda item: str(item["path"]).lower()),
                }
            )

    selected_records = sorted(
        selected_records,
        key=lambda item: (
            item["acquisition_datetime"] or "9999-99-99 99:99:99",
            str(item["scene_id"]).upper(),
        ),
    )
    format_counts = {"SAFE": 0, "TIFF": 0, "ZIP": 0}
    for record in selected_records:
        format_counts[str(record["format"])] += 1

    acquisition_dates = sorted(
        {
            str(record["acquisition_date"])
            for record in selected_records
            if record["acquisition_date"]
        }
    )

    return _store_scene_batch_info(input_path, {
        "input_path": str(input_path),
        "raw_count": len(discovered_records),
        "unique_count": len(selected_records),
        "skipped_count": len(discovered_records) - len(selected_records),
        "ignored_count": len(ignored_candidates),
        "ignored_inputs": ignored_candidates,
        "selected": selected_records,
        "duplicate_count": len(duplicate_groups),
        "duplicate_groups": duplicate_groups,
        "format_counts": format_counts,
        "acquisition_dates": acquisition_dates,
        "selection_rule": "Prefer .SAFE over .tif/.tiff over .zip when duplicate inputs are found. Ignore unsupported .SAFE/.zip/.tif/.tiff files.",
    })


def discover_scene_inputs(input_scene_path: str) -> list[str]:
    """Discover one or more scene inputs, de-duplicated for batch folders."""
    scene_batch_info = discover_scene_batch_info(input_scene_path)
    return [str(scene_record["path"]) for scene_record in scene_batch_info["selected"]]


def _scene_status_root_directory(scene_path: Path) -> Path:
    """Resolve the parent directory that should contain Done/Failed scene folders."""
    parent_dir = scene_path.parent
    if parent_dir.name.upper() in _SCENE_STATUS_FOLDER_NAMES and parent_dir.parent != parent_dir:
        return parent_dir.parent
    return parent_dir


def _build_unique_scene_destination(destination_dir: Path, scene_name: str) -> Path:
    """Return a non-conflicting destination path inside a status folder."""
    candidate_path = destination_dir / scene_name
    if not candidate_path.exists():
        return candidate_path

    name_path = Path(scene_name)
    suffixes = "".join(name_path.suffixes)
    stem = scene_name[:-len(suffixes)] if suffixes else scene_name
    for index in range(1, 10_000):
        candidate_path = destination_dir / f"{stem}_{index}{suffixes}"
        if not candidate_path.exists():
            return candidate_path

    raise FileExistsError(
        f"Could not find a free destination name for {scene_name} in {destination_dir}"
    )


def move_scene_input_to_status_folder(input_scene_path: str | Path, status: str) -> str:
    """Move one scene file/folder into a sibling Done/Failed directory and return the new path."""
    normalized_status = status.strip().lower()
    if normalized_status not in _SCENE_STATUS_FOLDERS:
        raise ValueError(
            f"Unsupported scene status '{status}'. Expected one of: {', '.join(_SCENE_STATUS_FOLDERS)}."
        )

    scene_path = Path(input_scene_path)
    if not scene_path.exists():
        raise FileNotFoundError(f"Scene input no longer exists: {scene_path}")

    target_root = _scene_status_root_directory(scene_path)
    target_dir = target_root / _SCENE_STATUS_FOLDERS[normalized_status]
    target_dir.mkdir(parents=True, exist_ok=True)

    if scene_path.parent == target_dir:
        return str(scene_path)

    target_path = _build_unique_scene_destination(target_dir, scene_path.name)
    shutil.move(str(scene_path), str(target_path))

    _SCENE_BATCH_CACHE.clear()
    if scene_path.suffix.lower() == ".zip":
        _ZIP_SCENE_CACHE.pop(_scene_path_cache_key(scene_path), None)

    return str(target_path)


@contextmanager
def _prepare_s2_scene_input(
    input_scene_path: str,
    status_callback: Callable[[str], None] | None = None,
):
    """Accept a .SAFE folder, zipped SAFE scene, or 12-band TIFF input."""
    input_path = Path(input_scene_path)

    if input_path.is_dir():
        if not _is_safe_directory_path(input_path):
            raise ValueError(
                "Input directory must be a valid Sentinel-2 .SAFE folder."
            )
        yield str(input_path)
        return

    if input_path.is_file() and input_path.suffix.lower() == ".zip":
        if not _is_zip_scene_path(input_path):
            raise ValueError(
                "Input zip archive does not contain a valid Sentinel-2 .SAFE scene."
            )
        _emit_status(
            status_callback,
            f"Extracting zipped Sentinel-2 scene from {input_scene_path}",
        )
        temp_dir = tempfile.mkdtemp(prefix="ice_creams_safe_")
        try:
            with zipfile.ZipFile(input_path) as archive:
                try:
                    archive.extractall(temp_dir)
                except zipfile.BadZipFile as exc:
                    raw_message = str(exc).strip()
                    if "Bad CRC-32 for file" in raw_message:
                        raise ValueError(
                            "Input zip archive is corrupted or incomplete. "
                            f"{raw_message}. "
                            "This usually means the scene did not finish downloading or syncing locally. "
                            "Re-download the scene, or if it is stored in a sync folder such as Nextcloud, "
                            "make sure the zip is fully available on disk before running."
                        ) from exc
                    raise ValueError(
                        "Input zip archive could not be extracted cleanly. "
                        f"{raw_message}. "
                        "Re-download the scene or extract it manually with a zip tool first."
                    ) from exc
            yield _locate_safe_directory(temp_dir)
        finally:
            _cleanup_temp_dir(temp_dir, status_callback)
        return

    if input_path.is_file() and input_path.suffix.lower() in MULTIBAND_TIFF_SUFFIXES:
        if not _is_multiband_tiff_path(input_path):
            raise ValueError(
                "Input TIFF must contain exactly 12 spectral bands."
            )
        yield str(input_path)
        return

    raise FileNotFoundError(
        "Input scene must be a .SAFE directory, a .zip archive containing one, or a 12-band .tif/.tiff image."
    )


def build_s2_mask_scl_mask(scl_data):
    """
    Takes Sentinel-2 SCL image as an xarray data array and returns a mask based on flags (True=valid data)

    S2 flags are:

    1: saturated or defective
    2: dark area pixels
    3: cloud shadows
    4: vegetation
    5: not vegetated
    6: water
    7: unclassified
    8: cloud medium probability
    9: cloud high probability
    10: thin cirrus
    11: snow

    Parameters
    ----------
    ds : xarray.Dataset with scl (i.e., cloud mask) variable.
    """
    mask = xarray.where(
        (scl_data == 1)
        | (scl_data == 11),
        True,
        False,
    )

    return mask


def _find_required_safe_file(input_s2_safe: str, pattern: str, description: str) -> str:
    """Find one required file inside a Sentinel-2 SAFE scene."""
    matches = glob.glob(os.path.join(input_s2_safe, "GRANULE/*/IMG_DATA", "R??m", pattern))
    if not matches:
        raise FileNotFoundError(
            f"Could not locate required Sentinel-2 {description} file in {input_s2_safe}."
        )
    return matches[0]


def _get_s2_files_from_safe(
    input_s2_safe,
    required_raw_bands: tuple[str, ...] | list[str] | None = None,
) -> dict:
    """Find the required raw-band and SCL jp2 files inside a .SAFE scene."""
    output_files_dict: dict[str, object] = {}
    ordered_raw_bands = tuple(
        dict.fromkeys(required_raw_bands or RAW_BANDS_BY_MODE[FEATURE_MODE_HIGH_SPECTRAL_COMPLEXITY])
    )

    if "B02" not in ordered_raw_bands:
        raise ValueError("Sentinel-2 feature extraction requires B02 as the 10 m reference grid.")

    for band_name in ordered_raw_bands:
        pattern = S2_RAW_BAND_FILE_PATTERNS.get(band_name)
        if pattern is None:
            raise ValueError(f"Unsupported Sentinel-2 raw band requested: {band_name}")
        output_files_dict[f"{band_name.lower()}_file"] = _find_required_safe_file(
            input_s2_safe,
            pattern,
            f"{band_name} band",
        )

    output_files_dict["scl_file"] = _find_required_safe_file(
        input_s2_safe,
        "*SCL_20m.jp2",
        "SCL mask",
    )
    output_files_dict["required_raw_bands"] = ordered_raw_bands
    output_files_dict["processing_base_line"] = int(
        os.path.basename(input_s2_safe.rstrip("/\\")).split("_")[3][1:5]
    )

    return output_files_dict


def _first_available_raw_reflectance_name(
    dataset: xarray.Dataset,
    preferred_feature_names: list[str] | tuple[str, ...] | None = None,
) -> str:
    """Return the first available raw reflectance variable in a dataset."""
    preferred_raw_names: list[str] = []
    for feature_name in preferred_feature_names or ():
        normalized = str(feature_name).strip()
        if normalized.startswith("Reflectance_B") and not normalized.startswith("Reflectance_Stan_"):
            preferred_raw_names.append(normalized)

    for variable_name in preferred_raw_names:
        if variable_name in dataset.data_vars:
            return variable_name

    for variable_name in dataset.data_vars:
        if variable_name.startswith("Reflectance_B") and not variable_name.startswith("Reflectance_Stan_"):
            return variable_name

    raise ValueError("No raw reflectance variables are available in the Sentinel-2 dataset.")


def _read_s2_data_xarray(
    input_s2_files,
    mask_vector_file=None,
    verbose_console: bool = True,
    status_callback: Callable[[str], None] | None = None,
):
    """
    Read s2 data to an xarray object, masking if a mask is provided.

    Takes a dictionary of input files, which can be anything rioxarray can open
     (e.g., .jp2 file, S3 bucket).
    """

    clip_bounds: tuple[float, float, float, float] | None = None
    mask_vector: geopandas.GeoDataFrame | None = None
    opened_raster_resources: list[object] = []
    ordered_raw_bands = tuple(
        dict.fromkeys(
            input_s2_files.get("required_raw_bands")
            or RAW_BANDS_BY_MODE[FEATURE_MODE_HIGH_SPECTRAL_COMPLEXITY]
        )
    )

    try:
        # Open the 10 m reference grid first so m1 can be aligned before any heavy resampling.
        b02 = _open_s2_band(
            input_s2_files["b02_file"],
            cleanup_resources=opened_raster_resources,
        )

        if mask_vector_file is not None:
            _emit_status(status_callback, f"Opening mask polygon (m1) from {mask_vector_file}")
            _console_log(f"Masking S2 scene to {mask_vector_file}", verbose_console)
            mask_vector = geopandas.read_file(mask_vector_file)
            mask_vector = _align_mask_vector_to_scene(
                mask_vector,
                mask_vector_file,
                b02,
                verbose_console=verbose_console,
            )

            mask_bounds = tuple(float(value) for value in mask_vector.total_bounds)
            scene_bounds = tuple(float(value) for value in b02.rio.bounds(recalc=True))
            clip_bounds = _bounds_intersection(
                _expand_bounds(
                    mask_bounds,
                    MASK_EXTENT_BUFFER_DISTANCE,
                    MASK_EXTENT_BUFFER_DISTANCE,
                ),
                scene_bounds,
            )
            if clip_bounds is None:
                raise ValueError(
                    "Buffered mask extent does not intersect the Sentinel-2 scene extent. "
                    f"Mask extent: {_format_bounds(mask_bounds)}. "
                    f"Scene extent: {_format_bounds(scene_bounds)}."
                )

            _emit_status(
                status_callback,
                f"Applying buffered extent mask (m2) with {int(MASK_EXTENT_BUFFER_DISTANCE)} m padding",
            )
            _console_log(
                f"Cropping native-resolution bands to mask extent + {MASK_EXTENT_BUFFER_DISTANCE:.0f} m buffer",
                verbose_console,
            )
            b02 = _open_s2_band(
                input_s2_files["b02_file"],
                clip_bounds=clip_bounds,
                cleanup_resources=opened_raster_resources,
            )

        # Read the remaining bands and apply the buffered-extent crop (m2) before resampling to 10 m.
        opened_raw_bands: dict[str, xarray.DataArray] = {"B02": b02}
        for band_name in ordered_raw_bands:
            if band_name == "B02":
                continue
            opened_raw_bands[band_name] = _open_s2_band(
                input_s2_files[f"{band_name.lower()}_file"],
                clip_bounds=clip_bounds,
                cleanup_resources=opened_raster_resources,
            )

        scl = _open_s2_band(
            input_s2_files["scl_file"],
            clip_bounds=clip_bounds,
            cleanup_resources=opened_raster_resources,
        )


        # Bias removed for images after 2022 using baseline processor after 400

        if input_s2_files["processing_base_line"] > 399:
            for band in opened_raw_bands.values():
                band.data = band.data - 1000

        _emit_status(status_callback, "Resampling buffered Sentinel-2 bands to 10 m")
        resampled_raw_bands: dict[str, xarray.DataArray] = {}
        for band_name, band_data in opened_raw_bands.items():
            if band_name in S2_NATIVE_10M_BANDS:
                resampled_raw_bands[band_name] = band_data
            else:
                resampled_raw_bands[band_name] = band_data.reindex_like(b02, method="nearest")
        scl_10m = scl.reindex_like(b02, method="nearest")

        # Save all to one Raw dataset
        dataset_vars = {
            raw_column_name(band_name): resampled_raw_bands[band_name]
            for band_name in ordered_raw_bands
        }
        dataset_vars["SCL"] = scl_10m
        s2_data_raw = xarray.Dataset(dataset_vars)
        # Set CRS
        s2_data_raw.rio.write_crs(b02.rio.crs)

        # Apply SCL mask to data
        scl_mask = build_s2_mask_scl_mask(scl_10m)
        s2_data_raw = s2_data_raw.where(~scl_mask)

        if mask_vector is not None:
            _emit_status(status_callback, "Applying original mask polygon (m1) to 10 m pixels")
            template_var_name = _first_available_raw_reflectance_name(s2_data_raw)
            mask_raster = rasterio.features.geometry_mask(
                mask_vector.geometry,
                out_shape=s2_data_raw[template_var_name].shape[1:],
                transform=s2_data_raw.rio.transform(recalc=True),
            )
            s2_data_raw = s2_data_raw.where(~mask_raster)
            s2_data_raw["study_site"] = xarray.DataArray(
                data=numpy.expand_dims(mask_raster, axis=0),
                dims=s2_data_raw.dims,
                coords=s2_data_raw.coords,
            )

        s2_data_raw.set_close(lambda: _close_xarray_resources(opened_raster_resources))
        return s2_data_raw
    except Exception:
        _close_xarray_resources(opened_raster_resources)
        raise


def read_s2_safe(
    input_s2_safe,
    mask_vector_file=None,
    verbose_console: bool = True,
    status_callback: Callable[[str], None] | None = None,
    required_raw_bands: tuple[str, ...] | list[str] | None = None,
):
    """
    Function to read S2 data from SAFE file and return as a Dataset with
    bands resampled to 10m
    """
    s2_files_dict = _get_s2_files_from_safe(
        input_s2_safe,
        required_raw_bands=required_raw_bands,
    )
    return _read_s2_data_xarray(
        s2_files_dict,
        mask_vector_file,
        verbose_console=verbose_console,
        status_callback=status_callback,
    )


def read_multiband_tiff(
    input_raster_path: str,
    mask_vector_file=None,
    verbose_console: bool = True,
    status_callback: Callable[[str], None] | None = None,
    required_raw_bands: tuple[str, ...] | list[str] | None = None,
):
    """Read a 12-band TIFF image and return a Dataset of raw reflectance bands."""
    clip_bounds: tuple[float, float, float, float] | None = None
    mask_vector: geopandas.GeoDataFrame | None = None
    opened_raster_resources: list[object] = []
    ordered_raw_bands = tuple(
        dict.fromkeys(required_raw_bands or RAW_BANDS_BY_MODE[FEATURE_MODE_HIGH_SPECTRAL_COMPLEXITY])
    )

    try:
        raster_stack = _open_multiband_raster(
            input_raster_path,
            cleanup_resources=opened_raster_resources,
        )
        band_count = int(raster_stack.sizes.get("band", 0))
        if band_count != len(TIFF_RAW_BAND_ORDER):
            raise ValueError(
                f"Input TIFF must contain exactly {len(TIFF_RAW_BAND_ORDER)} bands in Sentinel-2 spectral order. "
                f"Found {band_count} band(s)."
            )

        reference_band = (
            raster_stack.sel(band=1, drop=True)
            .expand_dims(dim={"band": [1]})
            .transpose("band", "y", "x")
        )

        if mask_vector_file is not None:
            _emit_status(status_callback, f"Opening mask polygon (m1) from {mask_vector_file}")
            _console_log(f"Masking raster to {mask_vector_file}", verbose_console)
            mask_vector = geopandas.read_file(mask_vector_file)
            mask_vector = _align_mask_vector_to_scene(
                mask_vector,
                mask_vector_file,
                reference_band,
                verbose_console=verbose_console,
            )

            mask_bounds = tuple(float(value) for value in mask_vector.total_bounds)
            scene_bounds = tuple(float(value) for value in reference_band.rio.bounds(recalc=True))
            clip_bounds = _bounds_intersection(
                _expand_bounds(
                    mask_bounds,
                    MASK_EXTENT_BUFFER_DISTANCE,
                    MASK_EXTENT_BUFFER_DISTANCE,
                ),
                scene_bounds,
            )
            if clip_bounds is None:
                raise ValueError(
                    "Buffered mask extent does not intersect the input raster extent. "
                    f"Mask extent: {_format_bounds(mask_bounds)}. "
                    f"Raster extent: {_format_bounds(scene_bounds)}."
                )

            _emit_status(
                status_callback,
                f"Applying buffered extent mask (m2) with {int(MASK_EXTENT_BUFFER_DISTANCE)} m padding",
            )
            raster_stack = _open_multiband_raster(
                input_raster_path,
                clip_bounds=clip_bounds,
                cleanup_resources=opened_raster_resources,
            )

        _emit_status(status_callback, "Using TIFF input grid directly (no spatial resampling)")
        dataset_vars: dict[str, xarray.DataArray] = {}
        for band_name in ordered_raw_bands:
            band_index = TIFF_RAW_BAND_ORDER.index(band_name) + 1
            dataset_vars[raw_column_name(band_name)] = (
                raster_stack.sel(band=band_index, drop=True)
                .expand_dims(dim={"band": [1]})
                .transpose("band", "y", "x")
            )

        raster_data = xarray.Dataset(dataset_vars)
        raster_data.rio.write_crs(raster_stack.rio.crs)

        if mask_vector is not None:
            _emit_status(status_callback, "Applying original mask polygon (m1) to raster pixels")
            template_var_name = _first_available_raw_reflectance_name(raster_data)
            mask_raster = rasterio.features.geometry_mask(
                mask_vector.geometry,
                out_shape=raster_data[template_var_name].shape[1:],
                transform=raster_data.rio.transform(recalc=True),
            )
            raster_data = raster_data.where(~mask_raster)
            raster_data["study_site"] = xarray.DataArray(
                data=numpy.expand_dims(mask_raster, axis=0),
                dims=raster_data.dims,
                coords=raster_data.coords,
            )

        _emit_status(status_callback, "No SCL mask is available for TIFF input; skipping SCL masking")
        raster_data.set_close(lambda: _close_xarray_resources(opened_raster_resources))
        return raster_data
    except Exception:
        _close_xarray_resources(opened_raster_resources)
        raise


def calc_ndvi_true(s2_data_raw):
    """
    Function to calculate NDVI from raw S2 data loaded as xarray Dataset

    Returns xarray DataArray
    """
    red_raw = s2_data_raw["Reflectance_B04"]
    nir_raw = s2_data_raw["Reflectance_B08"]

    ndvi_raw = (nir_raw - red_raw) / (nir_raw + red_raw)
    ndvi_raw.name = "NDVI"

    return ndvi_raw


def calc_ndwi(s2_data_raw):
    """
    Function to calculate NDWI from S2 data loaded as xarray Dataset

    Returns xarray DataArray
    """
    green_raw = s2_data_raw["Reflectance_B03"]
    nir_raw = s2_data_raw["Reflectance_B08"]

    ndwi_raw = (green_raw - nir_raw) / (nir_raw + green_raw)
    ndwi_raw.name = "NDWI"

    return ndwi_raw


def calc_spc(ndvi):
    """
    Function to calculate Seagrass Cover from an NDVI DataArray.

    Accepts the output of calc_ndvi_true() directly so the B04/B08 dask
    subgraph is shared rather than rebuilt from scratch.

    Returns xarray DataArray
    """
    spc = 172.06 * ndvi - 22.18
    spc.name = "SPC"

    return spc


def apply_post_classification_rules(
    predicted_classes: numpy.ndarray,
    ndvi_values: numpy.ndarray,
) -> tuple[numpy.ndarray, numpy.ndarray]:
    """
    Apply deterministic post-processing rules to zero-based class predictions.

    Rule: when a pixel is predicted as output Class 4 (zero-based index 3) and
    NDVI is below 0.25, reclassify it to output Class 5 (zero-based index 4).
    Returns the updated classes plus a mask of reclassified pixels.
    """
    updated_classes = numpy.asarray(predicted_classes, dtype=numpy.int16).copy()
    ndvi_array = numpy.asarray(ndvi_values, dtype=numpy.float32)
    reclassified_mask = (
        (updated_classes == MODEL_CLASS_INDEX_4)
        & numpy.isfinite(ndvi_array)
        & (ndvi_array < CLASS4_TO_CLASS5_NDVI_THRESHOLD)
    )
    if reclassified_mask.any():
        updated_classes[reclassified_mask] = MODEL_CLASS_INDEX_5
    return updated_classes, reclassified_mask


def apply_classification(
    input_xarray,
    class_model,
    model_metadata: dict[str, object],
    batch_size: int | None = None,
    dask_workers: int = DEFAULT_DASK_WORKERS,
    status_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[float], None] | None = None,
):
    """
    Apply classification to a list of input xarray DataArrays
    """
    worker_count = max(1, int(dask_workers))
    detected_model_family = str(model_metadata["model_family"])
    feature_mode = str(model_metadata["feature_mode"])
    required_feature_names = list(model_metadata["required_feature_names"])

    if batch_size is None:
        try:
            import torch
            _model_device = next(class_model.model.parameters()).device
        except Exception:
            import torch
            _model_device = torch.device("cpu")
        inference_batch_size = _optimal_batch_size(_model_device)
    else:
        inference_batch_size = max(1, int(batch_size))

    stacked_data = input_xarray.stack(pixel=("y", "x")).squeeze(dim="band", drop=True)
    pixel_template_name = _first_available_raw_reflectance_name(
        stacked_data,
        preferred_feature_names=list(required_feature_names),
    )
    pixel_template = stacked_data[pixel_template_name]
    valid_mask = pixel_template.notnull()
    valid_positions = numpy.flatnonzero(valid_mask.values)
    pixel_count = int(stacked_data.sizes.get("pixel", 0))

    out_class_values = numpy.full(pixel_count, -1, dtype=numpy.int16)
    class_probs_values = numpy.full(pixel_count, -1, dtype=numpy.float32)
    seagrass_cover_values = numpy.full(pixel_count, -1, dtype=numpy.float32)
    ndvi_out_values = numpy.full(pixel_count, numpy.nan, dtype=numpy.float32)

    if valid_positions.size:
        _emit_status(
            status_callback,
            f"Preparing {valid_positions.size:,} masked pixel(s) for model inference",
        )
        _emit_progress(progress_callback, 0.70)
        vars_to_load = list(
            dict.fromkeys(
                [
                    *[
                        feature_name
                        for feature_name in required_feature_names
                        if feature_name.startswith("Reflectance_B")
                        and not feature_name.startswith("Reflectance_Stan_")
                    ],
                    "NDVI",
                    "NDWI",
                    "SPC",
                ]
            )
        )
        valid_pixels = stacked_data[vars_to_load].isel(pixel=valid_positions)
        with dask.config.set(scheduler="threads", num_workers=worker_count):
            valid_pixels_computed = valid_pixels.compute()

        raw_feature_df = pandas.DataFrame(
            {var: valid_pixels_computed[var].values for var in valid_pixels_computed.data_vars}
        )
        if detected_model_family == "spectral_1d_cnn":
            valid_df = prepare_sequence_feature_dataframe(
                raw_feature_df,
                feature_mode=feature_mode,
                sequence_feature_names=model_metadata.get("sequence_feature_names"),
                sequence_channel_feature_names=model_metadata.get("sequence_channel_feature_names"),
                context="Apply pixel dataframe",
            )
        else:
            valid_df = prepare_feature_dataframe(
                raw_feature_df,
                feature_mode=feature_mode,
                required_feature_names=required_feature_names,
                context="Apply pixel dataframe",
                rebuild_standardised=True,
                rebuild_indices=True,
            ).fillna(0)

        _emit_status(
            status_callback,
            f"Running model inference on {len(valid_df):,} pixel(s)",
        )
        _emit_progress(progress_callback, 0.76)
        preds = predict_model_probabilities(
            class_model,
            valid_df,
            model_metadata,
            batch_size=inference_batch_size,
        )

        predicted_classes = preds.argmax(dim=1).cpu().numpy().astype(numpy.int16)
        predicted_probs = preds.max(dim=1).values.cpu().numpy().astype(numpy.float32)
        ndvi_values = raw_feature_df["NDVI"].to_numpy(dtype=numpy.float32, copy=True)
        predicted_classes, reclassified_mask = apply_post_classification_rules(
            predicted_classes,
            ndvi_values,
        )

        spc_values = raw_feature_df["SPC"].to_numpy(dtype=numpy.float32, copy=True)
        spc_values = numpy.clip(spc_values, 0, 100)
        seagrass_mask = (predicted_classes == 3) & (spc_values >= 20)
        spc20_values = numpy.where(seagrass_mask, spc_values, -1.0).astype(numpy.float32)
        if reclassified_mask.any():
            spc20_values[reclassified_mask] = -1.0

        _emit_status(status_callback, "Assembling classification rasters")
        _emit_progress(progress_callback, 0.84)
        out_class_values[valid_positions] = predicted_classes + 1
        class_probs_values[valid_positions] = predicted_probs
        seagrass_cover_values[valid_positions] = spc20_values
        # Extract NDVI eagerly from the already-loaded raw feature table so both
        # tabular and spectral-CNN models write the same NDVI output layer.
        ndvi_out_values[valid_positions] = ndvi_values

    output_vars = {}
    band_coord = input_xarray[pixel_template_name].coords["band"]
    output_specs = {
        "Out_Class": out_class_values,
        "Class_Probs": class_probs_values,
        "Seagrass_Cover": seagrass_cover_values,
        "NDVI": ndvi_out_values,
    }
    for variable_name, variable_values in output_specs.items():
        output_vars[variable_name] = (
            xarray.DataArray(
                variable_values,
                dims=("pixel",),
                coords={"pixel": pixel_template.coords["pixel"]},
                name=variable_name,
            )
            .unstack("pixel")
            .expand_dims(dim={"band": band_coord})
            .transpose("band", "y", "x")
        )

    # Drop the lazy dask NDVI from input_xarray before merging so the eager
    # numpy-backed NDVI in output_vars takes its place with no conflict.
    return xarray.merge([input_xarray.drop_vars("NDVI", errors="ignore"), xarray.Dataset(output_vars)])


def classify_s2_scene(
    input_s2_safe,
    output_gtiff,
    saved_model,
    mask_vector_file=None,
    debug=False,
    status_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[float], None] | None = None,
):
    """
    Function to classify an S2 scene netCDF file

    """
    ui_callback_mode = status_callback is not None or progress_callback is not None
    verbose_console = debug or not ui_callback_mode

    model_path = os.path.abspath(saved_model)
    cached_entry = _MODEL_CACHE.get(model_path)
    model_mtime = os.path.getmtime(model_path)
    if cached_entry and cached_entry[0] == model_mtime:
        _emit_status(status_callback, f"Reusing cached model from {saved_model}")
        _emit_progress(progress_callback, 0.12)
        class_model = cached_entry[1]
    else:
        _emit_status(status_callback, f"Loading model from {saved_model}")
        _emit_progress(progress_callback, 0.05)
        class_model = _load_cached_learner(saved_model, status_callback=status_callback)
        _emit_progress(progress_callback, 0.12)
    model_metadata = extract_model_metadata(class_model)
    detected_model_family = str(model_metadata["model_family"])
    detected_model_family_label = str(model_metadata["model_family_label"])
    detected_feature_mode = str(model_metadata["feature_mode"])
    detected_feature_mode_label = str(model_metadata["feature_mode_label"])
    required_raw_bands = RAW_BANDS_BY_MODE[detected_feature_mode]
    _emit_status(
        status_callback,
        f"Detected model method: {detected_model_family_label}",
    )
    _emit_status(
        status_callback,
        f"Detected model feature mode: {detected_feature_mode_label}",
    )
    if detected_model_family == "spectral_1d_cnn":
        _emit_status(
            status_callback,
            (
                "Detected spectral inputs: "
                f"{spectral_cnn_sequence_input_label(model_metadata.get('sequence_use_standardized_reflectance'))}"
            ),
        )

    # Keep the extracted zip contents alive for the full workflow.
    with _prepare_s2_scene_input(input_s2_safe, status_callback) as resolved_scene_path:
        s2_data_raw = None
        s2_data = None
        Out_Classified_s2_data = None
        output_raster = None
        try:
            resolved_input_path = Path(resolved_scene_path)
            # Open scene into xarray dataset
            # Specify chunksize so uses dask and doesn't load all data to RAM
            if resolved_input_path.suffix.lower() in MULTIBAND_TIFF_SUFFIXES:
                _emit_status(status_callback, f"Reading multi-band TIFF from {resolved_scene_path}")
                _console_log(f"Reading in TIFF data from {resolved_scene_path}...", verbose_console)
                s2_data_raw = read_multiband_tiff(
                    resolved_scene_path,
                    mask_vector_file,
                    verbose_console=verbose_console,
                    status_callback=status_callback,
                    required_raw_bands=required_raw_bands,
                )
            else:
                _emit_status(status_callback, f"Reading Sentinel-2 scene from {resolved_scene_path}")
                _console_log(f"Reading in data from {resolved_scene_path}...", verbose_console)
                s2_data_raw = read_s2_safe(
                    resolved_scene_path,
                    mask_vector_file,
                    verbose_console=verbose_console,
                    status_callback=status_callback,
                    required_raw_bands=required_raw_bands,
                )
            _emit_progress(progress_callback, 0.35)

            # Calculate NDVI, NDWI and SPC
            # Standardisation (Reflectance_Stan_*) is now computed inside
            # apply_classification using numpy on the already-loaded raw bands —
            # no separate dask graph or pre-compute step needed here.
            _emit_status(status_callback, "Calculating derived features (NDVI, NDWI, SPC)")
            ndwi_raw = calc_ndwi(s2_data_raw)
            ndvi_true_raw = calc_ndvi_true(s2_data_raw)
            spc_raw = calc_spc(ndvi_true_raw)  # reuse the already-built NDVI dask graph
            _emit_progress(progress_callback, 0.58)

            # Merge to a single xarray
            s2_data = xarray.merge([s2_data_raw, ndwi_raw, ndvi_true_raw, spc_raw])

            # Apply classification. Will print progress
            _emit_status(status_callback, "Applying the ICE CREAMS model")
            _emit_progress(progress_callback, 0.65)
            _console_log("Performing classification", verbose_console)
            Out_Classified_s2_data = apply_classification(
                s2_data,
                class_model,
                model_metadata,
                status_callback=status_callback,
                progress_callback=progress_callback,
            )
            _emit_progress(progress_callback, 0.86)

            # Set up output dataset
            # If running in debug mode don't subset and write out all variables.
            if debug:
                output_raster = Out_Classified_s2_data.squeeze(dim="band", drop=True)
            else:
                output_raster = Out_Classified_s2_data[
                    ["Out_Class", "Class_Probs", "Seagrass_Cover", "NDVI"]
                ].squeeze(dim="band", drop=True)

            output_raster = output_raster.assign_attrs(
                {
                    "description": "ICE CREAMS Model Output",
                    "class_ids": str(CLASSES_NUMBER_ID_DICT),
                }
            )
            output_raster = output_raster.rio.write_crs(s2_data.rio.crs)

            ## Write out to Geotiff
            output_dir = os.path.dirname(output_gtiff)
            if output_dir:
                os.makedirs(output_dir, exist_ok=True)

            _emit_status(status_callback, f"Writing Cloud-Optimised GeoTIFF to {output_gtiff}")
            _emit_progress(progress_callback, 0.92)
            _console_log("Writing out", verbose_console)
            progress_context = ProgressBar() if verbose_console else nullcontext()
            with progress_context:
                with dask.config.set(
                    scheduler="threads",
                    num_workers=DEFAULT_DASK_WORKERS,
                ):
                    output_raster.rio.to_raster(
                        output_gtiff,
                        driver="COG",
                        tiled=True,
                        windowed=True,
                        dtype=numpy.float32,
                    )

            if not debug:
                _emit_progress(progress_callback, 0.96)
                _apply_out_class_qgis_style(output_gtiff, status_callback)

            _console_log(f"Saved to {output_gtiff}", verbose_console)
            _emit_progress(progress_callback, 1.0)
            _emit_status(
                status_callback,
                (
                    f"Completed ({detected_model_family_label}, {detected_feature_mode_label}). "
                    f"Output saved to {output_gtiff}"
                ),
            )
            return output_gtiff
        finally:
            for dataset in (
                output_raster,
                Out_Classified_s2_data,
                s2_data,
                s2_data_raw,
            ):
                if dataset is not None and hasattr(dataset, "close"):
                    dataset.close()
            gc.collect()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Apply ICE CREAMS to an S2 image"
    )
    parser.add_argument(
        "insafe",
        help="Input .SAFE folder, .zip archive containing a .SAFE scene, or 12-band .tif/.tiff image",
    )
    parser.add_argument("outfile", help="Output file for classification")
    parser.add_argument(
        "--mask",
        required=False,
        default=None,
        help="Vector file specifying the bounds to run classification within. Will mask out areas outside polygon",
    )
    parser.add_argument(
        "--model",
        required=False,
        default=DEFAULT_FASTAI_MODEL,
        help="Fastai saved model file",
    )
    parser.add_argument(
        "--debug",
        required=False,
        default=False,
        action="store_true",
        help="Debug mode, writes out all layers to output file and prints more output",
    )
    args = parser.parse_args()

    classify_s2_scene(
        args.insafe,
        args.outfile,
        args.model,
        args.mask,
        debug=args.debug,
    )
