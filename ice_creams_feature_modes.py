#!/usr/bin/env python3
"""Shared feature-mode constants and preprocessing helpers for ICE CREAMS."""

from __future__ import annotations

from typing import Any, Iterable

import pandas as pd


FEATURE_MODE_HIGH_SPATIAL_ACCURACY = "high_spatial_accuracy"
FEATURE_MODE_HIGH_SPECTRAL_COMPLEXITY = "high_spectral_complexity"
DEFAULT_FEATURE_MODE = FEATURE_MODE_HIGH_SPECTRAL_COMPLEXITY

FEATURE_MODE_LABELS: dict[str, str] = {
    FEATURE_MODE_HIGH_SPATIAL_ACCURACY: "High Spatial Accuracy",
    FEATURE_MODE_HIGH_SPECTRAL_COMPLEXITY: "High Spectral Complexity",
}
FEATURE_MODE_CHOICES = tuple(FEATURE_MODE_LABELS)

SPATIAL_RAW_BANDS = ("B02", "B03", "B04", "B08")
SPECTRAL_RAW_BANDS = (
    "B01",
    "B02",
    "B03",
    "B04",
    "B05",
    "B06",
    "B07",
    "B08",
    "B8A",
    "B09",
    "B11",
    "B12",
)

# Preserve the legacy spectral feature ordering used by existing ICE CREAMS models.
_LEGACY_SPECTRAL_MODEL_RAW_BANDS = (
    "B02",
    "B03",
    "B04",
    "B08",
    "B05",
    "B06",
    "B07",
    "B11",
    "B12",
    "B8A",
    "B01",
    "B09",
)

RAW_BANDS_BY_MODE: dict[str, tuple[str, ...]] = {
    FEATURE_MODE_HIGH_SPATIAL_ACCURACY: SPATIAL_RAW_BANDS,
    FEATURE_MODE_HIGH_SPECTRAL_COMPLEXITY: SPECTRAL_RAW_BANDS,
}

MODEL_RAW_BANDS_BY_MODE: dict[str, tuple[str, ...]] = {
    FEATURE_MODE_HIGH_SPATIAL_ACCURACY: SPATIAL_RAW_BANDS,
    FEATURE_MODE_HIGH_SPECTRAL_COMPLEXITY: _LEGACY_SPECTRAL_MODEL_RAW_BANDS,
}


def raw_column_name(band_name: str) -> str:
    """Return the raw reflectance column name for a Sentinel-2 band."""
    return f"Reflectance_{str(band_name).strip()}"


def standardised_column_name(band_name: str) -> str:
    """Return the standardized reflectance column name for a Sentinel-2 band."""
    return f"Reflectance_Stan_{str(band_name).strip()}"


RAW_COLUMNS_BY_MODE: dict[str, tuple[str, ...]] = {
    mode_name: tuple(raw_column_name(band_name) for band_name in band_names)
    for mode_name, band_names in MODEL_RAW_BANDS_BY_MODE.items()
}
STANDARDISED_COLUMNS_BY_MODE: dict[str, tuple[str, ...]] = {
    mode_name: tuple(standardised_column_name(band_name) for band_name in band_names)
    for mode_name, band_names in MODEL_RAW_BANDS_BY_MODE.items()
}
FEATURE_COLUMNS_BY_MODE: dict[str, tuple[str, ...]] = {
    mode_name: (
        *RAW_COLUMNS_BY_MODE[mode_name],
        *STANDARDISED_COLUMNS_BY_MODE[mode_name],
        "NDVI",
        "NDWI",
    )
    for mode_name in FEATURE_MODE_CHOICES
}


def normalize_feature_mode(feature_mode: str | None) -> str:
    """Validate and canonicalize a feature-mode identifier."""
    normalized = str(feature_mode or DEFAULT_FEATURE_MODE).strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    if normalized not in FEATURE_MODE_LABELS:
        valid_values = ", ".join(FEATURE_MODE_CHOICES)
        raise ValueError(f"Invalid feature mode '{feature_mode}'. Use one of: {valid_values}.")
    return normalized


def feature_mode_label(feature_mode: str | None) -> str:
    """Return the user-facing label for a feature mode."""
    return FEATURE_MODE_LABELS[normalize_feature_mode(feature_mode)]


def _normalise_feature_name_list(feature_names: Iterable[Any]) -> list[str]:
    """Normalize and de-duplicate feature names while preserving order."""
    normalized_names: list[str] = []
    seen_names: set[str] = set()
    for feature_name in feature_names:
        normalized = str(feature_name).strip()
        if normalized and normalized not in seen_names:
            seen_names.add(normalized)
            normalized_names.append(normalized)
    return normalized_names


def extract_learner_required_feature_names(learner: Any) -> list[str]:
    """Extract tabular input feature names from an exported fastai learner."""
    metadata = getattr(learner, "ice_creams_model_metadata", None)
    if isinstance(metadata, dict):
        metadata_features = _normalise_feature_name_list(
            metadata.get("required_feature_names") or metadata.get("sequence_feature_names")
        )
        if metadata_features:
            return metadata_features

    discovered: list[str] = []
    seen: set[str] = set()

    def _append(values: Any) -> None:
        if values is None:
            return
        if isinstance(values, str):
            values_iter = [values]
        else:
            try:
                values_iter = list(values)
            except TypeError:
                return
        for name in values_iter:
            normalized = str(name).strip()
            if normalized and normalized not in seen:
                seen.add(normalized)
                discovered.append(normalized)

    dls = getattr(learner, "dls", None)
    train_ds = getattr(dls, "train_ds", None) if dls is not None else None

    if dls is not None:
        _append(getattr(dls, "cat_names", None))
        _append(getattr(dls, "cont_names", None))
        _append(getattr(dls, "x_names", None))

    if train_ds is not None:
        _append(getattr(train_ds, "cat_names", None))
        _append(getattr(train_ds, "cont_names", None))
        _append(getattr(train_ds, "x_names", None))

    if not discovered:
        raise ValueError("Unable to determine required model input features from the learner.")

    return discovered


def infer_feature_mode_from_feature_names(feature_names: Iterable[Any]) -> str:
    """Infer the ICE CREAMS feature mode from the learner's required feature set."""
    normalized_names = _normalise_feature_name_list(feature_names)
    feature_set = set(normalized_names)

    for mode_name in FEATURE_MODE_CHOICES:
        if feature_set == set(FEATURE_COLUMNS_BY_MODE[mode_name]):
            return mode_name

    supported_feature_set = set(FEATURE_COLUMNS_BY_MODE[FEATURE_MODE_HIGH_SPATIAL_ACCURACY]) | set(
        FEATURE_COLUMNS_BY_MODE[FEATURE_MODE_HIGH_SPECTRAL_COMPLEXITY]
    )
    unsupported_features = [name for name in normalized_names if name not in supported_feature_set]
    if unsupported_features:
        unsupported_preview = ", ".join(unsupported_features[:12])
        suffix = f", +{len(unsupported_features) - 12} more" if len(unsupported_features) > 12 else ""
        raise ValueError(
            "Unable to infer feature mode because the model requires unsupported input features: "
            f"{unsupported_preview}{suffix}"
        )

    raise ValueError(
        "Unable to infer feature mode from the learner's required feature names. "
        "The feature set does not match the supported spatial or spectral configurations."
    )


def infer_feature_mode_from_learner(learner: Any) -> tuple[str, list[str]]:
    """Infer feature mode and ordered feature names from an exported fastai learner."""
    metadata = getattr(learner, "ice_creams_model_metadata", None)
    if isinstance(metadata, dict):
        metadata_features = _normalise_feature_name_list(
            metadata.get("required_feature_names") or metadata.get("sequence_feature_names")
        )
        explicit_mode = metadata.get("feature_mode")
        if explicit_mode is not None and metadata_features:
            return normalize_feature_mode(explicit_mode), metadata_features

    required_feature_names = extract_learner_required_feature_names(learner)
    inferred_mode = infer_feature_mode_from_feature_names(required_feature_names)
    return inferred_mode, required_feature_names


def ensure_required_columns(
    frame: pd.DataFrame,
    required_columns: Iterable[str],
    *,
    context: str,
) -> list[str]:
    """Raise a clear error when required dataframe columns are missing."""
    required = _normalise_feature_name_list(required_columns)
    missing_columns = [column_name for column_name in required if column_name not in frame.columns]
    if missing_columns:
        missing_preview = ", ".join(missing_columns[:12])
        suffix = f", +{len(missing_columns) - 12} more" if len(missing_columns) > 12 else ""
        raise ValueError(f"{context} is missing required columns: {missing_preview}{suffix}")
    return required


def recompute_standardised_reflectance(
    frame: pd.DataFrame,
    raw_columns: Iterable[str],
) -> pd.DataFrame:
    """Recompute per-row standardized reflectance columns from raw reflectance inputs."""
    ordered_raw_columns = _normalise_feature_name_list(raw_columns)
    raw_values = frame.loc[:, ordered_raw_columns].apply(pd.to_numeric, errors="coerce")
    row_min = raw_values.min(axis=1, skipna=True)
    row_max = raw_values.max(axis=1, skipna=True)
    row_range = row_max - row_min
    variable_rows = row_range.notna() & row_range.gt(0)

    standardized = raw_values.sub(row_min, axis=0).div(
        row_range.where(variable_rows, other=1.0),
        axis=0,
    )
    standardized.loc[~variable_rows, :] = 0.0
    standardized = standardized.rename(
        columns={
            column_name: column_name.replace("Reflectance_", "Reflectance_Stan_")
            for column_name in ordered_raw_columns
        }
    )
    return standardized


def recompute_ndvi_ndwi(frame: pd.DataFrame) -> pd.DataFrame:
    """Recompute NDVI and NDWI from raw Sentinel-2 reflectance columns."""
    ensure_required_columns(
        frame,
        (raw_column_name("B03"), raw_column_name("B04"), raw_column_name("B08")),
        context="Derived-feature preprocessing",
    )

    green = pd.to_numeric(frame[raw_column_name("B03")], errors="coerce")
    red = pd.to_numeric(frame[raw_column_name("B04")], errors="coerce")
    nir = pd.to_numeric(frame[raw_column_name("B08")], errors="coerce")

    ndvi_denominator = nir + red
    ndwi_denominator = nir + green
    ndvi = (nir - red).div(ndvi_denominator).where(ndvi_denominator.ne(0), other=0.0)
    ndwi = (green - nir).div(ndwi_denominator).where(ndwi_denominator.ne(0), other=0.0)

    return pd.DataFrame({"NDVI": ndvi, "NDWI": ndwi}, index=frame.index)


def _overwrite_columns(
    frame: pd.DataFrame,
    replacement_frame: pd.DataFrame,
) -> pd.DataFrame:
    """Overwrite columns in a dataframe without introducing duplicate names."""
    updated_frame = frame.copy()
    for column_name in replacement_frame.columns:
        updated_frame[column_name] = replacement_frame[column_name]
    return updated_frame


def prepare_feature_dataframe(
    frame: pd.DataFrame,
    *,
    feature_mode: str | None = None,
    required_feature_names: Iterable[Any] | None = None,
    context: str,
    rebuild_standardised: bool | None = None,
    rebuild_indices: bool | None = None,
) -> pd.DataFrame:
    """
    Prepare a dataframe for training, apply, or validation using a feature mode.

    Spatial mode rebuilds standardized reflectance and NDVI/NDWI from raw 10 m
    reflectance inputs by default. Spectral mode can either keep the legacy
    full-feature CSV workflow (strict column checks) or rebuild standardized and
    derived features from raw inputs when requested by the caller.
    """
    if required_feature_names is not None:
        ordered_required_feature_names = _normalise_feature_name_list(required_feature_names)
        if not ordered_required_feature_names:
            raise ValueError("At least one required feature name is needed.")
    else:
        ordered_required_feature_names = []

    if feature_mode is None:
        if not ordered_required_feature_names:
            raise ValueError("Feature mode or required feature names must be provided.")
        resolved_feature_mode = infer_feature_mode_from_feature_names(ordered_required_feature_names)
    else:
        resolved_feature_mode = normalize_feature_mode(feature_mode)

    if not ordered_required_feature_names:
        ordered_required_feature_names = list(FEATURE_COLUMNS_BY_MODE[resolved_feature_mode])

    working_frame = frame.copy()
    raw_columns_for_mode = RAW_COLUMNS_BY_MODE[resolved_feature_mode]
    if rebuild_standardised is None:
        rebuild_standardised = resolved_feature_mode == FEATURE_MODE_HIGH_SPATIAL_ACCURACY
    if rebuild_indices is None:
        rebuild_indices = resolved_feature_mode == FEATURE_MODE_HIGH_SPATIAL_ACCURACY
    raw_input_context_suffix = (
        "spatial-mode raw inputs"
        if resolved_feature_mode == FEATURE_MODE_HIGH_SPATIAL_ACCURACY
        else "raw inputs"
    )

    if rebuild_standardised or rebuild_indices:
        ensure_required_columns(
            working_frame,
            raw_columns_for_mode,
            context=f"{context} ({raw_input_context_suffix})",
        )

    if rebuild_standardised:
        standardised_frame = recompute_standardised_reflectance(working_frame, raw_columns_for_mode)
        working_frame = _overwrite_columns(working_frame, standardised_frame)

    if rebuild_indices:
        indices_frame = recompute_ndvi_ndwi(working_frame)
        working_frame = _overwrite_columns(working_frame, indices_frame)

    if resolved_feature_mode == FEATURE_MODE_HIGH_SPECTRAL_COMPLEXITY and not (
        rebuild_standardised or rebuild_indices
    ):
        ensure_required_columns(
            working_frame,
            FEATURE_COLUMNS_BY_MODE[resolved_feature_mode],
            context=f"{context} (spectral-mode inputs)",
        )

    ensure_required_columns(
        working_frame,
        ordered_required_feature_names,
        context=f"{context} (prepared model inputs)",
    )
    return working_frame.loc[:, ordered_required_feature_names].copy()


def build_training_dataframe(
    frame: pd.DataFrame,
    *,
    feature_mode: str | None = None,
    label_column: str = "True_Class",
) -> tuple[pd.DataFrame, list[str], str]:
    """Build the exact training dataframe and ordered feature list for a feature mode."""
    resolved_feature_mode = normalize_feature_mode(feature_mode)
    ensure_required_columns(frame, (label_column,), context="Training data")

    feature_frame = prepare_feature_dataframe(
        frame,
        feature_mode=resolved_feature_mode,
        required_feature_names=FEATURE_COLUMNS_BY_MODE[resolved_feature_mode],
        context="Training data",
    )
    training_frame = pd.concat([frame.loc[:, [label_column]].copy(), feature_frame], axis=1)
    return training_frame, list(feature_frame.columns), resolved_feature_mode
