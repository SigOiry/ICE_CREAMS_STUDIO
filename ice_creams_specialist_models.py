#!/usr/bin/env python3
"""Helpers for the Magnoliopsida vs Microphytobenthos specialist post-processor."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd
from fastai.losses import CrossEntropyLossFlat
from fastai.tabular.all import (
    CategoryBlock,
    FillMissing,
    TabularPandas,
    accuracy,
    tabular_learner,
)
import torch

from ice_creams_feature_modes import (
    FEATURE_MODE_HIGH_SPATIAL_ACCURACY,
    ensure_required_columns,
    raw_column_name,
    recompute_ndvi_ndwi,
    recompute_standardised_reflectance,
    standardised_column_name,
)
from ice_creams_model_families import (
    MODEL_FAMILY_TABULAR_DENSE,
    attach_model_metadata,
    extract_model_metadata,
    predict_model_probabilities,
)
from train_icecreams import (
    TrainingProgressCallback,
    _emit_progress,
    _emit_status,
    _normalise_model_path,
    discover_training_csvs,
)


SPECIALIST_ROLE_CLASS4_CLASS5 = "class4_class5_postprocessor"
SPECIALIST_MODEL_FILENAME = "ICECREAMS_Magnoliopsida_vs_Microphytobenthos_V1_3_Cleaned.pkl"
SPECIALIST_DISPLAY_NAME = "Magnoliopsida vs Microphytobenthos Specialist"
SPECIALIST_FEATURE_PROFILE_LABEL = "B03/B04/B05/B08 + standardized + NDVI/NDWI"
SPECIALIST_TARGET_LABEL_MAGNOLIOPSIDA = "Magnoliopsida"
SPECIALIST_TARGET_LABEL_MICROPHYTOBENTHOS = "Microphytobenthos"
SPECIALIST_TARGET_LABELS = (
    SPECIALIST_TARGET_LABEL_MAGNOLIOPSIDA,
    SPECIALIST_TARGET_LABEL_MICROPHYTOBENTHOS,
)
SPECIALIST_MAIN_OUTPUT_CLASS_BY_LABEL: dict[str, int] = {
    SPECIALIST_TARGET_LABEL_MAGNOLIOPSIDA: 4,
    SPECIALIST_TARGET_LABEL_MICROPHYTOBENTHOS: 5,
}
SPECIALIST_RAW_BANDS = ("B03", "B04", "B05", "B08")
SPECIALIST_RAW_COLUMNS = tuple(raw_column_name(band_name) for band_name in SPECIALIST_RAW_BANDS)
SPECIALIST_STANDARDIZED_COLUMNS = tuple(
    standardised_column_name(band_name) for band_name in SPECIALIST_RAW_BANDS
)
SPECIALIST_FEATURE_COLUMNS = (
    *SPECIALIST_RAW_COLUMNS,
    *SPECIALIST_STANDARDIZED_COLUMNS,
    "NDVI",
    "NDWI",
)
SPECIALIST_DEFAULT_TRAINING_SOURCE = (
    Path(__file__).resolve().parent
    / "Data"
    / "Input"
    / "Training"
    / "V1_3_Cleaned"
)
SPECIALIST_DEFAULT_OUTPUT_MODEL = Path(__file__).resolve().parent / "models" / SPECIALIST_MODEL_FILENAME

_SPECIALIST_LABEL_NORMALIZATION: dict[str, str] = {
    "magnoliopsida": SPECIALIST_TARGET_LABEL_MAGNOLIOPSIDA,
    "magno": SPECIALIST_TARGET_LABEL_MAGNOLIOPSIDA,
    "seagrass": SPECIALIST_TARGET_LABEL_MAGNOLIOPSIDA,
    "zostera": SPECIALIST_TARGET_LABEL_MAGNOLIOPSIDA,
    "4": SPECIALIST_TARGET_LABEL_MAGNOLIOPSIDA,
    "microphytobenthos": SPECIALIST_TARGET_LABEL_MICROPHYTOBENTHOS,
    "mpb": SPECIALIST_TARGET_LABEL_MICROPHYTOBENTHOS,
    "bacillariophyceae": SPECIALIST_TARGET_LABEL_MICROPHYTOBENTHOS,
    "diatom": SPECIALIST_TARGET_LABEL_MICROPHYTOBENTHOS,
    "diatoms": SPECIALIST_TARGET_LABEL_MICROPHYTOBENTHOS,
    "5": SPECIALIST_TARGET_LABEL_MICROPHYTOBENTHOS,
}


def _normalize_label_text(value: Any) -> str:
    """Normalize label strings for robust comparisons."""
    return "".join(ch for ch in str(value).strip().lower() if ch.isalnum())


def canonicalize_specialist_label(value: Any) -> str | None:
    """Map raw label text to one of the two canonical specialist class labels."""
    return _SPECIALIST_LABEL_NORMALIZATION.get(_normalize_label_text(value))


def is_class45_specialist_model_path(model_path: str | Path | None) -> bool:
    """Return True when the path points to the dedicated class-4/5 specialist model."""
    if model_path is None:
        return False
    return Path(model_path).name.lower() == SPECIALIST_MODEL_FILENAME.lower()


def resolve_class45_specialist_model_path(main_model_path: str | Path | None) -> Path | None:
    """Resolve the default specialist model path alongside a selected main model."""
    if main_model_path is None:
        return None
    resolved_main_model = Path(main_model_path).expanduser().resolve()
    if is_class45_specialist_model_path(resolved_main_model):
        return None
    candidate = resolved_main_model.parent / SPECIALIST_MODEL_FILENAME
    if candidate.exists() and candidate.is_file():
        return candidate
    return None


def specialist_required_raw_columns() -> tuple[str, ...]:
    """Return the raw reflectance columns needed by the class-4/5 specialist."""
    return SPECIALIST_RAW_COLUMNS


def specialist_main_output_class_ids() -> tuple[int, int]:
    """Return the main-model output class ids covered by the specialist."""
    return tuple(SPECIALIST_MAIN_OUTPUT_CLASS_BY_LABEL.values())


def prepare_class45_specialist_feature_dataframe(
    frame: pd.DataFrame,
    *,
    context: str,
) -> pd.DataFrame:
    """Build the exact specialist feature table from raw reflectance inputs."""
    ensure_required_columns(frame, SPECIALIST_RAW_COLUMNS, context=f"{context} (specialist raw inputs)")
    raw_frame = frame.loc[:, SPECIALIST_RAW_COLUMNS].apply(pd.to_numeric, errors="coerce")
    standardized_frame = recompute_standardised_reflectance(raw_frame, SPECIALIST_RAW_COLUMNS)
    indices_frame = recompute_ndvi_ndwi(raw_frame)
    feature_frame = pd.concat([raw_frame, standardized_frame, indices_frame], axis=1)
    ensure_required_columns(
        feature_frame,
        SPECIALIST_FEATURE_COLUMNS,
        context=f"{context} (prepared specialist inputs)",
    )
    return feature_frame.loc[:, SPECIALIST_FEATURE_COLUMNS].fillna(0.0)


def build_class45_specialist_training_dataframe(
    frame: pd.DataFrame,
    *,
    label_column: str = "True_Class",
) -> pd.DataFrame:
    """Filter, canonicalize, and deduplicate training rows for the 4-vs-5 specialist."""
    ensure_required_columns(frame, (label_column,), context="Specialist training data")
    canonical_labels = frame[label_column].map(canonicalize_specialist_label)
    selected_mask = canonical_labels.notna()
    if not bool(selected_mask.any()):
        raise ValueError(
            "Specialist training data does not contain any Magnoliopsida or Microphytobenthos rows."
        )

    feature_frame = prepare_class45_specialist_feature_dataframe(
        frame.loc[selected_mask],
        context="Specialist training data",
    )
    training_frame = pd.concat(
        [
            canonical_labels.loc[selected_mask].rename(label_column).reset_index(drop=True),
            feature_frame.reset_index(drop=True),
        ],
        axis=1,
    )
    training_frame = training_frame.drop_duplicates(
        subset=[label_column, *SPECIALIST_FEATURE_COLUMNS]
    ).reset_index(drop=True)
    return training_frame


def extract_class45_specialist_metadata(learner: Any) -> dict[str, Any] | None:
    """Extract normalized metadata when a learner is the dedicated class-4/5 specialist."""
    raw_metadata = getattr(learner, "ice_creams_model_metadata", None)
    if not isinstance(raw_metadata, dict):
        return None
    if str(raw_metadata.get("specialist_role") or "").strip() != SPECIALIST_ROLE_CLASS4_CLASS5:
        return None

    model_metadata = extract_model_metadata(learner)
    return {
        **model_metadata,
        "specialist_role": SPECIALIST_ROLE_CLASS4_CLASS5,
        "specialist_display_name": str(raw_metadata.get("specialist_display_name") or SPECIALIST_DISPLAY_NAME),
        "specialist_feature_profile_label": str(
            raw_metadata.get("specialist_feature_profile_label") or SPECIALIST_FEATURE_PROFILE_LABEL
        ),
        "specialist_target_labels": [
            canonicalize_specialist_label(label_name) or str(label_name)
            for label_name in raw_metadata.get("specialist_target_labels") or SPECIALIST_TARGET_LABELS
        ],
        "specialist_target_output_class_ids": [
            int(class_id)
            for class_id in raw_metadata.get("specialist_target_output_class_ids")
            or specialist_main_output_class_ids()
        ],
        "specialist_raw_bands": [
            str(band_name) for band_name in raw_metadata.get("specialist_raw_bands") or SPECIALIST_RAW_BANDS
        ],
        "specialist_feature_columns": [
            str(column_name)
            for column_name in raw_metadata.get("specialist_feature_columns") or SPECIALIST_FEATURE_COLUMNS
        ],
        "specialist_use_standardized_reflectance": bool(
            raw_metadata.get("specialist_use_standardized_reflectance", True)
        ),
        "specialist_use_indices": bool(raw_metadata.get("specialist_use_indices", True)),
    }


def map_specialist_labels_to_main_output_class_ids(
    raw_labels: Iterable[Any],
) -> np.ndarray:
    """Map predicted specialist labels to the main model's one-based class ids."""
    mapped_class_ids: list[int] = []
    unsupported_labels: list[str] = []

    for raw_label in raw_labels:
        canonical_label = canonicalize_specialist_label(raw_label)
        if canonical_label is None:
            unsupported_labels.append(str(raw_label))
            continue
        mapped_class_ids.append(int(SPECIALIST_MAIN_OUTPUT_CLASS_BY_LABEL[canonical_label]))

    if unsupported_labels:
        unsupported_preview = ", ".join(unsupported_labels[:6])
        suffix = f", +{len(unsupported_labels) - 6} more" if len(unsupported_labels) > 6 else ""
        raise ValueError(
            "The specialist model predicted unsupported labels: "
            f"{unsupported_preview}{suffix}"
        )

    return np.asarray(mapped_class_ids, dtype=np.int16)


def predict_class45_specialist(
    frame: pd.DataFrame,
    specialist_learner: Any,
    specialist_model_metadata: dict[str, Any],
    *,
    batch_size: int,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Run the specialist model and return mapped main-model class ids plus confidences."""
    feature_frame = prepare_class45_specialist_feature_dataframe(
        frame,
        context="Class 4/5 specialist inference",
    )
    probabilities = predict_model_probabilities(
        specialist_learner,
        feature_frame,
        specialist_model_metadata,
        batch_size=batch_size,
    )
    class_indices = probabilities.argmax(dim=1).cpu().numpy().astype(np.int16)
    confidence_values = probabilities.max(dim=1).values.cpu().numpy().astype(np.float32)
    vocab = list(getattr(specialist_learner.dls, "vocab", []) or [])
    predicted_labels = [
        str(vocab[class_index]) if 0 <= class_index < len(vocab) else str(class_index)
        for class_index in class_indices
    ]
    mapped_main_class_ids = map_specialist_labels_to_main_output_class_ids(predicted_labels)
    return mapped_main_class_ids, confidence_values, predicted_labels


def _stratified_random_split(
    labels: pd.Series,
    *,
    valid_pct: float,
    seed: int,
) -> tuple[list[int], list[int]]:
    """Build reproducible per-class train/validation indices."""
    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    valid_indices: list[int] = []

    for _, class_indices in labels.groupby(labels, sort=True).groups.items():
        index_array = np.asarray(list(class_indices), dtype=np.int64)
        if len(index_array) < 2:
            raise ValueError(
                "Each specialist class needs at least two rows so both train and validation splits are non-empty."
            )
        shuffled = rng.permutation(index_array)
        valid_count = int(round(len(shuffled) * valid_pct))
        valid_count = max(1, min(len(shuffled) - 1, valid_count))
        valid_indices.extend(shuffled[:valid_count].tolist())
        train_indices.extend(shuffled[valid_count:].tolist())

    return sorted(train_indices), sorted(valid_indices)


def train_class45_specialist_model(
    training_source: str | list[str] | tuple[str, ...] = str(SPECIALIST_DEFAULT_TRAINING_SOURCE),
    output_model: str = str(SPECIALIST_DEFAULT_OUTPUT_MODEL),
    *,
    epochs: int = 10,
    valid_pct: float = 0.3,
    batch_size: int = 4096,
    seed: int = 42,
    status_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """Train the dedicated Magnoliopsida-vs-Microphytobenthos specialist model."""
    if epochs < 1:
        raise ValueError("Epochs must be at least 1.")
    if not 0 < valid_pct < 1:
        raise ValueError("Validation split must be between 0 and 1.")
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    csv_files = discover_training_csvs(training_source)
    _emit_status(status_callback, f"Found {len(csv_files)} specialist training CSV files")
    _emit_status(status_callback, f"Specialist feature profile: {SPECIALIST_FEATURE_PROFILE_LABEL}")
    _emit_progress(progress_callback, 0.04)

    required_columns = ["True_Class", *SPECIALIST_RAW_COLUMNS]
    frames: list[pd.DataFrame] = []
    for index, csv_file in enumerate(csv_files, start=1):
        _emit_status(
            status_callback,
            f"Loading specialist CSV {index}/{len(csv_files)}: {Path(csv_file).name}",
        )
        frames.append(pd.read_csv(csv_file, usecols=lambda column_name: column_name in required_columns))
        _emit_progress(progress_callback, 0.04 + (0.20 * (index / len(csv_files))))

    _emit_status(status_callback, "Combining and deduplicating specialist training rows")
    source_frame = pd.concat(frames, ignore_index=True).dropna(how="all")
    training_frame = build_class45_specialist_training_dataframe(source_frame)
    label_column = "True_Class"
    class_counts = {
        str(label_name): int(count)
        for label_name, count in training_frame[label_column].value_counts().sort_index().items()
    }
    _emit_status(
        status_callback,
        (
            "Specialist class counts after deduplication: "
            + ", ".join(f"{label_name}={count:,}" for label_name, count in class_counts.items())
        ),
    )
    _emit_progress(progress_callback, 0.30)

    feature_columns = list(SPECIALIST_FEATURE_COLUMNS)
    train_indices, valid_indices = _stratified_random_split(
        training_frame[label_column],
        valid_pct=valid_pct,
        seed=seed,
    )

    _emit_status(status_callback, "Preparing fastai tabular data loaders for the specialist model")
    to_nn = TabularPandas(
        training_frame,
        [FillMissing],
        [],
        feature_columns,
        splits=(train_indices, valid_indices),
        y_names=label_column,
        y_block=CategoryBlock(),
    )
    dls = to_nn.dataloaders(bs=batch_size)
    _emit_progress(progress_callback, 0.44)

    vocab = [str(label_name) for label_name in dls.vocab]
    train_label_counts = (
        training_frame.iloc[train_indices][label_column]
        .value_counts()
        .reindex(vocab, fill_value=1)
        .astype(np.float32)
    )
    class_weight_values = (
        train_label_counts.sum() / (len(train_label_counts) * train_label_counts)
    ).to_numpy(dtype=np.float32, copy=True)

    _emit_status(status_callback, "Building specialist tabular neural network")
    learn = tabular_learner(
        dls,
        n_out=len(vocab),
        metrics=accuracy,
        loss_func=CrossEntropyLossFlat(weight=torch.tensor(class_weight_values, dtype=torch.float32)),
    )
    attach_model_metadata(
        learn,
        model_family=MODEL_FAMILY_TABULAR_DENSE,
        feature_mode=FEATURE_MODE_HIGH_SPATIAL_ACCURACY,
        required_feature_names=feature_columns,
    )
    learn.ice_creams_model_metadata.update(
        {
            "specialist_role": SPECIALIST_ROLE_CLASS4_CLASS5,
            "specialist_display_name": SPECIALIST_DISPLAY_NAME,
            "specialist_feature_profile_label": SPECIALIST_FEATURE_PROFILE_LABEL,
            "specialist_target_labels": list(SPECIALIST_TARGET_LABELS),
            "specialist_target_output_class_ids": list(specialist_main_output_class_ids()),
            "specialist_raw_bands": list(SPECIALIST_RAW_BANDS),
            "specialist_feature_columns": list(SPECIALIST_FEATURE_COLUMNS),
            "specialist_use_standardized_reflectance": True,
            "specialist_use_indices": True,
            "specialist_class_counts": class_counts,
        }
    )
    _emit_progress(progress_callback, 0.52)

    progress_cb = TrainingProgressCallback(
        total_epochs=epochs + 1,
        base_progress=0.52,
        progress_span=0.34,
        status_callback=status_callback,
        progress_callback=progress_callback,
    )
    learn.add_cb(progress_cb)
    try:
        learn.fine_tune(epochs)
    finally:
        learn.remove_cb(progress_cb)

    _emit_status(status_callback, "Running final specialist validation")
    validation = learn.validate()
    accuracy_value = float(validation[1]) if len(validation) > 1 else None
    _emit_progress(progress_callback, 0.92)

    output_path = Path(_normalise_model_path(output_model))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    _emit_status(status_callback, f"Exporting specialist model to {output_path}")
    learn.export(str(output_path))
    _emit_progress(progress_callback, 1.0)
    _emit_status(status_callback, f"Completed. Specialist model saved to {output_path}")

    return {
        "model_path": str(output_path),
        "rows": int(len(training_frame)),
        "csv_files": int(len(csv_files)),
        "classes": int(len(vocab)),
        "accuracy": accuracy_value,
        "specialist_display_name": SPECIALIST_DISPLAY_NAME,
        "specialist_feature_profile_label": SPECIALIST_FEATURE_PROFILE_LABEL,
        "class_counts": class_counts,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the ICE CREAMS Magnoliopsida vs Microphytobenthos specialist model"
    )
    parser.add_argument(
        "training_source",
        nargs="?",
        default=str(SPECIALIST_DEFAULT_TRAINING_SOURCE),
        help="CSV file, list of CSV files, or folder containing the cleaned V1.3 training tables",
    )
    parser.add_argument(
        "output_model",
        nargs="?",
        default=str(SPECIALIST_DEFAULT_OUTPUT_MODEL),
        help="Output path for the exported specialist .pkl model",
    )
    parser.add_argument("--epochs", type=int, default=10, help="Number of fine-tuning epochs")
    parser.add_argument(
        "--valid-pct",
        type=float,
        default=0.3,
        help="Fraction of data to hold back for validation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4096,
        help="Batch size for fastai data loaders",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    result = train_class45_specialist_model(
        training_source=args.training_source,
        output_model=args.output_model,
        epochs=args.epochs,
        valid_pct=args.valid_pct,
        batch_size=args.batch_size,
        seed=args.seed,
        status_callback=print,
    )
    print(f"Validation accuracy: {result['accuracy']}")
