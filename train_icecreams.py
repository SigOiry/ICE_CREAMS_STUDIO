#!/usr/bin/env python3
"""Train an ICE CREAMS tabular or spectral 1D CNN model from CSV training data."""

from __future__ import annotations

import argparse
import glob
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from fastai.callback.core import Callback
from fastai.learner import Learner
from fastai.tabular.all import (
    CategoryBlock,
    FillMissing,
    RandomSplitter,
    TabularPandas,
    accuracy,
    range_of,
    tabular_learner,
)

from ice_creams_feature_modes import (
    DEFAULT_FEATURE_MODE,
    FEATURE_MODE_CHOICES,
    build_training_dataframe,
    feature_mode_label,
    normalize_feature_mode,
)
from ice_creams_model_families import (
    apply_sequence_normalization,
    DEFAULT_MODEL_FAMILY,
    DEFAULT_SPECTRAL_CNN_USE_STANDARDIZED_REFLECTANCE,
    MODEL_FAMILY_CHOICES,
    MODEL_FAMILY_SPECTRAL_1D_CNN,
    attach_model_metadata,
    build_spectral_cnn_learner,
    model_family_label,
    normalize_model_family,
    prepare_sequence_feature_dataframe,
    compute_sequence_normalization,
    spectral_cnn_sequence_input_label,
    sequence_channel_feature_names_for_mode,
)


def _emit_status(status_callback: Callable[[str], None] | None, message: str) -> None:
    """Safely emit a status update for CLI or UI callers."""
    if status_callback is not None:
        status_callback(message)


def _emit_progress(progress_callback: Callable[[float], None] | None, value: float) -> None:
    """Safely emit bounded progress values between 0 and 1."""
    if progress_callback is not None:
        progress_callback(max(0.0, min(1.0, value)))


def _normalise_model_path(output_model: str) -> str:
    """Ensure exported model paths always end with .pkl."""
    return output_model if output_model.lower().endswith(".pkl") else f"{output_model}.pkl"


def _normalise_csv_path(csv_path: str) -> str:
    """Validate and normalise a CSV file path."""
    candidate = Path(csv_path).expanduser()
    if not candidate.exists():
        raise FileNotFoundError(f"Training CSV not found: {csv_path}")
    if not candidate.is_file():
        raise ValueError(f"Training CSV path must point to a file: {csv_path}")
    if candidate.suffix.lower() != ".csv":
        raise ValueError(f"Training data file must be a CSV: {csv_path}")
    return str(candidate.resolve())


def discover_training_csvs(training_source: str | list[str] | tuple[str, ...]) -> list[str]:
    """Discover training CSV files from selected file(s) or a directory tree."""
    if isinstance(training_source, (list, tuple)):
        if not training_source:
            raise ValueError("At least one training CSV file is required.")
        normalised = sorted(
            {
                _normalise_csv_path(str(csv_path).strip())
                for csv_path in training_source
                if str(csv_path).strip()
            }
        )
        if not normalised:
            raise ValueError("At least one training CSV file is required.")
        return normalised

    source_value = str(training_source).strip() if training_source is not None else ""
    if not source_value:
        raise ValueError("A training data path is required.")

    if os.path.isfile(source_value):
        return [_normalise_csv_path(source_value)]

    if not os.path.isdir(source_value):
        raise FileNotFoundError(f"Training data path not found: {source_value}")

    csv_files = sorted(
        {
            str(Path(csv_path).resolve())
            for csv_path in glob.glob(os.path.join(source_value, "**", "*.csv"), recursive=True)
        }
    )
    if not csv_files:
        raise FileNotFoundError(f"No CSV files were found in {source_value}")

    return csv_files


class TrainingProgressCallback(Callback):
    """Push epoch-level progress updates out to a CLI or UI."""

    order = 60

    def __init__(
        self,
        total_epochs: int,
        base_progress: float,
        progress_span: float,
        status_callback: Callable[[str], None] | None = None,
        progress_callback: Callable[[float], None] | None = None,
    ) -> None:
        self.total_epochs = max(total_epochs, 1)
        self.base_progress = base_progress
        self.progress_span = progress_span
        self.status_callback = status_callback
        self.progress_callback = progress_callback
        self.completed_epochs = 0

    def before_fit(self) -> None:
        if self.completed_epochs == 0:
            _emit_status(self.status_callback, "Training model")
            _emit_progress(self.progress_callback, self.base_progress)

    def after_epoch(self) -> None:
        self.completed_epochs += 1
        fraction = min(self.completed_epochs / self.total_epochs, 1.0)
        _emit_status(
            self.status_callback,
            f"Completed epoch {self.completed_epochs}/{self.total_epochs}",
        )
        _emit_progress(
            self.progress_callback,
            self.base_progress + (self.progress_span * fraction),
        )


def train_model(
    training_source: str | list[str] | tuple[str, ...],
    output_model: str,
    epochs: int = 20,
    valid_pct: float = 0.3,
    batch_size: int = 4096,
    seed: int = 42,
    feature_mode: str = DEFAULT_FEATURE_MODE,
    model_family: str = DEFAULT_MODEL_FAMILY,
    spectral_cnn_use_standardized_reflectance: bool = DEFAULT_SPECTRAL_CNN_USE_STANDARDIZED_REFLECTANCE,
    status_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[float], None] | None = None,
) -> dict[str, Any]:
    """
    Train a fastai tabular learner from ICE CREAMS CSV training data.

    The training source can be a CSV path, a folder containing CSVs, or a list
    of CSV file paths. All CSVs are merged, and `True_Class` is used as target.
    """

    if epochs < 1:
        raise ValueError("Epochs must be at least 1.")
    if not 0 < valid_pct < 1:
        raise ValueError("Validation split must be between 0 and 1.")
    if batch_size < 1:
        raise ValueError("Batch size must be at least 1.")

    resolved_feature_mode = normalize_feature_mode(feature_mode)
    selected_mode_label = feature_mode_label(resolved_feature_mode)
    resolved_model_family = normalize_model_family(model_family)
    selected_model_family_label = model_family_label(resolved_model_family)
    selected_sequence_input_label = spectral_cnn_sequence_input_label(
        spectral_cnn_use_standardized_reflectance
    )

    csv_files = discover_training_csvs(training_source)
    _emit_status(status_callback, f"Found {len(csv_files)} training CSV files")
    _emit_status(status_callback, f"Training feature mode: {selected_mode_label}")
    _emit_status(status_callback, f"Training model method: {selected_model_family_label}")
    if resolved_model_family == MODEL_FAMILY_SPECTRAL_1D_CNN:
        _emit_status(status_callback, f"Training spectral inputs: {selected_sequence_input_label}")
    _emit_progress(progress_callback, 0.04)

    frames: list[pd.DataFrame] = []
    for index, csv_file in enumerate(csv_files, start=1):
        _emit_status(
            status_callback,
            f"Loading CSV {index}/{len(csv_files)}: {os.path.basename(csv_file)}",
        )
        frames.append(pd.read_csv(csv_file))
        _emit_progress(progress_callback, 0.04 + (0.31 * (index / len(csv_files))))

    _emit_status(status_callback, "Combining training tables")
    df_nn = pd.concat(frames, ignore_index=True).dropna(how="all")
    _emit_progress(progress_callback, 0.38)

    dep_var = "True_Class"
    learn: Learner
    n_classes: int
    feature_columns: list[str]
    label_series = df_nn[dep_var].astype("string").fillna("").str.strip()
    if label_series.eq("").any():
        raise ValueError("Training data contains missing values in the True_Class column.")

    if resolved_model_family == MODEL_FAMILY_SPECTRAL_1D_CNN:
        _emit_status(status_callback, f"Preparing {selected_mode_label} spectral sequences")
        sequence_channel_feature_names = sequence_channel_feature_names_for_mode(
            resolved_feature_mode,
            use_standardized_reflectance=spectral_cnn_use_standardized_reflectance,
        )
        sequence_frame = prepare_sequence_feature_dataframe(
            df_nn,
            feature_mode=resolved_feature_mode,
            sequence_channel_feature_names=sequence_channel_feature_names,
            context="Training data",
        )
        feature_columns = list(sequence_frame.columns)
        splits = RandomSplitter(valid_pct=valid_pct, seed=seed)(range_of(sequence_frame))
        train_indices, valid_indices = splits
        if not train_indices or not valid_indices:
            raise ValueError("Training/validation split produced an empty partition. Adjust the split or input data.")

        vocab = sorted(label_series.unique().tolist())
        if len(vocab) < 2:
            raise ValueError("At least two distinct classes are required to train a spectral 1D CNN model.")

        label_codes = pd.Categorical(label_series, categories=vocab).codes.astype(np.int64, copy=False)
        if (label_codes < 0).any():
            raise ValueError("Training data contains labels that could not be encoded for the spectral 1D CNN.")

        sequence_values = np.stack(
            [
                sequence_frame.loc[:, channel_feature_names].to_numpy(dtype=np.float32, copy=True)
                for channel_feature_names in sequence_channel_feature_names
            ],
            axis=1,
        )
        sequence_normalization = compute_sequence_normalization(sequence_values[train_indices])
        normalized_values = apply_sequence_normalization(
            sequence_values,
            sequence_normalization,
        )

        _emit_status(status_callback, "Preparing fastai spectral 1D CNN data loaders")
        learn = build_spectral_cnn_learner(
            normalized_values[train_indices],
            label_codes[train_indices],
            normalized_values[valid_indices],
            label_codes[valid_indices],
            vocab=vocab,
            sequence_feature_names=feature_columns,
            batch_size=batch_size,
        )
        n_classes = len(vocab)
        selected_sequence_input_label = spectral_cnn_sequence_input_label(
            len(sequence_channel_feature_names) > 1
        )
        _emit_progress(progress_callback, 0.5)

        _emit_status(status_callback, "Building spectral 1D CNN")
        attach_model_metadata(
            learn,
            model_family=resolved_model_family,
            feature_mode=resolved_feature_mode,
            required_feature_names=feature_columns,
            sequence_feature_names=feature_columns,
            sequence_channel_feature_names=sequence_channel_feature_names,
            sequence_normalization=sequence_normalization,
        )
        _emit_progress(progress_callback, 0.56)

        progress_cb = TrainingProgressCallback(
            total_epochs=epochs,
            base_progress=0.56,
            progress_span=0.34,
            status_callback=status_callback,
            progress_callback=progress_callback,
        )
        learn.add_cb(progress_cb)
        try:
            learn.fit_one_cycle(epochs, 1e-3)
        finally:
            learn.remove_cb(progress_cb)
    else:
        _emit_status(status_callback, f"Preparing {selected_mode_label} training features")
        df_nn, feature_columns, resolved_feature_mode = build_training_dataframe(
            df_nn,
            feature_mode=resolved_feature_mode,
            label_column=dep_var,
        )
        selected_mode_label = feature_mode_label(resolved_feature_mode)

        _emit_status(status_callback, "Preparing fastai tabular data loaders")
        splits = RandomSplitter(valid_pct=valid_pct, seed=seed)(range_of(df_nn))
        to_nn = TabularPandas(
            df_nn,
            [FillMissing],
            [],
            feature_columns,
            splits=splits,
            y_names=dep_var,
            y_block=CategoryBlock(),
        )
        dls = to_nn.dataloaders(bs=batch_size)
        _emit_progress(progress_callback, 0.5)

        n_classes = int(df_nn[dep_var].dropna().nunique())
        _emit_status(status_callback, "Building tabular neural network")
        learn = tabular_learner(dls, n_out=n_classes, metrics=accuracy)
        attach_model_metadata(
            learn,
            model_family=resolved_model_family,
            feature_mode=resolved_feature_mode,
            required_feature_names=feature_columns,
        )
        _emit_progress(progress_callback, 0.56)

        progress_cb = TrainingProgressCallback(
            total_epochs=epochs + 1,
            base_progress=0.56,
            progress_span=0.34,
            status_callback=status_callback,
            progress_callback=progress_callback,
        )

        learn.add_cb(progress_cb)
        try:
            learn.fine_tune(epochs)
        finally:
            learn.remove_cb(progress_cb)

    _emit_status(status_callback, "Running final validation")
    validation = learn.validate()
    accuracy_value = float(validation[1]) if len(validation) > 1 else None
    _emit_progress(progress_callback, 0.93)

    output_path = Path(_normalise_model_path(output_model))
    output_path.parent.mkdir(parents=True, exist_ok=True)

    _emit_status(status_callback, f"Exporting model to {output_path}")
    learn.export(str(output_path))
    _emit_progress(progress_callback, 1.0)
    _emit_status(status_callback, f"Completed. Model saved to {output_path}")

    return {
        "model_path": str(output_path),
        "rows": int(len(df_nn)),
        "csv_files": int(len(csv_files)),
        "classes": n_classes,
        "accuracy": accuracy_value,
        "model_family": resolved_model_family,
        "model_family_label": selected_model_family_label,
        "feature_mode": resolved_feature_mode,
        "feature_mode_label": selected_mode_label,
        "sequence_use_standardized_reflectance": (
            bool(len(sequence_channel_feature_names) > 1)
            if resolved_model_family == MODEL_FAMILY_SPECTRAL_1D_CNN
            else False
        ),
        "sequence_input_label": (
            selected_sequence_input_label
            if resolved_model_family == MODEL_FAMILY_SPECTRAL_1D_CNN
            else ""
        ),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train an ICE CREAMS model")
    parser.add_argument(
        "training_source",
        nargs="?",
        default=os.path.join("Data", "Input", "Training"),
        help="CSV file or folder containing ICE CREAMS training CSVs",
    )
    parser.add_argument(
        "output_model",
        nargs="?",
        default="ICECREAMS_custom.pkl",
        help="Output path for the exported .pkl model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of fine-tuning epochs",
    )
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
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for train/validation splitting",
    )
    parser.add_argument(
        "--feature-mode",
        choices=FEATURE_MODE_CHOICES,
        default=DEFAULT_FEATURE_MODE,
        help="Training feature mode",
    )
    parser.add_argument(
        "--model-family",
        choices=MODEL_FAMILY_CHOICES,
        default=DEFAULT_MODEL_FAMILY,
        help="Training model family",
    )
    return parser


if __name__ == "__main__":
    args = _build_parser().parse_args()
    result = train_model(
        training_source=args.training_source,
        output_model=args.output_model,
        epochs=args.epochs,
        valid_pct=args.valid_pct,
        batch_size=args.batch_size,
        seed=args.seed,
        feature_mode=args.feature_mode,
        model_family=args.model_family,
        status_callback=print,
    )
    print(f"Validation accuracy: {result['accuracy']}")
