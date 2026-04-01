#!/usr/bin/env python3
"""Shared model-family constants, metadata, and spectral 1D CNN helpers."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
from fastai.data.core import DataLoaders
from fastai.learner import Learner
from fastai.losses import CrossEntropyLossFlat
from fastai.metrics import accuracy
from torch import nn
from torch.utils.data import Dataset

from ice_creams_feature_modes import (
    FEATURE_MODE_CHOICES,
    ensure_required_columns,
    extract_learner_required_feature_names,
    feature_mode_label,
    infer_feature_mode_from_feature_names,
    normalize_feature_mode,
    raw_column_name,
    RAW_BANDS_BY_MODE,
    recompute_standardised_reflectance,
    standardised_column_name,
)


MODEL_FAMILY_TABULAR_DENSE = "tabular_dense"
MODEL_FAMILY_SPECTRAL_1D_CNN = "spectral_1d_cnn"
DEFAULT_MODEL_FAMILY = MODEL_FAMILY_TABULAR_DENSE
DEFAULT_SPECTRAL_CNN_USE_STANDARDIZED_REFLECTANCE = True

MODEL_FAMILY_LABELS: dict[str, str] = {
    MODEL_FAMILY_TABULAR_DENSE: "Tabular Dense Network",
    MODEL_FAMILY_SPECTRAL_1D_CNN: "Spectral 1D CNN",
}
MODEL_FAMILY_CHOICES = tuple(MODEL_FAMILY_LABELS)
SPECTRAL_CNN_SEQUENCE_INPUT_LABEL_RAW_ONLY = "Raw Reflectance Only"
SPECTRAL_CNN_SEQUENCE_INPUT_LABEL_RAW_AND_STANDARDIZED = "Raw + Standardized Reflectance"


def normalize_model_family(model_family: str | None) -> str:
    """Validate and canonicalize a model-family identifier."""
    normalized = str(model_family or DEFAULT_MODEL_FAMILY).strip().lower()
    normalized = normalized.replace("-", "_").replace(" ", "_")
    if normalized not in MODEL_FAMILY_LABELS:
        valid_values = ", ".join(MODEL_FAMILY_CHOICES)
        raise ValueError(f"Invalid model family '{model_family}'. Use one of: {valid_values}.")
    return normalized


def model_family_label(model_family: str | None) -> str:
    """Return the user-facing label for a model family."""
    return MODEL_FAMILY_LABELS[normalize_model_family(model_family)]


def _normalise_name_list(values: Iterable[Any] | None) -> list[str]:
    """Normalize and de-duplicate string values while preserving order."""
    normalized: list[str] = []
    seen: set[str] = set()
    for value in values or []:
        item = str(value).strip()
        if item and item not in seen:
            seen.add(item)
            normalized.append(item)
    return normalized


def spectral_cnn_uses_standardized_reflectance(
    sequence_channel_feature_names: Any | None = None,
    sequence_feature_names: Iterable[Any] | None = None,
) -> bool:
    """Return True when spectral CNN inputs include standardized reflectance."""
    channel_groups = _normalize_sequence_channel_feature_names(sequence_channel_feature_names)
    if channel_groups:
        return any(
            feature_name.startswith("Reflectance_Stan_")
            for channel_group in channel_groups
            for feature_name in channel_group
        )

    ordered_feature_names = _normalise_name_list(sequence_feature_names)
    return any(feature_name.startswith("Reflectance_Stan_") for feature_name in ordered_feature_names)


def spectral_cnn_sequence_input_label(use_standardized_reflectance: bool | None) -> str:
    """Return the user-facing spectral-input description for spectral CNN models."""
    return (
        SPECTRAL_CNN_SEQUENCE_INPUT_LABEL_RAW_AND_STANDARDIZED
        if bool(use_standardized_reflectance)
        else SPECTRAL_CNN_SEQUENCE_INPUT_LABEL_RAW_ONLY
    )


def sequence_feature_names_for_mode(
    feature_mode: str | None,
    *,
    use_standardized_reflectance: bool = DEFAULT_SPECTRAL_CNN_USE_STANDARDIZED_REFLECTANCE,
) -> list[str]:
    """Return the flattened ordered spectral CNN inputs for a feature mode."""
    channel_groups = sequence_channel_feature_names_for_mode(
        feature_mode,
        use_standardized_reflectance=use_standardized_reflectance,
    )
    return [feature_name for group in channel_groups for feature_name in group]


def sequence_channel_feature_names_for_mode(
    feature_mode: str | None,
    *,
    use_standardized_reflectance: bool = DEFAULT_SPECTRAL_CNN_USE_STANDARDIZED_REFLECTANCE,
) -> list[list[str]]:
    """Return ordered spectral CNN channel groups used by a feature mode."""
    resolved_mode = normalize_feature_mode(feature_mode)
    ordered_band_names = list(RAW_BANDS_BY_MODE[resolved_mode])
    raw_columns = [raw_column_name(band_name) for band_name in ordered_band_names]
    if not bool(use_standardized_reflectance):
        return [raw_columns]
    standardized_columns = [standardised_column_name(band_name) for band_name in ordered_band_names]
    return [raw_columns, standardized_columns]


def _normalize_sequence_channel_feature_names(value: Any) -> list[list[str]]:
    """Normalize nested sequence channel feature names while preserving order."""
    normalized_groups: list[list[str]] = []
    for group in value or []:
        normalized_group = _normalise_name_list(group)
        if normalized_group:
            normalized_groups.append(normalized_group)
    return normalized_groups


def _infer_sequence_channel_feature_names(
    sequence_feature_names: Iterable[Any] | None,
    feature_mode: str | None,
) -> list[list[str]]:
    """Infer channel-group metadata, preserving legacy raw-only spectral CNN ordering."""
    ordered_feature_names = _normalise_name_list(sequence_feature_names)
    if not ordered_feature_names:
        return sequence_channel_feature_names_for_mode(feature_mode)

    default_channel_groups = sequence_channel_feature_names_for_mode(feature_mode)
    default_flattened = [feature_name for group in default_channel_groups for feature_name in group]
    if ordered_feature_names == default_flattened:
        return default_channel_groups

    raw_only_channel_groups = sequence_channel_feature_names_for_mode(
        feature_mode,
        use_standardized_reflectance=False,
    )
    raw_only_flattened = [feature_name for group in raw_only_channel_groups for feature_name in group]
    if ordered_feature_names == raw_only_flattened:
        return raw_only_channel_groups

    return [ordered_feature_names]


def prepare_sequence_feature_dataframe(
    frame: pd.DataFrame,
    *,
    feature_mode: str | None = None,
    sequence_feature_names: Iterable[Any] | None = None,
    sequence_channel_feature_names: Any | None = None,
    context: str,
) -> pd.DataFrame:
    """Prepare the raw+standardized ordered spectral inputs used by spectral 1D CNN models."""
    resolved_mode = normalize_feature_mode(feature_mode)
    channel_groups = _normalize_sequence_channel_feature_names(sequence_channel_feature_names)
    ordered_feature_names = _normalise_name_list(sequence_feature_names)
    if not channel_groups:
        channel_groups = sequence_channel_feature_names_for_mode(resolved_mode)
    if not ordered_feature_names:
        ordered_feature_names = [feature_name for group in channel_groups for feature_name in group]

    required_raw_columns: list[str] = []
    for feature_name in ordered_feature_names:
        normalized_name = str(feature_name).strip()
        if normalized_name.startswith("Reflectance_Stan_"):
            required_raw_columns.append(normalized_name.replace("Reflectance_Stan_", "Reflectance_"))
        elif normalized_name.startswith("Reflectance_B"):
            required_raw_columns.append(normalized_name)
    required_raw_columns = _normalise_name_list(required_raw_columns)
    ensure_required_columns(
        frame,
        required_raw_columns,
        context=f"{context} (raw sequence inputs)",
    )
    raw_frame = frame.loc[:, required_raw_columns].apply(pd.to_numeric, errors="coerce")
    standardized_frame = recompute_standardised_reflectance(raw_frame, required_raw_columns)

    prepared_columns: dict[str, pd.Series] = {}
    for feature_name in ordered_feature_names:
        if feature_name.startswith("Reflectance_Stan_"):
            prepared_columns[feature_name] = pd.to_numeric(
                standardized_frame[feature_name],
                errors="coerce",
            )
        elif feature_name.startswith("Reflectance_B"):
            prepared_columns[feature_name] = pd.to_numeric(raw_frame[feature_name], errors="coerce")
        else:
            raise ValueError(
                f"{context} includes unsupported spectral 1D CNN feature '{feature_name}'. "
                "Only raw and standardized reflectance inputs are supported."
            )

    numeric_frame = pd.DataFrame(prepared_columns, index=frame.index)
    return numeric_frame.fillna(0.0)


def _normalize_sequence_normalization(
    normalization: Any,
    expected_shape: tuple[int, ...],
) -> dict[str, Any]:
    """Validate and normalize per-band sequence normalization metadata."""
    if not isinstance(normalization, dict):
        raise ValueError("Sequence normalization metadata must be a dictionary.")

    mean_values = np.asarray(normalization.get("mean", []), dtype=np.float32)
    std_values = np.asarray(normalization.get("std", []), dtype=np.float32)
    if len(expected_shape) == 2 and expected_shape[0] == 1:
        if mean_values.shape == (expected_shape[1],):
            mean_values = mean_values.reshape(expected_shape)
        if std_values.shape == (expected_shape[1],):
            std_values = std_values.reshape(expected_shape)
    elif mean_values.ndim == 1 and mean_values.size == int(np.prod(expected_shape)):
        mean_values = mean_values.reshape(expected_shape)
    elif std_values.ndim == 1 and std_values.size == int(np.prod(expected_shape)):
        std_values = std_values.reshape(expected_shape)

    if mean_values.shape != expected_shape or std_values.shape != expected_shape:
        raise ValueError(
            "Sequence normalization metadata does not match the expected number of spectral inputs."
        )

    std_values = np.where(np.isfinite(std_values) & (std_values > 0), std_values, 1.0)
    mean_values = np.where(np.isfinite(mean_values), mean_values, 0.0)
    return {
        "mean": mean_values.astype(np.float32).tolist(),
        "std": std_values.astype(np.float32).tolist(),
    }


def compute_sequence_normalization(sequence_values: np.ndarray) -> dict[str, Any]:
    """Compute per-band mean/std normalization statistics for spectral sequences."""
    values = np.asarray(sequence_values, dtype=np.float32)
    if values.ndim not in {2, 3}:
        raise ValueError("Sequence normalization expects a 2D [rows, bands] or 3D [rows, channels, bands] array.")
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    mean_values = values.mean(axis=0, dtype=np.float32)
    std_values = values.std(axis=0, dtype=np.float32)
    std_values = np.where(std_values > 0, std_values, 1.0).astype(np.float32)
    return {
        "mean": mean_values.astype(np.float32).tolist(),
        "std": std_values.astype(np.float32).tolist(),
    }


def apply_sequence_normalization(
    sequence_values: np.ndarray,
    normalization: dict[str, Any],
) -> np.ndarray:
    """Apply stored per-band normalization statistics to spectral sequence values."""
    values = np.asarray(sequence_values, dtype=np.float32)
    if values.ndim not in {2, 3}:
        raise ValueError("Sequence normalization expects a 2D [rows, bands] or 3D [rows, channels, bands] array.")
    expected_shape = values.shape[1:] if values.ndim == 3 else (values.shape[1],)
    stats = _normalize_sequence_normalization(normalization, expected_shape)
    mean_values = np.asarray(stats["mean"], dtype=np.float32)
    std_values = np.asarray(stats["std"], dtype=np.float32)
    values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
    normalized = (values - mean_values) / std_values
    return np.nan_to_num(normalized, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class SpectralSequenceDataset(Dataset):
    """Simple export-friendly dataset for spectral 1D CNN training and inference."""

    def __init__(
        self,
        features: Any,
        labels: Any | None = None,
        *,
        vocab: Iterable[str] | None = None,
        sequence_feature_names: Iterable[str] | None = None,
    ) -> None:
        feature_tensor = torch.as_tensor(features, dtype=torch.float32)
        if feature_tensor.ndim == 2:
            feature_tensor = feature_tensor.unsqueeze(1)
        if feature_tensor.ndim != 3:
            raise ValueError(
                "SpectralSequenceDataset expects features shaped [rows, bands] or [rows, channels, bands]."
            )

        self.features = feature_tensor.contiguous()
        self.labels = None if labels is None else torch.as_tensor(labels, dtype=torch.long).contiguous()
        self.vocab = list(vocab or [])
        self.sequence_feature_names = list(sequence_feature_names or [])

    def __len__(self) -> int:
        return int(self.features.shape[0])

    def __getitem__(self, index: int):
        if self.labels is None:
            return (self.features[index],)
        return self.features[index], self.labels[index]

    def new_empty(self) -> "SpectralSequenceDataset":
        """Return an empty dataset shell so FastAI export can strip training items."""
        empty_features = self.features[:0].clone()
        empty_labels = None if self.labels is None else self.labels[:0].clone()
        return SpectralSequenceDataset(
            empty_features,
            empty_labels,
            vocab=self.vocab,
            sequence_feature_names=self.sequence_feature_names,
        )


class Spectral1DCNN(nn.Module):
    """Compact 1D CNN that learns ordered spectral signatures from reflectance channels."""

    def __init__(self, sequence_length: int, n_classes: int, in_channels: int = 1) -> None:
        super().__init__()
        if sequence_length < 2:
            raise ValueError("Spectral1DCNN requires at least two ordered spectral inputs.")
        if n_classes < 2:
            raise ValueError("Spectral1DCNN requires at least two classes.")
        if in_channels < 1:
            raise ValueError("Spectral1DCNN requires at least one input channel.")

        hidden_channels = 32 if sequence_length >= 8 else 16
        deeper_channels = hidden_channels * 2
        self.network = nn.Sequential(
            nn.Conv1d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_channels, deeper_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(deeper_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.12),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(deeper_channels, max(32, hidden_channels)),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.18),
            nn.Linear(max(32, hidden_channels), n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2:
            x = x.unsqueeze(1)
        return self.network(x.float())


def build_spectral_cnn_learner(
    train_features: np.ndarray,
    train_labels: np.ndarray,
    valid_features: np.ndarray,
    valid_labels: np.ndarray,
    *,
    vocab: Iterable[str],
    sequence_feature_names: Iterable[str],
    batch_size: int,
) -> Learner:
    """Build a FastAI learner around the spectral 1D CNN model family."""
    vocab_list = list(vocab)
    feature_names = list(sequence_feature_names)
    train_ds = SpectralSequenceDataset(
        train_features,
        train_labels,
        vocab=vocab_list,
        sequence_feature_names=feature_names,
    )
    valid_ds = SpectralSequenceDataset(
        valid_features,
        valid_labels,
        vocab=vocab_list,
        sequence_feature_names=feature_names,
    )
    dls = DataLoaders.from_dsets(
        train_ds,
        valid_ds,
        bs=max(1, int(batch_size)),
        shuffle=True,
        num_workers=0,
    )
    model = Spectral1DCNN(
        sequence_length=train_ds.features.shape[-1],
        n_classes=len(vocab_list),
        in_channels=train_ds.features.shape[1],
    )
    return Learner(dls, model, loss_func=CrossEntropyLossFlat(), metrics=accuracy)


def attach_model_metadata(
    learner: Any,
    *,
    model_family: str,
    feature_mode: str,
    required_feature_names: Iterable[Any],
    sequence_feature_names: Iterable[Any] | None = None,
    sequence_channel_feature_names: Any | None = None,
    sequence_normalization: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Attach normalized ICE CREAMS model metadata to an exported learner."""
    resolved_model_family = normalize_model_family(model_family)
    resolved_feature_mode = normalize_feature_mode(feature_mode)
    required_names = _normalise_name_list(required_feature_names)
    sequence_names = _normalise_name_list(sequence_feature_names)
    sequence_channel_names = _normalize_sequence_channel_feature_names(sequence_channel_feature_names)

    metadata: dict[str, Any] = {
        "model_family": resolved_model_family,
        "feature_mode": resolved_feature_mode,
        "required_feature_names": required_names,
        "sequence_feature_names": sequence_names,
        "sequence_channel_feature_names": sequence_channel_names,
        "sequence_use_standardized_reflectance": False,
        "sequence_input_label": "",
        "sequence_normalization": {},
    }

    if resolved_model_family == MODEL_FAMILY_SPECTRAL_1D_CNN:
        if not sequence_channel_names:
            sequence_channel_names = _infer_sequence_channel_feature_names(
                sequence_names,
                resolved_feature_mode,
            )
            metadata["sequence_channel_feature_names"] = sequence_channel_names
        if not sequence_names:
            sequence_names = [feature_name for group in sequence_channel_names for feature_name in group]
            metadata["sequence_feature_names"] = sequence_names
        if not sequence_names:
            raise ValueError("Spectral 1D CNN metadata requires ordered sequence feature names.")
        channel_lengths = {len(channel_group) for channel_group in sequence_channel_names}
        if len(channel_lengths) != 1:
            raise ValueError("Spectral 1D CNN metadata requires channel groups with matching sequence lengths.")
        if sequence_normalization is None:
            raise ValueError("Spectral 1D CNN metadata requires sequence normalization statistics.")
        metadata["sequence_normalization"] = _normalize_sequence_normalization(
            sequence_normalization,
            (
                len(sequence_channel_names),
                len(sequence_channel_names[0]),
            ),
        )
        metadata["sequence_use_standardized_reflectance"] = spectral_cnn_uses_standardized_reflectance(
            sequence_channel_names,
            sequence_names,
        )
        metadata["sequence_input_label"] = spectral_cnn_sequence_input_label(
            metadata["sequence_use_standardized_reflectance"]
        )

    learner.ice_creams_model_metadata = metadata
    learner.ice_creams_model_family = resolved_model_family
    learner.ice_creams_feature_mode = resolved_feature_mode
    learner.ice_creams_required_feature_names = required_names
    learner.ice_creams_sequence_feature_names = sequence_names
    learner.ice_creams_sequence_channel_feature_names = sequence_channel_names
    return metadata


def _infer_cnn_feature_mode_from_sequence_names(sequence_feature_names: list[str]) -> str:
    """Infer feature mode for spectral 1D CNN models from raw or raw+standardized inputs."""
    feature_set = set(sequence_feature_names)
    for mode_name in FEATURE_MODE_CHOICES:
        raw_columns = sequence_channel_feature_names_for_mode(mode_name)[0]
        if feature_set in {set(raw_columns), set(sequence_feature_names_for_mode(mode_name))}:
            return mode_name
    raise ValueError(
        "Unable to infer feature mode from the spectral 1D CNN sequence inputs."
    )


def extract_model_metadata(learner: Any) -> dict[str, Any]:
    """Extract normalized ICE CREAMS model metadata from an exported learner."""
    explicit_metadata = getattr(learner, "ice_creams_model_metadata", None)
    if isinstance(explicit_metadata, dict) and explicit_metadata:
        resolved_model_family = normalize_model_family(explicit_metadata.get("model_family"))
        required_feature_names = _normalise_name_list(explicit_metadata.get("required_feature_names"))
        sequence_feature_names = _normalise_name_list(explicit_metadata.get("sequence_feature_names"))
        sequence_channel_feature_names = _normalize_sequence_channel_feature_names(
            explicit_metadata.get("sequence_channel_feature_names")
        )

        explicit_feature_mode = explicit_metadata.get("feature_mode")
        if explicit_feature_mode is None:
            if resolved_model_family == MODEL_FAMILY_TABULAR_DENSE:
                explicit_feature_mode = infer_feature_mode_from_feature_names(required_feature_names)
            else:
                explicit_feature_mode = _infer_cnn_feature_mode_from_sequence_names(
                    sequence_feature_names
                    or required_feature_names
                    or [feature_name for group in sequence_channel_feature_names for feature_name in group]
                )
        resolved_feature_mode = normalize_feature_mode(explicit_feature_mode)

        if resolved_model_family == MODEL_FAMILY_TABULAR_DENSE:
            if not required_feature_names:
                required_feature_names = extract_learner_required_feature_names(learner)
            sequence_normalization: dict[str, Any] = {}
        else:
            if not sequence_feature_names:
                if required_feature_names:
                    sequence_feature_names = required_feature_names
                elif sequence_channel_feature_names:
                    sequence_feature_names = [
                        feature_name
                        for channel_group in sequence_channel_feature_names
                        for feature_name in channel_group
                    ]
                else:
                    sequence_feature_names = sequence_feature_names_for_mode(resolved_feature_mode)
            if not sequence_channel_feature_names:
                sequence_channel_feature_names = _infer_sequence_channel_feature_names(
                    sequence_feature_names,
                    resolved_feature_mode
                )
            if not required_feature_names:
                required_feature_names = sequence_feature_names
            expected_shape = (
                (len(sequence_channel_feature_names), len(sequence_channel_feature_names[0]))
                if sequence_channel_feature_names
                else (len(sequence_feature_names),)
            )
            sequence_normalization = _normalize_sequence_normalization(
                explicit_metadata.get("sequence_normalization", {}),
                expected_shape,
            )

        return {
            "model_family": resolved_model_family,
            "model_family_label": model_family_label(resolved_model_family),
            "feature_mode": resolved_feature_mode,
            "feature_mode_label": feature_mode_label(resolved_feature_mode),
            "required_feature_names": required_feature_names,
            "sequence_feature_names": sequence_feature_names,
            "sequence_channel_feature_names": sequence_channel_feature_names,
            "sequence_use_standardized_reflectance": spectral_cnn_uses_standardized_reflectance(
                sequence_channel_feature_names,
                sequence_feature_names,
            ),
            "sequence_input_label": spectral_cnn_sequence_input_label(
                spectral_cnn_uses_standardized_reflectance(
                    sequence_channel_feature_names,
                    sequence_feature_names,
                )
            ),
            "sequence_normalization": sequence_normalization,
        }

    required_feature_names = extract_learner_required_feature_names(learner)
    inferred_feature_mode = infer_feature_mode_from_feature_names(required_feature_names)
    return {
        "model_family": MODEL_FAMILY_TABULAR_DENSE,
        "model_family_label": model_family_label(MODEL_FAMILY_TABULAR_DENSE),
        "feature_mode": inferred_feature_mode,
        "feature_mode_label": feature_mode_label(inferred_feature_mode),
        "required_feature_names": required_feature_names,
        "sequence_feature_names": [],
        "sequence_channel_feature_names": [],
        "sequence_use_standardized_reflectance": False,
        "sequence_input_label": "",
        "sequence_normalization": {},
    }


def predict_model_probabilities(
    learner: Any,
    model_input_frame: pd.DataFrame,
    model_metadata: dict[str, Any],
    *,
    batch_size: int,
) -> torch.Tensor:
    """Run model inference for either the legacy tabular learner or the spectral CNN."""
    resolved_model_family = normalize_model_family(model_metadata.get("model_family"))
    inference_batch_size = max(1, int(batch_size))

    if resolved_model_family == MODEL_FAMILY_TABULAR_DENSE:
        test_dl = learner.dls.test_dl(model_input_frame, bs=inference_batch_size)
        preds, _ = learner.get_preds(dl=test_dl)
        return preds

    sequence_feature_names = _normalise_name_list(
        model_metadata.get("sequence_feature_names") or model_metadata.get("required_feature_names")
    )
    sequence_channel_feature_names = _normalize_sequence_channel_feature_names(
        model_metadata.get("sequence_channel_feature_names")
    )
    if not sequence_channel_feature_names:
        sequence_channel_feature_names = [sequence_feature_names]
    sequence_df = prepare_sequence_feature_dataframe(
        model_input_frame,
        feature_mode=model_metadata.get("feature_mode"),
        sequence_feature_names=sequence_feature_names,
        sequence_channel_feature_names=sequence_channel_feature_names,
        context="Model inference",
    )
    sequence_values = np.stack(
        [
            sequence_df.loc[:, channel_feature_names].to_numpy(dtype=np.float32, copy=True)
            for channel_feature_names in sequence_channel_feature_names
        ],
        axis=1,
    )
    sequence_values = apply_sequence_normalization(
        sequence_values,
        model_metadata.get("sequence_normalization", {}),
    )
    sequence_tensor = torch.from_numpy(sequence_values)

    first_param = next(learner.model.parameters(), None)
    device = first_param.device if first_param is not None else torch.device("cpu")
    probability_batches: list[torch.Tensor] = []
    learner.model.eval()
    with torch.inference_mode():
        for start in range(0, len(sequence_tensor), inference_batch_size):
            batch_tensor = sequence_tensor[start : start + inference_batch_size].to(device)
            logits = learner.model(batch_tensor)
            probability_batches.append(torch.softmax(logits, dim=1).cpu())

    if probability_batches:
        return torch.cat(probability_batches, dim=0)

    n_classes = len(getattr(learner.dls, "vocab", []) or [])
    return torch.empty((0, n_classes), dtype=torch.float32)
