#!/usr/bin/env python3
"""Validate an ICE CREAMS exported model on a labelled table."""

from __future__ import annotations

import argparse
import difflib
import re
import unicodedata
from pathlib import Path
from typing import Any, Callable

import pandas as pd
from fastai.tabular.all import load_learner

from ice_creams_feature_modes import (
    prepare_feature_dataframe,
)
from ice_creams_model_families import (
    extract_model_metadata,
    predict_model_probabilities,
    prepare_sequence_feature_dataframe,
    spectral_cnn_sequence_input_label,
)


_CONCEPT_DISPLAY: dict[str, str] = {
    "bare_sediment": "Bare Sediment",
    "sand": "Sand",
    "chlorophyta": "Chlorophyta",
    "magnoliopsida": "Magnoliopsida",
    "microphytobenthos": "Microphytobenthos",
    "phaeophyceae": "Phaeophyceae",
    "rhodophyta": "Rhodophyta",
    "water": "Water",
    "vauch": "Vauch",
    "unknown": "Unknown",
}

# Numeric class mapping based on ICE_CREAMS_v1_1.qml and historical ICE CREAMS
# class naming used in scripts/notebooks.
_NUMERIC_CLASS_TO_CONCEPT: dict[int, str] = {
    0: "unknown",
    1: "bare_sediment",
    2: "sand",
    3: "chlorophyta",
    4: "magnoliopsida",
    5: "microphytobenthos",
    6: "phaeophyceae",
    7: "rhodophyta",
    8: "water",
    9: "vauch",
    10: "unknown",
}

_CONCEPT_ALIASES: dict[str, set[str]] = {
    "bare_sediment": {
        "bare sediment",
        "baresediment",
        "bare sediments",
        "sediment",
        "bare_sediment",
    },
    "sand": {"sand", "bare sand", "baresand"},
    "chlorophyta": {
        "chlorophyta",
        "chlorophyceae",
        "green algae",
        "greenalgae",
        "green_macroalgae",
        "green algae class",
        "ulvophyceae",
    },
    "magnoliopsida": {
        "magnoliopsida",
        "magno",
        "seagrass",
        "zostera",
    },
    "microphytobenthos": {
        "microphytobenthos",
        "bacillariophyceae",
        "mpb",
        "diatom",
        "diatoms",
    },
    "phaeophyceae": {
        "phaeophyceae",
        "brown",
        "brown algae",
        "brownalgae",
    },
    "rhodophyta": {
        "rhodophyta",
        "florideophyceae",
        "rodo",
        "rhodo",
        "red algae",
        "redalgae",
    },
    "water": {"water", "h2o"},
    "vauch": {"vauch", "xanthophyceae"},
    "unknown": {"unknown", "other", "unclassified", "nodata", "no data"},
}

_ALIAS_TO_CONCEPT: dict[str, str] = {}
for _concept_name, _aliases in _CONCEPT_ALIASES.items():
    for _alias in _aliases:
        _ALIAS_TO_CONCEPT["".join(ch for ch in _alias.lower() if ch.isalnum())] = _concept_name

VALIDATION_MODE_MULTICLASS = "multiclass"
VALIDATION_MODE_PRESENCE_ABSENCE = "presence_absence"
_VALIDATION_MODE_ALIASES: dict[str, str] = {
    "multiclass": VALIDATION_MODE_MULTICLASS,
    "full": VALIDATION_MODE_MULTICLASS,
    "asis": VALIDATION_MODE_MULTICLASS,
    "as-is": VALIDATION_MODE_MULTICLASS,
    "as_is": VALIDATION_MODE_MULTICLASS,
    "original": VALIDATION_MODE_MULTICLASS,
    "presenceabsence": VALIDATION_MODE_PRESENCE_ABSENCE,
    "presence_absence": VALIDATION_MODE_PRESENCE_ABSENCE,
    "presence-absence": VALIDATION_MODE_PRESENCE_ABSENCE,
    "binary": VALIDATION_MODE_PRESENCE_ABSENCE,
    "binarypresenceabsence": VALIDATION_MODE_PRESENCE_ABSENCE,
    "pva": VALIDATION_MODE_PRESENCE_ABSENCE,
}

PRESENCE_LABEL = "Presence"
ABSENCE_LABEL = "Absence"
DEFAULT_TARGET_CLASS = "Magnoliopsida"


def _emit_status(status_callback: Callable[[str], None] | None, message: str) -> None:
    """Safely emit a status update for CLI or UI callers."""
    if status_callback is not None:
        status_callback(message)


def _emit_progress(progress_callback: Callable[[float], None] | None, value: float) -> None:
    """Safely emit bounded progress values between 0 and 1."""
    if progress_callback is not None:
        progress_callback(max(0.0, min(1.0, value)))


def _normalise_label_text(value: Any) -> str:
    """Normalize label strings for robust comparisons."""
    text = str(value).strip().lower()
    text = unicodedata.normalize("NFKD", text)
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    return "".join(ch for ch in text if ch.isalnum())


def _parse_numeric_label(value: Any) -> int | None:
    """Parse integer class labels from values like '3', '3.0', or 3."""
    text = str(value).strip()
    if not text:
        return None
    if re.fullmatch(r"[+-]?\d+", text):
        return int(text)
    if re.fullmatch(r"[+-]?\d+\.0+", text):
        return int(float(text))
    return None


def _match_concept(raw_label: Any) -> str | None:
    """Map raw model/validation labels to internal concepts."""
    numeric_value = _parse_numeric_label(raw_label)
    if numeric_value is not None and numeric_value in _NUMERIC_CLASS_TO_CONCEPT:
        return _NUMERIC_CLASS_TO_CONCEPT[numeric_value]

    normalized = _normalise_label_text(raw_label)
    if not normalized:
        return None

    if normalized in _ALIAS_TO_CONCEPT:
        return _ALIAS_TO_CONCEPT[normalized]

    if normalized in _CONCEPT_ALIASES:
        return normalized

    partial_matches: list[str] = []
    for alias_value, concept_name in _ALIAS_TO_CONCEPT.items():
        if len(alias_value) >= 5 and (alias_value in normalized or normalized in alias_value):
            partial_matches.append(concept_name)
    if partial_matches:
        return partial_matches[0]

    close = difflib.get_close_matches(normalized, list(_ALIAS_TO_CONCEPT), n=1, cutoff=0.78)
    if close:
        return _ALIAS_TO_CONCEPT[close[0]]

    return None


def _build_validation_label_space(
    validation_labels: pd.Series,
) -> tuple[dict[str, str], dict[str, str], list[str]]:
    """
    Build mappings from concepts and normalized values to dataset label names.

    Returns:
    - concept_to_dataset_label
    - normalized_to_dataset_label
    - ordered_unique_dataset_labels
    """
    concept_to_dataset_label: dict[str, str] = {}
    normalized_to_dataset_label: dict[str, str] = {}
    ordered_unique: list[str] = []
    seen: set[str] = set()

    for raw_value in validation_labels.astype("string").fillna("<MISSING>").astype(str):
        label_value = str(raw_value).strip()
        if not label_value:
            continue

        normalized = _normalise_label_text(label_value)
        if normalized and normalized not in normalized_to_dataset_label:
            normalized_to_dataset_label[normalized] = label_value

        concept_name = _match_concept(label_value)
        if concept_name and concept_name not in concept_to_dataset_label:
            concept_to_dataset_label[concept_name] = label_value

        if label_value not in seen:
            seen.add(label_value)
            ordered_unique.append(label_value)

    return concept_to_dataset_label, normalized_to_dataset_label, ordered_unique


def _map_label_to_validation_space(
    raw_label: Any,
    concept_to_dataset_label: dict[str, str],
    normalized_to_dataset_label: dict[str, str],
) -> str:
    """Map raw labels to validation label names using numeric+text robust matching."""
    label_value = str(raw_label).strip()
    if not label_value:
        return "<MISSING>"

    concept_name = _match_concept(label_value)
    if concept_name:
        if concept_name in concept_to_dataset_label:
            return concept_to_dataset_label[concept_name]
        return _CONCEPT_DISPLAY.get(concept_name, label_value)

    normalized = _normalise_label_text(label_value)
    if normalized in normalized_to_dataset_label:
        return normalized_to_dataset_label[normalized]

    close = difflib.get_close_matches(
        normalized,
        list(normalized_to_dataset_label),
        n=1,
        cutoff=0.86,
    )
    if close:
        return normalized_to_dataset_label[close[0]]

    numeric_value = _parse_numeric_label(label_value)
    if numeric_value is not None:
        concept_from_numeric = _NUMERIC_CLASS_TO_CONCEPT.get(numeric_value)
        if concept_from_numeric:
            if concept_from_numeric in concept_to_dataset_label:
                return concept_to_dataset_label[concept_from_numeric]
            return _CONCEPT_DISPLAY.get(concept_from_numeric, f"Class_{numeric_value}")
        return f"Class_{numeric_value}"

    return label_value


def _normalise_dataset_path(dataset_path: str) -> Path:
    """Validate and normalize a validation-table path."""
    candidate = Path(str(dataset_path).strip()).expanduser()
    if not candidate.exists():
        raise FileNotFoundError(f"Validation dataset not found: {dataset_path}")
    if not candidate.is_file():
        raise ValueError(f"Validation dataset path must be a file: {dataset_path}")
    if candidate.suffix.lower() not in {".csv", ".xlsx"}:
        raise ValueError("Validation dataset must be a .csv or .xlsx file.")
    return candidate.resolve()


def _normalise_model_path(model_path: str) -> Path:
    """Validate and normalize a model path."""
    candidate = Path(str(model_path).strip()).expanduser()
    if not candidate.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not candidate.is_file():
        raise ValueError(f"Model path must be a file: {model_path}")
    if candidate.suffix.lower() != ".pkl":
        raise ValueError("Model file must be a .pkl file.")
    return candidate.resolve()


def _normalise_label_column(label_column: str) -> str:
    """Validate label-column input."""
    value = str(label_column).strip()
    if not value:
        raise ValueError("A label column name is required.")
    return value


def _normalise_validation_mode(validation_mode: str | None) -> str:
    """Validate and canonicalize validation aggregation mode."""
    normalized = _normalise_label_text(validation_mode or VALIDATION_MODE_MULTICLASS)
    mode = _VALIDATION_MODE_ALIASES.get(normalized)
    if mode is None:
        raise ValueError(
            "Invalid validation mode. Use 'multiclass' or 'presence_absence'."
        )
    return mode


def _normalise_target_class(target_class: str | None) -> str:
    """Validate and normalize target class for presence/absence mode."""
    value = str(target_class or DEFAULT_TARGET_CLASS).strip()
    if not value:
        raise ValueError("A target class is required for presence/absence validation mode.")
    return value


def _map_to_presence_absence(labels: pd.Series, target_class_label: str) -> pd.Series:
    """Convert labels to binary Presence/Absence for a selected class label."""
    normalized_target = str(target_class_label).strip()
    normalized_labels = labels.astype("string").fillna("<MISSING>").astype(str)
    return normalized_labels.map(
        lambda class_name: PRESENCE_LABEL if str(class_name).strip() == normalized_target else ABSENCE_LABEL
    ).astype("string")


def _read_validation_table(dataset_path: Path) -> pd.DataFrame:
    """Read a CSV or XLSX validation table. XLSX always uses the first sheet."""
    suffix = dataset_path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(dataset_path, low_memory=False)
    else:
        frame = pd.read_excel(dataset_path, sheet_name=0)

    if frame.empty:
        raise ValueError(f"Validation dataset is empty: {dataset_path}")
    return frame


def _extract_vocab(learner: Any) -> list[str]:
    """Extract class vocabulary from learner.dls.vocab."""
    dls = getattr(learner, "dls", None)
    vocab = getattr(dls, "vocab", None) if dls is not None else None
    if vocab is None:
        return []

    if isinstance(vocab, (list, tuple)):
        values = list(vocab)
    else:
        try:
            values = list(vocab)
        except TypeError:
            return []

    # Tabular exports can store nested vocab containers; flatten one level if needed.
    if values and all(isinstance(item, (list, tuple)) for item in values):
        flattened: list[str] = []
        for group in values:
            for item in group:
                normalized = str(item).strip()
                if normalized and normalized not in flattened:
                    flattened.append(normalized)
        return flattened

    class_names: list[str] = []
    for value in values:
        normalized = str(value).strip()
        if normalized and normalized not in class_names:
            class_names.append(normalized)
    return class_names


def _sanitize_output_stem(stem_value: str, fallback: str) -> str:
    """Create filesystem-safe output stems."""
    cleaned = "".join(
        ch if ch.isalnum() or ch in ("-", "_", ".") else "_"
        for ch in str(stem_value).strip()
    ).strip("._")
    return cleaned or fallback


def _compute_metrics_table(
    true_labels: pd.Series,
    predicted_labels: pd.Series,
    class_order_hint: list[str] | None = None,
    class_subset: list[str] | None = None,
) -> pd.DataFrame:
    """Build metrics table with overall and per-class precision/recall."""
    true_series = true_labels.astype("string").fillna("<MISSING>").astype(str)
    pred_series = predicted_labels.astype("string").fillna("<MISSING>").astype(str)

    overall_accuracy = float((true_series == pred_series).mean())
    confusion = pd.crosstab(true_series, pred_series, dropna=False)

    allowed_classes_raw = class_subset or sorted(set(true_series.tolist()))
    allowed_classes = [cls for cls in allowed_classes_raw if str(cls).strip() and str(cls) != "<MISSING>"]
    if not allowed_classes:
        raise ValueError("No labelled classes were found in the validation dataset.")

    ordered_classes: list[str] = []
    allowed_set = set(allowed_classes)
    seen_classes: set[str] = set()

    for class_name in class_order_hint or []:
        normalized = str(class_name).strip()
        if normalized and normalized in allowed_set and normalized not in seen_classes:
            seen_classes.add(normalized)
            ordered_classes.append(normalized)

    for class_name in allowed_classes:
        normalized = str(class_name).strip()
        if normalized and normalized not in seen_classes:
            seen_classes.add(normalized)
            ordered_classes.append(normalized)

    rows: list[dict[str, Any]] = [
        {
            "Class": "OVERALL",
            "Overall_Accuracy": overall_accuracy,
            "User_Accuracy_Precision": pd.NA,
            "Producer_Accuracy_Recall": pd.NA,
            "Support_True": int(len(true_series)),
            "Support_Pred": int(len(pred_series)),
        }
    ]

    for class_name in ordered_classes:
        true_support = int(confusion.loc[class_name].sum()) if class_name in confusion.index else 0
        pred_support = int(confusion[class_name].sum()) if class_name in confusion.columns else 0
        true_positive = (
            int(confusion.at[class_name, class_name])
            if class_name in confusion.index and class_name in confusion.columns
            else 0
        )

        precision = (true_positive / pred_support) if pred_support else pd.NA
        recall = (true_positive / true_support) if true_support else pd.NA

        rows.append(
            {
                "Class": class_name,
                "Overall_Accuracy": pd.NA,
                "User_Accuracy_Precision": precision,
                "Producer_Accuracy_Recall": recall,
                "Support_True": true_support,
                "Support_Pred": pred_support,
            }
        )

    return pd.DataFrame(
        rows,
        columns=[
            "Class",
            "Overall_Accuracy",
            "User_Accuracy_Precision",
            "Producer_Accuracy_Recall",
            "Support_True",
            "Support_Pred",
        ],
    )


def validate_model(
    dataset_path: str,
    model_path: str,
    label_column: str,
    output_dir: str,
    status_callback: Callable[[str], None] | None = None,
    progress_callback: Callable[[float], None] | None = None,
    validation_mode: str = VALIDATION_MODE_MULTICLASS,
    target_class: str = DEFAULT_TARGET_CLASS,
) -> dict[str, Any]:
    """
    Validate a trained fastai model against a labelled validation dataset.

    Returns paths to prediction and metric CSVs plus summary counts.
    Supports:
    - multiclass label validation (default)
    - binary presence/absence validation for a selected target class
    """
    _emit_status(status_callback, "Validating inputs")
    _emit_progress(progress_callback, 0.02)

    dataset_file = _normalise_dataset_path(dataset_path)
    model_file = _normalise_model_path(model_path)
    label_name = _normalise_label_column(label_column)
    mode_name = _normalise_validation_mode(validation_mode)
    target_class_name = _normalise_target_class(target_class)
    output_folder = Path(str(output_dir).strip()).expanduser()
    output_folder.mkdir(parents=True, exist_ok=True)

    _emit_status(status_callback, f"Loading validation table: {dataset_file.name}")
    validation_df = _read_validation_table(dataset_file)
    _emit_progress(progress_callback, 0.18)

    if label_name not in validation_df.columns:
        raise ValueError(
            f"Label column '{label_name}' was not found in validation dataset."
        )

    _emit_status(status_callback, f"Using label column '{label_name}'")
    (
        concept_to_dataset_label,
        normalized_to_dataset_label,
        ordered_validation_labels,
    ) = _build_validation_label_space(validation_df[label_name])
    if mode_name == VALIDATION_MODE_PRESENCE_ABSENCE:
        _emit_status(
            status_callback,
            (
                "Validation mode: presence/absence "
                f"(target class: {target_class_name})"
            ),
        )
    else:
        _emit_status(status_callback, "Validation mode: multiclass")
    _emit_progress(progress_callback, 0.24)

    _emit_status(status_callback, f"Loading model: {model_file.name}")
    learner = load_learner(str(model_file))
    model_metadata = extract_model_metadata(learner)
    detected_model_family = str(model_metadata["model_family"])
    detected_model_family_label = str(model_metadata["model_family_label"])
    detected_feature_mode = str(model_metadata["feature_mode"])
    detected_feature_mode_label = str(model_metadata["feature_mode_label"])
    required_features = list(model_metadata["required_feature_names"])
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
    _emit_progress(progress_callback, 0.32)

    _emit_status(status_callback, "Checking required feature columns")
    if detected_model_family == "spectral_1d_cnn":
        validation_model_df = prepare_sequence_feature_dataframe(
            validation_df,
            feature_mode=detected_feature_mode,
            sequence_feature_names=model_metadata.get("sequence_feature_names"),
            sequence_channel_feature_names=model_metadata.get("sequence_channel_feature_names"),
            context="Validation dataset",
        )
    else:
        validation_model_df = prepare_feature_dataframe(
            validation_df,
            feature_mode=detected_feature_mode,
            required_feature_names=required_features,
            context="Validation dataset",
        )
    _emit_progress(progress_callback, 0.42)

    _emit_status(status_callback, "Running model predictions")
    batch_size = max(512, min(8192, len(validation_model_df)))
    preds = predict_model_probabilities(
        learner,
        validation_model_df,
        model_metadata,
        batch_size=batch_size,
    )
    _emit_progress(progress_callback, 0.74)

    class_indices = preds.argmax(dim=1).cpu().numpy().astype(int)
    confidence_values = preds.max(dim=1).values.cpu().numpy().astype(float)
    vocab = _extract_vocab(learner)

    predicted_classes_raw = [
        str(vocab[class_idx]) if 0 <= class_idx < len(vocab) else str(class_idx)
        for class_idx in class_indices
    ]
    predicted_classes = [
        _map_label_to_validation_space(
            raw_label=raw_label,
            concept_to_dataset_label=concept_to_dataset_label,
            normalized_to_dataset_label=normalized_to_dataset_label,
        )
        for raw_label in predicted_classes_raw
    ]
    mapped_true_series = validation_df[label_name].map(
        lambda raw_label: _map_label_to_validation_space(
            raw_label=raw_label,
            concept_to_dataset_label=concept_to_dataset_label,
            normalized_to_dataset_label=normalized_to_dataset_label,
        )
    ).astype("string")
    predicted_series = pd.Series(predicted_classes, name="Predicted_Class", dtype="string")

    if mode_name == VALIDATION_MODE_PRESENCE_ABSENCE:
        target_class_label = _map_label_to_validation_space(
            raw_label=target_class_name,
            concept_to_dataset_label=concept_to_dataset_label,
            normalized_to_dataset_label=normalized_to_dataset_label,
        )
        _emit_status(
            status_callback,
            (
                "Collapsing labels to Presence/Absence "
                f"for class '{target_class_label}'"
            ),
        )
        mapped_true_series = _map_to_presence_absence(mapped_true_series, target_class_label)
        predicted_series = _map_to_presence_absence(
            predicted_series, target_class_label
        ).rename("Predicted_Class")
        predicted_classes = predicted_series.tolist()
        ordered_present_classes = [ABSENCE_LABEL, PRESENCE_LABEL]
        class_order_hint = [ABSENCE_LABEL, PRESENCE_LABEL]
    else:
        ordered_present_classes = []
        present_seen: set[str] = set()
        for raw_label in ordered_validation_labels:
            mapped_label = _map_label_to_validation_space(
                raw_label=raw_label,
                concept_to_dataset_label=concept_to_dataset_label,
                normalized_to_dataset_label=normalized_to_dataset_label,
            )
            if mapped_label != "<MISSING>" and mapped_label not in present_seen:
                present_seen.add(mapped_label)
                ordered_present_classes.append(mapped_label)

        class_order_hint = [
            _map_label_to_validation_space(
                raw_label=class_name,
                concept_to_dataset_label=concept_to_dataset_label,
                normalized_to_dataset_label=normalized_to_dataset_label,
            )
            for class_name in vocab
        ]

    _emit_status(status_callback, "Computing validation metrics")
    metrics_df = _compute_metrics_table(
        true_labels=mapped_true_series,
        predicted_labels=predicted_series,
        class_order_hint=class_order_hint,
        class_subset=ordered_present_classes,
    )
    _emit_progress(progress_callback, 0.88)

    overall_row = metrics_df.loc[metrics_df["Class"] == "OVERALL", "Overall_Accuracy"]
    overall_accuracy = float(overall_row.iloc[0]) if not overall_row.empty else float("nan")

    # Keep confusion-matrix display aligned with classes present in validation labels.
    confusion_full = pd.crosstab(
        mapped_true_series.astype("string").fillna("<MISSING>").astype(str),
        predicted_series.astype("string").fillna("<MISSING>").astype(str),
        dropna=False,
    )
    confusion_display = confusion_full.reindex(
        index=ordered_present_classes,
        columns=ordered_present_classes,
        fill_value=0,
    ).astype(int)
    predictions_outside_validation_space = int((~predicted_series.isin(ordered_present_classes)).sum())

    predictions_df = validation_df.copy()
    predictions_df[label_name] = mapped_true_series
    predictions_df["Predicted_Class"] = predicted_classes
    predictions_df["Predicted_Confidence"] = confidence_values

    dataset_stem = _sanitize_output_stem(dataset_file.stem, "validation")
    model_stem = _sanitize_output_stem(model_file.stem, "model")
    predictions_csv = output_folder / f"{dataset_stem}__{model_stem}__predictions.csv"
    metrics_csv = output_folder / f"{dataset_stem}__{model_stem}__metrics.csv"

    _emit_status(status_callback, f"Writing predictions CSV: {predictions_csv.name}")
    predictions_df.to_csv(predictions_csv, index=False)
    _emit_progress(progress_callback, 0.95)

    _emit_status(status_callback, f"Writing metrics CSV: {metrics_csv.name}")
    metrics_df.to_csv(metrics_csv, index=False)
    _emit_progress(progress_callback, 1.0)
    _emit_status(status_callback, "Validation workflow completed")

    class_count = len(ordered_present_classes)

    return {
        "predictions_csv": str(predictions_csv),
        "metrics_csv": str(metrics_csv),
        "rows": int(len(validation_df)),
        "classes": int(class_count),
        "overall_accuracy": overall_accuracy,
        "confusion_labels": ordered_present_classes,
        "confusion_matrix": confusion_display.values.tolist(),
        "predictions_outside_validation_classes": predictions_outside_validation_space,
        "validation_mode": mode_name,
        "target_class": target_class_name,
        "model_family": detected_model_family,
        "model_family_label": detected_model_family_label,
        "feature_mode": detected_feature_mode,
        "feature_mode_label": detected_feature_mode_label,
        "sequence_use_standardized_reflectance": bool(
            model_metadata.get("sequence_use_standardized_reflectance", False)
        ),
        "sequence_input_label": str(model_metadata.get("sequence_input_label") or ""),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate an ICE CREAMS model")
    parser.add_argument("dataset_path", help="Validation dataset path (.csv or .xlsx)")
    parser.add_argument("model_path", help="Trained model path (.pkl)")
    parser.add_argument(
        "--label-column",
        default="Label_Char",
        help="Name of the ground-truth label column in the validation table",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Folder where predictions and metrics CSV files will be written",
    )
    parser.add_argument(
        "--validation-mode",
        default=VALIDATION_MODE_MULTICLASS,
        help="Validation mode: 'multiclass' or 'presence_absence'",
    )
    parser.add_argument(
        "--target-class",
        default=DEFAULT_TARGET_CLASS,
        help="Target class for presence/absence mode (name or numeric class id)",
    )
    return parser


def _main() -> None:
    args = _build_parser().parse_args()
    result = validate_model(
        dataset_path=args.dataset_path,
        model_path=args.model_path,
        label_column=args.label_column,
        output_dir=args.output_dir,
        status_callback=lambda message: print(message, flush=True),
        validation_mode=args.validation_mode,
        target_class=args.target_class,
    )
    print(f"Predictions CSV: {result['predictions_csv']}")
    print(f"Metrics CSV: {result['metrics_csv']}")
    print(f"Rows evaluated: {result['rows']}")
    print(f"Classes: {result['classes']}")


if __name__ == "__main__":
    _main()
