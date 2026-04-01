import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import xarray as xr

from apply_ICECREAMS import apply_classification
from ice_creams_feature_modes import (
    FEATURE_COLUMNS_BY_MODE,
    FEATURE_MODE_HIGH_SPATIAL_ACCURACY,
    FEATURE_MODE_HIGH_SPECTRAL_COMPLEXITY,
    raw_column_name,
)
from ice_creams_model_families import (
    DEFAULT_SPECTRAL_CNN_USE_STANDARDIZED_REFLECTANCE,
    MODEL_FAMILY_SPECTRAL_1D_CNN,
    MODEL_FAMILY_TABULAR_DENSE,
    Spectral1DCNN,
    attach_model_metadata,
    extract_model_metadata,
    predict_model_probabilities,
    prepare_sequence_feature_dataframe,
    spectral_cnn_sequence_input_label,
    sequence_channel_feature_names_for_mode,
    sequence_feature_names_for_mode,
)


class ModelFamilyHelperTests(unittest.TestCase):
    def test_sequence_channel_feature_names_can_disable_standardized_reflectance(self) -> None:
        raw_only_channel_groups = sequence_channel_feature_names_for_mode(
            FEATURE_MODE_HIGH_SPATIAL_ACCURACY,
            use_standardized_reflectance=False,
        )

        self.assertEqual(len(raw_only_channel_groups), 1)
        self.assertEqual(
            raw_only_channel_groups[0],
            [
                raw_column_name("B02"),
                raw_column_name("B03"),
                raw_column_name("B04"),
                raw_column_name("B08"),
            ],
        )
        self.assertEqual(
            spectral_cnn_sequence_input_label(False),
            "Raw Reflectance Only",
        )
        self.assertTrue(DEFAULT_SPECTRAL_CNN_USE_STANDARDIZED_REFLECTANCE)

    def test_extract_model_metadata_infers_legacy_tabular_model(self) -> None:
        spectral_features = list(FEATURE_COLUMNS_BY_MODE[FEATURE_MODE_HIGH_SPECTRAL_COMPLEXITY])
        learner = SimpleNamespace(
            dls=SimpleNamespace(
                cat_names=[],
                cont_names=spectral_features,
                x_names=None,
                train_ds=SimpleNamespace(cat_names=[], cont_names=spectral_features, x_names=None),
            )
        )

        metadata = extract_model_metadata(learner)

        self.assertEqual(metadata["model_family"], MODEL_FAMILY_TABULAR_DENSE)
        self.assertEqual(metadata["feature_mode"], FEATURE_MODE_HIGH_SPECTRAL_COMPLEXITY)
        self.assertEqual(metadata["required_feature_names"], spectral_features)
        self.assertEqual(metadata["sequence_feature_names"], [])

    def test_extract_model_metadata_prefers_explicit_spectral_cnn_metadata(self) -> None:
        sequence_channel_feature_names = sequence_channel_feature_names_for_mode(
            FEATURE_MODE_HIGH_SPATIAL_ACCURACY
        )
        sequence_feature_names = sequence_feature_names_for_mode(FEATURE_MODE_HIGH_SPATIAL_ACCURACY)
        learner = SimpleNamespace()
        attach_model_metadata(
            learner,
            model_family=MODEL_FAMILY_SPECTRAL_1D_CNN,
            feature_mode=FEATURE_MODE_HIGH_SPATIAL_ACCURACY,
            required_feature_names=sequence_feature_names,
            sequence_feature_names=sequence_feature_names,
            sequence_channel_feature_names=sequence_channel_feature_names,
            sequence_normalization={
                "mean": [[1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4]],
                "std": [[0.5, 0.5, 0.5, 0.5], [0.2, 0.2, 0.2, 0.2]],
            },
        )

        metadata = extract_model_metadata(learner)

        self.assertEqual(metadata["model_family"], MODEL_FAMILY_SPECTRAL_1D_CNN)
        self.assertEqual(metadata["feature_mode"], FEATURE_MODE_HIGH_SPATIAL_ACCURACY)
        self.assertEqual(metadata["sequence_feature_names"], sequence_feature_names)
        self.assertEqual(metadata["sequence_channel_feature_names"], sequence_channel_feature_names)
        self.assertTrue(metadata["sequence_use_standardized_reflectance"])
        self.assertEqual(metadata["sequence_input_label"], "Raw + Standardized Reflectance")
        np.testing.assert_allclose(
            metadata["sequence_normalization"]["mean"],
            [[1.0, 2.0, 3.0, 4.0], [0.1, 0.2, 0.3, 0.4]],
        )

    def test_extract_model_metadata_supports_legacy_raw_only_spectral_cnn_metadata(self) -> None:
        legacy_sequence_feature_names = sequence_channel_feature_names_for_mode(
            FEATURE_MODE_HIGH_SPATIAL_ACCURACY
        )[0]
        learner = SimpleNamespace(
            ice_creams_model_metadata={
                "model_family": MODEL_FAMILY_SPECTRAL_1D_CNN,
                "feature_mode": FEATURE_MODE_HIGH_SPATIAL_ACCURACY,
                "required_feature_names": legacy_sequence_feature_names,
                "sequence_feature_names": legacy_sequence_feature_names,
                "sequence_normalization": {
                    "mean": [0.1, 0.2, 0.3, 0.4],
                    "std": [1.0, 1.0, 1.0, 1.0],
                },
            }
        )

        metadata = extract_model_metadata(learner)

        self.assertEqual(metadata["sequence_channel_feature_names"], [legacy_sequence_feature_names])
        self.assertFalse(metadata["sequence_use_standardized_reflectance"])
        self.assertEqual(metadata["sequence_input_label"], "Raw Reflectance Only")
        np.testing.assert_allclose(
            metadata["sequence_normalization"]["mean"],
            [[0.1, 0.2, 0.3, 0.4]],
        )

    def test_prepare_sequence_feature_dataframe_rebuilds_standardized_inputs(self) -> None:
        source_df = pd.DataFrame(
            [
                {
                    raw_column_name("B02"): 1.0,
                    raw_column_name("B03"): 2.0,
                    raw_column_name("B04"): 3.0,
                    raw_column_name("B08"): 4.0,
                    "NDVI": 999.0,
                    "Reflectance_Stan_B02": 999.0,
                }
            ]
        )

        sequence_df = prepare_sequence_feature_dataframe(
            source_df,
            feature_mode=FEATURE_MODE_HIGH_SPATIAL_ACCURACY,
            context="Validation dataset",
        )

        self.assertEqual(
            list(sequence_df.columns),
            sequence_feature_names_for_mode(FEATURE_MODE_HIGH_SPATIAL_ACCURACY),
        )
        self.assertAlmostEqual(sequence_df.loc[0, raw_column_name("B08")], 4.0)
        self.assertAlmostEqual(sequence_df.loc[0, "Reflectance_Stan_B02"], 0.0)
        self.assertAlmostEqual(sequence_df.loc[0, "Reflectance_Stan_B03"], 1.0 / 3.0)
        self.assertAlmostEqual(sequence_df.loc[0, "Reflectance_Stan_B04"], 2.0 / 3.0)
        self.assertAlmostEqual(sequence_df.loc[0, "Reflectance_Stan_B08"], 1.0)

    def test_predict_model_probabilities_runs_spectral_cnn_inference(self) -> None:
        torch.manual_seed(0)
        sequence_channel_feature_names = sequence_channel_feature_names_for_mode(
            FEATURE_MODE_HIGH_SPATIAL_ACCURACY
        )
        sequence_feature_names = sequence_feature_names_for_mode(FEATURE_MODE_HIGH_SPATIAL_ACCURACY)
        learner = SimpleNamespace(
            model=Spectral1DCNN(
                sequence_length=len(sequence_channel_feature_names[0]),
                n_classes=2,
                in_channels=len(sequence_channel_feature_names),
            ),
            dls=SimpleNamespace(vocab=["Sand", "Water"]),
        )
        metadata = attach_model_metadata(
            learner,
            model_family=MODEL_FAMILY_SPECTRAL_1D_CNN,
            feature_mode=FEATURE_MODE_HIGH_SPATIAL_ACCURACY,
            required_feature_names=sequence_feature_names,
            sequence_feature_names=sequence_feature_names,
            sequence_channel_feature_names=sequence_channel_feature_names,
            sequence_normalization={
                "mean": [[0.0] * len(sequence_channel_feature_names[0])] * len(sequence_channel_feature_names),
                "std": [[1.0] * len(sequence_channel_feature_names[0])] * len(sequence_channel_feature_names),
            },
        )
        input_df = pd.DataFrame(
            [
                {
                    raw_column_name("B02"): 0.1,
                    raw_column_name("B03"): 0.2,
                    raw_column_name("B04"): 0.3,
                    raw_column_name("B08"): 0.4,
                },
                {
                    raw_column_name("B02"): 0.5,
                    raw_column_name("B03"): 0.4,
                    raw_column_name("B04"): 0.3,
                    raw_column_name("B08"): 0.2,
                },
            ]
        )

        preds = predict_model_probabilities(
            learner,
            input_df,
            metadata,
            batch_size=2,
        )

        self.assertEqual(tuple(preds.shape), (2, 2))
        self.assertTrue(torch.allclose(preds.sum(dim=1), torch.ones(2), atol=1e-5))

    def test_predict_model_probabilities_supports_legacy_raw_only_spectral_cnn(self) -> None:
        torch.manual_seed(0)
        legacy_sequence_feature_names = sequence_channel_feature_names_for_mode(
            FEATURE_MODE_HIGH_SPATIAL_ACCURACY
        )[0]
        learner = SimpleNamespace(
            model=Spectral1DCNN(
                sequence_length=len(legacy_sequence_feature_names),
                n_classes=2,
                in_channels=1,
            ),
            dls=SimpleNamespace(vocab=["Sand", "Water"]),
            ice_creams_model_metadata={
                "model_family": MODEL_FAMILY_SPECTRAL_1D_CNN,
                "feature_mode": FEATURE_MODE_HIGH_SPATIAL_ACCURACY,
                "required_feature_names": legacy_sequence_feature_names,
                "sequence_feature_names": legacy_sequence_feature_names,
                "sequence_normalization": {
                    "mean": [0.0] * len(legacy_sequence_feature_names),
                    "std": [1.0] * len(legacy_sequence_feature_names),
                },
            },
        )
        metadata = extract_model_metadata(learner)
        input_df = pd.DataFrame(
            [
                {
                    raw_column_name("B02"): 0.1,
                    raw_column_name("B03"): 0.2,
                    raw_column_name("B04"): 0.3,
                    raw_column_name("B08"): 0.4,
                },
                {
                    raw_column_name("B02"): 0.4,
                    raw_column_name("B03"): 0.3,
                    raw_column_name("B04"): 0.2,
                    raw_column_name("B08"): 0.1,
                },
            ]
        )

        preds = predict_model_probabilities(
            learner,
            input_df,
            metadata,
            batch_size=2,
        )

        self.assertEqual(tuple(preds.shape), (2, 2))
        self.assertTrue(torch.allclose(preds.sum(dim=1), torch.ones(2), atol=1e-5))

    def test_apply_classification_keeps_ndvi_output_for_spectral_cnn(self) -> None:
        torch.manual_seed(0)
        sequence_channel_feature_names = sequence_channel_feature_names_for_mode(
            FEATURE_MODE_HIGH_SPATIAL_ACCURACY
        )
        sequence_feature_names = sequence_feature_names_for_mode(FEATURE_MODE_HIGH_SPATIAL_ACCURACY)
        learner = SimpleNamespace(
            model=Spectral1DCNN(
                sequence_length=len(sequence_channel_feature_names[0]),
                n_classes=2,
                in_channels=len(sequence_channel_feature_names),
            ),
            dls=SimpleNamespace(vocab=["Sand", "Water"]),
        )
        metadata = attach_model_metadata(
            learner,
            model_family=MODEL_FAMILY_SPECTRAL_1D_CNN,
            feature_mode=FEATURE_MODE_HIGH_SPATIAL_ACCURACY,
            required_feature_names=sequence_feature_names,
            sequence_feature_names=sequence_feature_names,
            sequence_channel_feature_names=sequence_channel_feature_names,
            sequence_normalization={
                "mean": [[0.0] * len(sequence_channel_feature_names[0])] * len(sequence_channel_feature_names),
                "std": [[1.0] * len(sequence_channel_feature_names[0])] * len(sequence_channel_feature_names),
            },
        )
        coords = {
            "band": [1],
            "y": [20.0, 10.0],
            "x": [100.0, 110.0],
        }
        input_ds = xr.Dataset(
            {
                raw_column_name("B02"): xr.DataArray([[[0.1, 0.2], [0.3, 0.4]]], dims=("band", "y", "x"), coords=coords),
                raw_column_name("B03"): xr.DataArray([[[0.2, 0.3], [0.4, 0.5]]], dims=("band", "y", "x"), coords=coords),
                raw_column_name("B04"): xr.DataArray([[[0.3, 0.4], [0.5, 0.6]]], dims=("band", "y", "x"), coords=coords),
                raw_column_name("B08"): xr.DataArray([[[0.6, 0.7], [0.8, 0.9]]], dims=("band", "y", "x"), coords=coords),
                "NDVI": xr.DataArray([[[0.33, 0.27], [0.23, 0.20]]], dims=("band", "y", "x"), coords=coords),
                "NDWI": xr.DataArray([[[-0.50, -0.40], [-0.33, -0.29]]], dims=("band", "y", "x"), coords=coords),
                "SPC": xr.DataArray([[[30.0, 35.0], [40.0, 45.0]]], dims=("band", "y", "x"), coords=coords),
            }
        )

        output_ds = apply_classification(
            input_ds,
            learner,
            metadata,
            batch_size=2,
        )
        try:
            self.assertIn("NDVI", output_ds.data_vars)
            self.assertTrue(
                torch.allclose(
                    torch.as_tensor(output_ds["NDVI"].values, dtype=torch.float32),
                    torch.as_tensor(input_ds["NDVI"].values, dtype=torch.float32),
                    atol=1e-6,
                )
            )
        finally:
            output_ds.close()

    def test_apply_classification_reclassifies_low_ndvi_class4_to_class5(self) -> None:
        coords = {
            "band": [1],
            "y": [20.0],
            "x": [100.0, 110.0],
        }
        input_ds = xr.Dataset(
            {
                raw_column_name("B02"): xr.DataArray([[[0.1, 0.2]]], dims=("band", "y", "x"), coords=coords),
                raw_column_name("B03"): xr.DataArray([[[0.2, 0.3]]], dims=("band", "y", "x"), coords=coords),
                raw_column_name("B04"): xr.DataArray([[[0.3, 0.4]]], dims=("band", "y", "x"), coords=coords),
                raw_column_name("B08"): xr.DataArray([[[0.5, 0.6]]], dims=("band", "y", "x"), coords=coords),
                "NDVI": xr.DataArray([[[0.24, 0.26]]], dims=("band", "y", "x"), coords=coords),
                "NDWI": xr.DataArray([[[-0.2, -0.3]]], dims=("band", "y", "x"), coords=coords),
                "SPC": xr.DataArray([[[42.0, 55.0]]], dims=("band", "y", "x"), coords=coords),
            }
        )
        model_metadata = {
            "model_family": MODEL_FAMILY_TABULAR_DENSE,
            "feature_mode": FEATURE_MODE_HIGH_SPATIAL_ACCURACY,
            "required_feature_names": list(FEATURE_COLUMNS_BY_MODE[FEATURE_MODE_HIGH_SPATIAL_ACCURACY]),
        }
        predicted_probabilities = torch.tensor(
            [
                [0.01, 0.01, 0.01, 0.90, 0.07],
                [0.01, 0.01, 0.01, 0.90, 0.07],
            ],
            dtype=torch.float32,
        )

        with patch("apply_ICECREAMS.predict_model_probabilities", return_value=predicted_probabilities):
            output_ds = apply_classification(
                input_ds,
                class_model=SimpleNamespace(),
                model_metadata=model_metadata,
                batch_size=2,
            )
        try:
            out_class = output_ds["Out_Class"].values.astype(np.int16)
            seagrass_cover = output_ds["Seagrass_Cover"].values.astype(np.float32)

            self.assertEqual(int(out_class[0, 0, 0]), 5)
            self.assertEqual(int(out_class[0, 0, 1]), 4)
            self.assertEqual(float(seagrass_cover[0, 0, 0]), -1.0)
            self.assertEqual(float(seagrass_cover[0, 0, 1]), 55.0)
        finally:
            output_ds.close()


if __name__ == "__main__":
    unittest.main()
