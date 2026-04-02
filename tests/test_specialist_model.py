import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
import xarray as xr

from apply_ICECREAMS import apply_classification
from ice_creams_feature_modes import FEATURE_COLUMNS_BY_MODE, FEATURE_MODE_HIGH_SPATIAL_ACCURACY, raw_column_name
from ice_creams_specialist_models import (
    SPECIALIST_FEATURE_COLUMNS,
    build_class45_specialist_training_dataframe,
    prepare_class45_specialist_feature_dataframe,
)


class SpecialistModelTests(unittest.TestCase):
    def test_prepare_class45_specialist_feature_dataframe_builds_expected_columns(self) -> None:
        source_df = pd.DataFrame(
            [
                {
                    raw_column_name("B03"): 2.0,
                    raw_column_name("B04"): 4.0,
                    raw_column_name("B05"): 5.0,
                    raw_column_name("B08"): 8.0,
                }
            ]
        )

        prepared_df = prepare_class45_specialist_feature_dataframe(
            source_df,
            context="Unit test specialist dataset",
        )

        self.assertEqual(list(prepared_df.columns), list(SPECIALIST_FEATURE_COLUMNS))
        self.assertAlmostEqual(prepared_df.loc[0, "Reflectance_Stan_B03"], 0.0)
        self.assertAlmostEqual(prepared_df.loc[0, "Reflectance_Stan_B04"], 2.0 / 6.0)
        self.assertAlmostEqual(prepared_df.loc[0, "Reflectance_Stan_B05"], 3.0 / 6.0)
        self.assertAlmostEqual(prepared_df.loc[0, "Reflectance_Stan_B08"], 1.0)
        self.assertAlmostEqual(prepared_df.loc[0, "NDVI"], (8.0 - 4.0) / (8.0 + 4.0))
        self.assertAlmostEqual(prepared_df.loc[0, "NDWI"], (2.0 - 8.0) / (8.0 + 2.0))

    def test_build_class45_specialist_training_dataframe_filters_and_deduplicates(self) -> None:
        training_df = pd.DataFrame(
            [
                {
                    "True_Class": "Magnoliopsida",
                    raw_column_name("B03"): 0.2,
                    raw_column_name("B04"): 0.3,
                    raw_column_name("B05"): 0.4,
                    raw_column_name("B08"): 0.5,
                },
                {
                    "True_Class": "Magnoliopsida",
                    raw_column_name("B03"): 0.2,
                    raw_column_name("B04"): 0.3,
                    raw_column_name("B05"): 0.4,
                    raw_column_name("B08"): 0.5,
                },
                {
                    "True_Class": "Microphytobenthos",
                    raw_column_name("B03"): 0.4,
                    raw_column_name("B04"): 0.3,
                    raw_column_name("B05"): 0.2,
                    raw_column_name("B08"): 0.1,
                },
                {
                    "True_Class": "Water",
                    raw_column_name("B03"): 0.6,
                    raw_column_name("B04"): 0.5,
                    raw_column_name("B05"): 0.4,
                    raw_column_name("B08"): 0.3,
                },
            ]
        )

        specialist_training_df = build_class45_specialist_training_dataframe(training_df)

        self.assertEqual(len(specialist_training_df), 2)
        self.assertEqual(
            specialist_training_df["True_Class"].tolist(),
            ["Magnoliopsida", "Microphytobenthos"],
        )

    def test_apply_classification_uses_specialist_model_for_class4_and_class5_candidates(self) -> None:
        coords = {
            "band": [1],
            "y": [20.0],
            "x": [100.0, 110.0, 120.0],
        }
        input_ds = xr.Dataset(
            {
                raw_column_name("B02"): xr.DataArray([[[0.1, 0.2, 0.3]]], dims=("band", "y", "x"), coords=coords),
                raw_column_name("B03"): xr.DataArray([[[0.2, 0.3, 0.4]]], dims=("band", "y", "x"), coords=coords),
                raw_column_name("B04"): xr.DataArray([[[0.3, 0.4, 0.5]]], dims=("band", "y", "x"), coords=coords),
                raw_column_name("B05"): xr.DataArray([[[0.35, 0.45, 0.55]]], dims=("band", "y", "x"), coords=coords),
                raw_column_name("B08"): xr.DataArray([[[0.7, 0.8, 0.9]]], dims=("band", "y", "x"), coords=coords),
                "NDVI": xr.DataArray([[[0.40, 0.35, 0.30]]], dims=("band", "y", "x"), coords=coords),
                "NDWI": xr.DataArray([[[-0.55, -0.45, -0.38]]], dims=("band", "y", "x"), coords=coords),
                "SPC": xr.DataArray([[[55.0, 65.0, 75.0]]], dims=("band", "y", "x"), coords=coords),
            }
        )
        main_model_metadata = {
            "model_family": "tabular_dense",
            "feature_mode": FEATURE_MODE_HIGH_SPATIAL_ACCURACY,
            "required_feature_names": list(FEATURE_COLUMNS_BY_MODE[FEATURE_MODE_HIGH_SPATIAL_ACCURACY]),
        }
        specialist_model_metadata = {
            "model_family": "tabular_dense",
            "feature_mode": FEATURE_MODE_HIGH_SPATIAL_ACCURACY,
            "required_feature_names": list(SPECIALIST_FEATURE_COLUMNS),
            "specialist_target_output_class_ids": [4, 5],
            "specialist_raw_bands": ["B03", "B04", "B05", "B08"],
        }
        main_probabilities = torch.tensor(
            [
                [0.01, 0.01, 0.01, 0.92, 0.05],
                [0.01, 0.01, 0.01, 0.02, 0.94],
                [0.01, 0.94, 0.01, 0.02, 0.02],
            ],
            dtype=torch.float32,
        )
        specialist_learner = SimpleNamespace(dls=SimpleNamespace(vocab=["Magnoliopsida", "Microphytobenthos"]))

        with patch("apply_ICECREAMS.predict_model_probabilities", return_value=main_probabilities), patch(
            "apply_ICECREAMS.predict_class45_specialist",
            return_value=(
                np.asarray([5, 4], dtype=np.int16),
                np.asarray([0.92, 0.88], dtype=np.float32),
                ["Microphytobenthos", "Magnoliopsida"],
            ),
        ):
            output_ds = apply_classification(
                input_ds,
                class_model=SimpleNamespace(),
                model_metadata=main_model_metadata,
                specialist_model=specialist_learner,
                specialist_model_metadata=specialist_model_metadata,
                batch_size=3,
            )
        try:
            out_class = output_ds["Out_Class"].values.astype(np.int16)
            class_probs = output_ds["Class_Probs"].values.astype(np.float32)

            self.assertEqual(out_class[0, 0, 0], 5)
            self.assertEqual(out_class[0, 0, 1], 4)
            self.assertEqual(out_class[0, 0, 2], 2)
            self.assertAlmostEqual(float(class_probs[0, 0, 0]), 0.92, places=6)
            self.assertAlmostEqual(float(class_probs[0, 0, 1]), 0.88, places=6)
            self.assertAlmostEqual(float(class_probs[0, 0, 2]), 0.94, places=6)
        finally:
            output_ds.close()


if __name__ == "__main__":
    unittest.main()
