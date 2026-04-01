import unittest

import pandas as pd

from ice_creams_feature_modes import (
    FEATURE_COLUMNS_BY_MODE,
    FEATURE_MODE_HIGH_SPATIAL_ACCURACY,
    FEATURE_MODE_HIGH_SPECTRAL_COMPLEXITY,
    build_training_dataframe,
    infer_feature_mode_from_feature_names,
    raw_column_name,
)


class FeatureModeHelperTests(unittest.TestCase):
    def test_infer_feature_mode_from_required_feature_names(self) -> None:
        spatial_features = list(FEATURE_COLUMNS_BY_MODE[FEATURE_MODE_HIGH_SPATIAL_ACCURACY])
        spectral_features = list(FEATURE_COLUMNS_BY_MODE[FEATURE_MODE_HIGH_SPECTRAL_COMPLEXITY])

        self.assertEqual(
            infer_feature_mode_from_feature_names(reversed(spatial_features)),
            FEATURE_MODE_HIGH_SPATIAL_ACCURACY,
        )
        self.assertEqual(
            infer_feature_mode_from_feature_names(reversed(spectral_features)),
            FEATURE_MODE_HIGH_SPECTRAL_COMPLEXITY,
        )

    def test_spatial_training_preprocessing_rebuilds_four_band_standardization(self) -> None:
        source_df = pd.DataFrame(
            [
                {
                    "True_Class": "Water",
                    raw_column_name("B02"): 1.0,
                    raw_column_name("B03"): 2.0,
                    raw_column_name("B04"): 3.0,
                    raw_column_name("B08"): 5.0,
                    raw_column_name("B01"): 10.0,
                    "Unrelated": 99.0,
                }
            ]
        )

        training_df, feature_columns, resolved_mode = build_training_dataframe(
            source_df,
            feature_mode=FEATURE_MODE_HIGH_SPATIAL_ACCURACY,
        )

        self.assertEqual(resolved_mode, FEATURE_MODE_HIGH_SPATIAL_ACCURACY)
        self.assertEqual(
            feature_columns,
            list(FEATURE_COLUMNS_BY_MODE[FEATURE_MODE_HIGH_SPATIAL_ACCURACY]),
        )
        self.assertNotIn(raw_column_name("B01"), training_df.columns)
        self.assertNotIn("Reflectance_Stan_B01", training_df.columns)
        self.assertNotIn("Unrelated", training_df.columns)
        self.assertAlmostEqual(training_df.loc[0, "Reflectance_Stan_B02"], 0.0)
        self.assertAlmostEqual(training_df.loc[0, "Reflectance_Stan_B03"], 0.25)
        self.assertAlmostEqual(training_df.loc[0, "Reflectance_Stan_B04"], 0.5)
        self.assertAlmostEqual(training_df.loc[0, "Reflectance_Stan_B08"], 1.0)
        self.assertAlmostEqual(training_df.loc[0, "NDVI"], 0.25)
        self.assertAlmostEqual(training_df.loc[0, "NDWI"], -3.0 / 7.0)

    def test_spatial_training_overwrites_existing_derived_columns_without_duplicates(self) -> None:
        source_df = pd.DataFrame(
            [
                {
                    "True_Class": "Water",
                    raw_column_name("B02"): 1.0,
                    raw_column_name("B03"): 2.0,
                    raw_column_name("B04"): 3.0,
                    raw_column_name("B08"): 5.0,
                    "Reflectance_Stan_B02": 999.0,
                    "Reflectance_Stan_B03": 999.0,
                    "Reflectance_Stan_B04": 999.0,
                    "Reflectance_Stan_B08": 999.0,
                    "NDVI": 999.0,
                    "NDWI": 999.0,
                }
            ]
        )

        training_df, _, _ = build_training_dataframe(
            source_df,
            feature_mode=FEATURE_MODE_HIGH_SPATIAL_ACCURACY,
        )

        self.assertEqual(training_df.columns.duplicated().sum(), 0)
        self.assertAlmostEqual(training_df.loc[0, "Reflectance_Stan_B02"], 0.0)
        self.assertAlmostEqual(training_df.loc[0, "Reflectance_Stan_B08"], 1.0)
        self.assertAlmostEqual(training_df.loc[0, "NDVI"], 0.25)
        self.assertAlmostEqual(training_df.loc[0, "NDWI"], -3.0 / 7.0)

    def test_spectral_training_selection_excludes_unrelated_extra_columns(self) -> None:
        spectral_features = list(FEATURE_COLUMNS_BY_MODE[FEATURE_MODE_HIGH_SPECTRAL_COMPLEXITY])
        source_row = {"True_Class": "Sand", "Ignored_Column": 123.0, "Another_Extra": "x"}
        for index, feature_name in enumerate(spectral_features, start=1):
            source_row[feature_name] = float(index)
        source_df = pd.DataFrame([source_row])

        training_df, feature_columns, resolved_mode = build_training_dataframe(
            source_df,
            feature_mode=FEATURE_MODE_HIGH_SPECTRAL_COMPLEXITY,
        )

        self.assertEqual(resolved_mode, FEATURE_MODE_HIGH_SPECTRAL_COMPLEXITY)
        self.assertEqual(
            list(training_df.columns),
            ["True_Class", *spectral_features],
        )
        self.assertEqual(feature_columns, spectral_features)
        self.assertNotIn("Ignored_Column", training_df.columns)
        self.assertNotIn("Another_Extra", training_df.columns)

    def test_spatial_mode_missing_raw_inputs_raise_clear_error(self) -> None:
        source_df = pd.DataFrame(
            [
                {
                    "True_Class": "Water",
                    raw_column_name("B02"): 1.0,
                    raw_column_name("B03"): 2.0,
                    raw_column_name("B04"): 3.0,
                }
            ]
        )

        with self.assertRaisesRegex(
            ValueError,
            r"Training data \(spatial-mode raw inputs\) is missing required columns: Reflectance_B08",
        ):
            build_training_dataframe(
                source_df,
                feature_mode=FEATURE_MODE_HIGH_SPATIAL_ACCURACY,
            )


if __name__ == "__main__":
    unittest.main()
