import tempfile
import unittest
from pathlib import Path

import numpy as np
import xarray as xr
from rasterio.crs import CRS

from apply_ICECREAMS import (
    ACOLITE_BAND_VARIABLES,
    discover_scene_batch_info,
    read_acolite_netcdf,
)


class ApplyAcoliteInputTests(unittest.TestCase):
    def test_discover_and_read_acolite_netcdf_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            netcdf_path = Path(temp_dir) / "S2B_MSI_2026_03_07_11_45_54_T28QCH_L2W.nc"
            projection_attrs = {
                "grid_mapping_name": "transverse_mercator",
                "crs_wkt": CRS.from_epsg(32628).to_wkt(),
            }
            coords = {
                "x": (
                    "x",
                    np.asarray([334055.0, 334065.0, 334075.0, 334085.0], dtype=np.float64),
                    {
                        "standard_name": "projection_x_coordinate",
                        "long_name": "x coordinate of projection",
                        "units": "m",
                    },
                ),
                "y": (
                    "y",
                    np.asarray([2209975.0, 2209965.0, 2209955.0], dtype=np.float64),
                    {
                        "standard_name": "projection_y_coordinate",
                        "long_name": "y coordinate of projection",
                        "units": "m",
                    },
                ),
                "transverse_mercator": xr.DataArray(0.0, attrs=projection_attrs),
            }
            data_vars = {}
            for band_index, variable_name in enumerate(ACOLITE_BAND_VARIABLES.values(), start=1):
                data_vars[variable_name] = xr.DataArray(
                    np.full((3, 4), float(band_index), dtype=np.float32),
                    dims=("y", "x"),
                    attrs={
                        "long_name": "Surface reflectance",
                        "units": "1",
                        "grid_mapping": "transverse_mercator",
                    },
                )

            dataset = xr.Dataset(
                data_vars,
                coords=coords,
                attrs={
                    "generated_by": "ACOLITE",
                    "acolite_file_type": "L2W",
                    "projection_key": "transverse_mercator",
                    "sensor": "S2B_MSI",
                },
            )
            dataset.to_netcdf(netcdf_path)

            batch_info = discover_scene_batch_info(str(netcdf_path))
            self.assertEqual(batch_info["unique_count"], 1)
            self.assertEqual(batch_info["format_counts"]["ACOLITE"], 1)
            self.assertEqual(batch_info["selected"][0]["format"], "ACOLITE")
            self.assertEqual(batch_info["selected"][0]["acquisition_date"], "2026-03-07")

            raster_data = read_acolite_netcdf(
                str(netcdf_path),
                required_raw_bands=("B02", "B03", "B04", "B08", "B11"),
            )
            try:
                self.assertEqual(
                    sorted(raster_data.data_vars),
                    [
                        "Reflectance_B02",
                        "Reflectance_B03",
                        "Reflectance_B04",
                        "Reflectance_B08",
                        "Reflectance_B11",
                    ],
                )
                self.assertEqual(raster_data["Reflectance_B02"].shape, (1, 3, 4))
                self.assertTrue(
                    np.allclose(
                        raster_data["Reflectance_B02"].values,
                        np.full((1, 3, 4), 20000.0, dtype=np.float32),
                    )
                )
                self.assertTrue(
                    np.allclose(
                        raster_data["Reflectance_B11"].values,
                        np.full((1, 3, 4), 100000.0, dtype=np.float32),
                    )
                )
            finally:
                raster_data.close()

    def test_discover_and_read_acolite_netcdf_with_sensor_specific_wavelength_names(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            netcdf_path = Path(temp_dir) / "S2C_MSI_2025_06_15_11_46_15_T28QCH_L2W.nc"
            projection_attrs = {
                "grid_mapping_name": "transverse_mercator",
                "crs_wkt": CRS.from_epsg(32628).to_wkt(),
            }
            coords = {
                "x": (
                    "x",
                    np.asarray([334055.0, 334065.0, 334075.0, 334085.0], dtype=np.float64),
                    {
                        "standard_name": "projection_x_coordinate",
                        "long_name": "x coordinate of projection",
                        "units": "m",
                    },
                ),
                "y": (
                    "y",
                    np.asarray([2209975.0, 2209965.0, 2209955.0], dtype=np.float64),
                    {
                        "standard_name": "projection_y_coordinate",
                        "long_name": "y coordinate of projection",
                        "units": "m",
                    },
                ),
                "transverse_mercator": xr.DataArray(0.0, attrs=projection_attrs),
            }
            sensor_specific_band_variables = {
                "B01": "rhos_444",
                "B02": "rhos_489",
                "B03": "rhos_561",
                "B04": "rhos_667",
                "B05": "rhos_707",
                "B06": "rhos_741",
                "B07": "rhos_785",
                "B08": "rhos_835",
                "B8A": "rhos_866",
                "B11": "rhos_1612",
                "B12": "rhos_2191",
            }
            data_vars = {}
            for band_index, variable_name in enumerate(sensor_specific_band_variables.values(), start=1):
                data_vars[variable_name] = xr.DataArray(
                    np.full((3, 4), float(band_index), dtype=np.float32),
                    dims=("y", "x"),
                    attrs={
                        "long_name": "Surface reflectance",
                        "units": "1",
                        "grid_mapping": "transverse_mercator",
                    },
                )

            dataset = xr.Dataset(
                data_vars,
                coords=coords,
                attrs={
                    "generated_by": "ACOLITE",
                    "acolite_file_type": "L2W",
                    "projection_key": "transverse_mercator",
                    "sensor": "S2C_MSI",
                },
            )
            dataset.to_netcdf(netcdf_path)

            batch_info = discover_scene_batch_info(str(netcdf_path))
            self.assertEqual(batch_info["unique_count"], 1)
            self.assertEqual(batch_info["format_counts"]["ACOLITE"], 1)
            self.assertEqual(batch_info["selected"][0]["format"], "ACOLITE")
            self.assertEqual(batch_info["selected"][0]["acquisition_date"], "2025-06-15")

            raster_data = read_acolite_netcdf(
                str(netcdf_path),
                required_raw_bands=("B02", "B03", "B04", "B08", "B11"),
            )
            try:
                self.assertEqual(
                    sorted(raster_data.data_vars),
                    [
                        "Reflectance_B02",
                        "Reflectance_B03",
                        "Reflectance_B04",
                        "Reflectance_B08",
                        "Reflectance_B11",
                    ],
                )
                self.assertEqual(raster_data["Reflectance_B02"].shape, (1, 3, 4))
                self.assertTrue(
                    np.allclose(
                        raster_data["Reflectance_B02"].values,
                        np.full((1, 3, 4), 20000.0, dtype=np.float32),
                    )
                )
                self.assertTrue(
                    np.allclose(
                        raster_data["Reflectance_B11"].values,
                        np.full((1, 3, 4), 100000.0, dtype=np.float32),
                    )
                )
            finally:
                raster_data.close()


if __name__ == "__main__":
    unittest.main()
