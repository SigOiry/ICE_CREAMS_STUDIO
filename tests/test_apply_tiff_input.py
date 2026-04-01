import tempfile
import unittest
from pathlib import Path

import numpy as np
import rasterio
from rasterio.transform import from_origin

from apply_ICECREAMS import discover_scene_batch_info, read_multiband_tiff


class ApplyTiffInputTests(unittest.TestCase):
    def test_discover_and_read_multiband_tiff_input(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            raster_path = Path(temp_dir) / "synthetic_stack.tif"
            profile = {
                "driver": "GTiff",
                "height": 3,
                "width": 4,
                "count": 12,
                "dtype": "float32",
                "crs": "EPSG:32630",
                "transform": from_origin(100.0, 200.0, 10.0, 10.0),
            }
            with rasterio.open(raster_path, "w", **profile) as raster:
                for band_index in range(1, 13):
                    raster.write(
                        np.full((3, 4), float(band_index), dtype=np.float32),
                        band_index,
                    )

            batch_info = discover_scene_batch_info(str(raster_path))
            self.assertEqual(batch_info["unique_count"], 1)
            self.assertEqual(batch_info["format_counts"]["TIFF"], 1)
            self.assertEqual(batch_info["selected"][0]["format"], "TIFF")

            raster_data = read_multiband_tiff(
                str(raster_path),
                required_raw_bands=("B02", "B03", "B04", "B08"),
            )
            try:
                self.assertEqual(
                    sorted(raster_data.data_vars),
                    [
                        "Reflectance_B02",
                        "Reflectance_B03",
                        "Reflectance_B04",
                        "Reflectance_B08",
                    ],
                )
                self.assertEqual(raster_data["Reflectance_B02"].shape, (1, 3, 4))
                self.assertTrue(
                    np.allclose(
                        raster_data["Reflectance_B02"].values,
                        np.full((1, 3, 4), 2.0, dtype=np.float32),
                    )
                )
                self.assertTrue(
                    np.allclose(
                        raster_data["Reflectance_B08"].values,
                        np.full((1, 3, 4), 8.0, dtype=np.float32),
                    )
                )
            finally:
                raster_data.close()


if __name__ == "__main__":
    unittest.main()
