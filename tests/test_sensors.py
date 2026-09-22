from pathlib import Path

import pandas as pd
import pytest
import geopandas as gpd
from shapely.geometry import box

from ice_creams_generic_raster import (
    configure_raster_schema_sensor,
    polygon_attribute_columns,
    prepare_generic_features,
)
from ice_creams_sensors import (
    SENTINEL_2,
    assign_model_sensor,
    bootstrap_existing_models,
    create_sensor,
    index_band_names,
    load_sensors,
    model_sensor,
)


def test_sensor_registry_associates_legacy_and_new_models(tmp_path):
    legacy = tmp_path / "legacy.pkl"
    legacy.touch()
    assert bootstrap_existing_models(tmp_path) == {"legacy.pkl": SENTINEL_2}
    new_model = tmp_path / "new.pkl"
    new_model.touch()
    assert model_sensor(tmp_path, new_model) is None
    definition = create_sensor(tmp_path, "Drone RGB-NIR", [490, 560, 665, 840])
    assert load_sensors(tmp_path)["Drone RGB-NIR"] == definition
    assign_model_sensor(tmp_path, new_model, "Drone RGB-NIR")
    assert model_sensor(tmp_path, new_model) == "Drone RGB-NIR"
    with pytest.raises(ValueError, match="already exists"):
        create_sensor(tmp_path, "drone rgb-nir", [490])


def test_custom_wavelengths_build_ndvi_and_ndwi_for_generic_raster(tmp_path):
    sensor = create_sensor(tmp_path, "Four band camera", [490, 560, 665, 840])
    schema = configure_raster_schema_sensor(
        {"band_count": 4, "band_descriptions": [], "feature_names": [f"Band_{i}" for i in range(1, 5)]},
        sensor,
    )
    frame = pd.DataFrame({"Band_1": [0.1], "Band_2": [0.2], "Band_3": [0.3], "Band_4": [0.6]})
    features = prepare_generic_features(frame, schema, standardized=True)
    assert features.loc[0, "NDVI"] == pytest.approx(1 / 3)
    assert features.loc[0, "NDWI"] == pytest.approx(-0.5)
    assert "NDVI_Standardized" in features and "NDWI_Standardized" in features
    assert index_band_names({"Band_1": 490, "Band_2": 665}) == {"red": "Band_2"}
    with pytest.raises(ValueError, match="defines 4 bands"):
        configure_raster_schema_sensor({"feature_names": ["Band_1"]}, sensor)


def test_training_requires_sensor_before_reading_data(tmp_path):
    from train_icecreams import train_model

    with pytest.raises(ValueError, match="Select a sensor"):
        train_model([], str(tmp_path / "model.pkl"))


def test_shapefile_attribute_names_are_available_for_class_selection(tmp_path):
    path = tmp_path / "labels.shp"
    gpd.GeoDataFrame(
        {"Habitat": ["sand"], "Class_ID": [2]},
        geometry=[box(0, 0, 1, 1)], crs="EPSG:3857",
    ).to_file(path)
    assert polygon_attribute_columns(str(path)) == ["Habitat", "Class_ID"]


def test_one_csv_trains_custom_sensor_model_in_chosen_folder(tmp_path):
    import numpy as np
    import rasterio
    from rasterio.transform import from_origin
    from apply_ICECREAMS import classify_s2_scene
    from train_icecreams import train_model
    from validate_icecreams import validate_model

    create_sensor(tmp_path, "Three band drone", [560, 665, 840])
    csv_path = tmp_path / "training.csv"
    pd.DataFrame({
        "True_Class": ["sand"] * 12 + ["vegetation"] * 12,
        "Green": [0.2] * 12 + [0.3] * 12,
        "Red": [0.35] * 12 + [0.15] * 12,
        "NIR": [0.2] * 12 + [0.7] * 12,
    }).to_csv(csv_path, index=False)
    model_path = tmp_path / "chosen_location" / "drone.pkl"
    result = train_model(
        str(csv_path), str(model_path), epochs=1, batch_size=8,
        feature_mode="generic_raster", sensor_name="Three band drone",
        sensor_registry_dir=str(tmp_path),
    )
    assert result["sensor_name"] == "Three band drone"
    assert model_sensor(tmp_path, model_path) == "Three band drone"
    validated = validate_model(
        str(csv_path), str(model_path), label_column="True_Class",
        output_dir=str(tmp_path / "validation"),
    )
    assert validated["rows"] == 24
    raster_path = tmp_path / "drone.tif"
    with rasterio.open(
        raster_path, "w", driver="GTiff", width=4, height=4, count=3,
        dtype="float32", crs="EPSG:3857", transform=from_origin(0, 4, 1, 1),
    ) as dst:
        for band, value in enumerate((0.2, 0.35, 0.2), start=1):
            dst.write(np.full((4, 4), value, dtype="float32"), band)
    output_path = tmp_path / "prediction.tif"
    classify_s2_scene(str(raster_path), str(output_path), str(model_path))
    with rasterio.open(output_path) as predicted:
        assert predicted.count == 2


def test_sentinel_csv_uses_existing_features_without_extra_ui_settings(tmp_path):
    from ice_creams_feature_modes import DEFAULT_FEATURE_MODE, FEATURE_COLUMNS_BY_MODE
    from train_icecreams import train_model

    rows = 24
    csv_path = tmp_path / "sentinel.csv"
    frame = pd.DataFrame({name: [0.2] * 12 + [0.7] * 12 for name in FEATURE_COLUMNS_BY_MODE[DEFAULT_FEATURE_MODE]})
    frame.insert(0, "True_Class", ["sand"] * 12 + ["vegetation"] * 12)
    frame.to_csv(csv_path, index=False)
    result = train_model(
        str(csv_path), str(tmp_path / "sentinel.pkl"), epochs=1, batch_size=8,
        sensor_name=SENTINEL_2,
    )
    assert result["rows"] == rows
    assert result["feature_mode"] == DEFAULT_FEATURE_MODE
