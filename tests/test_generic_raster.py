from __future__ import annotations

import geopandas as gpd
import numpy as np
import pytest
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import box

from ice_creams_generic_raster import (
    labelled_generic_raster_dataframe,
    matching_band_indices,
    prepare_generic_features,
)
from ice_creams_sensors import create_sensor


def _sample(tmp_path, labels=(1, 2)):
    create_sensor(tmp_path, "Drone", [665, 833])
    raster_path = tmp_path / "drone.tif"
    polygon_path = tmp_path / "classes.geojson"
    with rasterio.open(
        raster_path, "w", driver="GTiff", width=8, height=8, count=2,
        dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 8, 1, 1),
        nodata=0,
    ) as dst:
        first = np.full((8, 8), 20, dtype="uint8")
        second = np.full((8, 8), 80, dtype="uint8")
        first[:, 4:] = 80
        second[:, 4:] = 20
        dst.write(first, 1)
        dst.write(second, 2)
        dst.set_band_description(1, "Red")
        dst.set_band_description(2, "NIR")
    gpd.GeoDataFrame(
        {"habitat": list(labels)}, geometry=[box(0, 0, 4, 8), box(4, 0, 8, 8)],
        crs="EPSG:3857",
    ).to_file(polygon_path, driver="GeoJSON")
    return raster_path, polygon_path


def test_generic_raster_uses_all_input_bands_without_sentinel_features(tmp_path):
    raster, polygons = _sample(tmp_path)
    frame, schema = labelled_generic_raster_dataframe(str(raster), str(polygons), "habitat")
    assert len(frame) == 64
    assert schema["band_descriptions"] == ["Red", "NIR"]
    assert schema["feature_names"] == ["Band_1", "Band_2"]
    assert set(frame["habitat"]) == {"1", "2"}
    assert "NDVI" not in frame
    prepared = prepare_generic_features(frame, schema, standardized=True)
    assert list(prepared.columns) == ["Band_1", "Band_2", "Band_1_Standardized", "Band_2_Standardized"]


def test_generic_band_names_allow_reordering_and_reject_wrong_bands(tmp_path):
    raster, polygons = _sample(tmp_path)
    _, schema = labelled_generic_raster_dataframe(str(raster), str(polygons), "habitat")
    with rasterio.open(raster, "r+") as dst:
        dst.set_band_description(1, "NIR")
        dst.set_band_description(2, "Red")
    with rasterio.open(raster) as src:
        assert matching_band_indices(src, schema) == [2, 1]
    with rasterio.open(raster, "r+") as dst:
        dst.set_band_description(1, "Blue")
    with rasterio.open(raster) as src:
        with pytest.raises(ValueError, match="band descriptions"):
            matching_band_indices(src, schema)


def test_generic_train_validate_and_apply(tmp_path, monkeypatch):
    import torch

    from apply_ICECREAMS import classify_s2_scene
    from train_icecreams import train_model
    from validate_icecreams import validate_model

    torch.set_num_threads(1)
    raster, polygons = _sample(tmp_path)
    model_path = tmp_path / "drone_model.pkl"
    trained = train_model(
        training_source=[], output_model=str(model_path), epochs=1,
        valid_pct=0.25, batch_size=16, feature_mode="generic_raster",
        raster_path=str(raster), polygon_path=str(polygons), label_column="habitat",
        sensor_name="Drone",
    )
    assert trained["classes"] == 2
    assert trained["sensor_name"] == "Drone"
    assert model_path.is_file()
    from fastai.learner import load_learner
    from ice_creams_model_families import extract_model_metadata, predict_model_probabilities
    learner = load_learner(str(model_path))
    metadata = extract_model_metadata(learner)
    assert metadata["sensor_name"] == "Drone"
    assert "NDVI" in metadata["required_feature_names"]

    labelled, _ = labelled_generic_raster_dataframe(str(raster), str(polygons), "habitat")
    features = prepare_generic_features(labelled, metadata["raster_schema"])
    baseline, _ = learner.get_preds(dl=learner.dls.test_dl(features, bs=7))
    monkeypatch.setattr(learner, "get_preds", lambda **_: pytest.fail("fastai loader was used"))
    direct = predict_model_probabilities(learner, features, metadata, batch_size=7)
    assert torch.allclose(direct, baseline, atol=1e-6)

    validated = validate_model(
        dataset_path=str(raster), model_path=str(model_path),
        label_column="habitat", polygon_path=str(polygons),
        output_dir=str(tmp_path / "validation"),
    )
    assert validated["rows"] == 64
    assert (tmp_path / "validation" / f"{raster.stem}__{model_path.stem}__predictions.csv").is_file()

    output = tmp_path / "classified.tif"
    classify_s2_scene(str(raster), str(output), str(model_path), mask_vector_file=None)
    with rasterio.open(output) as result:
        assert result.count == 2
        assert result.tags()["CLASS_LABELS"]
        assert set(np.unique(result.read(1))) <= {1, 2}


def test_generic_cnn_train_validate_and_apply(tmp_path):
    import torch

    from apply_ICECREAMS import classify_s2_scene
    from train_icecreams import train_model
    from validate_icecreams import validate_model

    torch.set_num_threads(1)
    raster, polygons = _sample(tmp_path)
    model_path = tmp_path / "drone_cnn.pkl"
    trained = train_model(
        training_source=[], output_model=str(model_path), epochs=1,
        valid_pct=0.25, batch_size=16, feature_mode="generic_raster",
        model_family="spectral_1d_cnn",
        spectral_cnn_use_standardized_reflectance=True,
        raster_path=str(raster), polygon_path=str(polygons), label_column="habitat",
        sensor_name="Drone",
    )
    assert trained["classes"] == 2
    validated = validate_model(
        dataset_path=str(raster), model_path=str(model_path),
        label_column="habitat", polygon_path=str(polygons),
        output_dir=str(tmp_path / "validation"),
    )
    assert validated["rows"] == 64
    output = tmp_path / "classified_cnn.tif"
    classify_s2_scene(str(raster), str(output), str(model_path), mask_vector_file=None)
    with rasterio.open(output) as result:
        assert result.count == 2
        assert set(np.unique(result.read(1))) <= {1, 2}


def test_generic_validation_preserves_distinct_user_class_names(tmp_path):
    import torch

    from train_icecreams import train_model
    from validate_icecreams import validate_model

    torch.set_num_threads(1)
    raster, polygons = _sample(tmp_path, labels=("Bare Sand", "Sand"))
    model_path = tmp_path / "classes.pkl"
    train_model(
        training_source=[], output_model=str(model_path), epochs=1,
        valid_pct=0.25, batch_size=16, feature_mode="generic_raster",
        raster_path=str(raster), polygon_path=str(polygons), label_column="habitat",
        sensor_name="Drone",
    )
    result = validate_model(
        dataset_path=str(raster), model_path=str(model_path),
        label_column="habitat", polygon_path=str(polygons),
        output_dir=str(tmp_path / "validation"),
    )
    assert result["classes"] == 2
    assert result["confusion_labels"] == ["Bare Sand", "Sand"]


def test_apply_combines_small_storage_blocks_and_skips_unmasked_windows(tmp_path, monkeypatch):
    from types import SimpleNamespace

    import torch
    from ice_creams_generic_raster import classify_generic_raster

    raster_path = tmp_path / "small_blocks.tif"
    with rasterio.open(
        raster_path, "w", driver="GTiff", width=1024, height=512, count=2,
        dtype="uint8", crs="EPSG:3857", transform=from_origin(0, 512, 1, 1),
        tiled=True, blockxsize=16, blockysize=16,
    ) as dst:
        dst.write(np.full((2, 512, 1024), 50, dtype="uint8"))
    mask_path = tmp_path / "mask.geojson"
    gpd.GeoDataFrame(
        {"mask": [1]}, geometry=[box(0, 0, 512, 512)], crs="EPSG:3857",
    ).to_file(mask_path, driver="GeoJSON")

    calls = []

    def predict(_, frame, __, *, batch_size):
        calls.append(len(frame))
        return torch.tensor([[0.9, 0.1]], dtype=torch.float32).repeat(len(frame), 1)

    monkeypatch.setattr("ice_creams_model_families.predict_model_probabilities", predict)
    learner = SimpleNamespace(dls=SimpleNamespace(vocab=["sand", "vegetation"]))
    metadata = {
        "raster_schema": {
            "band_count": 2, "band_descriptions": [], "feature_names": ["Band_1", "Band_2"],
        },
    }
    progress = []
    output = tmp_path / "classified.tif"
    classify_generic_raster(
        str(raster_path), str(output), learner, metadata,
        mask_polygon_path=str(mask_path), progress_callback=progress.append,
    )
    assert calls == [512 * 512]
    assert progress == [0.5, 1.0]
    with rasterio.open(output) as result:
        assert result.block_shapes == [(256, 256), (256, 256)]
        assert result.read(1, window=rasterio.windows.Window(0, 0, 1, 1))[0, 0] == 1
        assert result.read(1, window=rasterio.windows.Window(600, 0, 1, 1))[0, 0] == -1
