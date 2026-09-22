"""Persistent sensor definitions and model-to-sensor associations."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

SENTINEL_2 = "Sentinel-2"
# Sentinel-2A MSI centres (nm), in the application's spectral band order:
# https://documentation.dataspace.copernicus.eu/APIs/SentinelHub/Data/S2L2A.html
SENTINEL_2_BANDS = {
    "B01": 442.7, "B02": 492.4, "B03": 559.8, "B04": 664.6,
    "B05": 704.1, "B06": 740.5, "B07": 782.8, "B08": 832.8,
    "B8A": 864.7, "B09": 945.1, "B11": 1613.7, "B12": 2202.4,
}


def _read_json(path: Path) -> dict:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Invalid sensor registry: {path}")
    return value


def _write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def sensor_registry_path(models_dir: Path) -> Path:
    return _registry_dir(models_dir) / "sensors.json"


def model_registry_path(models_dir: Path) -> Path:
    return _registry_dir(models_dir) / "model_sensors.json"


def _registry_dir(models_dir: Path) -> Path:
    # The bundled models live under the install directory, which may be replaced
    # by an update. Keep user-created sensors and assignments in user data.
    if models_dir.resolve() == (Path(__file__).resolve().parent / "models").resolve():
        user_data = Path(os.environ.get("APPDATA") or Path.home() / ".config")
        return user_data / "ICE_CREAMS_Studio"
    return models_dir


def load_sensors(models_dir: Path) -> dict[str, dict]:
    custom = _read_json(sensor_registry_path(models_dir))
    sensors = {SENTINEL_2: {"name": SENTINEL_2, "bands": SENTINEL_2_BANDS.copy()}}
    for name, definition in custom.items():
        sensors[name] = validate_sensor(name, definition.get("bands", {}))
    return sensors


def validate_sensor(name: str, bands: dict[str, float]) -> dict:
    name = str(name).strip()
    if not name or name.casefold() == "create new sensor":
        raise ValueError("Enter a sensor name.")
    if not isinstance(bands, dict) or not bands:
        raise ValueError("Enter the central wavelength of every band.")
    normalized = {}
    for band, wavelength in bands.items():
        band = str(band).strip()
        try:
            value = float(wavelength)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Enter a numeric wavelength for {band}.") from exc
        if not band or not math.isfinite(value) or value <= 0:
            raise ValueError(f"Enter a positive wavelength for {band}.")
        normalized[band] = value
    return {"name": name, "bands": normalized}


def create_sensor(models_dir: Path, name: str, wavelengths_nm: list[float]) -> dict:
    bands = {f"Band_{index}": wavelength for index, wavelength in enumerate(wavelengths_nm, 1)}
    definition = validate_sensor(name, bands)
    sensors = load_sensors(models_dir)
    if any(existing.casefold() == definition["name"].casefold() for existing in sensors):
        raise ValueError(f"A sensor named '{name}' already exists.")
    custom = _read_json(sensor_registry_path(models_dir))
    custom[definition["name"]] = definition
    _write_json(sensor_registry_path(models_dir), custom)
    return definition


def _model_key(model_path: Path, models_dir: Path) -> str:
    resolved = model_path.resolve()
    try:
        return resolved.relative_to(models_dir.resolve()).as_posix()
    except ValueError:
        return resolved.as_posix()


def load_model_sensors(models_dir: Path) -> dict[str, str]:
    return {str(key): str(value) for key, value in _read_json(model_registry_path(models_dir)).items()}


def assign_model_sensor(models_dir: Path, model_path: Path, sensor_name: str) -> None:
    if sensor_name not in load_sensors(models_dir):
        raise ValueError(f"Unknown sensor: {sensor_name}")
    mapping = load_model_sensors(models_dir)
    mapping[_model_key(model_path, models_dir)] = sensor_name
    _write_json(model_registry_path(models_dir), mapping)


def bootstrap_existing_models(models_dir: Path) -> dict[str, str]:
    """Associate legacy models with Sentinel-2 once; leave later imports unassigned."""
    registry = model_registry_path(models_dir)
    if not registry.exists():
        mapping = {
            _model_key(path, models_dir): SENTINEL_2
            for path in models_dir.rglob("*.pkl")
        } if models_dir.exists() else {}
        _write_json(registry, mapping)
        return mapping
    return load_model_sensors(models_dir)


def model_sensor(models_dir: Path, model_path: Path) -> str | None:
    try:
        return load_model_sensors(models_dir).get(_model_key(model_path, models_dir))
    except ValueError:
        return None


def index_band_names(bands: dict[str, float]) -> dict[str, str]:
    """Choose the closest green/red/NIR centres within physical wavelength windows."""
    regions = {"green": (500, 600, 560), "red": (620, 700, 665), "nir": (760, 900, 833)}
    chosen = {}
    for role, (lower, upper, target) in regions.items():
        candidates = [(abs(float(wavelength) - target), name) for name, wavelength in bands.items()
                      if lower <= float(wavelength) <= upper]
        if candidates:
            chosen[role] = min(candidates)[1]
    return chosen
