# Patch Notes

## Version 1.0.24 - 2026-04-02

### Added

- Added ecological post-processing that reclassifies any pixel with positive `NDWI` to output `Class 8` before patch smoothing and `SPC` computation
- Added an Apply-tab option to smooth small salt-and-pepper patches in the final class map

### Changed

- Small-patch smoothing now uses a broader ecological neighborhood and connected-component relabeling instead of single-pixel cleanup only
- The unfinished secondary specialist-model toggle has been removed from the Apply tab while the backend wiring remains disabled in the desktop UI
- The packaged desktop release has been rebuilt against the current `models` folder contents for version `1.0.24`

### Fixed

- Improved post-processing consistency so class-map cleanup happens before `SPC` is derived for output `Class 4`
- Improved apply performance for patch smoothing by moving connected-component detection to `scipy.ndimage.label`

### Notes

- Previous version logs are retained below for release history
- Packaged models from the `models` folder are bundled into the desktop app and installer for this release

## Version 1.0.23 - 2026-04-01

### Added

- Added an optional `Spectral 1D CNN` training method alongside the legacy tabular dense network
- Added a Train-tab option to include standardized reflectance as an extra spectral CNN input channel
- Added automatic model-family and spectral-input-layout detection for Apply and Validation
- Added backend tests covering raw-only and raw-plus-standardized spectral CNN metadata and inference

### Changed

- Newly trained spectral CNN models now store whether they use raw reflectance only or raw plus standardized reflectance
- Apply and Validation now surface the detected model method, feature mode, and spectral input layout in their status text and run history
- The packaged desktop release has been rebuilt against the current `models` folder contents for version `1.0.23`

### Fixed

- Fixed spectral CNN apply and validation compatibility so both older raw-only exports and newer raw-plus-standardized exports run without a manual override
- Fixed apply post-processing so any output `Class 4` pixel with `NDVI < 0.25` is reclassified to `Class 5` and receives `SPC = -1`

### Notes

- Spectral CNN models remain pixel-wise classifiers; they do not use neighboring pixels or spatial patches
- Packaged models from the `models` folder are bundled into the desktop app and installer for this release
