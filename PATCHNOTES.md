# Patch Notes

## Version 1.0.22 - 2026-04-01

### Added

- Added a training mode selector with `High Spatial Accuracy` and `High Spectral Complexity`
- Added automatic feature-mode inference from exported FastAI model inputs for Apply and Validation
- Added support for 12-band multi-band `.tif` and `.tiff` apply inputs
- Added automatic refresh of the Apply and Validation model lists after training a new model
- Added backend unittest coverage for feature-mode preprocessing and TIFF input discovery

### Changed

- Training now saves new models automatically into the default `models` folder
- High Spatial Accuracy training, validation, and apply runs now rebuild standardized `B02/B03/B04/B08`, `NDVI`, and `NDWI` from the raw 10 m bands
- High Spectral Complexity training now uses the intended spectral feature set explicitly instead of allowing unrelated CSV columns into the learner
- Apply no longer depends on `Reflectance_B01` when building masks, templates, or band coordinates
- The in-app update manifest now points to the `1.0.22` installer and patch notes

### Fixed

- Fixed the spatial-mode training failure caused by duplicate derived columns producing the pandas ambiguous-Series error

### Notes

- TIFF apply inputs are expected to contain the 12 Sentinel-2 spectral bands on a shared grid, so no spatial resampling is performed
- SCL masking remains available for SAFE inputs; TIFF inputs run without an SCL mask because that layer is not present in the raster stack
