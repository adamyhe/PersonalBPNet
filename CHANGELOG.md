# Changelog

## [0.6.7] - 2026-02-23

### Fixed
- `procapnet.py`: mask was extracted from `y` after slicing to `n_outputs` channels, causing the last signal channel (e.g. minus strand) to be used as the mask instead of the actual mask track. Mask is now extracted before slicing.
- `procapnet.py`: removed stray characters at end of `fit()` that caused a `SyntaxError` on import.
- `losses.py` (`_mixture_loss_masked`): when `labels` was provided, `mask` was not filtered alongside `y` and `y_hat_logits`, causing mask rows to be misaligned with the filtered examples. `mask` is now filtered with `mask[labels == 1]`.
- `losses.py` (`MNLLLoss_masked`): per-example shape after masking was compared against the full-batch tensor shape instead of the per-example pre-masking shape, causing the unsqueeze branch to always trigger. Fixed to compare against `logits_i.shape`.

### Added
- Optional `[tf]` install extra (`pip install "PersonalBPNet[tf]"`) that includes `h5py`, required for `CLIPNET_TF.from_tf()`.
- `ProCapNet` documented in README with usage example, data format details, and note on weight interchangeability with `bpnetlite.bpnet.BPNet`.
- Reference copy of the original ProCapNet implementation (`procapnet_orig.py`) noted in README for reproducibility.

### Changed
- `personal_bpnet/__init__.py`: `__version__` is now derived dynamically from package metadata via `importlib.metadata` rather than being hardcoded. `pyproject.toml` is now the sole source of truth for the version number.
- `losses.py` (`_mixture_loss_masked`): when `mask=None`, now delegates directly to bpnetlite's `_mixture_loss` rather than reimplementing the no-mask path. `MNLLLoss_masked` consequently no longer handles `mask=None`.

## [0.6.6] - prior release
