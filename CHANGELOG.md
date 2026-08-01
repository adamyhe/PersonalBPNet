# Changelog

## [0.7.0] - 2026-08-01

### Changed (breaking)
- Repository restructured to a `src/` layout: `src/personal_bpnet/` (library) and `src/cli/` (CLI entry points, previously `personal_bpnet/cli_*.py`).
- `PersonalBPNet` is now a subclass of `bpnetlite.bpnet.BPNet` (inherits `__init__`/`forward` unchanged) instead of duplicating its architecture; the `alpha` constructor kwarg is renamed to `count_loss_weight` to match `BPNet`.
- `CLIPNET` is now a subclass of `PersonalBPNet`, adding only batch normalization layers; its `fit()` is inherited rather than duplicated. State_dict keys/shapes are unchanged, so existing `f1.torch`..`f9.torch` checkpoints still load.
- `PauseNet` is now a subclass of `bpnetlite.bpnet.CountWrapper` instead of wrapping one internally; access the wrapped base model via `.model` instead of `.transfer_model`.
- Removed `personal_bpnet/_DEPRECATED_bnbpnet.py` (unused, superseded by `clipnet_pytorch.py`).
- Replaced the `pyfastx` dependency with `pyfaidx` (already an indirect dependency via `tangermeme`), removing a redundant FASTA-parsing library from the dependency tree.
- `utils.twohot_encode`: rewritten from a per-character Python dict lookup to a `numba.njit`-compiled byte-lookup-table gather (`_twohot_gather`) — ~80x faster on a 2114bp sequence (204us -> 2.5us), with bit-for-bit identical output. `numba` is now an explicit dependency (previously only pulled in transitively via `tangermeme`).
- `utils.get_twohot_fasta_sequences`: now encodes all records in one batched call to `_twohot_gather` instead of looping per-sequence (optionally across a multiprocessing pool). Batching plus the faster per-position gather made the process pool a net loss (measured >10x slower than single-threaded once encoding itself got fast), so multiprocessing was removed; the `cores`/`desc`/`silence` parameters are gone. The function now requires every record to be the same length and raises `ValueError` up front otherwise (previously this requirement was implicit and failed later, more confusingly, at the final `np.stack`).

### Fixed
- `PauseNet`: the replacement counts-head `linear`/`cbn` layers were attached to the `CountWrapper` wrapper object instead of the wrapped base model, so they were never actually used in the forward pass — the base model's original (pretrained) counts head kept running unchanged. Layers are now assigned onto the wrapped model itself.
- `PauseNet`: `n_filters` was a hardcoded default (512) for the new head's input width rather than inferred from the base model, silently causing a shape-mismatch crash for any non-default `n_filters`. Now defaults to the base model's existing `linear.in_features`.
- `PauseNet.fit()`: crashed immediately on the common no-control-tracks case (unpacking a 2-tuple into 3 names); also hardcoded `.cuda()` instead of respecting a `device` argument.
- `PauseNet.__init__`: `param.require_grad = True` typo (no-op) silently left the new head frozen when `base_trainable=False`; fixed to `requires_grad`.
- `PersonalBPNet.fit()`: training loss values were logged as raw tensors instead of `.item()` floats.
- `utils.reverse_complement_twohot`: mutated its input array in place via numpy view aliasing, and incorrectly swapped the self-complementary `W` (A/T) and `S` (C/G) IUPAC heterozygote codes instead of leaving them unchanged. Plain axis reversal already handles all IUPAC codes correctly; the special-case correction was removed.
- `personal_bpnet/__init__.py` exported nothing but `__version__`, so the top-level import shown in the README (`from personal_bpnet import CLIPNET, PauseNet`) was broken. Now re-exports `CLIPNET`, `CLIPNET_TF`, `PauseNet`, `PersonalBPNet`, `ProCapNet`, `TwoHotToOneHot`.
- `clipnet` CLI (`predict`/`predict_tss`) and `pausenet` CLI (`predict`): ensembling averaged already-softmaxed/exponentiated per-replicate tracks, biasing the ensembled profile toward uniformity whenever replicates disagreed. Now averages raw profile logits and log-counts across replicates before applying softmax/exp and rescaling.
- `clipnet_tensorflow.py`: `h5py` was imported at module level, so importing `personal_bpnet` (or `CLIPNET_TF`) unconditionally required the `[tf]` extra even when TF weight loading was never used. Import moved inside `from_tf()`.

### Added
- `tests/` (pytest): coverage for model construction/forward shapes, state_dict key parity against bare `bpnetlite` classes, loss functions, twohot encoding utilities, and the CLI ensemble-averaging helpers.
- `.github/workflows/ci.yml`: runs the test suite on push/PR across Python 3.9–3.13 (all confirmed working; classifiers in `pyproject.toml` updated to include 3.13).
- `.github/workflows/publish.yml`: builds and publishes to PyPI on GitHub Release via `uv build`/`uv publish` (OIDC trusted publishing, no stored token), gated on the full `ci.yml` test matrix passing.
- `src/cli/_common.py`: shared argparse and ensemble-loading helpers, replacing ~300 lines of near-identical code duplicated across the three CLI entry points.
- `uv.lock`, committed for reproducible local dev installs; see `CONTRIBUTING.md` for the `uv sync`/`uv run` workflow.
- `docs/`: per-model usage docs (`clipnet.md`, `clipnet-tf.md`, `procapnet.md`, `pausenet.md`), plus `cli.md`. `README.md` is now a short overview, with PyPI-based install instructions (`uv add`/`pip install personalbpnet`) directly in it; the from-source/local-dev workflow moved to the new `CONTRIBUTING.md`.
- `pyproject.toml`: added `readme`, `keywords`, `classifiers`, and expanded `urls` (Repository/Issues/Changelog/Documentation) for proper PyPI metadata.
- `CITATION.cff`: cites this software plus the BPNet, CLIPNET, and ProCapNet papers, so GitHub's "Cite this repository" widget and tools like `cffconvert` can surface them.

## [0.6.7] - 2026-02-23

### Fixed
- `procapnet.py`: mask was extracted from `y` after slicing to `n_outputs` channels, causing the last signal channel (e.g. minus strand) to be used as the mask instead of the actual mask track. Mask is now extracted before slicing.
- `procapnet.py`: `y_valid` was not sliced to `n_outputs` channels before validation, causing a shape mismatch when `y_has_mask=True` and `y_valid` included the mask channel. `y_valid` is now sliced to `n_outputs` channels at the start of `fit()` when `y_has_mask=True`.
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
