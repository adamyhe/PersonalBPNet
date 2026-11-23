# Changelog

## [Unreleased]

### Added
- `personal_bpnet.clipnet_tf_fast.dfim` and `dfim_shuffle`, with `clipnet_tf dfim` and `clipnet_tf dfim_shuffle`: Deep Feature Interaction Maps (Greenside et al. 2018), as the original CLIPNET's `clipnet epistasis` computes them.
  - `dfim`: single-nucleotide maps (every substitution at every source position), reduced to the most extreme FIS per source position (`reduce="extrema"`, as CLIPNET) or kept per substitution.
  - `dfim_shuffle`: window-shuffle maps, filling the window from a dinucleotide shuffle of the whole sequence (as CLIPNET) or shuffling the window itself.
  - Each variant equals tangermeme's `deep_lift_shap` on it with its sequence's references. Those references are shared, so they go through the forward pass once per step instead of once per variant. On an RTX 3080, 600 variants take 0.37 s, against 4.36 s with tangermeme (~12x).
  - Any target (`counts`, `profile`, `shape`). The CLI averages the unreduced maps over the ensemble before reducing.
- `clipnet_tf_fast.dfim_extrema`: CLIPNET's reduction, for maps computed with `reduce=None`, e.g. after averaging an ensemble.

## [0.8.0] - 2026-11-22

### Added
- `personal_bpnet.clipnet_tf_fast.deep_lift_shap`: a DeepLIFT/SHAP engine written for `CLIPNET_TF`, after Cherimoya's and promoterai-torch's: joint sequence/reference rows (each sequence forwarded once instead of once per reference), a hand-written backward over only the sequence half of each pair, fused DeepLIFT rules, and references drawn on the CPU while the GPU runs. It returns the same attributions as tangermeme's `deep_lift_shap` (same references for a given seed), for 1024 sequences × 20 shuffles: 1.35 s vs 6.3 s for tangermeme at its fastest batch size (4.7x; 9x vs its default batch size) on an RTX 3080, and 3.6 s vs 12.4 s (3.4x) on a TITAN X (Pascal), in ~5x less memory per sequence–reference pair. The rules are fused with `torch.compile` on GPUs of compute capability >= 7.0 and with TorchScript elsewhere (`compile=` to choose).
- `personal_bpnet.clipnet_tf_fast.saturation_mutagenesis`: incremental (fastISM) in silico saturation mutagenesis for `CLIPNET_TF`. It caches each reference's layers and recomputes only the window each substitution can reach, giving the same predictions as tangermeme's `saturation_mutagenesis` (to float rounding) ~2.7x faster on an RTX 3080. Its forward folds the BatchNorms into the adjacent layers, caches pre-padded activations and fuses each residual epilogue (TorchScript by default, `compile=True` for torch.compile). For `output="profile"`, attributions are the change at every profile position, unreduced, `(N, 4, end - start, n_profile)`, rather than tangermeme's mean over the outputs (only the change in the profile's total, which predictions normalize away).
- `clipnet_tf attribute --profile_rule {product,rescale}`.
- `clipnet_tf ism`: in silico saturation mutagenesis through the incremental engine (`-a counts|profile|both`, `--start/--end`, `--raw_outputs`, `-y`), averaged across the ensemble.
- `clipnet_tf_fast.deep_lift_shap(target="shape")` and `clipnet_tf attribute -a shape`: DeepLIFT/SHAP of the profile's shape, `sum_j w_j p_j`, with `p = profile / (sum(profile) + 1e-3)` normalized as `clipnet_tf predict` normalizes it and `w` the sequence's own `p` (midpoint product rule, rescale rule at the reciprocal). Unlike `-a profile`'s softmax summary, it doesn't treat CLIPNET's sum-normalized profile as logits.
- `personal_bpnet.clipnet_tf_fast.pisa` and `clipnet_tf pisa`: PISA (DeepLIFT/SHAP of every profile output position) for `CLIPNET_TF`, after Cherimoya's PISA engine. Each sequence is forwarded once with its references, every rule's (activation-only) coefficient is computed once per sequence–reference pair and shared across output positions, and the backward runs over the sequence half only, for chunks of output positions. By default it attributes the predicted track `counts * profile / (sum(profile) + 1e-3)` (midpoint product rule at the products, rescale rule at the reciprocal); `target="profile"` attributes the raw profile head. Equal to tangermeme's `pisa` on the equivalent wrapper (~1e-15 in float64), and 17-20x faster on an RTX 3080 (0.155 s per sequence and strand at 20 shuffles).
- `tf32=` on `clipnet_tf_fast.deep_lift_shap`, `saturation_mutagenesis` and `pisa`, and `--tf32` on every `clipnet_tf` subcommand. All default to off (full float32).
- `examples/promoter_interpretation/`: scripts comparing DeepLIFT/SHAP and PISA reference sets for `CLIPNET_TF` against in silico mutagenesis at 11 lymphoblastoid-expressed promoters, with TSSs taken from ENCODE B-cell PRO-cap. They aren't part of the package.

### Changed (breaking)
- Supported Python versions are now 3.11–3.14 (`requires-python = ">= 3.11"`; CI matrix updated). 3.9 and 3.10 are past end of life, 3.9 could no longer build bpnet-lite's `macs3` dependency, and current torch requires >= 3.11.
- `clipnet_tf attribute` uses the fast engine. `-bs/--batch_size` now counts sequences per step (each with all `-n` references) rather than sequence–reference pairs, and defaults to the engine's own choice (the smallest step that saturates the GPU) instead of 16.
- `clipnet_tf attribute -a profile` now attributes the profile summary `sum(c * softmax(c))` with tangermeme's (>= 1.5) softmax and midpoint product rules, instead of the elementwise rescale rule over the whole product (bpnet-lite's `ProfileWrapper` with `_ProfileLogitScaling` registered as `_nonlinear`). The rescale rule does not fit a softmax, which couples every position. Profile attributions therefore change; `--profile_rule rescale` reproduces the old ones.
- `clipnet_tf` and the `clipnet_tf_fast` engines now disable TF32 by default, so float32 results on Ampere and newer GPUs change slightly. PyTorch runs cuDNN convolutions in TF32 by default on Ampere and newer GPUs, which adds error to these engines. With the real CLIPNET ensemble, the error against float64 was ~0.2–0.4% (up to ~2%) for DeepLIFT/SHAP and PISA, and ~1.4% (up to 2.5%) for ISM. With random weights it was 30–50% for ISM, and batch-size dependent. Turning TF32 off cost no measurable speed on an RTX 3080. Server GPUs may gain ~1.3–1.6x from `--tf32` (an estimate; see `docs/clipnet-tf.md`). The engines restore the previous global setting on return.

### Changed
- `uv.lock`: tangermeme 1.4.0 -> 1.6.0.
- `[tool.uv] exclude-dependencies = ["macs3"]`: bpnet-lite declares `macs3`, but only its `bpnet` pipeline command uses it (it runs the `macs3` executable to call peaks). Neither bpnet-lite's library nor this package imports it. `macs3` ships no wheels, and its source build fails on Python 3.11 (newest Cython against an older numpy), so uv installs (the lockfile, development, CI) now leave it and its dependencies (`cykhash`, `hmmlearn`) out, and every locked package installs from a wheel. `pip install personalbpnet` still pulls it in through bpnet-lite.

### Fixed
- `clipnet_tf_fast.pisa` (and `saturation_mutagenesis(compile=True)`) could fail on CUDA after repeated calls that changed the dtype or the TF32 setting. cuDNN returns some convolution-transpose outputs in a layout that depends on the algorithm it picks, so the `torch.compile`-fused steps recompiled until they hit Dynamo's default recompile limit of 8. The outputs are now made contiguous, and both engines raise the limit as `deep_lift_shap` does.
- `CLIPNET_TF.__init__` declared `sconv` as `Conv1d(32, 128, 4)` while the TF weights (and the rest of the network) are `Conv1d(64, 64, 4)`. A model built directly, rather than through `from_tf` (which overwrote the tensor), could not run a forward pass or load a state dict saved from a `from_tf` model.

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
