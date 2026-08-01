# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Installation

This repo uses [uv](https://docs.astral.sh/uv/) for local development. It uses a `src/` layout: the library lives in `src/personal_bpnet/`, and the CLI entry points live in a separate top-level package, `src/cli/`.

```sh
uv sync                       # create .venv and install core deps from uv.lock
uv sync --extra tf            # also install h5py, for TF weight loading
uv run clipnet -h             # run CLI commands inside the managed environment
uv run python                 # or drop into a Python shell with the package importable
```

Run `uv lock` after changing dependencies in `pyproject.toml` to refresh `uv.lock`. As a plain dependency (not for local dev), the package installs via `uv add personalbpnet` or `pip install personalbpnet` (PyPI) — see `CONTRIBUTING.md` for the from-source workflow above.

Core dependencies: `bpnet-lite>=1.0.0`, `tangermeme>=1.0.0`, `pyfaidx`, `numba`. The `[tf]` extra installs `h5py`, required only for `CLIPNET_TF.from_tf()` (imported lazily inside that method, so the base install never requires `h5py`).

## CLI Commands

```sh
clipnet predict -f genome.fa -b regions.bed -o out.npz -m model_dir/
clipnet predict_tss -f genome.fa -b regions.bed -o out.npz -m model_dir/
clipnet attribute -f genome.fa -b regions.bed -o out.npz -m model_dir/ -a counts
clipnet_tf ...   # same interface, for TF-ported models

pausenet predict -f genome.fa -b regions.bed -o out.npz -m model.torch
pausenet attribute -f genome.fa -b regions.bed -o out.npz -m model.torch
```

Model directories are expected to contain files named `f1.torch` through `f9.torch`. The CLI averages predictions/attributions across all replicates.

### CLI implementation pattern

`src/cli/clipnet.py`, `src/cli/pausenet.py`, and `src/cli/tf.py` each define a single `cli()` function using `argparse` subparsers (`predict`, `predict_tss`, `attribute`). Shared scaffolding lives in `src/cli/_common.py`:
- `build_parent_parser()` / `add_shape_args()` / `add_attribute_args()` — the common `-f/-b/-o/-m/-c/-bs/-v`, `--in_window/--out_window/--n_filters/--n_layers`, and `-a/-s/-y/-n/-r` argparse groups, reused across subcommands and CLIs.
- `resolve_model_paths()` — expands a model directory into its `f1.torch..f9.torch` (or `fold_{i}.h5` for the TF CLI) replicate paths, or passes through a single file.
- `load_ensemble()` / `load_torch_model()` — a generator that loads one replicate at a time (freeing it before the next is loaded, to avoid VRAM buildup) and handles the "bare state dict vs. full serialized module" checkpoint ambiguity.
- `average_profile_and_counts()` — ensembles per-replicate `(profile_logits, log_counts)` predictions by averaging the raw logits and log-counts *before* applying softmax/exp, then rescaling. Averaging after the nonlinearity (i.e. averaging already-softmaxed, already-rescaled per-replicate tracks) biases the ensembled profile toward uniformity whenever replicates disagree — always average before the nonlinearity when ensembling logit-space or log-space model outputs.

Each CLI's subcommand bodies otherwise follow the same shape: `tangermeme.io.extract_loci` pulls sequence (and optionally signal) tensors, `load_ensemble` iterates replicates through `tangermeme.predict.predict`/`tangermeme.deep_lift_shap.deep_lift_shap` (wrapping in `bpnetlite.bpnet.CountWrapper`/`ProfileWrapper` for attribution), and results are averaged and written to `.npz` via `np.savez_compressed` (`predict` also writes a `*_metrics.npz` with Pearson/Spearman/JSD when `--signal_fname` is given). When adding a new subcommand or CLI, extend `_common.py` rather than re-copying this scaffolding.

## Architecture Overview

This is a PyTorch genomics deep learning library extending [bpnetlite](https://github.com/jmschrei/bpnet-lite). All models predict base-resolution genomic signal (e.g., PRO-cap read coverage) from DNA sequence. Model classes are subclasses of bpnetlite's classes wherever possible — override only what actually differs (architecture layers, `forward`, `fit`) rather than copying bpnetlite's implementation. `ProCapNet` (subclassing `bpnetlite.bpnet.BPNet`) is the template this convention follows.

### Model classes

**`src/personal_bpnet/personal_bpnet.py`** — `PersonalBPNet(bpnetlite.bpnet.BPNet)`: identical architecture to `BPNet` (inherits `__init__`/`forward` unchanged, so state_dicts are interchangeable with plain `BPNet` checkpoints), with a rewritten `fit()` that accepts a PyTorch `DataLoader` for validation (instead of loading the whole validation set into memory) and checkpoints the optimizer state dict + epoch number to support training resumption. Defaults differ from `BPNet`: `n_outputs=1`, `n_control_tracks=0`, `trimming` defaults to `2**n_layers` (vs BPNet's own default formula) when not given explicitly.

**`src/personal_bpnet/clipnet_pytorch.py`**
- `CLIPNET(PersonalBPNet)`: adds batch normalization after each conv and linear layer (overrides `__init__`/`forward` only to add the BN layers/calls; inherits `PersonalBPNet.fit()` unchanged). Default config: 512 filters, 8 dilated residual layers, input 2114 bp → output 1000 bp (`trimming=(2114-1000)//2`). Dual-head: profile head (stranded log-softmax) + counts head (scalar log-counts).
- `PauseNet(bpnetlite.bpnet.CountWrapper)`: transfer-learning wrapper around `CLIPNET` (or any BPNet-like model) for fine-tuning to a single scalar phenotype per locus. Replaces the wrapped base model's `linear` (and `cbn`, if present) counts-head layers in place with freshly initialized ones — inspect `self.model` (the wrapped base model), not `self`, when looking for the counts head. The new head's input width defaults to the base model's existing `linear.in_features` unless `n_filters` is given explicitly. `base_trainable=False` freezes the rest of the base model but never the new head.

**`src/personal_bpnet/procapnet.py`** — `ProCapNet(bpnetlite.bpnet.BPNet)`: subclass with per-position masking in the profile loss, which improves model attributions. Forward pass is identical to BPNet, so weights are interchangeable with plain `BPNet` checkpoints. Differences from BPNet defaults: adds masked profile loss, `count_loss_weight=100` (vs bpnetlite's `1`), and `n_filters=512`. The `y_has_mask` param (default `True`): training `y` must have shape `(batch, n_outputs+1, out_len)` where the final channel is a boolean mask; `y_valid` must have shape `(batch, n_outputs, out_len)` with no mask channel.

**`src/personal_bpnet/clipnet_tensorflow.py`** — `CLIPNET_TF`: Faithfully ports the original TensorFlow `rnn_v10` CLIPNET architecture to PyTorch using `from_tf(filename)`. A genuinely different architecture from the PyTorch `CLIPNET` above (uses MaxPool, ELU, 1000 bp input → 500 bp output, two-hot encoded inputs) — not a `BPNet` subclass. `TwoHotToOneHot` wrapper multiplies inputs by 2 for compatibility with one-hot pipelines.

All of the above are re-exported from the top-level `personal_bpnet` package (`from personal_bpnet import CLIPNET, PauseNet, PersonalBPNet, ProCapNet, CLIPNET_TF, TwoHotToOneHot`).

### Loss functions (`src/personal_bpnet/losses.py`)

`_mixture_loss_masked` and `MNLLLoss_masked`: Modified versions of bpnetlite's `_mixture_loss` that support masking specific positions out of the MNLL profile loss. Used by `ProCapNet`. Unmasked positions use `MNLLLoss` from bpnetlite; count loss uses `log1pMSELoss`.

### Training data format

DataLoader batches are expected as tuples:
- Without controls: `(X, y, labels)` — 3-element
- With controls: `(X, X_ctl, y, labels)` — 4-element

`X` shape: `(batch, 4, seq_len)` — (half) two-hot or one-hot encoded sequence
`y` shape: `(batch, n_outputs[+1], out_len)` — signal tracks (absolute value taken internally); when `y_has_mask=True` in `ProCapNet`, the final channel is a boolean mask track that is stripped before computing the loss
`labels`: boolean mask selecting which examples to include in profile loss

### Data utilities (`src/personal_bpnet/utils.py`)

- `twohot_encode`/`reverse_complement_twohot`: encode a DNA string (handling IUPAC heterozygote codes like `M`/`R`/`W`) into the half two-hot `(4, len)` format described above, and return its reverse complement as a new array (does not mutate the input). Both are byte-level lookup-table gathers (`_twohot_gather`, a `numba.njit` kernel, and a plain array-reversal, respectively) rather than per-character Python loops — plain axis reversal already self-complements the `W`/`S` heterozygote codes correctly, so no special-casing is needed.
- `get_twohot_fasta_sequences`: encodes every record in a fasta file in a single batched call to `_twohot_gather` (not a per-sequence loop or process pool — batching made multiprocessing net-negative once encoding itself got fast). Requires every record to be the same length (raises `ValueError` otherwise), since the output must be a single rectangular `(n, 4, len)` array.
- `ChunkedDataset` + `ChunkSampler` + `ChunkedDataLoader`: dataset for training on data split across many small FASTA/`.npz` chunk files (e.g. per-individual or per-region shards) too large to hold in memory at once. Only one chunk is materialized at a time (`current_chunk_index` tracks which); `ChunkSampler` shuffles chunk order and within-chunk order each epoch. Jitter and reverse-complement augmentation are applied per-example in `__getitem__`.
- `ScalarLoader`: dataset for `PauseNet`-style single-scalar-per-locus training. Expects pre-extracted sequences padded by `2*max_jitter` beyond `in_window` and slices the jittered window at `__getitem__` time.

### Testing

`tests/` (pytest) covers model construction/forward shapes, state_dict key parity with bare `bpnetlite` classes, loss functions, and the twohot encoding utilities — all CPU-only, no real data or pretrained weights required. Run with `uv run pytest`. `.github/workflows/ci.yml` runs this suite on push/PR across every Python version listed in `pyproject.toml`'s classifiers (currently 3.9–3.13; re-verify all still pass with `uv run --python X pytest` before adding/dropping a version there). For anything not covered by the suite (actual training runs, real genomic data), verify by running the relevant CLI command or a small script against sample data.

### Docs and publishing

User-facing documentation lives in `docs/` (one file per model, plus `cli.md`), with `README.md` kept as a short pointer/overview (install instructions live directly in the README, since they're the first thing a new user needs) — add new user-facing content to `docs/`, not the README. Installing from source / local development lives in `CONTRIBUTING.md`, not `docs/`. `.github/workflows/publish.yml` builds and publishes to PyPI via `uv build`/`uv publish` (OIDC trusted publishing, no stored token) on GitHub Release; it calls `ci.yml` as a reusable workflow first (`workflow_call`) so a release can't publish without the full test matrix passing.

`CITATION.cff` (validated with `uvx cffconvert --validate -i CITATION.cff`) lists the BPNet, CLIPNET, and ProCapNet papers under `references`, alongside the software's own metadata — GitHub surfaces this automatically as a "Cite this repository" widget. Update it if the version, authors, or cited papers change.

### Checkpoint format

Best model: `{name}.torch` — state dict only
Checkpoint: `{name}.checkpoint.torch` — dict with `epoch`, `early_stop_count`, `optimizer_state_dict`
Final model: `{name}.final.torch` — full serialized module

### File status notes

- `src/personal_bpnet/procapnet_orig.py`: Original ProCapNet implementation (reference).

### Version

Version is defined solely in `pyproject.toml`. `personal_bpnet/__init__.py` exposes `__version__` via `importlib.metadata` — no changes needed there when bumping versions. Version bumps are accompanied by a new dated entry (`## [x.y.z] - YYYY-MM-DD`) at the top of `CHANGELOG.md` summarizing fixes/additions/changes since the prior release.
