# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Installation

```sh
pip install -e /path/to/PersonalBPNet          # core
pip install -e "/path/to/PersonalBPNet[tf]"    # with h5py for TF weight loading
```

Core dependencies: `bpnet-lite>=1.0.0`, `tangermeme>=1.0.0`, `pyfastx`. The `[tf]` extra installs `h5py`, required only for `CLIPNET_TF.from_tf()`.

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

## Architecture Overview

This is a PyTorch genomics deep learning library extending [bpnetlite](https://github.com/jmschrei/bpnet-lite). All models predict base-resolution genomic signal (e.g., PRO-cap read coverage) from DNA sequence.

### Model classes

**`personal_bpnet/clipnet_pytorch.py`** — `CLIPNET`: BPNet architecture with added batch normalization after each conv and linear layer. Default config: 512 filters, 8 dilated residual layers, input 2114 bp → output 1000 bp (`trimming=(2114-1000)//2`). Dual-head: profile head (stranded log-softmax) + counts head (scalar log-counts).
- `PauseNet`: Transfer learning wrapper around `CLIPNET` (or any BPNet-like model) for fine-tuning to a single scalar phenotype per locus. Replaces the counts head linear+BN layers and wraps the base model with `bpnetlite.bpnet.CountWrapper`.

**`personal_bpnet/personal_bpnet.py`** — `PersonalBPNet`: Direct port of `bpnetlite.bpnet.BPNet` with a rewritten `fit()` method that accepts a PyTorch `DataLoader` for validation (instead of loading the whole validation set into memory). Checkpoints save optimizer state dict + epoch number to support training resumption.

**`personal_bpnet/procapnet.py`** — `ProCapNet`: Subclass of `bpnetlite.bpnet.BPNet` with per-position masking in the profile loss, which improves model attributions. Forward pass is identical to BPNet, so weights are interchangeable with plain `BPNet` checkpoints. Differences from BPNet defaults: adds masked profile loss, `count_loss_weight=100` (vs bpnetlite's `alpha=1`), and `n_filters=512`. The `y_has_mask` param (default `True`): training `y` must have shape `(batch, n_outputs+1, out_len)` where the final channel is a boolean mask; `y_valid` must have shape `(batch, n_outputs, out_len)` with no mask channel.

**`personal_bpnet/clipnet_tensorflow.py`** — `CLIPNET_TF`: Faithfully ports the original TensorFlow `rnn_v10` CLIPNET architecture to PyTorch using `from_tf(filename)`. Different architecture than the PyTorch CLIPNET (uses MaxPool, ELU, 1000 bp input → 500 bp output, two-hot encoded inputs). `TwoHotToOneHot` wrapper multiplies inputs by 2 for compatibility with one-hot pipelines.

### Loss functions (`personal_bpnet/losses.py`)

`_mixture_loss_masked` and `MNLLLoss_masked`: Modified versions of bpnetlite's `_mixture_loss` that support masking specific positions out of the MNLL profile loss. Used by `ProCapNet`. Unmasked positions use `MNLLLoss` from bpnetlite; count loss uses `log1pMSELoss`.

### Training data format

DataLoader batches are expected as tuples:
- Without controls: `(X, y, labels)` — 3-element
- With controls: `(X, X_ctl, y, labels)` — 4-element

`X` shape: `(batch, 4, seq_len)` — (half) two-hot or one-hot encoded sequence
`y` shape: `(batch, n_outputs[+1], out_len)` — signal tracks (absolute value taken internally); when `y_has_mask=True` in `ProCapNet`, the final channel is a boolean mask track that is stripped before computing the loss
`labels`: boolean mask selecting which examples to include in profile loss

### Checkpoint format

Best model: `{name}.torch` — state dict only
Checkpoint: `{name}.checkpoint.torch` — dict with `epoch`, `early_stop_count`, `optimizer_state_dict`
Final model: `{name}.final.torch` — full serialized module

### File status notes

- `personal_bpnet/_DEPRECATED_bnbpnet.py`: Deprecated, kept for backwards compat. The current version is `personal_bpnet/clipnet_pytorch.py`.
- `personal_bpnet/procapnet_orig.py`: Verbatim copy of the original ProCapNet implementation, included for full reproducibility of the original project's results. Do not modify.
- `personal_bpnet/_motif_call_old/`: Old motif calling scripts, unmaintained.

### Version

Version is defined solely in `pyproject.toml`. `personal_bpnet/__init__.py` exposes `__version__` via `importlib.metadata` — no changes needed there when bumping versions.
