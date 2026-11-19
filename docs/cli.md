# Command line interface

See [the README](../README.md#install) for installation. Three CLI entry points are installed: `clipnet`, `pausenet`, and `clipnet_tf` (for TensorFlow-ported models — requires the `tf` extra). Each takes a `predict`, `predict_tss`, or `attribute` subcommand (`clipnet_tf` and `clipnet` support all three; `pausenet` supports `predict`/`attribute`, since TSS prediction doesn't apply to scalar phenotypes). `clipnet_tf` also has `ism`.

Every subcommand of every CLI accepts:

| Flag | Description |
| --- | --- |
| `-f, --fa_fname` | Path to an uncompressed FASTA file (required). |
| `-b, --bed_fname` | Path to a BED file of regions to predict/attribute (required). |
| `-o, --out_fname` | Path to the output `.npz` file (required). |
| `-m, --model_fname` | Path to a model directory (containing `f1.torch`..`f9.torch`, averaged across all 9) or to a single model file. |
| `-c, --chroms` | Restrict to specific chromosomes. Defaults to all. |
| `-bs, --batch_size` | Batch size, to control VRAM usage. Default 16. |
| `-v, --verbose` | Print progress bars. |

`clipnet` and `pausenet` additionally accept `--in_window`/`--out_window`/`--n_filters`/`--n_layers` on every subcommand, used to reconstruct the model architecture when `-m` points to a bare state dict rather than a full serialized module — you shouldn't need these unless using a custom (non-default-shaped) model.

Get the full flag list for any subcommand with `-h`, e.g. `clipnet predict -h`.

## `clipnet`

```sh
clipnet predict -f genome.fa -b regions.bed -o out.npz -m model_dir/
clipnet predict_tss -f genome.fa -b regions.bed -o out.npz -m model_dir/
clipnet attribute -f genome.fa -b regions.bed -o out.npz -m model_dir/ -a counts
```

- `predict`: writes the ensembled `(N, 2, out_window)` profile track to `out.npz`. Pass `-s/--signal_fname plus.bw minus.bw` to also benchmark against experimental bigWigs (writes Pearson/Spearman/JSD to `*_metrics.npz`). `-r/--restrict_out` crops the output to a smaller centered window after prediction.
- `predict_tss`: like `predict`, but applies aggressive jittering (`-j/--max_jitter`, default 500 bp) before predicting, then reports TSS-position Pearson correlation per strand when `--signal_fname` is given.
- `attribute`: computes DeepLIFT/SHAP attributions. `-a/--attribute_type` is `counts` (default) or `profile`. `-y/--hypothetical` and `-s/--save_ohe` are for producing [tfmodisco-lite](https://github.com/jmschrei/tfmodisco-lite)-compatible output. `-n/--n_shuffles` controls the number of dinucleotide shuffles used as a reference (default 20).

Ensembling averages raw profile logits and log-counts across replicates *before* applying softmax/exp, then rescales — not the other way around (see `personal_bpnet.utils` / `cli._common.average_profile_and_counts` if you're adapting this pattern elsewhere).

## `pausenet`

```sh
pausenet predict -f genome.fa -b regions.bed -o out.npz -m model.torch
pausenet attribute -f genome.fa -b regions.bed -o out.npz -m model.torch
```

Same shape as `clipnet`, but for [PauseNet](pausenet.md)'s single-scalar-per-locus output. `predict` assumes the BED file's 6th column (if present, i.e. a BED6+ file) is the target phenotype, and reports Pearson/Spearman against it.

## `clipnet_tf`

```sh
clipnet_tf predict -f genome.fa -b regions.bed -o out.npz -m model_dir/
clipnet_tf attribute -f genome.fa -b regions.bed -o out.npz -m model_dir/
clipnet_tf ism -f genome.fa -b regions.bed -o out.npz -m model_dir/
```

For [TensorFlow-ported CLIPNET models](clipnet-tf.md) — model directories are expected to contain `fold_1.h5`..`fold_9.h5`. Input/output windows are fixed at 1000/500 bp (the original architecture's shape), so there are no `--in_window`/`--n_filters`/etc. flags. Pass `--counts_head_only` for counts-only ("PauseNet"-style) TF models.

`clipnet_tf attribute` runs the CLIPNET_TF-specific DeepLIFT/SHAP engine ([details](clipnet-tf.md#fast-deepliftshap-and-ism)), which computes the same attributions as tangermeme's `deep_lift_shap` (same references for a given `-r/--random_state`) about 3x faster on a GPU. Two differences from the other CLIs:

- `-bs/--batch_size` is the number of *sequences* per step, each with all of its `-n` references (tangermeme's `batch_size` counts sequence–reference pairs). By default the engine picks the smallest step that saturates the GPU (16 on a TITAN X, scaled by SM count on larger GPUs) if it fits in half the free memory. Memory grows ~70 MiB per sequence at 20 shuffles. A step that runs out of memory is retried at half the size.
- `-a profile` uses tangermeme's (≥ 1.5) softmax and midpoint product rules for the profile summary `sum(c * softmax(c))`. Up to version 0.7.0 it applied the elementwise rescale rule to the whole product, which does not fit a softmax (it couples every position, so the attributions don't exactly sum to the change in prediction). Pass `--profile_rule rescale` to reproduce those older attributions.

`clipnet_tf ism` runs in silico saturation mutagenesis with the incremental (fastISM) engine. It returns the same predictions as tangermeme's `saturation_mutagenesis`, 2–2.3x faster on a GPU. Predictions are averaged across the ensemble.

- `-a/--output`: `counts` (default) or `profile`. Each saves tangermeme-style ISM attributions of shape `(n, 4, end - start)`: each substitution's change in prediction, centered across the four bases at each position and averaged over the output. These are multiplied by the observed sequence unless `-y/--hypothetical` is given. `both` needs `--raw_outputs`.
- `--raw_outputs` saves the predictions instead: `y0` (reference) and `y_hat` (`(n, 4, end - start, n_outputs)`, every substitution), or `y0_profile`/`y0_counts`/`y_hat_profile`/`y_hat_counts` for `-a both`. Profile predictions for a full window take ~16 MB per region.
- `--start`/`--end` restrict the substituted positions within the 1000 bp window (default: all of them).
- `-bs/--batch_size` is the number of edited sequences per step (default 1024), and `-s/--save_ohe` saves the one-hot sequences.
