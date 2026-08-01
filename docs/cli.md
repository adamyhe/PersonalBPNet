# Command line interface

See [the README](../README.md#install) for installation. Three CLI entry points are installed: `clipnet`, `pausenet`, and `clipnet_tf` (for TensorFlow-ported models — requires the `tf` extra). Each takes a `predict`, `predict_tss`, or `attribute` subcommand (`clipnet_tf` and `clipnet` support all three; `pausenet` supports `predict`/`attribute`, since TSS prediction doesn't apply to scalar phenotypes).

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
```

For [TensorFlow-ported CLIPNET models](clipnet-tf.md) — model directories are expected to contain `fold_1.h5`..`fold_9.h5`. Input/output windows are fixed at 1000/500 bp (the original architecture's shape), so there are no `--in_window`/`--n_filters`/etc. flags. Pass `--counts_head_only` for counts-only ("PauseNet"-style) TF models.
