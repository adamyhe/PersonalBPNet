# PersonalBPNet

A PyTorch library of [BPNet](https://github.com/jmschrei/bpnet-lite)-family models for predicting base-resolution genomic signal (e.g., PRO-cap read coverage) from DNA sequence, plus CLI tools for prediction and attribution.

- **PersonalBPNet** — `bpnetlite.bpnet.BPNet`, with a `fit()` that validates against a PyTorch `DataLoader` instead of holding the whole validation set in memory.
- **CLIPNET** — `PersonalBPNet` plus batch normalization; PyTorch port of the original TensorFlow [CLIPNET](https://github.com/Danko-Lab/clipnet).
- **ProCapNet** — `BPNet` with a masked profile loss, improving attributions (from the [ProCapNet paper](https://www.biorxiv.org/content/10.1101/2024.05.28.596138v2)).
- **PauseNet** — transfer-learns a base-resolution model to predict a single scalar phenotype per locus.

## Install

With [uv](https://docs.astral.sh/uv/):

```sh
uv add personalbpnet
```

Or with pip:

```sh
pip install personalbpnet
```

To load TensorFlow-trained weights (requires `h5py`), install with the `tf` extra: `uv add "personalbpnet[tf]"` or `pip install "personalbpnet[tf]"`.

Installing from source, for local development, is covered in [CONTRIBUTING.md](CONTRIBUTING.md).

## Documentation

| | |
| --- | --- |
| [docs/clipnet.md](docs/clipnet.md) | `PersonalBPNet` and `CLIPNET`, incl. pretrained weights |
| [docs/clipnet-tf.md](docs/clipnet-tf.md) | Porting the original TensorFlow CLIPNET weights |
| [docs/procapnet.md](docs/procapnet.md) | `ProCapNet` |
| [docs/pausenet.md](docs/pausenet.md) | `PauseNet` |
| [docs/cli.md](docs/cli.md) | `clipnet`, `pausenet`, and `clipnet_tf` command line tools |
| [CONTRIBUTING.md](CONTRIBUTING.md) | Installing from source and local development |

This package is under active development; APIs may change between releases. See [CHANGELOG.md](CHANGELOG.md).

## Citing

If you use `PersonalBPNet`/`CLIPNET`/`ProCapNet`, or the underlying `BPNet` architecture (via the `bpnet-lite` dependency), please cite the corresponding paper (see [CITATION.cff](CITATION.cff)):

- Avsec et al. (2021). [Base-resolution models of transcription-factor binding reveal soft motif syntax](https://doi.org/10.1038/s41588-021-00782-6). *Nature Genetics* 53:354–366. — BPNet
- He & Danko (2024). [Dissection of core promoter syntax through single nucleotide resolution modeling of transcription initiation](https://www.biorxiv.org/content/10.1101/2024.03.13.583868). *bioRxiv*. — CLIPNET
- Cochran et al. (2024). [Dissecting the cis-regulatory syntax of transcription initiation with deep learning](https://www.biorxiv.org/content/10.1101/2024.05.28.596138). *bioRxiv*. — ProCapNet

## License

[MIT](LICENSE)
