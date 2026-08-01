# PersonalBPNet

A PyTorch library of [BPNet](https://github.com/jmschrei/bpnet-lite)-family models for predicting base-resolution genomic signal (e.g., PRO-cap read coverage) from DNA sequence, plus CLI tools for prediction and attribution.

- **PersonalBPNet** — `bpnetlite.bpnet.BPNet`, with a `fit()` that validates against a PyTorch `DataLoader` instead of holding the whole validation set in memory.
- **CLIPNET** — `PersonalBPNet` plus batch normalization; PyTorch port of the original TensorFlow [CLIPNET](https://github.com/Danko-Lab/clipnet).
- **ProCapNet** — `BPNet` with a masked profile loss, improving attributions (from the [ProCapNet paper](https://www.biorxiv.org/content/10.1101/2024.05.28.596138v2)).
- **PauseNet** — transfer-learns a base-resolution model to predict a single scalar phenotype per locus.

## Install

```sh
pip install git+https://github.com/adamyhe/personalbpnet.git
```

See [docs/installation.md](docs/installation.md) for the `[tf]` extra (TensorFlow weight loading) and the local development setup (this repo uses [uv](https://docs.astral.sh/uv/)).

## Documentation

| | |
| --- | --- |
| [docs/installation.md](docs/installation.md) | Installing as a dependency, or for local development |
| [docs/clipnet.md](docs/clipnet.md) | `PersonalBPNet` and `CLIPNET`, incl. pretrained weights |
| [docs/clipnet-tf.md](docs/clipnet-tf.md) | Porting the original TensorFlow CLIPNET weights |
| [docs/procapnet.md](docs/procapnet.md) | `ProCapNet` |
| [docs/pausenet.md](docs/pausenet.md) | `PauseNet` |
| [docs/cli.md](docs/cli.md) | `clipnet`, `pausenet`, and `clipnet_tf` command line tools |

This package is under active development; APIs may change between releases. See [CHANGELOG.md](CHANGELOG.md).

## License

[MIT](LICENSE)
