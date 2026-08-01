# Installation

## As a dependency

Install directly from GitHub with pip:

```sh
pip install git+https://github.com/adamyhe/personalbpnet.git
```

To load TensorFlow-trained weights (requires `h5py`), install with the `tf` optional dependency:

```sh
pip install "git+https://github.com/adamyhe/personalbpnet.git[tf]"
```

## Local development

This repo uses [uv](https://docs.astral.sh/uv/) to manage the local development environment. After [installing uv](https://docs.astral.sh/uv/getting-started/installation/):

```sh
git clone https://github.com/adamyhe/personalbpnet.git
cd personalbpnet
uv sync                # create .venv and install core deps from uv.lock
uv sync --extra tf     # also include h5py, for TF weight loading
uv run clipnet -h      # run CLI commands inside the managed environment
uv run python          # or drop into a Python shell with the package importable
```

`uv sync` re-resolves and installs whenever `pyproject.toml` changes; run `uv lock` after adding or updating a dependency to refresh `uv.lock`.

Run the test suite with:

```sh
uv run pytest
```
