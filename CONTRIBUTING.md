# Contributing

## Installing from source

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

## Running tests

```sh
uv run pytest
```

CI (`.github/workflows/ci.yml`) runs this suite on every push and pull request, across every Python version listed in `pyproject.toml`'s classifiers. If you change the minimum/maximum supported Python version, verify the suite still passes under it first, e.g. `uv run --python 3.13 pytest`.

## Releasing

Version is defined solely in `pyproject.toml`. A release is a dated entry at the top of `CHANGELOG.md` (`## [x.y.z] - YYYY-MM-DD`) plus a version bump, tagged and published as a GitHub Release — `.github/workflows/publish.yml` then builds and publishes to PyPI automatically (gated on the full CI matrix passing).
