# _common.py
# Author: Adam He <adamyhe@gmail.com>

"""
Argparse and ensemble-loading helpers shared by the clipnet, pausenet, and
clipnet_tf CLIs.
"""

import argparse
import os

import torch


def build_parent_parser(bed_fname_help=None):
    """The -f/-b/-o/-m/-c/-bs/-v arguments common to every subcommand of every CLI
    in this project. Callers may add further arguments to the returned parser
    before passing it as a `parents=[...]` entry to `subparsers.add_parser`."""

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "-f",
        "--fa_fname",
        type=str,
        required=True,
        help="Path to uncompressed fasta file.",
    )
    parser.add_argument(
        "-b",
        "--bed_fname",
        type=str,
        required=True,
        help=bed_fname_help
        or "Path to bed file of regions to calculate predictions/attributions for.",
    )
    parser.add_argument(
        "-o", "--out_fname", type=str, required=True, help="Path to output npz file"
    )
    parser.add_argument(
        "-m",
        "--model_fname",
        type=str,
        required=True,
        help="Path to model directory or to specific model file to predict/attribute. "
        "If a directory, loads and calculates average predictions/attributions across "
        "all models in directory. If a specific model file, will only predict/attribute "
        "that model. ",
    )
    parser.add_argument(
        "-c",
        "--chroms",
        type=str,
        nargs="+",
        default=None,
        help="Chromosomes to calculate attributions for. Defaults to all chromosomes.",
    )
    parser.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=16,
        help="Batch size to control VRAM usage. Defaults to 16.",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true", help="Whether to print progress bars."
    )
    return parser


def add_shape_args(parser):
    """The --in_window/--out_window/--n_filters/--n_layers arguments used to
    reconstruct a model from a bare state dict, shared across clipnet/pausenet
    predict/predict_tss/attribute subcommands."""

    parser.add_argument(
        "--in_window",
        type=int,
        default=2114,
        help="Used to specify model input size. "
        "Should not be needed unless using a custom model.",
    )
    parser.add_argument(
        "--out_window",
        type=int,
        default=1000,
        help="Used to specify model output size. "
        "Should not be needed unless using a custom model.",
    )
    parser.add_argument(
        "--n_filters",
        type=int,
        default=512,
        help="Used to specify model convolutions. "
        "Should not be needed unless using a custom model.",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=8,
        help="Used to specify model layers. "
        "Should not be needed unless using a custom model.",
    )


def add_attribute_args(parser):
    """The -a/-s/-y/-n/-r arguments shared by every CLI's attribute subcommand."""

    parser.add_argument(
        "-a",
        "--attribute_type",
        type=str,
        default="counts",
        choices={"counts", "profile"},
        help="The type of attribution to calculate.",
    )
    parser.add_argument(
        "-s",
        "--save_ohe",
        type=str,
        default=None,
        help="Where to save OHE of sequences. Defaults to not saving. "
        "Set this & hypothetical if you intend to use these attributions for "
        "tfmodisco-lite.",
    )
    parser.add_argument(
        "-y",
        "--hypothetical",
        action="store_true",
        help="Whether to use hypothetical attributions. Defaults to False. "
        "Set this & save_ohe if you intend to use these attributions for "
        "tfmodisco-lite.",
    )
    parser.add_argument(
        "-n",
        "--n_shuffles",
        type=int,
        default=20,
        help="Number of dinucleotide shuffles for DeepLIFT/SHAP. Defaults to 20.",
    )
    parser.add_argument(
        "-r",
        "--random_state",
        type=int,
        default=47,
        help="Random seed. Defaults to 47.",
    )


def resolve_model_paths(model_fname, pattern="f{i}.torch", n=9):
    """If `model_fname` is a directory, returns the `n` replicate paths matching
    `pattern` (formatted with i=1..n) inside it; otherwise returns `[model_fname]`
    unchanged."""

    if os.path.isdir(model_fname):
        return [os.path.join(model_fname, pattern.format(i=i)) for i in range(1, n + 1)]
    return [model_fname]


def set_cpu_threads():
    """Use all available CPU cores for inference when no GPU is present, capped by
    SLURM's allocation when running under a SLURM job."""

    if not torch.cuda.is_available():
        if "SLURM_CPUS_PER_TASK" in os.environ:
            n = min(int(os.environ["SLURM_CPUS_PER_TASK"]), os.cpu_count())
        else:
            n = os.cpu_count()
        torch.set_num_threads(n)
        torch.set_num_interop_threads(n)


def resolve_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_torch_model(path, build_model_fn):
    """Loads a `.torch` file that is either a full serialized module or a bare
    state dict; in the latter case, builds a fresh model via `build_model_fn` and
    loads the state dict into it."""

    checkpoint = torch.load(path)
    if isinstance(checkpoint, torch.nn.Module):
        return checkpoint
    model = build_model_fn()
    model.load_state_dict(checkpoint)
    return model


def load_ensemble(model_paths, load_fn):
    """Yields one model per path in `model_paths`, loaded via `load_fn(path)`.
    Frees each replicate before loading the next to avoid accumulating VRAM/RAM
    across the ensemble; the caller should fully consume each yielded model
    (e.g. run prediction/attribution on it) before continuing the loop."""

    for path in model_paths:
        model = load_fn(path)
        yield model
        del model
        torch.cuda.empty_cache()


def average_profile_and_counts(predictions):
    """Ensembles (profile_logits, log_counts) predictions from multiple replicates.

    Averages the raw profile logits and log-counts across replicates *before*
    applying the softmax/exp nonlinearity, then rescales. Averaging after the
    nonlinearity instead (i.e. averaging already-softmaxed, already-rescaled
    per-replicate tracks) biases the ensembled profile toward uniformity whenever
    replicates disagree, which understates real signal.

    Parameters
    ----------
    predictions: list of (profile_logits, log_counts) tuples, one per replicate, as
        returned by tangermeme.predict.predict.

    Returns
    -------
    track: torch.Tensor, shape=profile_logits.shape
        The ensembled, rescaled profile track.
    """

    profiles, counts = zip(*predictions)
    shape = profiles[0].shape

    mean_profile = torch.stack(
        [profile.reshape(profile.shape[0], -1) for profile in profiles]
    ).mean(dim=0)
    mean_count = torch.stack(list(counts)).mean(dim=0)

    track = torch.nn.functional.softmax(mean_profile, dim=-1) * (
        torch.exp(mean_count) - 1
    )
    return track.reshape(*shape)
