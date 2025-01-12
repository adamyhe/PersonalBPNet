#!/usr/bin/env python
# CLIPNET CLI
# Author: Adam He <adamyhe@gmail.com>

"""
Wrapper script to calculate attributions and predictions for CLIPNET models
"""

import argparse
import os
import warnings

import numpy as np
import torch
from bpnetlite.bpnet import CountWrapper, ProfileWrapper, _ProfileLogitScaling
from bpnetlite.performance import pearson_corr, spearman_corr
from scipy.spatial.distance import jensenshannon
from tangermeme.deep_lift_shap import _nonlinear, deep_lift_shap
from tangermeme.io import extract_loci
from tangermeme.predict import predict

from .clipnet_pytorch import CLIPNET, PauseNet

_help = """
The following commands are available:
    predict         Calculate predictions for a CLIPNET model
    attribute       Calculate DeepLIFT/SHAP attributions for a CLIPNET model
Planned but not implemented commands:
    vep             Calculate variant effect prediction using a CLIPNET model
    predict_tss     Calculate TSS predictions (uses aggressive jittering).
    predict_scalar  Calculate predictions for a scalar output model (i.e., PauseNet)
"""


def cli():
    # DUMMY PARSER FOR COMMON PARAMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    parser_parent = argparse.ArgumentParser(add_help=False)
    parser_parent.add_argument(
        "-f",
        "--fa_fname",
        type=str,
        required=True,
        help="Path to uncompressed fasta file.",
    )
    parser_parent.add_argument(
        "-b",
        "--bed_fname",
        type=str,
        required=True,
        help="Path to bed file of peaks to calculate predictions/attributions for.",
    )
    parser_parent.add_argument(
        "-o", "--out_fname", type=str, required=True, help="Path to output npz file"
    )
    parser_parent.add_argument(
        "-md",
        "--model_dir",
        type=str,
        default=None,
        help="Path to model directory. "
        "Loads and calculates average predictions across all models in directory. "
        "Use either model_dir or model_fname",
    )
    parser_parent.add_argument(
        "-mf",
        "--model_fname",
        type=str,
        default=None,
        help="Path to model file. Use either model_dir or model_fname",
    )
    parser_parent.add_argument(
        "-c",
        "--chroms",
        type=str,
        nargs="+",
        default=None,
        help="Chromosomes to calculate attributions for. Defaults to all chromosomes.",
    )
    parser_parent.add_argument(
        "-bs",
        "--batch_size",
        type=int,
        default=16,
        help="Batch size to control VRAM usage. Defaults to 16.",
    )
    parser_parent.add_argument(
        "-v", "--verbose", action="store_true", help="Whether to print progress bars."
    )

    # MAIN PARSER ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(
        help="The following commands are available:", required=True, dest="cmd"
    )

    # PREDICT PARAMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    parser_predict = subparsers.add_parser(
        "predict",
        help="Calculate predictions for a given set of peaks.",
        parents=[parser_parent],
    )
    parser_predict.add_argument(
        "-s",
        "--signal_fname",
        type=str,
        nargs=2,
        default=None,
        help="Signal files containing experimental data to benchmark model "
        "predictions against. Expected order is [plus_bigWig, minus_bigWig]. "
        "If not provided, will not calculate performance metrics.",
    )
    parser_predict.add_argument(
        "-r",
        "--restrict_out",
        type=int,
        default=None,
        help="Restrict output prediction to a specific size. Default is None.",
    )
    parser_predict.add_argument(
        "--in_window",
        type=int,
        default=2114,
        help="Used to specify model input size. "
        "Should not be needed unless using a custom model.",
    )
    parser_predict.add_argument(
        "--out_window",
        type=int,
        default=1000,
        help="Used to specify model output size. "
        "Should not be needed unless using a custom model.",
    )
    parser_predict.add_argument(
        "--n_filters",
        type=int,
        default=512,
        help="Used to specify model convolutions. "
        "Should not be needed unless using a custom model.",
    )
    parser_predict.add_argument(
        "--n_layers",
        type=int,
        default=8,
        help="Used to specify model layers. "
        "Should not be needed unless using a custom model.",
    )

    # ATTRIBUTE PARAMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    parser_attribute = subparsers.add_parser(
        "attribute",
        help="Calculate attributions for a given set of peaks.",
        parents=[parser_parent],
    )
    parser_attribute.add_argument(
        "-m",
        "--mode",
        type=str,
        default="counts",
        choices={"counts", "profile", "scalar"},
        help="The type of attribution to calculate. "
        "Counts and profile are for CLIPNET, scalar is for PauseNet",
    )
    parser_attribute.add_argument(
        "-s",
        "--save_ohe",
        type=str,
        default=None,
        help="Where to save OHE of sequences. Defaults to not saving.",
    )
    parser_attribute.add_argument(
        "-n",
        "--n_shuffles",
        type=int,
        default=20,
        help="Number of dinucleotide shuffles for DeepLIFT/SHAP. Defaults to 20.",
    )
    parser_attribute.add_argument(
        "-r",
        "--random_state",
        type=int,
        default=47,
        help="Random seed. Defaults to 47.",
    )
    parser_attribute.add_argument(
        "--in_window",
        type=int,
        default=2114,
        help="Used to specify model input size. "
        "Should not be needed unless using a custom model.",
    )
    parser_attribute.add_argument(
        "--out_window",
        type=int,
        default=1000,
        help="Used to specify model output size. "
        "Should not be needed unless using a custom model.",
    )
    parser_attribute.add_argument(
        "--n_filters",
        type=int,
        default=512,
        help="Used to specify model convolutions. "
        "Should not be needed unless using a custom model.",
    )
    parser_attribute.add_argument(
        "--n_layers",
        type=int,
        default=8,
        help="Used to specify model layers. "
        "Should not be needed unless using a custom model.",
    )
    args = parser.parse_args()

    # MAIN CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    # Get model paths
    if (args.model_dir is not None and args.model_fname is not None) or (
        args.model_dir is None and args.model_fname is None
    ):
        raise ValueError("Use exactly one of model_dir or model_fname, not both.")
    if args.model_fname is not None:
        model_names = [args.model_fname]
    else:
        model_names = [
            os.path.join(args.model_dir, f"f{i}.torch") for i in range(1, 10)
        ]

    if args.cmd == "predict":
        # Load data
        data = extract_loci(
            loci=args.bed_fname,
            sequences=args.fa_fname,
            signals=args.signal_fname,
            chroms=args.chroms,
            in_window=args.in_window,
            out_window=args.out_window,
            verbose=args.verbose,
            ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        )
        if args.signal_fname is not None:
            X, signals = data
        else:
            X = data

        predictions = []
        for f in model_names:
            # Load model inside of for loop to prevent VRAM leak
            m = torch.load(f)
            # Check if models are state_dict or module
            if isinstance(m, torch.nn.Module):
                model = m
            else:
                model = CLIPNET(
                    n_filters=args.n_filters,
                    n_outputs=2,
                    n_control_tracks=0,
                    n_layers=args.n_layers,
                    trimming=(args.in_window - args.out_window) // 2,
                )
                model.load_state_dict(m)

            # Calculate and log predictions
            predictions.append(
                predict(model, X, batch_size=args.batch_size, verbose=args.verbose)
            )

            # clear VRAM
            del m
            torch.cuda.empty_cache()

        # Rescale predictions
        z = predictions[0][0].shape
        tracks = [
            torch.nn.functional.softmax(profile.reshape(profile.shape[0], -1), dim=-1)
            * (torch.exp(count))
            for profile, count in predictions
        ]
        if len(tracks) > 1:
            track = torch.stack(tracks).mean(dim=0).cpu().reshape(*z)
        else:
            track = tracks[0].cpu().reshape(*z)

        # Save predictions
        np.savez_compressed(args.out_fname, track.numpy())

        if args.restrict_out is not None:
            if args.restrict_out < args.out_window:
                track = track[
                    :,
                    :,
                    (args.out_window - args.restrict_out) // 2 : (
                        args.out_window + args.restrict_out
                    )
                    // 2,
                ]
                signals = signals[
                    :,
                    :,
                    (args.out_window - args.restrict_out) // 2 : (
                        args.out_window + args.restrict_out
                    )
                    // 2,
                ]
            else:
                warnings.warn(
                    f"{args.restrict_out} >= {args.out_window}. Ignoring restrict_out."
                )

        # Calculate metrics
        if args.signal_fname is not None:
            signals_flattened = torch.abs(signals).reshape(signals.shape[0], -1)
            track_flattened = track.reshape(track.shape[0], -1)
            pred_log_counts = torch.log1p(track_flattened.sum(dim=-1))

            profile_pearson = pearson_corr(track_flattened, signals_flattened)
            profile_jsd = jensenshannon(
                track_flattened.numpy(), signals_flattened.numpy(), axis=1
            )
            counts_pearson = pearson_corr(
                pred_log_counts, torch.log1p(signals_flattened.sum(dim=-1))
            )
            counts_spearman = spearman_corr(
                pred_log_counts, torch.log1p(signals_flattened.sum(dim=-1))
            )

            print(
                f"Mean profile Pearson: {np.nanmean(profile_pearson)} "
                "+/- {np.nanstd(profile_pearson, ddof=1)}"
            )
            print(f"Median profile Pearson: {np.nanmedian(profile_pearson)}")
            print(f"Mean profile JSD: {np.nanmean(profile_jsd)}")
            print(f"Median profile JSD: {np.nanmedian(profile_jsd)}")
            print(f"Count Pearson: {counts_pearson}")
            print(f"Count Spearman: {counts_spearman}")

            np.savez_compressed(
                args.out_fname.replace(".npz", "_metrics.npz"),
                profile_pearson=profile_pearson,
                profile_jsd=profile_jsd,
                counts_pearson=counts_pearson,
                counts_spearman=counts_spearman,
            )

    elif args.cmd == "attribute":
        # Disable TF32
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

        # Load data
        X = extract_loci(
            loci=args.bed_fname,
            sequences=args.fa_fname,
            chroms=args.chroms,
            in_window=args.in_window,
            out_window=args.out_window,
            verbose=args.verbose,
            ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        ).to(torch.float)
        if args.save_ohe is not None:
            np.savez_compressed(args.save_ohe, X.to(int).numpy())

        attributions = []
        for f in model_names:
            # Load model inside of for loop to prevent VRAM leak
            m = torch.load(f)
            # Check if models are state_dict or module
            if isinstance(m, torch.nn.Module):
                model = m
            else:
                model = CLIPNET(
                    n_filters=args.n_filters,
                    n_outputs=2,
                    n_control_tracks=0,
                    n_layers=args.n_layers,
                    trimming=(args.in_window - args.out_window) // 2,
                )
                if args.mode == "scalar":
                    model = PauseNet(m)
                model.load_state_dict(m)

            additional_nonlinear_ops = None
            # Wrap models depending on args.mode
            if args.mode == "counts":
                model = CountWrapper(model)
            elif args.mode == "profile":
                model = ProfileWrapper(model)
                additional_nonlinear_ops = {_ProfileLogitScaling: _nonlinear}

            # Calculate and log attributions
            attributions.append(
                deep_lift_shap(
                    model,
                    X,
                    batch_size=args.batch_size,
                    n_shuffles=args.n_shuffles,
                    random_state=args.random_state,
                    verbose=args.verbose,
                    additional_nonlinear_ops=additional_nonlinear_ops,
                ).numpy()
            )

            # clear VRAM
            del m
            torch.cuda.empty_cache()

        # Save
        np.savez_compressed(args.out_fname, np.stack(attributions).mean(axis=0))

    else:
        raise ValueError(_help)


if __name__ == "__main__":
    cli()
