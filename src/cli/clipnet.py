#!/usr/bin/env python
# CLIPNET CLI
# Author: Adam He <adamyhe@gmail.com>

"""
Wrapper script to calculate attributions and predictions for CLIPNET models
"""

import random
import warnings

import numpy as np
import torch
from argparse import ArgumentParser
from bpnetlite.bpnet import CountWrapper, ProfileWrapper, _ProfileLogitScaling
from bpnetlite.performance import pearson_corr, spearman_corr
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LinearRegression
from tangermeme.deep_lift_shap import _nonlinear, deep_lift_shap
from tangermeme.io import extract_loci
from tangermeme.predict import predict

from personal_bpnet.clipnet_pytorch import CLIPNET

from ._common import (
    add_attribute_args,
    add_shape_args,
    average_profile_and_counts,
    build_parent_parser,
    load_ensemble,
    load_torch_model,
    resolve_device,
    resolve_model_paths,
    set_cpu_threads,
)

_help = """
The following commands are available:
    predict         Calculate predictions for a CLIPNET model
    predict_tss     Calculate TSS predictions (uses aggressive jittering).
    attribute       Calculate DeepLIFT/SHAP attributions for a CLIPNET model
Planned but not implemented commands:
    vep             Calculate variant effect prediction using a CLIPNET model
"""


def _build_clipnet(args):
    return CLIPNET(
        n_filters=args.n_filters,
        n_outputs=2,
        n_control_tracks=0,
        n_layers=args.n_layers,
        trimming=(args.in_window - args.out_window) // 2,
    )


def cli():
    parser_parent = build_parent_parser()

    parser = ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(
        help="The following commands are available:", required=True, dest="cmd"
    )

    # PREDICT PARAMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    parser_predict = subparsers.add_parser(
        "predict",
        help="Calculate predictions for a given set of regions.",
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
    add_shape_args(parser_predict)

    # PREDICT_TSS PARAMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    parser_predict_tss = subparsers.add_parser(
        "predict_tss",
        help="Calculate TSS predictions for a given set of jittered regions.",
        parents=[parser_parent],
    )
    parser_predict_tss.add_argument(
        "-s",
        "--signal_fname",
        type=str,
        nargs=2,
        default=None,
        help="Signal files containing experimental data to benchmark model "
        "predictions against. Expected order is [plus_bigWig, minus_bigWig]. "
        "If not provided, will not calculate performance metrics.",
    )
    parser_predict_tss.add_argument(
        "-j",
        "--max_jitter",
        type=int,
        default=500,
        help="Maximum number of bp to jitter. Default is 500.",
    )
    add_shape_args(parser_predict_tss)

    # ATTRIBUTE PARAMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    parser_attribute = subparsers.add_parser(
        "attribute",
        help="Calculate attributions for a given set of regions.",
        parents=[parser_parent],
    )
    add_attribute_args(parser_attribute)
    add_shape_args(parser_attribute)
    args = parser.parse_args()

    # MAIN CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    model_paths = resolve_model_paths(args.model_fname)
    set_cpu_threads()
    device = resolve_device()

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

        predictions = [
            predict(
                model,
                X,
                batch_size=args.batch_size,
                verbose=args.verbose,
                device=device,
            )
            for model in load_ensemble(
                model_paths, lambda f: load_torch_model(f, lambda: _build_clipnet(args))
            )
        ]

        # Ensemble and rescale predictions
        track = average_profile_and_counts(predictions)

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
            lm = LinearRegression(fit_intercept=True).fit(
                pred_log_counts.reshape(-1, 1),
                torch.log1p(signals_flattened.sum(dim=-1).reshape(-1, 1)),
            )

            print(
                f"Mean profile Pearson: {np.nanmean(profile_pearson)} "
                f"+/- {np.nanstd(profile_pearson, ddof=1)}"
            )
            print(f"Median profile Pearson: {np.nanmedian(profile_pearson)}")
            print(f"Mean profile JSD: {np.nanmean(profile_jsd)}")
            print(f"Median profile JSD: {np.nanmedian(profile_jsd)}")
            print(f"Count Pearson: {counts_pearson}")
            print(f"Count Spearman: {counts_spearman}")
            print(f"Count slope: {lm.coef_[0]}")
            print(f"Count intercept: {lm.intercept_}")

            np.savez_compressed(
                args.out_fname.replace(".npz", "_metrics.npz"),
                profile_pearson=profile_pearson,
                profile_jsd=profile_jsd,
                counts_pearson=counts_pearson,
                counts_spearman=counts_spearman,
            )

    elif args.cmd == "predict_tss":
        # Load data
        loci = extract_loci(
            loci=args.bed_fname,
            sequences=args.fa_fname,
            signals=args.signal_fname,
            chroms=args.chroms,
            in_window=args.in_window,
            out_window=args.out_window,
            max_jitter=args.max_jitter,
            verbose=args.verbose,
        )
        if args.signal_fname is not None:
            seqs, signals = loci
        else:
            seqs = loci
        # Jitter
        seqs_jitter = []
        signals_jitter = []
        for i in range(seqs.shape[0]):
            j = random.randint(0, args.max_jitter * 2 - 1)
            seqs_jitter.append(seqs[i, :, j : j + args.in_window])
            signals_jitter.append(signals[i, :, j : j + args.out_window])

        X = torch.stack(seqs_jitter)
        signals = torch.abs(torch.stack(signals_jitter))

        # Calculate predictions
        predictions = [
            predict(
                model,
                X,
                batch_size=args.batch_size,
                verbose=args.verbose,
                device=device,
            )
            for model in load_ensemble(
                model_paths, lambda f: load_torch_model(f, lambda: _build_clipnet(args))
            )
        ]

        # Ensemble and rescale predictions
        track = average_profile_and_counts(predictions)

        # Calculate metrics
        if args.signal_fname is not None:
            pred_tss = torch.argmax(track, dim=-1).to(torch.float)
            expt_tss = torch.argmax(signals, dim=-1).to(torch.float)
            np.savez_compressed(args.out_fname, pred=pred_tss, expt=expt_tss)

            print(
                f"+ strand TSS Pearson: {pearson_corr(pred_tss[:, 0], expt_tss[:, 0])}"
            )
            print(
                f"- strand TSS Pearson: {pearson_corr(pred_tss[:, 1], expt_tss[:, 1])}"
            )

    elif args.cmd == "attribute":
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
        for model in load_ensemble(
            model_paths, lambda f: load_torch_model(f, lambda: _build_clipnet(args))
        ):
            additional_nonlinear_ops = None
            # Wrap models depending on args.attribute_type
            if args.attribute_type == "counts":
                model = CountWrapper(model)
            else:
                model = ProfileWrapper(model)
                additional_nonlinear_ops = {_ProfileLogitScaling: _nonlinear}

            # Calculate and log attributions
            attributions.append(
                deep_lift_shap(
                    model,
                    X,
                    hypothetical=args.hypothetical,
                    batch_size=args.batch_size,
                    n_shuffles=args.n_shuffles,
                    random_state=args.random_state,
                    verbose=args.verbose,
                    additional_nonlinear_ops=additional_nonlinear_ops,
                    device=device,
                ).numpy()
            )

        # Save
        np.savez_compressed(args.out_fname, np.stack(attributions).mean(axis=0))

    else:
        raise ValueError(_help)


if __name__ == "__main__":
    cli()
