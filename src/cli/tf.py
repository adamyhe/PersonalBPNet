#!/usr/bin/env python
# CLIPNET_TF CLI
# Author: Adam He <adamyhe@gmail.com>

"""
Wrapper script to calculate attributions and predictions for CLIPNET TensorFlow models
"""

import random
from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from bpnetlite.bpnet import CountWrapper, ProfileWrapper, _ProfileLogitScaling
from bpnetlite.performance import pearson_corr, spearman_corr
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LinearRegression
from tangermeme.deep_lift_shap import _nonlinear, deep_lift_shap
from tangermeme.io import extract_loci
from tangermeme.predict import predict

from personal_bpnet.clipnet_tensorflow import CLIPNET_TF, TwoHotToOneHot

from ._common import (
    add_attribute_args,
    build_parent_parser,
    load_ensemble,
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


def cli():
    parser_parent = build_parent_parser()
    parser_parent.add_argument(
        "--counts_head_only",
        action="store_true",
        help="Use this flag to load a model with only the counts head. This is mostly "
        "used for 'PauseNet' models, which trim the profile head and fine-tune the just"
        "the counts head.",
    )

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
        default=250,
        help="Maximum number of bp to jitter. Default is 500.",
    )

    # ATTRIBUTE PARAMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    parser_attribute = subparsers.add_parser(
        "attribute",
        help="Calculate attributions for a given set of regions.",
        parents=[parser_parent],
    )
    add_attribute_args(parser_attribute)
    args = parser.parse_args()

    # MAIN CODE ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    model_paths = resolve_model_paths(args.model_fname, pattern="fold_{i}.h5")
    set_cpu_threads()
    device = resolve_device()

    def _load_tf_model(f):
        return TwoHotToOneHot(
            CLIPNET_TF.from_tf(f, counts_head_only=args.counts_head_only)
        )

    if args.cmd == "predict":
        # Load data
        data = extract_loci(
            loci=args.bed_fname,
            sequences=args.fa_fname,
            signals=args.signal_fname,
            chroms=args.chroms,
            in_window=1000,
            out_window=500,
            verbose=args.verbose,
            ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        )
        if args.signal_fname is not None:
            X, signals = data
        else:
            X = data

        counts = []
        profiles = []
        for model in load_ensemble(model_paths, _load_tf_model):
            pred = predict(
                model, X, batch_size=args.batch_size, verbose=args.verbose, device=device
            )
            if args.counts_head_only:
                counts.append(pred)
            else:
                profiles.append(pred[0])
                counts.append(pred[1])

        # Average predictions
        count = torch.mean(torch.stack(counts), dim=0)
        if args.counts_head_only:
            # Save predictions
            np.savez_compressed(args.out_fname, count.numpy())
        else:
            profile = torch.mean(torch.stack(profiles), dim=0)
            # Rescale predictions
            track = count * profile / (profile.sum(dim=-1, keepdim=True) + 1e-3)
            # Save predictions
            if args.signal_fname is not None:
                signals_flattened = torch.cat(
                    [signals[:, 0, :], torch.abs(signals[:, 1, :])], dim=-1
                )
                np.savez_compressed(
                    args.out_fname, pred=track.numpy(), obs=signals_flattened.numpy()
                )
            else:
                np.savez_compressed(args.out_fname, track.numpy())

        # Calculate metrics
        if args.signal_fname is not None:
            pred_log_counts = torch.log10(track.sum(dim=-1) + 1e-3)

            profile_pearson = pearson_corr(track, signals_flattened)
            profile_jsd = jensenshannon(
                track.numpy(), signals_flattened.numpy(), axis=1
            )
            counts_pearson = pearson_corr(
                pred_log_counts, torch.log10(signals_flattened.sum(dim=-1) + 1e-3)
            )
            counts_linear_pearson = pearson_corr(
                track.sum(dim=-1), signals_flattened.sum(dim=-1)
            )
            counts_spearman = spearman_corr(
                pred_log_counts, torch.log10(signals_flattened.sum(dim=-1) + 1e-3)
            )
            lm = LinearRegression(fit_intercept=True).fit(
                torch.log10(signals_flattened.sum(dim=-1).reshape(-1, 1) + 1e-3),
                pred_log_counts.reshape(-1, 1),
            )

            print(
                f"Mean profile Pearson: {np.nanmean(profile_pearson)} "
                f"+/- {np.nanstd(profile_pearson, ddof=1)}"
            )
            print(f"Median profile Pearson: {np.nanmedian(profile_pearson)}")
            print(f"Mean profile JSD: {np.nanmean(profile_jsd)}")
            print(f"Median profile JSD: {np.nanmedian(profile_jsd)}")
            print(f"Count Pearson: {counts_pearson}")
            print(f"Count linear Pearson: {counts_linear_pearson}")
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

        elif args.counts_head_only:
            bed = pd.read_csv(args.bed_fname, sep="\t", header=None)
            bed_test = bed[bed.iloc[:, 0].isin(args.chroms)]
            expt = torch.from_numpy(bed_test.iloc[:, 4].to_numpy())
            pred = count.squeeze()

            counts_pearson = pearson_corr(torch.log1p(pred), torch.log1p(expt))
            counts_spearman = spearman_corr(torch.log1p(pred), torch.log1p(expt))
            lm = LinearRegression(fit_intercept=True).fit(
                torch.log1p(expt).reshape(-1, 1), torch.log1p(pred).reshape(-1, 1)
            )

            print(f"Pearson: {counts_pearson.item()}")
            print(f"Spearman: {counts_spearman.item()}")
            print(f"Slope: {lm.coef_[0][0]}")
            print(f"Intercept: {lm.intercept_[0]}")

            np.savez_compressed(
                args.out_fname.replace(".npz", "_metrics.npz"),
                pearson=counts_pearson,
                spearman=counts_spearman,
                slope=lm.coef_[0][0],
                intercept=lm.intercept_[0],
                expt=expt,
                pred=pred,
            )

    elif args.cmd == "predict_tss":
        if args.counts_head_only:
            raise TypeError("Counts head only models do not support TSS prediction.")
        # Load data
        loci = extract_loci(
            loci=args.bed_fname,
            sequences=args.fa_fname,
            signals=args.signal_fname,
            chroms=args.chroms,
            in_window=1000,
            out_window=500,
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
            seqs_jitter.append(seqs[i, :, j : j + 1000])
            signals_jitter.append(signals[i, :, j : j + 500])

        X = torch.stack(seqs_jitter)
        signals = torch.abs(torch.stack(signals_jitter))

        # Calculate predictions
        profiles = [
            predict(
                model, X, batch_size=args.batch_size, verbose=args.verbose, device=device
            )[0]
            for model in load_ensemble(model_paths, _load_tf_model)
        ]

        # Average predictions
        profile = torch.mean(torch.stack(profiles), dim=0)
        profile = torch.stack([profile[:, :500], profile[:, 500:]], dim=1)

        # Calculate metrics
        if args.signal_fname is not None:
            pred_tss = torch.argmax(profile, dim=-1).to(torch.float)
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
            in_window=1000,
            verbose=args.verbose,
            ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        ).to(torch.float)
        # Save OHE
        if args.save_ohe is not None:
            np.savez_compressed(args.save_ohe, X.to(int).numpy())

        attributions = []
        for model in load_ensemble(model_paths, _load_tf_model):
            additional_nonlinear_ops = None
            # Wrap models depending on args.attribute_type
            if args.counts_head_only:
                pass
            elif args.attribute_type == "counts":
                model = CountWrapper(model)
            elif args.attribute_type == "profile":
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
