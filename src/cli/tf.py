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
from bpnetlite.performance import pearson_corr, spearman_corr
from scipy.spatial.distance import jensenshannon
from sklearn.linear_model import LinearRegression
from tangermeme.io import extract_loci
from tangermeme.predict import predict

from personal_bpnet.clipnet_tensorflow import CLIPNET_TF, TwoHotToOneHot
from personal_bpnet.clipnet_tf_fast import deep_lift_shap as fast_deep_lift_shap
from personal_bpnet.clipnet_tf_fast import saturation_mutagenesis

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
    ism             Calculate in silico saturation mutagenesis for a CLIPNET model
Planned but not implemented commands:
    vep             Calculate variant effect prediction using a CLIPNET model
"""


def _parent_parser():
    """The arguments every clipnet_tf subcommand takes. Built fresh per call:
    subparsers share their parents' argument objects, so `attribute` gets its
    own copy to change the -bs default on."""

    parser_parent = build_parent_parser()
    parser_parent.add_argument(
        "--counts_head_only",
        action="store_true",
        help="Use this flag to load a model with only the counts head. This is mostly "
        "used for 'PauseNet' models, which trim the profile head and fine-tune the just"
        "the counts head.",
    )
    return parser_parent


def cli():
    parser_parent = _parent_parser()

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

    attribute_parent = _parent_parser()
    # The fast DeepLIFT/SHAP engine picks its own step size (the smallest that
    # saturates the GPU, within its memory), so -bs defaults to that here.
    for action in attribute_parent._actions:
        if action.dest == "batch_size":
            action.default = None
            action.help = (
                "Sequences per step, each with all of its -n references. Defaults "
                "to the smallest step that saturates the GPU (16 on a 28-SM TITAN "
                "X, scaled by SM count), if it fits in half the free memory "
                "(~70 MiB a sequence at 20 shuffles); 8 on the CPU."
            )
    parser_attribute = subparsers.add_parser(
        "attribute",
        help="Calculate attributions for a given set of regions.",
        parents=[attribute_parent],
    )
    add_attribute_args(parser_attribute)
    parser_attribute.add_argument(
        "--profile_rule",
        type=str,
        default="product",
        choices=("product", "rescale"),
        help="DeepLIFT rule for the profile summary sum(c * softmax(c)). 'product' "
        "(default) uses tangermeme's (>= 1.5) softmax and midpoint product rules; "
        "'rescale' applies the elementwise rescale rule to the whole product, which "
        "reproduces attributions from personalbpnet <= 0.7.0.",
    )
    # ISM PARAMS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ism_parent = _parent_parser()
    for action in ism_parent._actions:
        if action.dest == "batch_size":
            action.default = 1024
            action.help = (
                "Edited sequences per step, approximately (the edits at a few "
                "consecutive positions of up to batch_size / 24 regions). "
                "Defaults to 1024."
            )
    parser_ism = subparsers.add_parser(
        "ism",
        help="Calculate in silico saturation mutagenesis for a given set of regions.",
        parents=[ism_parent],
    )
    parser_ism.add_argument(
        "-a",
        "--output",
        type=str,
        default="counts",
        choices=("counts", "profile", "both"),
        help="Which predictions to mutagenize. 'counts' (default) and 'profile' "
        "save tangermeme-style ISM attributions (each substitution's change in "
        "prediction, centered across bases and averaged over the output); 'both' "
        "needs --raw_outputs.",
    )
    parser_ism.add_argument(
        "--start",
        type=int,
        default=0,
        help="First position of the 1000 bp window to substitute. Defaults to 0.",
    )
    parser_ism.add_argument(
        "--end",
        type=int,
        default=-1,
        help="Position after the last to substitute (-1: the end of the window). "
        "Defaults to -1.",
    )
    parser_ism.add_argument(
        "--raw_outputs",
        action="store_true",
        help="Save the predictions for the reference (y0) and for every "
        "substitution (y_hat, shape (n, 4, end - start, n_outputs)) instead of "
        "attributions. Profile predictions take ~16 MB per region for a full "
        "window.",
    )
    parser_ism.add_argument(
        "-y",
        "--hypothetical",
        action="store_true",
        help="Save the attribution of every base at every position instead of "
        "only the observed base's.",
    )
    parser_ism.add_argument(
        "-s",
        "--save_ohe",
        type=str,
        default=None,
        help="Where to save the one-hot encoded sequences. Defaults to not saving.",
    )

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
            # The CLIPNET_TF-specific engine: same references and rules as
            # tangermeme's deep_lift_shap on CountWrapper/ProfileWrapper, with a
            # hand-written forward/backward (see personal_bpnet.clipnet_tf_fast).
            attributions.append(
                fast_deep_lift_shap(
                    model,
                    X,
                    target="counts"
                    if args.counts_head_only
                    else args.attribute_type,
                    hypothetical=args.hypothetical,
                    batch_size=args.batch_size,
                    n_shuffles=args.n_shuffles,
                    random_state=args.random_state,
                    verbose=args.verbose,
                    device=device,
                    profile_rule=args.profile_rule,
                ).numpy()
            )

        # Save
        np.savez_compressed(args.out_fname, np.stack(attributions).mean(axis=0))

    elif args.cmd == "ism":
        if args.counts_head_only and args.output != "counts":
            raise ValueError("Counts head only models only support --output counts.")
        if args.output == "both" and not args.raw_outputs:
            raise ValueError("--output both needs --raw_outputs.")
        X = extract_loci(
            loci=args.bed_fname,
            sequences=args.fa_fname,
            chroms=args.chroms,
            in_window=1000,
            verbose=args.verbose,
            ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        ).to(torch.float)
        if args.save_ohe is not None:
            np.savez_compressed(args.save_ohe, X.to(int).numpy())

        # Ensemble by averaging across replicates as we go (a full-window profile
        # ISM is ~16 MB per region per replicate). ISM attributions are linear in
        # the predictions, so averaging them equals the attributions of the
        # averaged predictions.
        total, n_models = None, 0
        for model in load_ensemble(model_paths, _load_tf_model):
            out = saturation_mutagenesis(
                model,
                X,
                start=args.start,
                end=args.end,
                output=args.output,
                raw_outputs=args.raw_outputs,
                hypothetical=args.hypothetical,
                batch_size=args.batch_size,
                device=device,
                verbose=args.verbose,
            )
            flat = []
            for o in out if args.raw_outputs else [out]:
                flat.extend(o if isinstance(o, list) else [o])
            total = flat if total is None else [t + f for t, f in zip(total, flat)]
            n_models += 1
        mean = [(t / n_models).numpy() for t in total]

        if not args.raw_outputs:
            np.savez_compressed(args.out_fname, mean[0])
        elif args.output == "both":
            y0_profile, y0_counts, y_hat_profile, y_hat_counts = mean
            np.savez_compressed(
                args.out_fname,
                y0_profile=y0_profile,
                y0_counts=y0_counts,
                y_hat_profile=y_hat_profile,
                y_hat_counts=y_hat_counts,
            )
        else:
            np.savez_compressed(args.out_fname, y0=mean[0], y_hat=mean[1])

    else:
        raise ValueError(_help)


if __name__ == "__main__":
    cli()
