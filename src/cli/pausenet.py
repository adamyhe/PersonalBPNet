#!/usr/bin/env python
# PauseNet CLI
# Author: Adam He <adamyhe@gmail.com>

"""
Wrapper script to calculate attributions and predictions for PauseNet models
"""

from argparse import ArgumentParser

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.io import extract_loci
from tangermeme.predict import predict

from personal_bpnet.clipnet_pytorch import CLIPNET, PauseNet

from ._common import (
    add_attribute_args,
    add_shape_args,
    build_parent_parser,
    load_ensemble,
    load_torch_model,
    resolve_device,
    resolve_model_paths,
    set_cpu_threads,
)

_help = """
The following commands are available:
    predict         Calculate predictions for a PauseNet model
    attribute       Calculate DeepLIFT/SHAP attributions for a PauseNet model
Planned but not implemented commands:
    vep             Calculate variant effect prediction using a PauseNet model
"""


def _build_pausenet(args):
    base_model = CLIPNET(
        n_filters=args.n_filters,
        n_outputs=2,
        n_control_tracks=0,
        n_layers=args.n_layers,
        trimming=(args.in_window - args.out_window) // 2,
    )
    return PauseNet(base_model)


def cli():
    parser_parent = build_parent_parser(
        bed_fname_help=(
            "Path to bed file of regions to calculate predictions/attributions for. "
            "Will assume that the 6th column (if provided for a bed6+ file) is the "
            "target phenotype for this model, and will compute performance metrics "
            "with it."
        )
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
    add_shape_args(parser_predict)

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
        loci = pd.read_csv(args.bed_fname, sep="\t", header=None)
        loci.rename({0: "chrom", 1: "start", 2: "end"}, axis=1, inplace=True)

        X = extract_loci(
            loci=loci,
            sequences=args.fa_fname,
            chroms=args.chroms,
            in_window=args.in_window,
            out_window=args.out_window,
            verbose=args.verbose,
            ignore=list("QWERYUIOPSDFHJKLZXVBNM"),
        )

        log_predictions = [
            predict(
                model,
                X,
                batch_size=args.batch_size,
                verbose=args.verbose,
                device=device,
            )
            for model in load_ensemble(
                model_paths, lambda f: load_torch_model(f, lambda: _build_pausenet(args))
            )
        ]

        # Average log-scale predictions across replicates before exponentiating
        prediction = (
            torch.exp(torch.stack(log_predictions).mean(dim=0) - 1).cpu().numpy()
        )

        # Save predictions
        np.savez_compressed(args.out_fname, prediction)

        # Calculate performance metrics if score column is present in bed file
        if loci.shape[1] >= 5:
            if args.chroms is not None:
                loci = loci[loci["chrom"].isin(args.chroms)]
            p = np.log1p(prediction.squeeze())
            signals = np.log1p(loci.iloc[:, 4].to_numpy())
            pearson = pearsonr(p, signals)
            spearman = spearmanr(p, signals)

            print(f"Pearson: {pearson[0]}, p-value: {pearson[1]}")
            print(f"Spearman: {spearman[0]}, p-value: {spearman[1]}")

            np.savez_compressed(
                args.out_fname.replace(".npz", "_metrics.npz"),
                pearson=pearson,
                spearman=spearman,
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

        attributions = [
            deep_lift_shap(
                model,
                X,
                hypothetical=args.hypothetical,
                batch_size=args.batch_size,
                n_shuffles=args.n_shuffles,
                random_state=args.random_state,
                verbose=args.verbose,
                device=device,
            ).numpy()
            for model in load_ensemble(
                model_paths, lambda f: load_torch_model(f, lambda: _build_pausenet(args))
            )
        ]

        # Save
        np.savez_compressed(args.out_fname, np.stack(attributions).mean(axis=0))

    else:
        raise ValueError(_help)


if __name__ == "__main__":
    cli()
