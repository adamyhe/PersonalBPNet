#!/usr/bin/env python
# PauseNet CLI
# Author: Adam He <adamyhe@gmail.com>

"""
Wrapper script to calculate attributions and predictions for PauseNet models
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from tangermeme.deep_lift_shap import deep_lift_shap
from tangermeme.io import extract_loci
from tangermeme.predict import predict

from .clipnet_pytorch import CLIPNET, PauseNet

_help = """
The following commands are available:
    predict         Calculate predictions for a PauseNet model
    attribute       Calculate DeepLIFT/SHAP attributions for a PauseNet model
Planned but not implemented commands:
    vep             Calculate variant effect prediction using a PauseNet model
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
        help="Path to bed file of regions to calculate predictions/attributions for. "
        "Will assume that the 6th column (if provided for a bed6+ file) is the target "
        "phenotype for this model, and will compute performance metrics with it.",
    )
    parser_parent.add_argument(
        "-o", "--out_fname", type=str, required=True, help="Path to output npz file"
    )
    parser_parent.add_argument(
        "-m",
        "--model_fname",
        type=str,
        required=True,
        help="Path to model directory or to specific model file to predict/attribute. "
        "If a directory, loads and calculates average predictions/attributions across "
        "all models in directory. If a specific model file, will only predict/attribute "
        "that model. ",
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
        help="Calculate attributions for a given set of regions.",
        parents=[parser_parent],
    )
    parser_attribute.add_argument(
        "-s",
        "--save_ohe",
        type=str,
        default=None,
        help="Where to save OHE of sequences. Defaults to not saving. "
        "Set this & hypothetical if you intend to use these attributions for "
        "tfmodisco-lite.",
    )
    parser_attribute.add_argument(
        "-y",
        "--hypothetical",
        action="store_true",
        help="Whether to use hypothetical attributions. Defaults to False. "
        "Set this & save_ohe if you intend to use these attributions for "
        "tfmodisco-lite.",
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
    if os.path.isdir(args.model_fname):
        model_names = [
            os.path.join(args.model_fname, f"f{i}.torch") for i in range(1, 10)
        ]
    else:
        model_names = [args.model_fname]

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

        prediction_ = []
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
                model = PauseNet(model)
                model.load_state_dict(m)

            # Calculate and log predictions
            prediction_.append(
                predict(model, X, batch_size=args.batch_size, verbose=args.verbose)
            )

            # clear VRAM
            del m
            torch.cuda.empty_cache()

        # Rescale predictions and take average
        prediction = torch.exp(torch.stack(prediction_) - 1).mean(dim=0).cpu().numpy()

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
                model = PauseNet(model)
                model.load_state_dict(m)

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
