"""A single-nucleotide DFIM heatmap of one promoter with the CLIPNET 9-fold
ensemble (personal_bpnet.clipnet_tf_fast.dfim).

The unreduced maps are averaged over the folds and then reduced to the most
extreme FIS per source position, as `clipnet_tf dfim` does; every fold uses
the same references (20 dinucleotide shuffles of the window). The figure
shows the map over the core promoter, with the sequence's DeepLIFT/SHAP
logo along the target axis and its attribution along the source axis.

usage:
  dfim_heatmap.py MODELS_DIR HG38_FASTA OUT_PREFIX [--chrom chr6] [--tss 391752]
      [--strand +] [--lo -100] [--hi 60] [--target counts]
      [--motifs "GC box:-65:-55,GC box:-50:-40,TATA:-32:-27,Inr:-2:2"]

Defaults are the IRF4 promoter (TSS from ENCODE B-cell PRO-cap, as in
observed_tss.py). Writes OUT_PREFIX.npz and OUT_PREFIX.png.
"""

import argparse
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pyfaidx  # noqa: E402
import torch  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from tangermeme.plot import plot_logo  # noqa: E402
from tangermeme.utils import one_hot_encode  # noqa: E402

from personal_bpnet import CLIPNET_TF, TwoHotToOneHot  # noqa: E402
from personal_bpnet.clipnet_tf_fast import (  # noqa: E402
    _shuffle_references,
    deep_lift_shap,
    dfim,
    dfim_extrema,
)

warnings.filterwarnings("ignore")
T = 500  # the TSS's position in the 1 kb window
COMP = str.maketrans("ACGTN", "TGCAN")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
DIV = LinearSegmentedColormap.from_list("div", ["#2a78d6", "#f0efec", "#e34948"])


def window(fa, chrom, strand, tss):
    """1 kb with the TSS at position 500, reverse-complemented for - strand."""

    if strand == "+":
        return str(fa[chrom][tss - 1 - 500 : tss - 1 + 500]).upper()
    return str(fa[chrom][tss - 500 : tss + 500]).upper().translate(COMP)[::-1]


def compute(args):
    fa = pyfaidx.Fasta(args.fasta, read_ahead=0)
    X = one_hot_encode(window(fa, args.chrom, args.strand, args.tss)).float()[None]
    refs = _shuffle_references(X, 20, 47)  # shared by every fold
    start, end = T + args.lo, T + args.hi + 1
    fis, attr = 0, 0
    for i in range(1, 10):
        m = TwoHotToOneHot(CLIPNET_TF.from_tf(f"{args.models}/fold_{i}.h5")).eval()
        fis = fis + dfim(m, X, start, end, target=args.target, references=refs,
                         reduce=None) / 9
        attr = attr + deep_lift_shap(m, X, target=args.target, references=refs) / 9
        print(f"fold {i} done", flush=True)
    ext = dfim_extrema(fis, X, start, end)
    np.savez_compressed(f"{args.out}.npz", ohe=X[0].numpy(), fis=fis[0].numpy(),
                        extrema=ext[0].numpy(), attribution=attr[0].numpy(),
                        start=start, end=end)
    return X[0].numpy(), ext[0].numpy(), attr[0].numpy(), start, end


def plot(args, X, ext, attr, start, end):
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "text.color": INK, "font.size": 9, "axes.titlesize": 10,
        "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.spines.top": False, "axes.spines.right": False,
    })
    motifs = []
    for item in filter(None, args.motifs.split(",")):
        name, a, b = item.rsplit(":", 2)
        motifs.append((name, int(a), int(b)))
    M = ext[:, start:end].copy()  # sources x targets over the same span
    np.fill_diagonal(M, np.nan)  # a base's own attribution changes trivially
    v = np.nanquantile(np.abs(M), 0.995)
    obs = (attr * X).sum(0)[start:end]
    W = end - start
    rel = np.arange(start, end) - T

    fig = plt.figure(figsize=(12, 12.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1, 6], width_ratios=[0.6, 6, 0.15],
                          hspace=0.04, wspace=0.04)
    ax_logo = fig.add_subplot(gs[0, 1])
    ax_side = fig.add_subplot(gs[1, 0])
    ax = fig.add_subplot(gs[1, 1])
    cax = fig.add_subplot(gs[1, 2])

    im = ax.imshow(M, cmap=DIV, vmin=-v, vmax=v, aspect="auto", interpolation="nearest",
                   extent=(rel[0] - 0.5, rel[-1] + 0.5, rel[-1] + 0.5, rel[0] - 0.5))
    for name, a, b in motifs:
        for line in (ax.axvline, ax.axhline):
            line(a - 0.5, color=INK2, lw=0.6, ls=(0, (2, 2)))
            line(b + 0.5, color=INK2, lw=0.6, ls=(0, (2, 2)))
    ticks = np.arange(int(np.ceil(rel[0] / 20) * 20), rel[-1] + 1, 20)
    labels = [f"{t:+d}" if t else "TSS" for t in ticks]
    ax.set_xticks(ticks)
    ax.set_xticklabels(labels)
    ax.set_yticks(ticks)
    ax.set_yticklabels([])
    ax.set_xlabel(f"target position relative to the TSS (bp): whose attribution changes")
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("FIS = attribution in sequence − in variant (most extreme of 3)")
    cb.outline.set_visible(False)

    plot_logo((attr * X)[:, start:end], ax=ax_logo)
    ax_logo.set_xlim(-0.5, W - 0.5)
    ax_logo.set_xticks([])
    ax_logo.set_ylabel("DeepLIFT/\nSHAP")
    for name, a, b in motifs:
        ax_logo.axvspan(a - rel[0] - 0.5, b - rel[0] + 0.5, color=GRID, alpha=0.6, lw=0,
                        zorder=0)
        ax_logo.text((a + b) / 2 - rel[0], ax_logo.get_ylim()[1], name, ha="center",
                     va="bottom", fontsize=8, color=INK2)
    strand = " (reverse-complemented)" if args.strand == "-" else ""
    ax_logo.set_title(
        f"{args.name} promoter, CLIPNET 9-fold ensemble: single-nucleotide DFIM, "
        f"{args.target} target{strand}\n{args.chrom}:{args.tss:,} ({args.strand}); "
        "diagonal masked; dashed lines mark motifs", pad=16)

    ax_side.barh(rel, obs, height=0.9, color=np.where(obs >= 0, "#e34948", "#2a78d6"))
    ax_side.set_ylim(rel[-1] + 0.5, rel[0] - 0.5)
    ax_side.invert_xaxis()
    ax_side.set_yticks(ticks)
    ax_side.set_yticklabels(labels)
    ax_side.set_ylabel("source position relative to the TSS (bp): substituted")
    ax_side.set_xticks([])
    ax_side.spines["left"].set_visible(False)
    for name, a, b in motifs:
        ax_side.axhspan(a - 0.5, b + 0.5, color=GRID, alpha=0.6, lw=0, zorder=0)
    fig.savefig(f"{args.out}.png", dpi=180, bbox_inches="tight")
    print("wrote", f"{args.out}.png")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("models")
    p.add_argument("fasta")
    p.add_argument("out")
    p.add_argument("--name", default="IRF4")
    p.add_argument("--chrom", default="chr6")
    p.add_argument("--tss", type=int, default=391752)
    p.add_argument("--strand", default="+", choices=("+", "-"))
    p.add_argument("--lo", type=int, default=-100)
    p.add_argument("--hi", type=int, default=60)
    p.add_argument("--target", default="counts", choices=("counts", "profile", "shape"))
    p.add_argument("--motifs", default="GC box:-65:-55,GC box:-50:-40,TATA:-32:-27,Inr:-2:2")
    args = p.parse_args()
    plot(args, *compute(args))
