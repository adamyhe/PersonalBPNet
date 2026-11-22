"""Figures for the promoter study: a summary of agreement with ISM, and one
figure per promoter (ISM vs DeepLIFT/SHAP and PISA under each reference set).

usage: plots.py RESULTS_DIR OUT_DIR
"""

import ast
import glob
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from tangermeme.plot import plot_logo  # noqa: E402

res_dir, out = sys.argv[1:3]
os.makedirs(out, exist_ok=True)
T, Z0, Z1 = 500, 400, 560
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
REF_COLOR = {"standard": "#2a78d6", "quiet": "#eb6834", "mass": "#1baf7a"}
REF_LABEL = {"standard": "20 dinucleotide shuffles", "quiet": "20 low-activity shuffles",
             "mass": "20 mass-matched shuffles"}
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
})
NO_PISA = set()
TARGETS = [("counts", "counts", "DeepLIFT/SHAP, counts head"),
           ("shape", "shape", "DeepLIFT/SHAP, shape target"),
           ("pisa", "track", "PISA, predicted track at the TSS"),
           ("pisa_profile", "profile", "PISA, raw profile at the TSS")]


def observed(a, X):
    return (a * X).sum(0)


def ism_obs(d, X):
    return ((d - d.mean(0, keepdims=True)) * X).sum(0)


def r_core(a, b):
    return float(np.corrcoef(a[Z0:Z1], b[Z0:Z1])[0, 1])


files = sorted(glob.glob(f"{res_dir}/*.npz"))
data = {os.path.basename(f)[:-4]: np.load(f) for f in files}
order = sorted(data, key=lambda n: (data[n]["strand"] == "-", n))

# ------------------------------------------------------------- summary
fig, axes = plt.subplots(1, 4, figsize=(13, 4.6), sharey=True, gridspec_kw={"wspace": 0.08})
ypos = {n: i for i, n in enumerate(order)}
for ax, (key, ism_key, title) in zip(axes, TARGETS):
    for j, ref in enumerate(REF_COLOR):
        xs, ys = [], []
        for n in order:
            if key.startswith("pisa") and n in NO_PISA:
                continue
            r = data[n]
            X = r["ohe"]
            xs.append(r_core(observed(r[f"{ref}__{key}"], X), ism_obs(r[f"ism__{ism_key}"], X)))
            ys.append(ypos[n] + (j - 1) * 0.22)
        ax.scatter(xs, ys, s=36, color=REF_COLOR[ref], edgecolor=SURFACE, linewidth=1.5,
                   zorder=3, label=REF_LABEL[ref])
        ax.axvline(np.mean(xs), color=REF_COLOR[ref], lw=1.5, ls=(0, (2, 2)), zorder=2)
    ax.set_title(title, fontsize=9.5)
    ax.set_xlim(-0.1, 1.0)
    ax.grid(axis="x", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.set_xlabel("Pearson r with ISM, core −100..+60")
axes[0].set_yticks(range(len(order)))
axes[0].set_yticklabels([f"{n} ({data[n]['strand']})" for n in order])
axes[0].invert_yaxis()
axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.1), ncol=3, frameon=False)
fig.suptitle("CLIPNET 9-fold ensemble at 11 LCL promoters: agreement of each attribution with "
             "reference-free ISM of the same target (dashed: mean)", x=0.125, ha="left",
             y=1.06, fontsize=11, fontweight="bold")
fig.savefig(f"{out}/summary_agreement.png", dpi=200, bbox_inches="tight")
plt.close(fig)

# ------------------------------------------------------------- per promoter
def logo(ax, a, title):
    plot_logo(a[:, Z0:Z1], ax=ax)
    ax.axvline(T - Z0, color=INK2, lw=0.8, ls=(0, (3, 3)), zorder=0)
    pos = np.arange(Z0, Z1, 10)
    ax.set_xticks(pos - Z0)
    ax.set_xticklabels([f"{p - T:+d}" if p != T else "TSS" for p in pos])
    ax.set_title(title, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)


for n in order:
    r = data[n]
    X = r["ohe"]
    s = ast.literal_eval(str(r["summary"]))
    peak = int(r["peak"])
    has_pisa = n not in NO_PISA
    panels = [("ism", "counts"), ("standard", "counts"), ("ism", "shape"), ("standard", "shape")]
    if has_pisa:
        panels += [("ism", "pisa"), ("standard", "pisa"), ("quiet", "pisa"), ("mass", "pisa")]
    fig = plt.figure(figsize=(13, 1.75 * (len(panels) + 1.2)))
    gs = fig.add_gridspec(len(panels) + 1, 1, height_ratios=[1.2] + [1] * len(panels), hspace=0.9)
    ax = fig.add_subplot(gs[0])
    tr = r["track"]
    xx = np.arange(250, 750)
    ax.fill_between(xx, tr[:500], color="#2a78d6", lw=0, label="+ strand")
    ax.fill_between(xx, -tr[500:], color="#eb6834", lw=0, label="− strand")
    ax.axhline(0, color=INK2, lw=0.6)
    ax.axvline(T, color=INK2, lw=0.8, ls=(0, (3, 3)))
    ax.axvspan(Z0, Z1, color=GRID, alpha=0.5, lw=0, zorder=0)
    pos = np.arange(250, 750, 50)
    ax.set_xticks(pos)
    ax.set_xticklabels([f"{p - T:+d}" if p != T else "TSS" for p in pos])
    ax.set_xlim(250, 749)
    ax.legend(loc="upper left", frameon=False, ncol=2)
    ax.set_ylabel("reads")
    strand_note = "reverse-complemented, " if r["strand"] == "-" else ""
    ax.set_title(f"{n} ({r['strand']}), observed TSS (B-cell PRO-cap) {r['chrom']}:{int(r['tss']):,}; {strand_note}"
                 f"predicted counts {float(r['counts']):.1f}; main + site TSS{peak + 250 - T:+d}; "
                 f"raw profile mass: sequence {s['sequence']['mass']:.1f}, standard "
                 f"{s['standard']['mass']:.1f}, quiet {s['quiet']['mass']:.1f}, matched "
                 f"{s['mass']['mass']:.1f}", fontsize=9)
    for i, (ref, key) in enumerate(panels, start=1):
        g = gs[i]
        ax = fig.add_subplot(g)
        ism_key = {"counts": "counts", "shape": "shape", "pisa": "track"}[key]
        ism = r[f"ism__{ism_key}"]
        ism_a = (ism - ism.mean(0, keepdims=True)) * X
        name = {"counts": "counts", "shape": "shape target", "pisa": "PISA track at TSS"
                f"{peak + 250 - T:+d}"}[key]
        if ref == "ism":
            logo(ax, ism_a, f"ISM, {name} (observed base: minus the mean effect of mutating it)")
        else:
            a = r[f"{ref}__{key}"] * X
            rr = r_core(a.sum(0), ism_a.sum(0))
            logo(ax, a, f"{'DeepLIFT/SHAP' if key != 'pisa' else 'PISA'}, {name}, "
                        f"{REF_LABEL[ref]}  (r with ISM {rr:.2f})")
    ax.set_xlabel("position relative to the TSS (bp)")
    fig.savefig(f"{out}/{n}.png", dpi=170, bbox_inches="tight")
    plt.close(fig)
print("wrote", out)
