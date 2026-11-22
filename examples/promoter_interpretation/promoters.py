"""CLIPNET (9-fold ensemble) interpretation at several LCL promoters: three
reference sets (20 standard dinucleotide shuffles, 20 low-activity shuffles
selected as the PRO-cap atlas selects them, 20 mass-matched shuffles), scored
against reference-free in silico mutagenesis (ISM) of the same target:

  counts   DeepLIFT/SHAP of the counts head          vs ISM of the counts
  shape    DeepLIFT/SHAP, target="shape" (sum w p)   vs ISM of sum w p
  pisa     PISA row of the predicted track at the    vs ISM of that track value
           promoter's main + strand initiation site

Windows are 1 kb centered on the TSS observed in B-cell PRO-cap (not RefSeq's
start, nor the start codon);
minus-strand promoters are reverse-complemented so that every promoter reads
5'->3' with its TSS at position 500, on the + strand outputs.

usage: promoters.py MODELS_DIR HG38_FASTA OUT_DIR
"""

import os
import sys
import time
import warnings

import numpy as np
import pyfaidx
import torch
from tangermeme.ersatz import dinucleotide_shuffle
from tangermeme.utils import one_hot_encode

from personal_bpnet import CLIPNET_TF, TwoHotToOneHot
from personal_bpnet.clipnet_tf_fast import deep_lift_shap, pisa, saturation_mutagenesis

warnings.filterwarnings("ignore")
torch.backends.cudnn.allow_tf32 = False
EPS = 1e-3
# name, chrom, strand, TSS (1-based): the dominant TSS observed in ENCODE B-cell
# PRO-cap (ENCSR083AMN) within 300 bp of the gene's RefSeq starts, from
# observed_tss.py. RefSeq's own starts were off by up to 797 bp (MYC) here.
PROMOTERS = [
    ("IRF4", "chr6", "+", 391752), ("GAPDH", "chr12", "+", 6534517),
    ("CD79A", "chr19", "+", 41877280), ("JUNB", "chr19", "+", 12791486),
    ("MYC", "chr8", "+", 127736231), ("HLA-DRA", "chr6", "+", 32439887),
    ("ACTB", "chr7", "-", 5530601), ("EEF1A1", "chr6", "-", 73521032),
    ("CD74", "chr5", "-", 150412751), ("NFKBIA", "chr14", "-", 35404749),
    ("RPLP0", "chr12", "-", 120201107),
]
COMP = str.maketrans("ACGTN", "TGCAN")


def window(fa, chrom, strand, tss):
    if strand == "+":
        start0 = tss - 1 - 500
        return str(fa[chrom][start0 : start0 + 1000]).upper(), start0
    start0 = tss - 500
    return str(fa[chrom][start0 : start0 + 1000]).upper().translate(COMP)[::-1], start0


def robust_z(v, high_is_bad=True):
    med = np.median(v)
    mad = np.median(np.abs(v - med))
    scale = 1.4826 * mad if mad > 0 else (np.std(v) or 1.0)
    z = (v - med) / scale
    return z if high_is_bad else -z


def diverse(order, bases, n=20, min_frac=0.1):
    chosen = []
    for i in order:
        if all(np.mean(bases[i] != bases[j]) >= min_frac for j in chosen):
            chosen.append(int(i))
        if len(chosen) == n:
            break
    return chosen


def metrics(models, rows):
    keys = ("counts", "mass", "max_20bp", "max_5bp", "max_p", "imbalance", "entropy")
    acc = {k: 0.0 for k in keys}
    for m in models:
        with torch.no_grad():
            y, c = m(rows.cuda())
        c, S = c[:, 0], y.sum(-1)
        p = y / (S[:, None] + EPS)
        t = c[:, None] * p

        def wmax(a, w):
            return torch.nn.functional.conv1d(a[:, None], torch.ones(1, 1, w, device=a.device)).amax(-1)[:, 0]

        met = {"counts": c, "mass": S,
               "max_20bp": torch.maximum(wmax(t[:, :500], 20), wmax(t[:, 500:], 20)),
               "max_5bp": torch.maximum(wmax(t[:, :500], 5), wmax(t[:, 500:], 5)),
               "max_p": p.amax(-1),
               "imbalance": (p[:, :500].sum(-1) - p[:, 500:].sum(-1)).abs(),
               "entropy": -(p.clamp_min(1e-12) * p.clamp_min(1e-12).log()).sum(-1)}
        for k in keys:
            acc[k] = acc[k] + met[k].cpu().numpy() / len(models)
    return acc


def select(X, models):
    cand = dinucleotide_shuffle(X, n=500, random_state=0)[0]
    met = metrics(models, torch.cat([X, cand]))
    seq = {k: float(v[0]) for k, v in met.items()}
    met = {k: v[1:] for k, v in met.items()}
    bases = cand.numpy().argmax(1)
    # quiet: the atlas's clipped robust penalties, ties broken by the unclipped sum
    parts = [robust_z(np.log1p(met["counts"])), robust_z(met["max_20bp"]),
             robust_z(met["max_5bp"]), robust_z(met["max_p"]),
             robust_z(met["imbalance"]), robust_z(met["entropy"], high_is_bad=False)]
    score = sum(np.maximum(z, 0) for z in parts)
    quiet = diverse(np.lexsort((sum(parts), score)), bases)
    # mass-matched: no louder than the median in counts and peaks, then the
    # raw profile mass closest to the sequence's
    ok = ((met["counts"] <= np.median(met["counts"]))
          & (met["max_5bp"] <= np.median(met["max_5bp"])))
    gap = np.abs(np.log(met["mass"] + EPS) - np.log(seq["mass"] + EPS))
    gap[~ok] = np.inf
    mass = diverse(np.argsort(gap, kind="stable"), bases)
    standard = dinucleotide_shuffle(X, n=20, random_state=47)[0]
    sets = {"standard": standard, "quiet": cand[quiet], "mass": cand[mass]}
    summary = {"sequence": seq}
    for name, idx in (("quiet", quiet), ("mass", mass)):
        summary[name] = {k: float(met[k][idx].mean()) for k in met}
    summary["standard"] = {k: float(v.mean()) for k, v in metrics(models, standard).items()}
    return sets, summary


def ism_targets(m, X, peak):
    """Per-model ISM of the three targets: deltas (4, L) of counts, of the
    track at output `peak`, and of sum w p with w the sequence's own p."""

    (y0p, y0c), (yhp, yhc) = saturation_mutagenesis(m, X, output="both", raw_outputs=True)
    y0p, y0c, yhp, yhc = y0p[0], y0c[0, 0], yhp[0], yhc[0, ..., 0]
    S0, Sh = y0p.sum() + EPS, yhp.sum(-1) + EPS
    w = y0p / S0
    return {
        "counts": yhc - y0c,
        "track": yhc * yhp[..., peak] / Sh - y0c * y0p[peak] / S0,
        "shape": (yhp * w).sum(-1) / Sh - (y0p * w).sum() / S0,
    }


def main():
    models_dir, fasta, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    fa = pyfaidx.Fasta(fasta, read_ahead=0)
    models = [TwoHotToOneHot(CLIPNET_TF.from_tf(f"{models_dir}/fold_{i}.h5")).cuda().eval()
              for i in range(1, 10)]
    for name, chrom, strand, tss in PROMOTERS:
        t0 = time.time()
        seq, start0 = window(fa, chrom, strand, tss)
        X = one_hot_encode(seq).float()[None]
        with torch.no_grad():
            preds = [m(X.cuda()) for m in models]
        prof = torch.stack([p for p, _ in preds]).mean(0)[0]
        cnt = torch.stack([c for _, c in preds]).mean(0)[0]
        track = (cnt * prof / (prof.sum() + EPS)).cpu()
        # the main + strand initiation site within 20 bp of the annotated TSS
        lo = 250 - 20
        peak = lo + int(track[lo : 250 + 21].argmax())
        sets, summary = select(X, models)
        res = {}
        for m in models:
            for ref, R in sets.items():
                R = R[None].float()
                for key, f in (
                    ("counts", lambda: deep_lift_shap(m, X, target="counts", references=R,
                                                      hypothetical=True)),
                    ("shape", lambda: deep_lift_shap(m, X, target="shape", references=R,
                                                     hypothetical=True)),
                    ("pisa", lambda: pisa(m, X, outputs=[peak], target="track",
                                          references=R, hypothetical=True)[:, 0]),
                ):
                    k = f"{ref}__{key}"
                    res[k] = res.get(k, 0) + f()[0].numpy() / len(models)
            for key, d in ism_targets(m, X, peak).items():
                k = f"ism__{key}"
                res[k] = res.get(k, 0) + d.cpu().numpy() / len(models)
        np.savez_compressed(
            f"{out}/{name}.npz", seq=np.array(seq), ohe=X[0].numpy(), track=track.numpy(),
            counts=float(cnt), peak=peak, chrom=chrom, strand=strand, tss=tss,
            start0=start0, summary=np.array(repr(summary)), **res,
            **{f"refs__{k}": v.numpy() for k, v in sets.items()},
        )
        s = summary
        print(f"{name:8s} {strand} peak at TSS{peak + 250 - 500:+d} ({float(track[peak]):.1f} reads, "
              f"counts {float(cnt):.1f}) | mass seq {s['sequence']['mass']:.1f}, standard "
              f"{s['standard']['mass']:.1f}, quiet {s['quiet']['mass']:.1f}, matched "
              f"{s['mass']['mass']:.1f} | counts std {s['standard']['counts']:.1f} quiet "
              f"{s['quiet']['counts']:.1f} matched {s['mass']['counts']:.1f} "
              f"({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
