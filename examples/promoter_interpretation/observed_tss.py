"""Dominant TSS of each promoter from observed PRO-cap (ENCODE ENCSR083AMN,
primary B cells, GRCh38), searched on the gene's strand within 300 bp of any of
its RefSeq transcript starts. Also the model's strongest predicted site in the
same region, as a cross-check.

usage: observed_tss.py GTF_GZ BIGWIG_DIR HG38_FASTA MODELS_DIR OUT_TSV
"""

import gzip
import re
import sys
import warnings

import numpy as np
import pyBigWig
import pyfaidx
import torch
from tangermeme.utils import one_hot_encode

from personal_bpnet import CLIPNET_TF, TwoHotToOneHot

warnings.filterwarnings("ignore")
torch.backends.cudnn.allow_tf32 = False
gtf, bwdir, fasta, models_dir, out = sys.argv[1:6]
GENES = ["IRF4", "GAPDH", "CD79A", "JUNB", "MYC", "HLA-DRA", "ACTB", "EEF1A1", "CD74",
         "NFKBIA", "RPLP0"]
USED = {"IRF4": 391752, "GAPDH": 6534517, "CD79A": 41877279, "JUNB": 12791486,
        "MYC": 127735434, "HLA-DRA": 32439887, "ACTB": 5530601, "EEF1A1": 73521032,
        "CD74": 150412936, "NFKBIA": 35404754, "RPLP0": 120201111}
PLUS = ["ENCFF746KYF", "ENCFF812MRP"]
MINUS = ["ENCFF408NIK", "ENCFF074SSA"]
FLANK = 300

starts = {}
with gzip.open(gtf, "rt") as f:
    for line in f:
        p = line.split("\t")
        if len(p) < 9 or p[2] != "transcript" or not re.fullmatch(r"chr[0-9XY]+", p[0]):
            continue  # primary chromosomes only (HLA genes also sit on alt haplotypes)
        g = re.search(r'gene_name "([^"]+)"', p[8]).group(1)
        if g in GENES:
            tss = int(p[3]) if p[6] == "+" else int(p[4])
            starts.setdefault(g, (p[0], p[6], set()))[2].add(tss)

bws = {s: [pyBigWig.open(f"{bwdir}/{a}.bigWig") for a in acc] for s, acc in (("+", PLUS), ("-", MINUS))}
fa = pyfaidx.Fasta(fasta, read_ahead=0)
models = [TwoHotToOneHot(CLIPNET_TF.from_tf(f"{models_dir}/fold_{i}.h5")).cuda().eval()
          for i in range(1, 10)]


def predicted(chrom, strand, lo, hi):
    """Ensemble predicted track on `strand` over 0-based [lo, hi), averaged
    over windows that cover it (each window predicts its central 500 bp)."""

    cov, n = np.zeros(hi - lo), np.zeros(hi - lo)
    for c in range(lo - 250, hi + 250, 100):
        s0 = c - 500
        X = one_hot_encode(str(fa[chrom][s0 : s0 + 1000]).upper()).float()[None]
        with torch.no_grad():
            pr = [m(X.cuda()) for m in models]
        p = torch.stack([a for a, _ in pr]).mean(0)[0].cpu().numpy()
        k = float(torch.stack([b for _, b in pr]).mean())
        t = k * p / (p.sum() + 1e-3)
        t = t[:500] if strand == "+" else t[500:]
        g = np.arange(s0 + 250, s0 + 750) - lo
        ok = (g >= 0) & (g < hi - lo)
        cov[g[ok]] += t[ok]
        n[g[ok]] += 1
    return cov / np.maximum(n, 1)


rows = []
print(f"{'gene':8s} {'str':3s} {'observed TSS':>14s} {'reads':>7s} {'RefSeq used':>12s} {'Δ used':>7s}"
      f" {'nearest RefSeq':>14s} {'Δ':>5s} {'model peak':>11s} {'Δ obs':>6s}")
for g in GENES:
    chrom, strand, ts = starts[g]
    lo, hi = min(ts) - 1 - FLANK, max(ts) + FLANK  # 0-based half-open
    obs = sum(np.nan_to_num(np.abs(np.array(bw.values(chrom, lo, hi)))) for bw in bws[strand])
    i = int(np.argmax(obs))
    tss = lo + i + 1  # 1-based
    pred = predicted(chrom, strand, lo, hi)
    j = int(np.argmax(pred))
    model_peak = lo + j + 1
    nearest = min(ts, key=lambda t: abs(t - tss))
    sign = 1 if strand == "+" else -1  # offsets in transcript direction (+ = downstream)
    rows.append((g, chrom, strand, tss, float(obs[i]), USED[g], nearest, model_peak))
    print(f"{g:8s} {strand:3s} {chrom}:{tss:>11,} {obs[i]:7.0f} {USED[g]:12,} "
          f"{sign * (USED[g] - tss):+7d} {nearest:14,} {sign * (nearest - tss):+5d} "
          f"{model_peak:11,} {sign * (model_peak - tss):+6d}")
with open(out, "w") as f:
    f.write("gene\tchrom\tstrand\tobserved_tss\tobserved_reads\trefseq_used\tnearest_refseq\tmodel_peak\n")
    for r in rows:
        f.write("\t".join(str(x) for x in r) + "\n")
print("offsets are in the transcript's direction: + means the site lies downstream of the observed TSS")
