"""Agreement of each attribution with ISM of the same target, per promoter and
reference set: Pearson r of the observed-base attributions over the core
promoter (-100..+60) and over the whole 1 kb window.

usage: analyze.py RESULTS_DIR
"""

import glob
import os
import sys

import numpy as np

T, CORE = 500, slice(400, 561)
REFS = ("standard", "quiet", "mass")
TARGETS = ("counts", "shape", "pisa", "pisa_profile")
ISM_KEY = {"counts": "counts", "shape": "shape", "pisa": "track", "pisa_profile": "profile"}
NO_PISA = set()


def observed(attr, X):
    return (attr * X).sum(0)


def ism_observed(d, X):
    return ((d - d.mean(0, keepdims=True)) * X).sum(0)


def pearson(a, b):
    return float(np.corrcoef(a, b)[0, 1])


rows = []
for f in sorted(glob.glob(os.path.join(sys.argv[1], "*.npz"))):
    r = np.load(f)
    name = os.path.basename(f)[:-4]
    X = r["ohe"]
    for tgt in TARGETS:
        if tgt.startswith("pisa") and name in NO_PISA:
            continue
        ism = ism_observed(r[f"ism__{ISM_KEY[tgt]}"], X)
        for ref in REFS:
            a = observed(r[f"{ref}__{tgt}"], X)
            neg = np.abs(np.minimum(a[CORE], 0)).sum() / np.abs(a[CORE]).sum()
            rows.append((name, tgt, ref, pearson(a[CORE], ism[CORE]), pearson(a, ism), neg))

names = sorted({r[0] for r in rows}, key=[r[0] for r in rows].index)
for tgt in TARGETS:
    print(f"\n{tgt}: Pearson r with ISM, core -100..+60 (whole 1 kb)")
    print(f"  {'promoter':9s}" + "".join(f"{ref:>18s}" for ref in REFS))
    for n in names:
        vals = {r[2]: r for r in rows if r[0] == n and r[1] == tgt}
        if vals:
            print(f"  {n:9s}" + "".join(f"{vals[ref][3]:10.2f} ({vals[ref][4]:.2f})" for ref in REFS))
    sub = [r for r in rows if r[1] == tgt]
    means = {ref: np.mean([r[3] for r in sub if r[2] == ref]) for ref in REFS}
    fulls = {ref: np.mean([r[4] for r in sub if r[2] == ref]) for ref in REFS}
    print(f"  {'mean':9s}" + "".join(f"{means[ref]:10.2f} ({fulls[ref]:.2f})" for ref in REFS))
    best = {ref: 0 for ref in REFS}
    for n in names:
        vals = [r for r in sub if r[0] == n]
        if vals:
            best[max(vals, key=lambda r: r[3])[2]] += 1
    print(f"  {'best core':9s}" + "".join(f"{best[ref]:>18d}" for ref in REFS))
    negs = {ref: np.mean([r[5] for r in sub if r[2] == ref]) for ref in REFS}
    print(f"  {'neg share':9s}" + "".join(f"{negs[ref]:18.2f}" for ref in REFS)
          + "   (mean share of |attr| in the core that is negative)")
np.save(os.path.join(sys.argv[1], "agreement.npy"), np.array(rows, dtype=object))
