"""Add PISA of the raw profile output (target="profile", no normalization) at
each promoter's initiation site, and ISM of that output, to the results files
(same reference sets as promoters.py).

usage: pisa_profile_addon.py MODELS_DIR RESULTS_DIR
"""

import glob
import sys
import warnings

import numpy as np
import torch

from personal_bpnet import CLIPNET_TF, TwoHotToOneHot
from personal_bpnet.clipnet_tf_fast import pisa, saturation_mutagenesis

warnings.filterwarnings("ignore")
torch.backends.cudnn.allow_tf32 = False
models_dir, results = sys.argv[1:3]
models = [TwoHotToOneHot(CLIPNET_TF.from_tf(f"{models_dir}/fold_{i}.h5")).cuda().eval()
          for i in range(1, 10)]
for f in sorted(glob.glob(f"{results}/*.npz")):
    r = dict(np.load(f))
    X = torch.from_numpy(r["ohe"]).float()[None]
    peak = int(r["peak"])
    new = {}
    for m in models:
        for ref in ("standard", "quiet", "mass"):
            R = torch.from_numpy(r[f"refs__{ref}"]).float()[None]
            a = pisa(m, X, outputs=[peak], target="profile", references=R, hypothetical=True)
            k = f"{ref}__pisa_profile"
            new[k] = new.get(k, 0) + a[0, 0].numpy() / len(models)
        y0, yh = saturation_mutagenesis(m, X, output="profile", raw_outputs=True)
        d = (yh[0, ..., peak] - y0[0, peak]).cpu().numpy()
        new["ism__profile"] = new.get("ism__profile", 0) + d / len(models)
    r.update(new)
    np.savez_compressed(f, **r)
    print("updated", f.split("/")[-1], flush=True)
