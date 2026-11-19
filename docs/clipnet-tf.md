# Porting the original TensorFlow CLIPNET weights

See [the README](../README.md#install) for installation — this requires the `tf` extra (`uv add "personalbpnet[tf]"` or `pip install "personalbpnet[tf]"`).

`CLIPNET_TF` is a faithful port of the original TensorFlow `rnn_v10` CLIPNET architecture to PyTorch — a genuinely different architecture from the [PyTorch `CLIPNET`](clipnet.md) (uses MaxPool, ELU, 1000 bp input → 500 bp output, two-hot encoded inputs). The `from_tf` class method loads hdf5 weights directly into a PyTorch module, without requiring TensorFlow to be installed.

```python
import os

from personal_bpnet import CLIPNET_TF

os.makedirs("clipnet_models/", exist_ok=True)
for i in range(1, 10):
    os.system(f"wget https://zenodo.org/records/10408623/files/fold_{i}.h5 -P clipnet_models/")

models = [CLIPNET_TF.from_tf(f"clipnet_models/fold_{i}.h5") for i in range(1, 10)]
```

These models expect inputs of shape `(N, 4, 1000)`. **IMPORTANT:** models loaded this way still expect inputs to be two-hot encoded (see the description in the [TensorFlow CLIPNET README](https://github.com/Danko-Lab/clipnet/blob/main/README.md)). For compatibility with packages that only allow one-hot encoded sequences, use the `TwoHotToOneHot` wrapper:

```python
from personal_bpnet import TwoHotToOneHot

ohe_models = [TwoHotToOneHot(m) for m in models]
```

This works for all models trained using the `rnn_v10` architecture in the [original CLIPNET repo](https://github.com/Danko-Lab/clipnet/blob/main/clipnet/rnn_v10.py). At present, this includes:

- the original [LCL PRO-cap models](https://zenodo.org/records/10408623),
- [K562 PRO-cap models](https://zenodo.org/records/14037356) (fine-tuned from the above),
- [ablated LCL PRO-cap models](https://zenodo.org/records/14037356).

For a full worked example, see the [Google Colab notebook](https://github.com/Danko-Lab/clipnet/blob/main/clipnet_basic_tutorial.ipynb) from the original TensorFlow CLIPNET repo.

## Fast DeepLIFT/SHAP and ISM

`personal_bpnet.clipnet_tf_fast` has three functions written for the CLIPNET_TF architecture, after the engines in [Cherimoya](https://github.com/jmschrei/cherimoya) and [promoterai-torch](https://github.com/genomicsxai/promoterai-torch). They take a `CLIPNET_TF` or `TwoHotToOneHot(CLIPNET_TF)` model (run in eval mode, not modified):

```python
from personal_bpnet.clipnet_tf_fast import deep_lift_shap, pisa, saturation_mutagenesis

model = TwoHotToOneHot(CLIPNET_TF.from_tf("clipnet_models/fold_1.h5")).cuda()

# X: (N, 4, 1000) one-hot
attr = deep_lift_shap(model, X, target="counts", n_shuffles=20, random_state=47)
attr = deep_lift_shap(model, X, target="profile", n_shuffles=20, random_state=47,
                      hypothetical=True)
ism = saturation_mutagenesis(model, X)  # (N, 4, 1000) counts ISM attributions
prof = saturation_mutagenesis(model, X, output="profile")  # (N, 4, 1000, 1000), per profile position
y0, y_hat = saturation_mutagenesis(model, X, output="both", raw_outputs=True)
P = pisa(model, X, channel=0, random_state=47)  # (N, 500, 4, 1000), + strand
```

- **`deep_lift_shap`** returns what `tangermeme.deep_lift_shap.deep_lift_shap(CountWrapper(model), X, ...)` returns, with the same dinucleotide-shuffled references for a given `random_state`. It doesn't use hooks or autograd. Each sequence is forwarded once instead of once per reference, the backward runs only over the sequence half of each sequence–reference pair, and the DeepLIFT rules are written out and fused with TorchScript. On a TITAN X (Pascal), 1024 sequences × 20 shuffles take 3.6 s against 12.4 s for tangermeme 1.6 at its fastest batch size (3.4x), in 1.1 GiB of GPU memory against 9.5 GiB. Memory grows linearly with `batch_size` (~70 MiB a sequence at 20 shuffles, ~5x less per sequence–reference pair than tangermeme). The default is the smallest step that saturates the GPU (16 sequences on a TITAN X, scaled by SM count on larger GPUs). The two agree to ~1e-5 relative L2 in float32 (~1e-8 in float64, see `tests/test_clipnet_tf_fast.py`). `target="profile"` attributes `ProfileWrapper`'s summary `sum(c * softmax(c))` (`c` the mean-centered profile) with tangermeme's softmax and midpoint product rules (`profile_rule="product"`, what tangermeme ≥ 1.5 gives a wrapper written with `torch.nn.Softmax` and `BilinearOp`). `profile_rule="rescale"` reproduces the elementwise rescale rule of bpnet-lite's `ProfileWrapper` with `_ProfileLogitScaling` registered as `_nonlinear`, which the CLI used up to version 0.7.0.
- **`saturation_mutagenesis`** returns what `tangermeme.saturation_mutagenesis.saturation_mutagenesis` returns on `CountWrapper(model)` (`output="counts"`), the profile head (`"profile"`), or the bare model (`"both"`, with `raw_outputs=True`). The one exception is the attributions for `output="profile"`: tangermeme averages each substitution's (base-centered) change over the 1000 profile outputs, which leaves only the change in the profile's total. The model's predictions normalize that total away, since the counts head sets the scale. So `"profile"` returns the change at every profile position, unreduced, with shape `(N, 4, end - start, 1000)` (~16 MB per region for a full window). Averaging it over the last axis gives tangermeme's attributions.

  It is incremental (fastISM): each reference is run once with its layers cached, and each substitution recomputes only the window of each layer it can change. That window spans the whole sequence only from the 7th dilated residual layer on, and the profile dense layer always needs all of it, so a step still does ~45% of a full forward's multiply-adds. To keep the rest of the step from dominating, the BatchNorms are folded into the adjacent convolutions and dense layers, the cached activations are stored pre-padded, and each residual layer's elementwise epilogue is one fused kernel (TorchScript by default, `compile=True` for torch.compile, which is no faster here but takes a few seconds to compile). On an RTX 3080 (float32, TF32 off), counts ISM of 128 sequences × 1000 positions takes 4.2 s against 11.5 s for tangermeme's standard ISM (2.7x), and profile ISM of 32 sequences 1.1 s against 3.1 s (2.8x). The predictions agree with tangermeme's to float32 rounding (~3e-7 relative), and to ~1e-15 in float64. tangermeme 1.6's own generic fastISM (`fast=True`, its default) falls back to standard ISM on this model on a GPU, where it is slower.

- **`pisa`** (pairwise influence by sequence attribution) runs DeepLIFT/SHAP once for every profile output position. It returns the attributions of the prediction at each position to every input nucleotide, with shape `(N, n_out, 4, 1000)`: `n_out` is 500 for `channel=0` (+ strand) or `1` (- strand), and 1000 for `channel=None` (both, the default), or as many as `outputs=` lists. That is ~8 MB per sequence and strand.
  - **`target="track"`** (default) attributes the predicted track that `clipnet_tf predict` reports: `c * y_j / (sum(y) + 1e-3)`, the profile head's output `y` (both strands) normalized to a distribution and scaled by the counts head's `c`. Its products get the midpoint product rule and its reciprocal the rescale rule, so each output's attributions still sum to its change. **`target="profile"`** attributes `y_j` itself, whose scale the predictions normalize away.
  - It equals tangermeme's `pisa` on a wrapper returning those outputs (written with `BilinearOp` and a reciprocal module registered as `_nonlinear`), with the same references for a given `random_state` (~1e-15 in float64).
  - After Cherimoya's PISA engine, each sequence is forwarded once with its references. Every rule's coefficient depends only on the activations, so it is computed once per sequence–reference pair and shared by all output positions. The backward is then a chain of convolution transposes and fused elementwise products over (references × output positions) rows, sequence half only.
  - On an RTX 3080, one strand takes 0.155 s per sequence at 20 shuffles against 2.6–3.1 s for tangermeme at its fastest batch size (17–20x), in ~1.1 GiB.

Both take `tf32=False` (the default), which turns TF32 convolutions and matmuls off for the duration of the call and restores the previous global setting on return. PyTorch otherwise runs cuDNN convolutions in TF32 on Ampere and newer GPUs. Its rounding noise is about the size of a single base's effect, so it swamps float32 ISM (30–50% error against float64 in our tests, batch-size dependent) and adds error to DeepLIFT/SHAP. Pass `tf32=True` where TF32 is much faster than float32 (e.g. server GPUs) and the noise is acceptable. On an RTX 3080, `deep_lift_shap` takes 1.35 s for 1024 sequences × 20 shuffles with `torch.compile` (1.65 s with TorchScript), against 6.3 s for tangermeme at its fastest batch size, whether TF32 is on or off.

Both default to CUDA when available. `deep_lift_shap` fuses its rules with `torch.compile` on GPUs of compute capability ≥ 7.0 (the first call at each shape compiles for a few seconds to a minute) and with TorchScript elsewhere; pass `compile=True`/`False` to choose. On GPUs older than compute capability 7.0 (e.g. Pascal), use a torch build with kernels for them (e.g. the `cu126` wheels): the default CUDA 13 wheels have none.
