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

`personal_bpnet.clipnet_tf_fast` has two functions written for the CLIPNET_TF architecture, after the engines in [Cherimoya](https://github.com/jmschrei/cherimoya) and [promoterai-torch](https://github.com/genomicsxai/promoterai-torch). They take a `CLIPNET_TF` or `TwoHotToOneHot(CLIPNET_TF)` model (run in eval mode, not modified):

```python
from personal_bpnet.clipnet_tf_fast import deep_lift_shap, saturation_mutagenesis

model = TwoHotToOneHot(CLIPNET_TF.from_tf("clipnet_models/fold_1.h5")).cuda()

# X: (N, 4, 1000) one-hot
attr = deep_lift_shap(model, X, target="counts", n_shuffles=20, random_state=47)
attr = deep_lift_shap(model, X, target="profile", n_shuffles=20, random_state=47,
                      hypothetical=True)
ism = saturation_mutagenesis(model, X)  # (N, 4, 1000) counts ISM attributions
y0, y_hat = saturation_mutagenesis(model, X, output="both", raw_outputs=True)
```

- **`deep_lift_shap`** returns what `tangermeme.deep_lift_shap.deep_lift_shap(CountWrapper(model), X, ...)` returns, with the same dinucleotide-shuffled references for a given `random_state`. It doesn't use hooks or autograd. Each sequence is forwarded once instead of once per reference, the backward runs only over the sequence half of each sequence–reference pair, and the DeepLIFT rules are written out and fused with TorchScript. On a TITAN X (Pascal), 1024 sequences × 20 shuffles take 3.6 s against 12.4 s for tangermeme 1.6 at its fastest batch size (3.4x), in 1.1 GiB of GPU memory against 9.5 GiB. Memory grows linearly with `batch_size` (~70 MiB a sequence at 20 shuffles, ~5x less per sequence–reference pair than tangermeme). The default is the smallest step that saturates the GPU (16 sequences on a TITAN X, scaled by SM count on larger GPUs). The two agree to ~1e-5 relative L2 in float32 (~1e-8 in float64, see `tests/test_clipnet_tf_fast.py`). `target="profile"` attributes `ProfileWrapper`'s summary `sum(c * softmax(c))` (`c` the mean-centered profile) with tangermeme's softmax and midpoint product rules (`profile_rule="product"`, what tangermeme ≥ 1.5 gives a wrapper written with `torch.nn.Softmax` and `BilinearOp`). `profile_rule="rescale"` reproduces the elementwise rescale rule of bpnet-lite's `ProfileWrapper` with `_ProfileLogitScaling` registered as `_nonlinear`, which the CLI used up to version 0.7.0.
- **`saturation_mutagenesis`** returns what `tangermeme.saturation_mutagenesis.saturation_mutagenesis` returns on `CountWrapper(model)` (`output="counts"`), the profile head (`"profile"`), or the bare model (`"both"`, with `raw_outputs=True`). It is incremental (fastISM): each reference is run once with its layers cached, and each substitution recomputes only the window of each layer it can change. That window spans the whole sequence only from the 7th dilated residual layer on. On a TITAN X (Pascal), ISM of 32 sequences × 1000 positions takes 2.4 s against 5.6 s for tangermeme's standard ISM (2.3x) for the counts, and 3.1 s against 6.0 s (2.0x) for both heads. The predictions are bit-identical to tangermeme's in float32 in our tests.

Both default to CUDA when available. `deep_lift_shap` fuses its rules with `torch.compile` on GPUs of compute capability ≥ 7.0 (the first call at each shape compiles for a few seconds to a minute) and with TorchScript elsewhere; pass `compile=True`/`False` to choose. On GPUs older than compute capability 7.0 (e.g. Pascal), use a torch build with kernels for them (e.g. the `cu126` wheels): the default CUDA 13 wheels have none.
