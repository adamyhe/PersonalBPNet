"""The fast CLIPNET_TF DeepLIFT/SHAP and incremental ISM engines agree with
tangermeme's generic implementations (float64, random weights, CPU)."""

import pytest
import torch
from bpnetlite.bpnet import CountWrapper, ProfileWrapper, _ProfileLogitScaling
from tangermeme.deep_lift_shap import _nonlinear
from tangermeme.deep_lift_shap import deep_lift_shap as tm_deep_lift_shap
from tangermeme.saturation_mutagenesis import (
    saturation_mutagenesis as tm_saturation_mutagenesis,
)
from tangermeme.utils import random_one_hot

from personal_bpnet import CLIPNET_TF, TwoHotToOneHot
from personal_bpnet.clipnet_tf_fast import deep_lift_shap, saturation_mutagenesis


def _model(counts_head_only=False, wrap=True, seed=0):
    torch.manual_seed(seed)
    m = CLIPNET_TF(counts_head_only=counts_head_only)
    for mod in m.modules():  # non-trivial eval-mode BatchNorm statistics
        if isinstance(mod, torch.nn.BatchNorm1d):
            n = mod.num_features
            mod.running_mean.uniform_(-0.5, 0.5)
            mod.running_var.uniform_(0.5, 2.0)
            mod.weight.data.uniform_(0.5, 1.5)
            mod.bias.data.uniform_(-0.2, 0.2 if n > 1 else 1.0)
    m = m.double().eval()
    return TwoHotToOneHot(m).eval() if wrap else m


def _X(n=2, seed=0):
    return random_one_hot((n, 4, 1000), random_state=seed).double()


def _rel(a, b):
    return float((a - b).norm() / b.norm())


try:  # tangermeme >= 1.5
    from tangermeme.deep_lift_shap import BilinearOp
except ImportError:
    BilinearOp = None


class _ProductProfileWrapper(torch.nn.Module):
    """bpnet-lite's ProfileWrapper with its logits * softmax(logits) written as
    a torch.nn.Softmax and a BilinearOp, as Cherimoya's ProfileWrapper is, so
    tangermeme applies its softmax and midpoint product rules."""

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.softmax = torch.nn.Softmax(dim=-1)
        self.product = BilinearOp("...,...->...")

    def forward(self, X):
        logits = self.model(X)[0]
        logits = logits - logits.mean(dim=-1, keepdim=True)
        return self.product(logits, self.softmax(logits)).sum(dim=-1, keepdim=True)


def _check(model, X, wrapper, ops, hypothetical, tol, **kwargs):
    expected = tm_deep_lift_shap(
        wrapper, X, n_shuffles=3, random_state=5, hypothetical=hypothetical,
        additional_nonlinear_ops=ops, device="cpu", warning_threshold=1e9,
    )
    got, deltas = deep_lift_shap(
        model, X, n_shuffles=3, random_state=5, hypothetical=hypothetical,
        return_deltas=True, device="cpu", **kwargs,
    )
    assert got.shape == X.shape
    assert _rel(got, expected) < tol
    assert float(deltas.max()) < tol


@pytest.mark.parametrize("hypothetical", [True, False])
def test_deep_lift_shap_counts_matches_tangermeme(hypothetical):
    model = _model()
    _check(model, _X(), CountWrapper(model), None, hypothetical, 1e-8)


@pytest.mark.skipif(BilinearOp is None, reason="needs tangermeme >= 1.5")
@pytest.mark.parametrize("hypothetical", [True, False])
def test_deep_lift_shap_profile_matches_tangermeme_product_rules(hypothetical):
    model = _model()
    _check(model, _X(), _ProductProfileWrapper(model), None, hypothetical, 1e-8,
           target="profile")


@pytest.mark.parametrize("hypothetical", [True, False])
def test_deep_lift_shap_profile_rescale_matches_bpnetlite_wrapper(hypothetical):
    # The elementwise rescale rule over the whole product is not exactly
    # conservative here (a logit whose change is tiny gets an unbounded secant),
    # so this legacy rule only agrees with tangermeme to ~1e-6.
    model = _model()
    _check(model, _X(), ProfileWrapper(model), {_ProfileLogitScaling: _nonlinear},
           hypothetical, 1e-4, target="profile", profile_rule="rescale")


def test_deep_lift_shap_counts_head_only_and_given_references():
    model, X = _model(counts_head_only=True), _X()
    refs = torch.stack([random_one_hot((3, 4, 1000), random_state=i) for i in range(2)]).double()
    expected = tm_deep_lift_shap(model, X, references=refs, device="cpu", warning_threshold=1e9)
    got = deep_lift_shap(model, X, references=refs, device="cpu", batch_size=1)
    assert _rel(got, expected) < 1e-8


def test_deep_lift_shap_unwrapped_twohot():
    """A bare CLIPNET_TF is attributed in its own (two-hot) encoding: the
    multipliers still sum to the change in the prediction."""

    model = _model(wrap=False)
    X = _X() * 2
    refs = torch.stack([random_one_hot((2, 4, 1000), random_state=i) for i in range(2)]).double() * 2
    _, deltas = deep_lift_shap(model, X, references=refs, return_deltas=True, device="cpu")
    assert float(deltas.max()) < 1e-8


def test_deep_lift_shap_rejects_bad_inputs():
    with pytest.raises(TypeError):
        deep_lift_shap(torch.nn.Linear(4, 1), _X(), device="cpu")
    with pytest.raises(ValueError):
        deep_lift_shap(_model(counts_head_only=True), _X(), target="profile", device="cpu")


def test_train_mode_model_is_attributed_in_eval_and_restored():
    model, X = _model(counts_head_only=True), _X(1)
    expected = deep_lift_shap(model, X, n_shuffles=2, random_state=0, device="cpu")
    model.train()
    got = deep_lift_shap(model, X, n_shuffles=2, random_state=0, device="cpu")
    assert model.training and model.model.training
    assert torch.equal(got, expected)


@pytest.mark.parametrize("batch_size", [30, 200])  # 1 and 3 sequences per step
@pytest.mark.parametrize("start,end", [(0, 24), (480, 520), (970, -1)])
def test_saturation_mutagenesis_matches_tangermeme(start, end, batch_size):
    model = _model()
    X = _X(3, seed=1)
    X[0, :, start + 3] = 0  # an N: all four substitutions are kept
    expected = tm_saturation_mutagenesis(
        model, X, start=start, end=end, raw_outputs=True, device="cpu"
    )
    got = saturation_mutagenesis(
        model, X, start=start, end=end, output="both", raw_outputs=True,
        batch_size=batch_size, device="cpu",
    )
    for g, e in zip(got[0], expected[0]):
        assert torch.allclose(g, e, rtol=0, atol=1e-10)
    for g, e in zip(got[1], expected[1]):
        assert g.shape == e.shape
        assert float((g - e).abs().max()) < 1e-10


def test_saturation_mutagenesis_attributions_match_tangermeme():
    model = _model(counts_head_only=True)
    X = _X(seed=2)
    for hypothetical in (True, False):
        expected = tm_saturation_mutagenesis(
            model, X, start=100, end=140, hypothetical=hypothetical, device="cpu"
        )
        got = saturation_mutagenesis(
            model, X, start=100, end=140, hypothetical=hypothetical, device="cpu"
        )
        assert float((got - expected).abs().max()) < 1e-10


@pytest.mark.parametrize("compile", [True, False])
def test_deep_lift_shap_fusion_modes_agree(compile):
    """torch.compile (Inductor; its C++ backend on the CPU, Triton on CUDA) and
    TorchScript fuse the same rules: both match the unfused rules."""

    import personal_bpnet.clipnet_tf_fast as fast

    model, X = _model(), _X(3)
    kwargs = dict(n_shuffles=2, random_state=1, device="cpu", batch_size=2,
                  hypothetical=True)
    got = deep_lift_shap(model, X, target="profile", compile=compile, **kwargs)
    eager = fast._fused
    try:
        fast._fused = lambda fn, mode="script": fn
        expected = deep_lift_shap(model, X, target="profile", **kwargs)
    finally:
        fast._fused = eager
    assert _rel(got, expected) < 1e-12


def test_resolve_compile():
    from personal_bpnet.clipnet_tf_fast import _resolve_compile

    cpu = torch.device("cpu")
    assert _resolve_compile(None, cpu) == "script"
    assert _resolve_compile(False, cpu) == "script"
    assert _resolve_compile(True, cpu) == "compile"
