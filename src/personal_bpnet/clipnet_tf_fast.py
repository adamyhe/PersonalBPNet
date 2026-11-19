# clipnet_tf_fast.py
# Author: Adam He <adamyhe@gmail.com>

"""
Fast DeepLIFT/SHAP and in silico saturation mutagenesis for `CLIPNET_TF`.

Both functions here compute what a generic tangermeme call computes for a
`CLIPNET_TF` model, but with the forward and backward passes written out for
this one architecture, following the engines in Cherimoya
(`cherimoya/_deep_lift_shap_engine.py`) and promoterai-torch
(`promoterai_torch/attribution.py`, `promoterai_torch/incremental.py`).

`deep_lift_shap` matches, up to floating-point rounding::

    tangermeme.deep_lift_shap.deep_lift_shap(CountWrapper(TwoHotToOneHot(m)), X,
        n_shuffles=K, random_state=s, hypothetical=...)

for target="counts". For target="profile" it attributes `ProfileWrapper`'s
summary sum(c * softmax(c)), c the mean-centered profile, with tangermeme's
(>= 1.5) softmax rule and midpoint product rule -- what tangermeme gives a
wrapper that writes the product with `torch.nn.Softmax` and `BilinearOp`, as
Cherimoya's `ProfileWrapper` does -- or, with profile_rule="rescale", with the
elementwise rescale rule that bpnet-lite's `ProfileWrapper` plus
`{_ProfileLogitScaling: _nonlinear}` gives (the behavior of `clipnet_tf
attribute -a profile` up to personalbpnet 0.7.0). The rescale rule does not
fit a softmax, which couples every position: its multipliers are not exactly
conservative and grow without bound for logits that barely move.

What changes is how the work is done:

* Joint rows. tangermeme forwards each sequence once per reference, as half of
  a 2-row pair. Here each step forwards S sequences once and their S*K
  references once, so the sequence half of the forward is done K times less.
* Sequence-half backward. Every op in CLIPNET_TF is row-wise and tangermeme's
  rules for the sequence row of a pair read the reference row's activations
  but not its gradient, so the backward runs over the S*K pairs instead of
  2*S*K rows. The one exception is the max-pool rule, which routes the
  reference half's gradient as well; here that gradient is taken to equal the
  sequence half's. tangermeme's two halves carry the same gradient except
  downstream of a rescale rule's fallback (|x - r| < 1e-6) at a unit whose
  input straddles 0 between the sequence and the reference, which does not
  measurably occur (`tests/test_clipnet_tf_fast.py` agrees with tangermeme to
  1e-8 in float64).
* Hand-written rules, no hooks. The nonlinearities are the rescale rule at
  every ELU/ReLU (secant multiplier, or the ordinary gradient where the input
  moves by < 1e-6), tangermeme's max-pool rule, and for the profile summary
  the softmax and midpoint product rules; everything else (eval-mode
  BatchNorm, convolutions, the dense layers, the residual adds, the global
  average pool) is affine, so its ordinary transpose is exact.
* Fused rules. The rules are memory-bound elementwise work, fused by
  torch.compile where its Triton backend runs (CUDA capability >= 7.0) and by
  TorchScript elsewhere, whose fuser also runs on older GPUs such as Pascal
  (~3x faster than eager there). Both specialize on shapes, so a short last
  step is padded to the full step size.
* References are drawn on the CPU for the next step while the GPU runs the
  current one.

`saturation_mutagenesis` is incremental (fastISM, Nair et al. 2022): it runs
each reference sequence once, caches every layer, and for each block of
substituted positions recomputes only the window of each layer that the
substitutions can reach, splicing it into the cached reference activations.
The window covers the whole sequence only from the 7th dilated residual layer
on (dilation 64), so the stem and the first six residual layers -- which hold
most of the convolution work -- are computed on a small fraction of their
positions. A step takes a few positions of many sequences, to keep windows
narrow and batches large, and every window has the same width wherever the
edit is (shifted inward at the sequence ends rather than clipped), since
cuDNN's first call at each new shape costs far more than the call itself.
"""

import contextlib
import warnings

import torch
import torch.nn.functional as F

from .clipnet_tensorflow import CLIPNET_TF, TwoHotToOneHot

_RESCALE_EPS = 1e-6  # tangermeme's threshold for falling back to the gradient


# ---------------------------------------------------------------- model access


def _unwrap(model):
    """(CLIPNET_TF, input scale): TwoHotToOneHot multiplies its input by 2."""

    scale = 1.0
    while isinstance(model, TwoHotToOneHot):
        model, scale = model.model, scale * 2.0
    if not isinstance(model, CLIPNET_TF):
        raise TypeError(
            "model must be a CLIPNET_TF or TwoHotToOneHot(CLIPNET_TF), got "
            f"{type(model).__name__}"
        )
    return model, scale


@contextlib.contextmanager
def _eval_mode(m):
    """Run with `m` in eval mode (BatchNorm on its running statistics, which is
    what makes every layer row-wise), restoring its previous mode afterwards."""

    training = m.training
    m.eval()
    try:
        yield
    finally:
        m.train(training)


def _bn_scale(bn):
    """The per-channel multiplier of an eval-mode BatchNorm (also its transpose)."""

    return bn.weight / torch.sqrt(bn.running_var + bn.eps)


def _same_pad(pad):
    """The symmetric padding of an `AsymmetricPad`, as a conv `padding` value."""

    from .clipnet_tensorflow import _get_tf_same_padding

    left, right = _get_tf_same_padding(pad.kernel_size, pad.dilation)
    if left != right:
        raise ValueError("only symmetric 'same' padding (odd kernels) is supported")
    return left


def _pool(x):
    """MaxPool1d(2) and, for each output, whether its max is the window's second
    entry (`F.max_pool1d`'s tie-breaking: the first maximum wins)."""

    out, idx = F.max_pool1d(x, 2, return_indices=True)
    return out, (idx % 2).bool()


# ------------------------------------------------------------------- rules
#
# The rules are written as plain tensor functions so that torch.compile or
# TorchScript's fuser can fuse each into a few kernels (see `_fused`): they are
# memory-bound elementwise work over (S, K, C, L) tensors, ~3x faster fused on
# a GPU. torch.compile's Triton backend needs CUDA capability >= 7.0, and
# TorchScript's fuser also runs on older GPUs.


def _relu_rescale(g, a_x, a_r, scale):
    """tangermeme's rescale rule through a ReLU for the sequence half of each
    pair, times `scale` (the next BatchNorm's per-channel multiplier, (C, 1)).

    g (S, K, ...) is the gradient at the ReLU's output, a_x (S, 1, ...) the
    sequences' ReLU inputs and a_r (S, K, ...) the references'. Returns the
    gradient at the ReLU's input: g times the secant, or times the ordinary
    gradient where the input moves by less than 1e-6.
    """

    delta = a_x - a_r
    small = delta.abs() < 1e-6
    secant = (torch.relu(a_x) - torch.relu(a_r)) / torch.where(
        small, torch.ones_like(delta), delta
    )
    return g * torch.where(small, (a_x > 0).to(g.dtype), secant) * scale


def _elu_rescale(g, a_x, a_r, scale):
    """`_relu_rescale` through an ELU (alpha=1), whose ordinary gradient is 1
    for x > 0 and exp(x) otherwise (torch's ELU backward)."""

    delta = a_x - a_r
    small = delta.abs() < 1e-6
    secant = (F.elu(a_x) - F.elu(a_r)) / torch.where(
        small, torch.ones_like(delta), delta
    )
    grad = torch.where(a_x > 0, torch.ones_like(a_x), torch.exp(a_x))
    return g * torch.where(small, grad, secant) * scale


def _maxpool_rule(g, in_x, in_r, sel_x, sel_r):
    """tangermeme's `_maxpool` rule for MaxPool1d(2) on the sequence half.

    g (S, K, C, Lo) is the gradient at the pool's output, in_x (S, 1, C, L) and
    in_r (S, K, C, L) its inputs, sel_x and sel_r their `_pool` selections
    (whether each window's max is its second entry). The change in the pooled
    output is routed to each half's own argmax and divided by the change in the
    input, or the ordinary gradient is used where the input does not change.
    The reference half's gradient is taken to equal g (see the module
    docstring). Computed in the windows' (..., Lo, 2) layout.
    """

    L = in_x.shape[-1]
    Lo = g.shape[-1]
    wx = in_x[..., : 2 * Lo].unflatten(-1, (Lo, 2))
    wr = in_r[..., : 2 * Lo].unflatten(-1, (Lo, 2))
    out_x = wx.amax(-1)
    out_r = wr.amax(-1)
    top = torch.maximum(out_x, out_r)
    sx = sel_x.to(g.dtype).unsqueeze(-1)
    sr = sel_r.to(g.dtype).unsqueeze(-1)
    hot_x = torch.cat([1 - sx, sx], -1)
    hot_r = torch.cat([1 - sr, sr], -1)
    routed = g.unsqueeze(-1) * (
        (top - out_r).unsqueeze(-1) * hot_x + (out_x - top).unsqueeze(-1) * hot_r
    )
    delta = wx - wr
    zero = delta == 0
    out = torch.where(
        zero,
        g.unsqueeze(-1) * hot_x,
        routed / torch.where(zero, torch.ones_like(delta), delta),
    ).flatten(-2)
    return F.pad(out, (0, L - 2 * Lo))


_FUSED = {}


def _torch_compile_available(device):
    """Whether torch.compile's default (Inductor) backend can fuse on `device`:
    on CUDA it generates Triton kernels, which need compute capability >= 7.0
    and the triton package; the passes here also need torch >= 2.6."""

    import importlib.util

    version = tuple(int(v) for v in torch.__version__.split("+")[0].split(".")[:2])
    if version < (2, 6) or device.type != "cuda":
        return False
    return (
        torch.cuda.get_device_capability(device) >= (7, 0)
        and importlib.util.find_spec("triton") is not None
    )


def _resolve_compile(compile, device):
    """"compile" (torch.compile) or "script" (TorchScript) for the `compile`
    argument of the public functions: None picks torch.compile where Triton can
    run on `device`, and TorchScript (whose fuser also runs on older GPUs, and
    which is correct but unfused on the CPU) everywhere else."""

    if compile is None:
        return "compile" if _torch_compile_available(device) else "script"
    if compile and device.type == "cuda" and not _torch_compile_available(device):
        raise RuntimeError(
            "compile=True needs torch >= 2.6 and, on CUDA, the triton package and "
            "a GPU of compute capability >= 7.0; pass compile=False to fuse the "
            "rules with TorchScript instead"
        )
    return "compile" if compile else "script"


def _fused(fn, mode="script"):
    """`fn` fused: by torch.compile (mode="compile"; static shapes, which the
    callers keep constant, and one compile per distinct shape) or TorchScript
    (mode="script"; fused on CUDA after a couple of calls). If TorchScript
    cannot script `fn`, `fn` itself. TorchScript is deprecated in recent torch
    and warns; the warning is silenced since nothing else here uses it."""

    key = (fn, mode)
    if key not in _FUSED:
        if mode == "compile":
            _FUSED[key] = torch.compile(fn, dynamic=False, fullgraph=True)
        else:
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _FUSED[key] = torch.jit.script(fn)
            except Exception:  # pragma: no cover - depends on the torch build
                _FUSED[key] = fn
    return _FUSED[key]


def _recompile_limit(mode):
    """Room for one torch.compile specialization per distinct rule shape (a few
    per call, more if a step size is halved after running out of memory). The
    option is `recompile_limit` in newer torch, `cache_size_limit` in older."""

    if mode != "compile":
        return contextlib.nullcontext()
    config = torch._dynamo.config
    name = next(
        (n for n in ("recompile_limit", "cache_size_limit") if hasattr(config, n)), None
    )
    if name is None or getattr(config, name) >= 64:
        return contextlib.nullcontext()
    return config.patch(**{name: 64})


# --------------------------------------------------------------- the forward


def _forward(m, x):
    """The forward of CLIPNET_TF `m` on rows x (R, 4, L), keeping what the
    backward rules read. Operation for operation the model's own forward."""

    s = {}
    a0 = m.ibn(m.iconv(m.ibnorm(m.input(x))))
    s["a0"] = a0
    e0 = m.irelu(a0)
    p0, s["sel0"] = _pool(e0)
    a1 = m.sbnorm(m.sconv(p0))
    s["a1"] = a1
    r1 = m.srelu(a1)
    p1, s["sel1"] = _pool(r1)
    h = m.rconv(p1)
    s["c"], s["s"] = [], []
    for i in range(m.n_layers):
        c = m.rbnorms[i](m.rconvs[i](m.rpads[i](h)))
        s["c"].append(c)
        h = torch.add(h, m.rrelus[i](c))
        sk = m.skipbnorms[i](h)
        s["s"].append(sk)
        h = m.skiprelus[i](sk)
    s["h"] = h
    q, s["sel2"] = _pool(h)
    s["q"] = q
    s["z"] = m.cbnorm(m.clinear(m.cavgpool(q).squeeze(-1)))  # counts pre-ReLU
    if not m.counts_head_only:
        flat = q.permute(0, 2, 1).reshape(q.shape[0], -1)
        s["zp"] = m.pbnorm(m.plinear(flat))  # profile pre-ReLU
    return s


def _split(t, S, K):
    """Rows [x_0..x_{S-1}, r_00..r_{S-1,K-1}] as (S, 1, ...) and (S, K, ...)."""

    return t[:S].unsqueeze(1), t[S:].reshape(S, K, *t.shape[1:])


# ---------------------------------------------------------- output targets


def _counts_target(m, s, S, K):
    """The attributed output of each row and the multipliers at the final pool's
    output q, (S, K, C, Lo), for `CountWrapper` (or a counts-head-only model)."""

    z = s["z"][:, 0]
    y = F.relu(z)
    zx, zr = _split(z, S, K)
    g = _relu_rescale(torch.ones_like(zr), zx, zr, 1.0)  # (S, K)
    w = m.clinear.weight[0] * _bn_scale(m.cbnorm)[0]  # (C,)
    Lo = s["q"].shape[-1]
    G = (g[..., None] * w / Lo)[..., None].expand(S, K, w.shape[0], Lo)
    return y, G


def _softmax_rule(g, x, x_ref):
    """tangermeme's (>= 1.5) DeepLIFT rule for a softmax over the last axis, for
    the sequence half: the softmax decomposed as a_i = exp(x_i - c), s = sum a,
    v = 1 / s, y_i = a_i v, with an exact multiplier for each step chained.

    g (S, K, n) is the gradient at the softmax's output, x (S, 1, n) and
    x_ref (S, K, n) its inputs. Returns the gradient at its input.
    """

    c = torch.maximum(x.amax(-1, keepdim=True), x_ref.amax(-1, keepdim=True))
    a, a_ref = torch.exp(x - c), torch.exp(x_ref - c)
    tiny = torch.finfo(a.dtype).tiny
    v = a.sum(-1, keepdim=True).clamp_min(tiny).reciprocal()
    v_ref = a_ref.sum(-1, keepdim=True).clamp_min(tiny).reciprocal()
    y, y_ref = a * v, a_ref * v_ref

    dx = x - x_ref
    mult_x_to_a = torch.where(dx.abs() > _RESCALE_EPS, (a - a_ref) / dx, a_ref)
    dy, dv = y - y_ref, v - v_ref
    dlog_v = torch.log(v) - torch.log(v_ref)
    dlog_y = dx + dlog_v
    dy_dlog_y = torch.where(dlog_y.abs() > _RESCALE_EPS, dy / dlog_y, y_ref)
    dlog_v_dv = torch.where(dv.abs() > _RESCALE_EPS, dlog_v / dv, v_ref.reciprocal())
    mult_v_to_y = dy_dlog_y * dlog_v_dv
    return g * dy_dlog_y + mult_x_to_a * (g * mult_v_to_y * (-v * v_ref)).sum(
        -1, keepdim=True
    )


def _profile_target(m, s, S, K, rule="product"):
    """As `_counts_target`, for the profile summary of `ProfileWrapper`: the
    sum of c * softmax(c) over the flattened profile, c = p - mean(p) and p the
    profile output.

    rule="product" attributes c * softmax(c) as a softmax and an elementwise
    product, with tangermeme's (>= 1.5) softmax rule and midpoint product rule,
    which is what a wrapper written with `torch.nn.Softmax` and `BilinearOp`
    (as Cherimoya's `ProfileWrapper` is) gets. rule="rescale" applies the
    rescale rule elementwise to the whole product, which is what bpnet-lite's
    `ProfileWrapper` with `_ProfileLogitScaling` registered as `_nonlinear`
    gets (the `clipnet_tf attribute -a profile` behavior before this engine).
    The rescale rule does not fit an operation that couples every position: a
    logit whose change is small gets a secant multiplier that grows without
    bound, so "product" is the default.
    """

    zp = s["zp"]
    p = F.relu(zp)
    c = p - p.mean(dim=-1, keepdim=True)
    sm = torch.softmax(c, dim=-1)
    out = c * sm
    y = out.sum(dim=-1)
    cx, cr = _split(c, S, K)

    if rule == "product":
        smx, smr = _split(sm, S, K)
        # midpoint product rule; the summed output's gradient is 1 everywhere
        g_left = (0.5 * (smx + smr)).expand_as(cr)
        g_right = (0.5 * (cx + cr)).expand_as(cr)
        g = g_left + _softmax_rule(g_right, cx, cr)
    elif rule == "rescale":
        ox, orf = _split(out, S, K)
        # Where the input barely moves, the fallback is the scaling's ordinary
        # gradient at c_x, s_i (1 + c_i - sum_j c_j s_j). (tangermeme 1.4 also
        # hooks the scaling's inner Softmax, which can change this fallback; it
        # applies only where |c_x - c_r| < 1e-6.)
        delta = cx - cr
        small = delta.abs() < _RESCALE_EPS
        smx = torch.softmax(cx, dim=-1)
        grad = smx * (1 + cx - (cx * smx).sum(dim=-1, keepdim=True))
        g = torch.where(
            small, grad.expand_as(cr), (ox - orf) / torch.where(small, 1.0, delta)
        )
    else:
        raise ValueError("profile_rule must be 'product' or 'rescale'")

    g = g - g.mean(dim=-1, keepdim=True)  # transpose of the mean-centering
    zx, zr = _split(zp, S, K)
    g = _relu_rescale(g, zx, zr, _bn_scale(m.pbnorm))
    g = g @ m.plinear.weight  # (S, K, Lo * C), position-major
    Lo, C = s["q"].shape[-1], s["q"].shape[1]
    return y, g.reshape(S, K, Lo, C).transpose(-1, -2)


# -------------------------------------------------------------- the backward


def _backward(m, s, G, S, K, in_scale, length, mode="script"):
    """From the multipliers G (S, K, C, Lo) at the final pool's output to the
    multipliers at the one-hot input, (S * K, 4, length), with the rules fused
    as `mode` says (see `_fused`)."""

    relu, elu, pool = (_fused(f, mode) for f in (_relu_rescale, _elu_rescale, _maxpool_rule))

    def pair(name):
        return _split(s[name], S, K)

    def col(bn):
        return _bn_scale(bn)[:, None]

    hx, hr = pair("h")
    sx, sr = pair("sel2")
    G = pool(G, hx, hr, sx, sr)
    del hx, hr

    for i in reversed(range(m.n_layers)):
        ax, ar = _split(s["s"][i], S, K)
        G = relu(G, ax, ar, col(m.skipbnorms[i]))
        cx, cr = _split(s["c"][i], S, K)
        Gc = relu(G, cx, cr, col(m.rbnorms[i]))
        conv = m.rconvs[i]
        Gc = torch.nn.grad.conv1d_input(
            (S * K, conv.in_channels, Gc.shape[-1]),
            conv.weight,
            Gc.reshape(S * K, *Gc.shape[2:]),
            padding=_same_pad(m.rpads[i]),
            dilation=conv.dilation[0],
        )
        G = G + Gc.reshape(G.shape)

    G = torch.einsum("skol,oi->skil", G, m.rconv.weight[:, :, 0])

    ax, ar = pair("a1")
    sx, sr = pair("sel1")
    G = pool(G, F.relu(ax), F.relu(ar), sx, sr)
    G = relu(G, ax, ar, col(m.sbnorm))
    G = torch.nn.grad.conv1d_input(
        (S * K, m.sconv.in_channels, s["sel0"].shape[-1]),
        m.sconv.weight,
        G.reshape(S * K, *G.shape[2:]),
    ).reshape(S, K, m.sconv.in_channels, -1)

    ax, ar = pair("a0")
    sx, sr = pair("sel0")
    G = pool(G, F.elu(ax), F.elu(ar), sx, sr)
    G = elu(G, ax, ar, col(m.ibn))
    G = torch.nn.grad.conv1d_input(
        (S * K, 4, length), m.iconv.weight, G.reshape(S * K, *G.shape[2:])
    )
    return G * (_bn_scale(m.ibnorm)[:, None] * in_scale)


# -------------------------------------------------------------- references


def _shuffle_references(X, K, random_state):
    """tangermeme's references: reference k of each sequence is its dinucleotide
    shuffle with seed random_state + k (as `deep_lift_shap` draws them), or
    unseeded shuffles if random_state is None. (N, K, 4, L)."""

    from tangermeme.ersatz import dinucleotide_shuffle

    if random_state is None:
        return dinucleotide_shuffle(X, n=K)
    rows = X.repeat_interleave(K, dim=0)
    seeds = [random_state + k for _ in range(len(X)) for k in range(K)]
    try:  # tangermeme >= 1.5 takes one seed per sequence
        refs = dinucleotide_shuffle(rows, n=1, random_state=seeds)[:, 0]
    except (TypeError, ValueError):
        refs = torch.cat(
            [
                dinucleotide_shuffle(rows[i : i + 1], n=1, random_state=seed)[:, 0]
                for i, seed in enumerate(seeds)
            ]
        )
    return refs.reshape(len(X), K, *X.shape[1:])


# ------------------------------------------------------------------ driver


def _attribute_step(m, target, X, R, in_scale, profile_rule="product", mode="script"):
    """Hypothetical attributions (S, 4, L) averaged over references, deltas
    (S, K), for sequences X (S, 4, L) and their references R (S, K, 4, L)."""

    S, K = R.shape[:2]
    L = X.shape[-1]
    rows = torch.cat([X, R.reshape(S * K, *X.shape[1:])]) * in_scale
    s = _forward(m, rows)
    if target == "counts":
        y, G = _counts_target(m, s, S, K)
    else:
        y, G = _profile_target(m, s, S, K, profile_rule)
    M = _backward(m, s, G, S, K, in_scale, L, mode).reshape(S, K, 4, L)
    del s, G
    yx, yr = y[:S], y[S:].reshape(S, K)
    Xs = X.unsqueeze(1)
    delta = ((yx[:, None] - yr) - ((Xs - R) * M).sum(dim=(-2, -1))).abs()
    hyp = M - (R * M).sum(dim=-2, keepdim=True)
    return hyp.mean(dim=1), delta


@torch.no_grad()
def deep_lift_shap(
    model,
    X,
    target="counts",
    references=None,
    n_shuffles=20,
    random_state=None,
    batch_size=None,
    hypothetical=False,
    warning_threshold=1e-3,
    return_deltas=False,
    device=None,
    verbose=False,
    profile_rule="product",
    compile=None,
):
    """DeepLIFT/SHAP attributions of a CLIPNET_TF model's counts or profile.

    Equal, up to floating-point rounding, to tangermeme's `deep_lift_shap` on
    `CountWrapper(model)` (target="counts"; a counts-head-only model is
    attributed directly) or on the profile summary of `ProfileWrapper(model)`
    (target="profile"; see `profile_rule`), with the same references for a
    given `random_state`. See the module docstring for how it differs.

    Parameters
    ----------
    model: CLIPNET_TF or TwoHotToOneHot
        The model. It is run in eval mode, as tangermeme runs it, and is not
        moved or changed (a copy is used if it is not on `device`).
    X: torch.Tensor, shape=(N, 4, L)
        The sequences to attribute, in the encoding `model` takes (one-hot for
        a TwoHotToOneHot-wrapped model).
    target: "counts" or "profile"
        Which head to attribute. Default "counts".
    references: torch.Tensor, shape=(N, K, 4, L), optional
        The references. If None, `n_shuffles` dinucleotide shuffles of each
        sequence, reference k drawn with seed `random_state + k` as tangermeme
        draws them.
    n_shuffles: int
        References per sequence when `references` is None. Default 20.
    random_state: int, optional
        The references' seed. Default None (unseeded).
    batch_size: int, optional
        Sequences per step (each with all of its references). Default: 8 on
        the CPU; on CUDA, enough to saturate the GPU (16 on a 28-SM TITAN X,
        scaled by SM count) if they fit in half the free memory (~70 MiB a
        sequence at 20 shuffles), at most 64. A step that runs out of memory
        is retried at half the size.
    hypothetical: bool
        Return the attributions of every base at every position instead of
        multiplying them by X. Default False.
    warning_threshold: float or None
        Warn once if a convergence delta exceeds it. Default 1e-3.
    return_deltas: bool
        Also return the convergence deltas
        |(y(x) - y(r)) - sum((x - r) * multipliers)|, (N, K).
    device: str or torch.device, optional
        Where to compute. Default: CUDA if available, else the CPU.
    verbose: bool
        Show a progress bar. Default False.
    profile_rule: "product" or "rescale"
        The DeepLIFT rule for the profile summary sum(c * softmax(c)), see
        `_profile_target`. "product" (default) matches tangermeme >= 1.5 on a
        wrapper written with `torch.nn.Softmax` and `BilinearOp`; "rescale"
        matches bpnet-lite's `ProfileWrapper` with `_ProfileLogitScaling`
        registered as `_nonlinear`, the previous `clipnet_tf attribute -a
        profile` behavior.
    compile: bool or None
        How to fuse the DeepLIFT rules (memory-bound elementwise work, ~3x
        faster fused). None (default) uses torch.compile where its Triton
        backend runs (CUDA, compute capability >= 7.0, triton installed, torch
        >= 2.6) and TorchScript elsewhere (whose fuser also runs on older GPUs
        such as Pascal). True forces torch.compile (on the CPU, Inductor's C++
        backend), False TorchScript. torch.compile compiles once per distinct
        shape on the first call, which takes seconds to a minute.

    Returns
    -------
    attributions: torch.Tensor, shape=(N, 4, L), on the CPU
    deltas: torch.Tensor, shape=(N, K), on the CPU, if `return_deltas`
    """

    m, in_scale = _unwrap(model)
    with _eval_mode(m):
        return _deep_lift_shap(
            m, in_scale, X, target, references, n_shuffles, random_state,
            batch_size, hypothetical, warning_threshold, return_deltas, device,
            verbose, profile_rule, compile,
        )


def _deep_lift_shap(m, in_scale, X, target, references, n_shuffles, random_state,
                    batch_size, hypothetical, warning_threshold, return_deltas,
                    device, verbose, profile_rule, compile):
    from tqdm import tqdm

    if target not in ("counts", "profile"):
        raise ValueError("target must be 'counts' or 'profile'")
    if profile_rule not in ("product", "rescale"):
        raise ValueError("profile_rule must be 'product' or 'rescale'")
    if target == "profile" and m.counts_head_only:
        raise ValueError("a counts-head-only model has no profile to attribute")
    if X.ndim != 3 or X.shape[1] != 4:
        raise ValueError(f"X must have shape (N, 4, L), got {tuple(X.shape)}")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    dtype = next(m.parameters()).dtype
    param_device = next(m.parameters()).device
    if param_device != device:
        import copy

        m = copy.deepcopy(m).to(device)

    mode = _resolve_compile(compile, device)
    N, _, L = X.shape
    K = n_shuffles if references is None else references.shape[1]
    if references is not None and (
        references.ndim != 4 or references.shape[0] != N or references.shape[2:] != X.shape[1:]
    ):
        raise ValueError(
            f"references must have shape ({N}, K, 4, {L}), got {tuple(references.shape)}"
        )

    S = batch_size or _default_batch_size(m, K, L, device, dtype)
    attr = torch.empty(N, 4, L, dtype=dtype)
    deltas = torch.empty(N, K, dtype=dtype)
    progress = tqdm(total=N, disable=not verbose, unit="seq")

    def draw(i, j):
        if references is None:
            return _shuffle_references(X[i:j].to(dtype), K, random_state)
        return references[i:j]

    i, Rb = 0, draw(0, min(S, N))
    while i < N:
        j = min(i + S, N)
        if len(Rb) != j - i:  # the step size changed after running out of memory
            Rb = draw(i, j)
        Xb, Rp = X[i:j], Rb
        if j - i < S and N > S:
            # Pad a short last step to S rows by repeating its last sequence, so
            # that every step has the same shapes (the fused rules specialize on
            # them, and recompiling costs seconds); the padding is dropped below.
            pad = S - (j - i)
            Xb = torch.cat([Xb, Xb[-1:].expand(pad, -1, -1)])
            Rp = torch.cat([Rb, Rb[-1:].expand(pad, -1, -1, -1)])
        try:
            with _recompile_limit(mode):
                hyp, d = _attribute_step(
                    m, target, Xb.to(device, dtype), Rp.to(device, dtype), in_scale,
                    profile_rule, mode,
                )
        except torch.cuda.OutOfMemoryError:
            if S == 1:
                raise
            S = max(1, S // 2)
            torch.cuda.empty_cache()
            warnings.warn(f"out of memory; retrying with batch_size={S}", stacklevel=2)
            continue
        # The step runs asynchronously on CUDA: draw the next step's references
        # on the CPU before waiting for its results.
        if j < N:
            Rb = draw(j, min(j + S, N))
        attr[i:j] = hyp[: j - i].cpu()
        deltas[i:j] = d[: j - i].cpu()
        progress.update(j - i)
        i = j
    progress.close()

    if warning_threshold is not None and bool((deltas > warning_threshold).any()):
        warnings.warn(
            f"Convergence deltas too high: {int((deltas > warning_threshold).sum())} "
            f"of {deltas.numel()} pairs > {warning_threshold:g}, the largest "
            f"{float(deltas.max()):.4g}",
            RuntimeWarning,
            stacklevel=2,
        )
    if not hypothetical:
        attr = attr * X.cpu().to(dtype)
    return (attr, deltas) if return_deltas else attr


_ROWS_PER_SM = 12
"""Joint rows per step (S * (K + 1)) per streaming multiprocessor at which a
step saturates the GPU. Measured on a TITAN X (Pascal, 28 SMs), 1024 sequences
x 20 shuffles: 8 / 16 / 32 / 64 sequences a step took 3.96 / 3.58 / 3.54 /
3.60 s at 0.57 / 1.10 / 2.16 / 4.27 GiB peak, so 16 sequences (336 rows) a
step saturate it. Larger steps cost memory (~70 MiB a sequence at 20 shuffles)
without being faster. Scaling by SM count is a heuristic for larger GPUs."""


def _default_batch_size(m, K, L, device, dtype):
    """Sequences per step: 8 on the CPU. On CUDA, enough to saturate the GPU
    (`_ROWS_PER_SM`), but no more than an estimate of the saved activations
    (and as much again for the backward's transients, ~70 MiB a sequence at 20
    shuffles) says fits in half the available memory; between 1 and 64."""

    if device.type != "cuda":
        return 8
    free, _ = torch.cuda.mem_get_info(device)
    # memory this process's caching allocator holds but isn't using is free too
    free += torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device)
    itemsize = torch.tensor([], dtype=dtype).element_size()
    # saved per row: a0 (C x ~L), a1 (C x ~L/2), 2 per residual layer + h (C x ~L/4)
    C = m.rconv.out_channels
    per_row = C * (L + L // 2 + (2 * m.n_layers + 1) * (L // 4))
    per_seq = 2 * (K + 1) * per_row * itemsize
    n_sm = torch.cuda.get_device_properties(device).multi_processor_count
    saturating = -(-_ROWS_PER_SM * n_sm // (K + 1))
    return int(max(1, min(64, saturating, 0.5 * free // per_seq)))


# ================================================== incremental (fastISM) ISM


def _reference_cache(m, x):
    """A full forward of reference rows x (n, 4, L), already scaled to the
    model's input encoding, keeping every layer that a windowed recompute
    reads: the stem pool's output p0, the input of every residual layer, the
    residual stack's output, and the final pool's output q."""

    e0 = m.irelu(m.ibn(m.iconv(m.ibnorm(m.input(x)))))
    p0 = m.imp(e0)
    h = m.rconv(m.smp(m.srelu(m.sbnorm(m.sconv(p0)))))
    hs = [h]
    for i in range(m.n_layers):
        c = m.rrelus[i](m.rbnorms[i](m.rconvs[i](m.rpads[i](h))))
        h = m.skiprelus[i](m.skipbnorms[i](torch.add(h, c)))
        hs.append(h)
    return {"p0": p0, "h": hs, "q": m.skipmp(h)}


def _splice(ref, a, b, alt, lo, hi, seq):
    """ref[seq, :, a:b], the cached reference (n, C, L) of each alt row's
    sequence, with alt (B, C, hi - lo) written over the positions [lo, hi) that
    fall inside [a, b)."""

    seg = ref[seq, :, a:b]
    c, d = max(lo, a), min(hi, b)
    if c < d:
        seg[..., c - a : d - a] = alt[..., c - lo : d - lo]
    return seg


def _pooled_span(lo, hi, n_out):
    """The output positions of MaxPool1d(2) that [lo, hi) can change."""

    return lo // 2, min(n_out, (hi + 1) // 2)


def _head(m, q, profile=True):
    """The model's heads on the final pool's output q (B, C, Lo); the profile
    (the costliest layer at this point) is None if not `profile`."""

    y_counts = m.crelu(m.cbnorm(m.clinear(m.cavgpool(q).squeeze(-1))))
    if m.counts_head_only or not profile:
        return None, y_counts
    flat = q.permute(0, 2, 1).reshape(q.shape[0], -1)
    return m.prelu(m.pbnorm(m.plinear(flat))), y_counts


def _widen(lo, hi, width, n):
    """[lo, hi) widened to `width` positions inside [0, n) (shifted inward at
    the ends rather than clipped), so that windows have one size wherever the
    edit is: cuDNN sets up each new convolution shape on its first call, which
    takes far longer than the call itself. A wider window is still exact."""

    if width is None or hi - lo >= width:
        return lo, hi
    lo = max(0, min(lo, n - width))
    return lo, lo + width


def _spans(m, L, lo, hi, widths=None):
    """The window of each stage that an edit at input positions [lo, hi) can
    change: the stem pool's output, the second pool's output, each residual
    layer's output and the final pool's output, each widened to `widths`."""

    k0, k1 = m.iconv.kernel_size[0], m.sconv.kernel_size[0]
    n0 = L - k0 + 1  # iconv's output
    n_p0 = n0 // 2  # the stem pool's
    n1 = n_p0 - k1 + 1  # sconv's
    n = n1 // 2  # the second pool's, and every residual layer's
    w = iter(widths or ())

    def widen(span, length):
        return _widen(*span, next(w, None), length)

    # iconv output j reads inputs [j, j + k0), so the edit changes
    # [lo - k0 + 1, hi) of it, and then the pool windows covering that
    spans = [widen(_pooled_span(max(0, lo - k0 + 1), min(n0, hi), n_p0), n_p0)]
    plo, phi = spans[-1]
    spans.append(widen(_pooled_span(max(0, plo - k1 + 1), min(n1, phi), n), n))
    for i in range(m.n_layers):  # output j reads j - d, j, j + d
        d = _same_pad(m.rpads[i])
        lo, hi = spans[-1]
        spans.append(widen((max(0, lo - d), min(n, hi + d)), n))
    spans.append(widen(_pooled_span(*spans[-1], n // 2), n // 2))
    return spans


def _window_forward(m, cache, x_alt, seq, spans, profile=True):
    """The model's outputs (profile or None, counts) for rows x_alt (B, 4, L),
    in the model's input encoding, each equal to the cached reference of its
    sequence `seq` (B,) outside the input positions an edit's `_spans` were
    computed for. Only those windows of each layer are computed; the rest is
    read from `cache`."""

    k0, k1 = m.iconv.kernel_size[0], m.sconv.kernel_size[0]
    (plo, phi), (qlo, qhi) = spans[0], spans[1]

    # Stem: iconv, ELU and pool over the stem pool's window.
    e = m.ibnorm(x_alt[..., 2 * plo : 2 * phi + k0 - 1])
    p0 = m.imp(m.irelu(m.ibn(m.iconv(e))))

    # sconv, ReLU and pool over the second pool's window.
    seg = _splice(cache["p0"], 2 * qlo, 2 * qhi + k1 - 1, p0, plo, phi, seq)
    h = m.rconv(m.smp(m.srelu(m.sbnorm(m.sconv(seg)))))
    lo, hi = qlo, qhi

    # Dilated residual layers, each reading its input (zero-padded at the true
    # sequence ends, as AsymmetricPad pads it) d positions beyond its window.
    n = cache["h"][0].shape[-1]
    for i in range(m.n_layers):
        d = _same_pad(m.rpads[i])
        olo, ohi = spans[2 + i]
        a, b = max(0, olo - d), min(n, ohi + d)
        seg = _splice(cache["h"][i], a, b, h, lo, hi, seq)
        conv_in = F.pad(seg, (a - (olo - d), (ohi + d) - b))
        c = m.rrelus[i](m.rbnorms[i](m.rconvs[i](conv_in)))
        h_in = seg[..., olo - a : ohi - a]
        h = m.skiprelus[i](m.skipbnorms[i](torch.add(h_in, c)))
        lo, hi = olo, ohi

    # Final pool, then the heads at full width (spliced into the cached q).
    mlo, mhi = spans[-1]
    seg = _splice(cache["h"][-1], 2 * mlo, 2 * mhi, h, lo, hi, seq)
    q = _splice(cache["q"], 0, cache["q"].shape[-1], m.skipmp(seg), mlo, mhi, seq)
    return _head(m, q, profile)


@torch.no_grad()
def saturation_mutagenesis(
    model,
    X,
    start=0,
    end=-1,
    output="counts",
    raw_outputs=False,
    hypothetical=False,
    batch_size=1024,
    device=None,
    verbose=False,
):
    """In silico saturation mutagenesis of a CLIPNET_TF model, incrementally.

    Equal, up to floating-point rounding, to tangermeme's
    `saturation_mutagenesis(CountWrapper(model), X, ...)` (output="counts"),
    the same on the profile head (output="profile"), or on the bare model with
    `raw_outputs=True` (output="both"): every single-base substitution in
    [start, end) of each sequence is predicted, the substitutions that
    re-apply the base already present are skipped and filled from the
    reference prediction, as tangermeme does. Instead of running every edited
    sequence through the whole model, each reference is run once with its
    layers cached, and each block of consecutive substituted positions
    recomputes only the window of each layer those positions can change (see
    the module docstring).

    Parameters
    ----------
    model: CLIPNET_TF or TwoHotToOneHot
        The model. Run in eval mode; not moved or changed.
    X: torch.Tensor, shape=(N, 4, L)
        One-hot sequences (cast to int8, as tangermeme does). An edit sets the
        position to the one-hot of a base; a TwoHotToOneHot wrapper turns that
        into the two-hot encoding the model takes.
    start, end: int
        The positions to substitute, [start, end); end=-1 means the end of the
        sequence. Default: every position.
    output: "counts", "profile" or "both"
        Which predictions to return. "both" returns lists [profile, counts],
        like tangermeme on the bare model. Default "counts".
    raw_outputs: bool
        Return (y0, y_hat) predictions instead of attributions. Default False.
    hypothetical: bool
        With raw_outputs=False, return the attribution of every base instead of
        multiplying by X. Default False.
    batch_size: int
        Edited sequences per step, approximately. A step takes the edits at a
        few consecutive positions of up to batch_size / 24 sequences at once
        (fewer positions keep the recomputed windows narrow). Default 1024.
    device: str or torch.device, optional
        Default: CUDA if available, else the CPU.
    verbose: bool
        Show a progress bar over groups of sequences. Default False.

    Returns
    -------
    attr: torch.Tensor, shape=(N, 4, end - start)
        As tangermeme: each substitution's change in prediction, centered across
        the alphabet at each position and averaged over the output's entries.
    -- or, if raw_outputs=True --
    y0: torch.Tensor (or list of [profile, counts] for output="both")
        The reference predictions, (N, n_outputs).
    y_hat: torch.Tensor (or list), shape=(N, 4, end - start, n_outputs)
        The predictions for each substitution.
    """

    m, in_scale = _unwrap(model)
    with _eval_mode(m):
        return _saturation_mutagenesis(
            m, in_scale, X, start, end, output, raw_outputs, hypothetical,
            batch_size, device, verbose,
        )


def _saturation_mutagenesis(m, in_scale, X, start, end, output, raw_outputs,
                            hypothetical, batch_size, device, verbose):
    from tqdm import trange

    if output not in ("counts", "profile", "both"):
        raise ValueError("output must be 'counts', 'profile' or 'both'")
    if output != "counts" and m.counts_head_only:
        raise ValueError("a counts-head-only model has only counts to return")
    if output == "both" and not raw_outputs:
        raise ValueError("output='both' needs raw_outputs=True, as in tangermeme")
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    if next(m.parameters()).device != device:
        import copy

        m = copy.deepcopy(m).to(device)
    dtype = next(m.parameters()).dtype

    X = X.to(torch.int8).cpu()
    N, A, L = X.shape
    if end < 0:
        end = L + 1 + end
    if not 0 <= start < end <= L:
        raise ValueError(f"need 0 <= start < end <= {L}; got start={start}, end={end}")
    W = end - start
    # A step takes the edits at P consecutive positions of G sequences: few
    # positions keep each layer's window narrow, many sequences keep the batch
    # large. G sequences' caches are held at once.
    G = max(1, min(N, batch_size // (8 * (A - 1))))
    P = max(1, min(W, batch_size // ((A - 1) * G)))

    # Every step computes windows of the widths an interior block of P
    # positions needs, and B rows (a block's edits when every position holds a
    # base; a block with an N, which has four, runs at its own size).
    mid = max(0, L // 2 - P // 2)
    widths = [hi - lo for lo, hi in _spans(m, L, mid, min(L, mid + P))]
    B = (A - 1) * P * G

    heads = {"profile": [0], "counts": [1], "both": [0, 1]}[output]
    need_profile = output != "counts"
    y0s = [[] for _ in heads]
    y_hats = [[] for _ in heads]
    for g0 in trange(0, N, G, disable=not verbose):
        Xg = X[g0 : g0 + G]
        xg = Xg.to(device, dtype) * in_scale
        cache = _reference_cache(m, xg)
        ref_out = _head(m, cache["q"], need_profile)
        y0 = [ref_out[h].cpu() for h in heads]  # (n, n_outputs)
        grid = [y[:, None, None].expand(-1, A, W, -1).clone() for y in y0]

        ref = Xg[:, :, start:end]
        identity = (ref == 1) & (ref.sum(dim=1, keepdim=True) == 1)
        Xd = Xg.to(device)
        for p in range(0, W, P):
            seq, chars, pos = torch.where(~identity[:, :, p : p + P])
            if len(seq) == 0:
                continue
            n_rows = len(seq)
            if n_rows < B:
                # Pad to B rows by repeating the last edit (dropped below), so
                # that every step has the same shapes.
                fill = torch.full((B - n_rows,), n_rows - 1)
                seq, chars, pos = (torch.cat([t, t[fill]]) for t in (seq, chars, pos))
            pos = pos + p
            lo = p + start  # the block's nominal span, for a constant window size
            spans = _spans(m, L, lo, min(lo + P, end), widths)
            seq_d, chars_d, col = seq.to(device), chars.to(device), (pos + start).to(device)
            rows = torch.arange(len(seq), device=device)
            x_alt = Xd[seq_d]
            x_alt[rows, :, col] = 0
            x_alt[rows, chars_d, col] = 1
            out = _window_forward(
                m, cache, x_alt.to(dtype) * in_scale, seq_d, spans, need_profile
            )
            seq, chars, pos = seq[:n_rows], chars[:n_rows], pos[:n_rows]
            for gr, h in zip(grid, heads):
                gr[seq, chars, pos] = out[h][:n_rows].cpu()
        for k in range(len(heads)):
            y0s[k].append(y0[k])
            y_hats[k].append(grid[k])

    y0 = [torch.cat(y) for y in y0s]
    y_hat = [torch.cat(y) for y in y_hats]
    if output != "both":
        y0, y_hat = y0[0], y_hat[0]
    if raw_outputs:
        return y0, y_hat
    attr = y_hat - y0[:, None, None]
    attr = attr - attr.mean(dim=1, keepdim=True)
    attr = attr.mean(dim=-1)
    return attr if hypothetical else X[:, :, start:end] * attr
