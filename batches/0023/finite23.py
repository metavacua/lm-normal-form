# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Batch 0023, the finite instance. A Llama-style decoder of batch 0015 (model.py) so small that its function can be written down whole: vocabulary 5, contexts of every length 1 to 4
# (5 + 25 + 125 + 625 = 780 contexts, each giving the next-token logits), hidden 4, two heads in one key/value group, head dimension 2 (one rotary plane), 4 feed-forward units, two layers, an
# untied head, the norm weights frozen at 1 (so they are not parameters): 224 trainable parameters. "The function" is the 340 x 4 array of logits at the last position of every context: the whole
# input space of the model at these lengths, so that two parameter vectors are the same model exactly when they give the same array.
# What is here: the forward pass for a whole batch of parameter vectors at once (a leading axis B on every weight), in numpy and valid on complex numbers, so that the Jacobian of the function with
# respect to the parameters is the imaginary part of one pass per parameter divided by h = 1e-30 (the complex step: no subtraction, no cancellation); and four optional variants of the position structure
# for the experiments of covariance (batches 0023 `cov23.py`): `posdep` (the queries and keys have their own projection at every position, and no rotary embedding), `theta` (the frequencies of
# the rotary planes are parameters), `delta` (the position of token t is the proper time tau_t = sum over i < t of exp(b + w . x_i), x_i the normalized input of the layer, in place of t), and `theta+delta`.
# The forward pass is checked against model.py (selftest, run first by every script).
import itertools, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "0015"))
from model import Arch, shapes, forward as ref_forward, _norm      # noqa: E402
from ops import FloatOps                                           # noqa: E402

ARCH = Arch(d=4, n_heads=2, n_kv=1, hd=2, d_ff=4, n_layers=2, vocab=5, tied=False)
EPS = 1e-6
TMAX = 4
H = 1e-30


def ops_of(eps=EPS):
    return FloatOps(eps=eps)


def contexts(a=ARCH, tmax=TMAX):
    """One array (V^t, t) of token ids per length t: every context of that length."""
    return [np.array(list(itertools.product(range(a.vocab), repeat=t)), dtype=np.int64) for t in range(1, tmax + 1)]


def make_spec(a=ARCH, variant="base", tmax=TMAX):
    """key -> shape of the trainable parameters, in a fixed order (the norm weights are frozen at 1 and are not here)."""
    spec = {k: s for k, s in shapes(a).items() if not (k.endswith("norm") or k == "norm_f")}
    out = {}
    for k, s in spec.items():
        out[k] = s
        if variant == "posdep" and k.endswith(".wq"):
            del out[k]
            out[k + "_t"] = (tmax, a.n_heads * a.hd, a.d)
        if variant == "posdep" and k.endswith(".wk"):
            del out[k]
            out[k + "_t"] = (tmax, a.n_kv * a.hd, a.d)
    for i in range(a.n_layers):
        if "theta" in variant:
            out[f"l{i}.theta"] = (a.hd // 2,)
        if "delta" in variant:
            out[f"l{i}.dlt_b"] = (1,)
            out[f"l{i}.dlt_w"] = (a.d,)
    return out


def n_of(spec):
    return int(sum(int(np.prod(s)) for s in spec.values()))


def init_theta(spec, rng, a=ARCH, ops=None):
    """The initialization of batch 0015's lm_optim.py (matrices normal with standard deviation 0.4 / sqrt(in)); the frequencies near the standard ones, the proper-time parameters small."""
    ops = ops or ops_of()
    h2 = a.hd // 2
    std = ops.base ** (-2.0 * np.arange(h2) / a.hd)
    parts = []
    for k, s in spec.items():
        if k.endswith(".theta"):
            parts.append(std * np.exp(0.1 * rng.standard_normal(h2)))
        elif k.endswith(".dlt_b"):
            parts.append(0.2 * rng.standard_normal(1))
        elif k.endswith(".dlt_w"):
            parts.append(0.5 * rng.standard_normal(s) / np.sqrt(a.d))
        elif len(s) == 3:                                       # position-dependent projections (tmax, out, in)
            parts.append((rng.standard_normal(s) * 0.4 / np.sqrt(s[-1])).reshape(-1))
        else:
            parts.append((rng.standard_normal(s) * 0.4 / np.sqrt(s[-1])).reshape(-1))
    return np.concatenate(parts)


def unpack(theta_b, spec):
    """theta_b (B, n) -> {key: view (B, *shape)}."""
    B = theta_b.shape[0]
    P, o = {}, 0
    for k, s in spec.items():
        n = int(np.prod(s))
        P[k] = theta_b[:, o:o + n].reshape((B,) + tuple(s))
        o += n
    return P


def to_dict(theta, spec, a=ARCH):
    """One parameter vector as the dict of model.py (the frozen norm weights added, equal to 1): for the gauge functions of batch 0015 and for the reference forward pass."""
    P = {k: v[0] for k, v in unpack(theta[None, :], spec).items()}
    P["norm_f"] = np.ones(a.d)
    for i in range(a.n_layers):
        P[f"l{i}.attn_norm"], P[f"l{i}.mlp_norm"] = np.ones(a.d), np.ones(a.d)
    return P


def from_dict(P, spec):
    return np.concatenate([np.asarray(P[k]).reshape(-1) for k in spec])


def _rope_b(x, C, S):
    h2 = x.shape[-1] // 2
    x1, x2 = x[..., :h2], x[..., h2:]
    return np.concatenate([x1 * C - x2 * S, x2 * C + x1 * S], axis=-1)


def _tr(W):
    """(B, out, in) -> (B, 1, in, out), the right operand of x @ W^T for x (B, N, T, in)."""
    return W.transpose(0, 2, 1)[:, None]


def layer_pb(a, P, i, h, ops):
    """Layer i applied to the stream h (B, N, T, d): the same layer as model.forward (attention with the rotary embedding, then the gated block), for B parameter vectors at once."""
    B, N, T, _ = h.shape
    h2 = a.hd // 2
    mask = np.tril(np.ones((T, T)))
    scale = ops.attn_scale(a.hd)
    std = ops.base ** (-2.0 * np.arange(h2) / a.hd)
    x = _norm(a, ops, h, 1.0)
    posdep = f"l{i}.wq_t" in P
    if posdep:
        q = np.einsum("btod,bntd->bnto", P[f"l{i}.wq_t"][:, :T], x)
        k = np.einsum("btod,bntd->bnto", P[f"l{i}.wk_t"][:, :T], x)
    else:
        q = x @ _tr(P[f"l{i}.wq"])
        k = x @ _tr(P[f"l{i}.wk"])
    v = x @ _tr(P[f"l{i}.wv"])
    q = q.reshape(B, N, T, a.n_heads, a.hd).transpose(0, 1, 3, 2, 4)
    k = k.reshape(B, N, T, a.n_kv, a.hd).transpose(0, 1, 3, 2, 4)
    v = v.reshape(B, N, T, a.n_kv, a.hd).transpose(0, 1, 3, 2, 4)
    if not posdep:
        theta = P[f"l{i}.theta"][:, None, None, :] if f"l{i}.theta" in P else std[None, None, None, :]          # (B or 1, 1, 1, h2)
        if f"l{i}.dlt_b" in P:
            delta = np.exp(P[f"l{i}.dlt_b"][:, None, :] + (x @ P[f"l{i}.dlt_w"][:, None, :, None])[..., 0])        # (B, N, T)
            tau = np.cumsum(delta, axis=2) - delta                                                                  # tau_t = sum over i < t of delta_i
        else:
            tau = np.arange(T, dtype=np.float64)[None, None, :]                                                    # (1, 1, T)
        ang = theta * tau[..., None]                                                                               # (B or 1, N or 1, T, h2)
        C, S = np.cos(ang)[:, :, None], np.sin(ang)[:, :, None]                                                    # (., ., 1, T, h2)
        q, k = _rope_b(q, C, S), _rope_b(k, C, S)
    k, v = np.repeat(k, a.rep, axis=2), np.repeat(v, a.rep, axis=2)
    s = (q @ k.transpose(0, 1, 2, 4, 3)) * scale
    w = np.exp(s) * mask
    w = w / w.sum(-1, keepdims=True)
    o = (w @ v).transpose(0, 1, 3, 2, 4).reshape(B, N, T, a.n_heads * a.hd)
    h = h + o @ _tr(P[f"l{i}.wo"])
    x = _norm(a, ops, h, 1.0)
    m = ops.act_silu(x @ _tr(P[f"l{i}.wg"])) * (x @ _tr(P[f"l{i}.wu"]))
    return h + m @ _tr(P[f"l{i}.wd"])


def embed_pb(P, toks):
    return P["embed"][:, toks]


def readout_pb(a, P, h, ops):
    return _norm(a, ops, h, 1.0) @ _tr(P["lm_head"])


def forward_pb(a, P, toks, ops):
    """Logits (B, N, T, V) of the N token sequences `toks` (N, T) under B parameter vectors at once. Every entry of P has a leading axis B."""
    h = embed_pb(P, toks)
    for i in range(a.n_layers):
        h = layer_pb(a, P, i, h, ops)
    return readout_pb(a, P, h, ops)


def function(theta_b, spec, groups, a=ARCH, ops=None):
    """The function of B parameter vectors (B, n): the logits at the last position of every context, (B, n_ctx, V) (complex if theta_b is)."""
    ops = ops or ops_of()
    P = unpack(theta_b, spec)
    return np.concatenate([forward_pb(a, P, toks, ops)[:, :, -1, :] for toks in groups], axis=1)


def logits_of(theta, spec, groups, a=ARCH, ops=None):
    return function(np.asarray(theta, dtype=np.float64)[None, :], spec, groups, a, ops)[0].real


def jacobian(theta, spec, groups, a=ARCH, ops=None, chunk=48):
    """d(function) / d(parameters), (n_ctx * V, n): the complex step."""
    ops = ops or ops_of()
    n = theta.size
    J = None
    for s in range(0, n, chunk):
        idx = np.arange(s, min(n, s + chunk))
        TH = np.broadcast_to(theta.astype(np.complex128), (len(idx), n)).copy()
        TH[np.arange(len(idx)), idx] += 1j * H
        z = function(TH, spec, groups, a, ops)
        col = z.imag.reshape(len(idx), -1) / H
        if J is None:
            J = np.empty((col.shape[1], n))
        J[:, idx] = col.T
    return J


def softmax(z):
    e = np.exp(z - z.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def selftest(seed=0):
    """forward_pb (B = 2, one of them perturbed) against model.forward on every context of lengths 1 to 4, to 1e-12; and the Jacobian against a central difference on a few parameters."""
    a, ops = ARCH, ops_of()
    spec, groups = make_spec(a), contexts(a)
    rng = np.random.default_rng(seed)
    th = init_theta(spec, rng, a, ops)
    P = to_dict(th, spec, a)
    z = logits_of(th, spec, groups, a, ops)
    worst, row = 0.0, 0
    for toks in groups:
        for s in toks[:: max(1, len(toks) // 17)]:
            ref = ref_forward(a, P, [int(t) for t in s], ops)[-1]
            idx = row + int(np.flatnonzero((toks == s).all(axis=1))[0])
            worst = max(worst, float(np.max(np.abs(z[idx] - ref))))
        row += len(toks)
    assert worst < 1e-12, f"forward_pb differs from model.forward by {worst:.2e}"
    J = jacobian(th, spec, groups, a, ops)
    for j in (0, 17, 101, th.size - 1):
        e = np.zeros_like(th)
        e[j] = 1e-6
        fd = (logits_of(th + e, spec, groups, a, ops) - logits_of(th - e, spec, groups, a, ops)).reshape(-1) / 2e-6
        assert np.max(np.abs(fd - J[:, j])) < 1e-6 * max(1.0, np.max(np.abs(fd))), f"Jacobian column {j}"
    return worst


if __name__ == "__main__":
    print("selftest: max |forward_pb - model.forward| =", selftest())
