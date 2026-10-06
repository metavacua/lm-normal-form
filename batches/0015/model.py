# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# One definition of a Llama-style decoder-only Transformer, generic over the arithmetic (ops.py). It is the definition that the claims of
# batch 0015 are about: pre-norm residual blocks; grouped-query attention with optional rotary embeddings (rotate-half pairing, the
# pairing of Hugging Face's Llama) and an optional causal mask; a gated (SwiGLU-shaped) feed-forward block; a final norm; a head that is the
# embedding or its own matrix. No biases. The weights of a layer are kept in the shapes of Hugging Face's Llama (out x in):
#   embed (V, d)   lm_head (V, d) unless tied   norm_f (d)
#   l{i}.attn_norm (d)  .wq (n_heads*hd, d)  .wk (n_kv*hd, d)  .wv (n_kv*hd, d)  .wo (d, n_heads*hd)
#   l{i}.mlp_norm (d)   .wg (d_ff, d)  .wu (d_ff, d)  .wd (d, d_ff)
# Query head h reads key/value group h // (n_heads // n_kv), as in Hugging Face's repeat_kv.
# Run over float64 with the real nonlinearities the forward pass is checked against Hugging Face (conform.py), which ties the claims that
# are proved over F_p to the model that is run.
from dataclasses import dataclass, replace
import numpy as np


@dataclass(frozen=True)
class Arch:
    d: int
    n_heads: int
    n_kv: int
    hd: int
    d_ff: int
    n_layers: int
    vocab: int
    tied: bool = False
    pos: str = "rope"        # "rope" or "none"
    causal: bool = True
    norm: str = "rms"        # "rms", or "ln" (the mean is subtracted first; no bias)

    @property
    def rep(self):
        return self.n_heads // self.n_kv

    def with_(self, **kw):
        return replace(self, **kw)


def shapes(a):
    s = {"embed": (a.vocab, a.d), "norm_f": (a.d,)}
    if not a.tied:
        s["lm_head"] = (a.vocab, a.d)
    for i in range(a.n_layers):
        s.update({f"l{i}.attn_norm": (a.d,), f"l{i}.wq": (a.n_heads * a.hd, a.d), f"l{i}.wk": (a.n_kv * a.hd, a.d), f"l{i}.wv": (a.n_kv * a.hd, a.d),
                  f"l{i}.wo": (a.d, a.n_heads * a.hd), f"l{i}.mlp_norm": (a.d,), f"l{i}.wg": (a.d_ff, a.d), f"l{i}.wu": (a.d_ff, a.d), f"l{i}.wd": (a.d, a.d_ff)})
    return s


def n_params(a):
    return int(sum(int(np.prod(v)) for v in shapes(a).values()))


def init_params(a, ops, rng):
    P = {}
    for k, shp in shapes(a).items():
        if k.endswith("norm") or k == "norm_f":
            P[k] = ops.rand_nonzero(rng, shp)
        else:
            P[k] = ops.rand(rng, shp) if ops.exact else rng.standard_normal(shp) / np.sqrt(shp[-1])
    return P


def head_matrix(a, P):
    return P["embed"] if a.tied else P["lm_head"]


def _norm(a, ops, h, gamma):
    d_inv = ops.const_inv(a.d)
    x = h
    if a.norm == "ln":
        x = ops.red(x - ops.red(x.sum(axis=-1, keepdims=True) * d_inv))
    msq = ops.red(ops.red(x * x).sum(axis=-1, keepdims=True) * d_inv)
    return ops.red(ops.red(x * ops.norm_scale(msq)) * gamma)


def _rope(ops, x, C, S):
    h2 = x.shape[-1] // 2
    x1, x2 = x[..., :h2], x[..., h2:]
    return np.concatenate([ops.red(x1 * C - x2 * S), ops.red(x2 * C + x1 * S)], axis=-1)


def forward(a, P, tokens, ops, trace=None):
    """Logits (T, vocab) of the token sequence `tokens`. `trace`, if a dict, receives the residual stream after every block (key i) and the KV cache of every layer (keys ("k", i), ("v", i))."""
    T = len(tokens)
    h = P["embed"][np.asarray(tokens)]
    C = S = None
    if a.pos == "rope":
        C, S = ops.rope_tables(T, a.hd)
    mask = np.tril(np.ones((T, T), dtype=np.int64)) if a.causal else np.ones((T, T), dtype=np.int64)
    mask = mask.astype(object) if ops.exact else mask.astype(np.float64)
    scale = ops.const(ops.attn_scale(a.hd))
    for i in range(a.n_layers):
        x = _norm(a, ops, h, P[f"l{i}.attn_norm"])
        q = ops.mm(x, P[f"l{i}.wq"].T).reshape(T, a.n_heads, a.hd).transpose(1, 0, 2)
        k = ops.mm(x, P[f"l{i}.wk"].T).reshape(T, a.n_kv, a.hd).transpose(1, 0, 2)
        v = ops.mm(x, P[f"l{i}.wv"].T).reshape(T, a.n_kv, a.hd).transpose(1, 0, 2)
        if a.pos == "rope":
            q, k = _rope(ops, q, C, S), _rope(ops, k, C, S)
        if trace is not None:
            trace[("k", i)], trace[("v", i)] = k, v          # the KV cache of the layer: keys after the rotary embedding, and values, (n_kv, T, hd)
        k, v = np.repeat(k, a.rep, axis=0), np.repeat(v, a.rep, axis=0)
        s = ops.red(ops.mm(q, k.transpose(0, 2, 1)) * scale)
        w = ops.red(ops.act_exp(s) * mask)
        w = ops.red(w * ops.inv(ops.red(w.sum(axis=-1, keepdims=True))))
        o = ops.mm(w, v).transpose(1, 0, 2).reshape(T, a.n_heads * a.hd)
        h = ops.red(h + ops.mm(o, P[f"l{i}.wo"].T))
        x = _norm(a, ops, h, P[f"l{i}.mlp_norm"])
        m = ops.red(ops.act_silu(ops.mm(x, P[f"l{i}.wg"].T)) * ops.mm(x, P[f"l{i}.wu"].T))
        h = ops.red(h + ops.mm(m, P[f"l{i}.wd"].T))
        if trace is not None:
            trace[i] = h
    return ops.mm(_norm(a, ops, h, P["norm_f"]), head_matrix(a, P).T)


def fold_norms(a, P, ops):
    """The same function with every norm weight equal to 1: each norm weight is multiplied into the columns of the matrices that read its
    output (q, k, v; gate, up; the head, which is made a matrix of its own if it was the embedding). Returns (arch, parameters)."""
    Q = dict(P)
    one = lambda n: np.array([1] * n, dtype=object) if ops.exact else np.ones(n)
    for i in range(a.n_layers):
        g = P[f"l{i}.attn_norm"]
        for n in ("wq", "wk", "wv"):
            Q[f"l{i}.{n}"] = ops.red(P[f"l{i}.{n}"] * g[None, :])
        Q[f"l{i}.attn_norm"] = one(a.d)
        g = P[f"l{i}.mlp_norm"]
        for n in ("wg", "wu"):
            Q[f"l{i}.{n}"] = ops.red(P[f"l{i}.{n}"] * g[None, :])
        Q[f"l{i}.mlp_norm"] = one(a.d)
    Q["lm_head"] = ops.red(head_matrix(a, P) * P["norm_f"][None, :])
    Q["norm_f"] = one(a.d)
    return a.with_(tied=False), Q
