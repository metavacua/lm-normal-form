# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The claims of batch 0015 about the algebra of a Llama-style Transformer, each as an executable statement. A claim says, for an architecture
# (the scope) and a family of transformations, whether the function of model.py is unchanged by them, or changed:
#   expect = True    the identity f(g . theta) = f(theta) holds for generic theta and g (a theorem; the run is a randomized proof over F_p and a check in float64)
#   expect = False   it does not (a counterexample to something that is pasted, or to an over-general version of a theorem)
# `build(ops, rng, arch)` returns the two arrays of results that the claim says are equal. Over F_p they are equal exactly or they differ; a claim
# that expects equality passes only if every trial gives equality, and a claim that expects a difference passes only if no trial gives equality.
# Over float64, with the real nonlinearities, equality is to a tolerance and a difference means a relative difference above 1e-6.
import json, sys
from dataclasses import dataclass
from typing import Callable
import numpy as np
import gauge as G
from model import Arch, init_params, forward, fold_norms

BASE = Arch(d=8, n_heads=4, n_kv=2, hd=4, d_ff=12, n_layers=2, vocab=7, tied=False, pos="rope", causal=True, norm="rms")


@dataclass
class Claim:
    id: str
    family: str
    text: str
    scope: dict
    expect: bool
    build: Callable
    source: str = ""        # where the statement comes from: the pasted text (its number), or derived


CLAIMS = []


def claim(id, family, text, scope, expect, source=""):
    def deco(fn):
        CLAIMS.append(Claim(id, family, text, scope, expect, fn, source))
        return fn
    return deco


def _tokens(ops, rng, arch, T=5):
    """T distinct tokens (so that a non-trivial permutation of them is a different sequence)."""
    return [int(t) for t in rng.permutation(arch.vocab)[:T]]


def _perm(rng, n):
    return [int(i) for i in rng.permutation(n)]


def _nontrivial_perm(rng, n):
    while True:
        p = _perm(rng, n)
        if p != list(range(n)):
            return p


def _signs(rng, n):
    return [int(s) for s in rng.choice([1, -1], n)]


def _per_layer_per_group(ops, arch, rng, f):
    return [[f() for _ in range(arch.n_kv)] for _ in range(arch.n_layers)]


# ---- T: permutation of the tokens ------------------------------------------------------------------------------------------------------------

def _token_equivariance(ops, rng, arch):
    P = init_params(arch, ops, rng)
    x = _tokens(ops, rng, arch, 6)
    pi = _nontrivial_perm(rng, len(x))
    y = [x[i] for i in pi]
    return forward(arch, P, x, ops)[pi], forward(arch, P, y, ops)


@claim("T1", "T", "Without a causal mask and without positional information, f(x o pi) = f(x) o pi: the Transformer is permutation equivariant in its tokens.",
       {"pos": "none", "causal": False}, True, "pasted 1: 'a vanilla Transformer encoder is fundamentally permutation equivariant'")
def t1(ops, rng, arch):
    return _token_equivariance(ops, rng, arch)


@claim("T2", "T", "With a causal mask and no positional information, token permutation equivariance fails.", {"pos": "none", "causal": True}, False, "pasted 1, applied to a decoder")
def t2(ops, rng, arch):
    return _token_equivariance(ops, rng, arch)


@claim("T3", "T", "With rotary positional embeddings and no mask, token permutation equivariance fails.", {"pos": "rope", "causal": False}, False, "pasted 1, applied to RoPE")
def t3(ops, rng, arch):
    return _token_equivariance(ops, rng, arch)


@claim("T4", "T", "With a causal mask and rotary positional embeddings (a Llama-style decoder), token permutation equivariance fails.", {"pos": "rope", "causal": True}, False,
       "pasted 1, applied to the models of this repository")
def t4(ops, rng, arch):
    return _token_equivariance(ops, rng, arch)


def _last_position_prefix_invariance(ops, rng, arch):
    P = init_params(arch, ops, rng)
    x = _tokens(ops, rng, arch, 6)
    pre = _nontrivial_perm(rng, len(x) - 1)
    y = [x[i] for i in pre] + [x[-1]]
    return forward(arch, P, x, ops)[-1:], forward(arch, P, y, ops)[-1:]


@claim("T5", "T", "A causal decoder without positional information and with ONE layer: the logits at the last position are invariant under permutations of the earlier tokens.",
       {"pos": "none", "causal": True, "n_layers": 1}, True, "derived: what survives of T2")
def t5(ops, rng, arch):
    return _last_position_prefix_invariance(ops, rng, arch)


@claim("T6", "T", "The same with TWO layers: the invariance of T5 fails (the second layer reads the first layer's prefix-dependent states).",
       {"pos": "none", "causal": True, "n_layers": 2}, False, "derived: the depth at which T5 stops")
def t6(ops, rng, arch):
    return _last_position_prefix_invariance(ops, rng, arch)


# ---- H: the hidden coordinates --------------------------------------------------------------------------------------------------------------

def _compare(ops, rng, arch, make, tokens=None):
    P = init_params(arch, ops, rng)
    a2, P2 = make(arch, P, ops, rng)
    x = tokens or _tokens(ops, rng, arch)
    return forward(arch, P, x, ops), forward(a2, P2, x, ops)


@claim("H1", "H", "A signed permutation of the hidden coordinates, applied to every matrix that reads or writes the residual stream (embedding and head included) and to the norm weights, leaves f unchanged.",
       {}, True, "pasted 1 ('the exact same transformation to the weight matrices'), stated correctly")
def h1(ops, rng, arch):
    return _compare(ops, rng, arch, lambda a, P, o, r: G.residual_signed_perm(a, P, o, _perm(r, a.d), _signs(r, a.d)))


@claim("H1t", "H", "H1 with the head tied to the embedding.", {"tied": True}, True, "pasted 1, SmolLM2-style")
def h1t(ops, rng, arch):
    return h1(ops, rng, arch)


@claim("H2", "H", "The rule as pasted, W_new = P W P^T, applied to the square weight matrices (here q and o) and nothing else, does not leave f unchanged.",
       {"n_heads": 2, "n_kv": 2, "hd": 4}, False, "pasted 1: 'W_new = P_C W P_C^T'")
def h2(ops, rng, arch):
    return _compare(ops, rng, arch, lambda a, P, o, r: G.pasted_rule(a, P, o, _perm(r, a.d)))


@claim("H3", "H", "A signed permutation of the hidden coordinates applied to the layers and the final norm but not to the embedding and the head ('encrypting the body') changes f.",
       {}, False, "pasted 1: 'applying a secret permutation matrix to an LLM's weights ... the model becomes entirely broken'")
def h3(ops, rng, arch):
    return _compare(ops, rng, arch, lambda a, P, o, r: G.body_only_perm(a, P, o, _perm(r, a.d), _signs(r, a.d)))


@claim("H3k", "H", "Applying the permutation of H3 to the embedding and the head as well (the key) restores f exactly: f(body-only permuted, then keyed) = f.",
       {}, True, "pasted 1: 'an authorized user who possesses the permutation key ... perfectly decrypting'")
def h3k(ops, rng, arch):
    def make(a, P, o, r):
        perm, signs = _perm(r, a.d), _signs(r, a.d)
        a2, R = G.body_only_perm(a, P, o, perm, signs)
        a3, full = G.residual_signed_perm(a, P, o, perm, signs)
        for k in ("embed", "lm_head"):
            if k in full:
                R[k] = full[k]
        return a2, R
    return _compare(ops, rng, arch, make)


@claim("H4", "H", "With the norm weights folded into the matrices that read them (the head made its own matrix), any orthogonal Q applied to the residual stream leaves f unchanged.",
       {}, True, "derived: 'computational invariance' (SliceGPT), pasted 4 (change of basis)")
def h4(ops, rng, arch):
    return _compare(ops, rng, arch, lambda a, P, o, r: G.residual_orth(a, P, o, o.rand_orthogonal(a.d, r), fold=True))


@claim("H4c", "H", "In the unfolded parameterization (norm weights kept), with an untied head, an orthogonal Q applied to the stream with the readers conjugated, W -> W diag(gamma) Q diag(gamma)^-1, leaves f unchanged.",
       {}, True, "derived: the action of fold, rotate, unfold; found independently in the research notes")
def h4c(ops, rng, arch):
    return _compare(ops, rng, arch, lambda a, P, o, r: G.residual_orth_conj(a, P, o, o.rand_orthogonal(a.d, r)))


@claim("H6c", "H", "The conjugated action of H4c on a model with a tied head and a learned final norm weight changes f.", {"tied": True}, False,
       "derived: what the tie takes away from H4c")
def h6c(ops, rng, arch):
    return _compare(ops, rng, arch, lambda a, P, o, r: G.residual_orth_conj(a, P, o, o.rand_orthogonal(a.d, r)))


@claim("H5", "H", "Without the fold, a general orthogonal Q applied to the stream changes f.", {}, False, "derived: the necessity of the fold in H4")
def h5(ops, rng, arch):
    return _compare(ops, rng, arch, lambda a, P, o, r: G.residual_orth(a, P, o, o.rand_orthogonal(a.d, r), fold=False))


@claim("H6", "H", "With a tied head and a final norm weight that is not 1, an orthogonal Q applied to the stream (the tied head following) changes f: the final norm cannot be folded without untying.",
       {"tied": True}, False, "derived: what the tie takes away from H4")
def h6(ops, rng, arch):
    def make(a, P, o, r):
        Q = o.rand_orthogonal(a.d, r)
        R = dict(P)
        for i in range(a.n_layers):                      # fold the layer norms only: the final norm weight stays, and the head stays the embedding
            g = P[f"l{i}.attn_norm"]
            for n in ("wq", "wk", "wv"):
                R[f"l{i}.{n}"] = o.red(P[f"l{i}.{n}"] * g[None, :])
            R[f"l{i}.attn_norm"] = np.array([1] * a.d, dtype=object) if o.exact else np.ones(a.d)
            g = P[f"l{i}.mlp_norm"]
            for n in ("wg", "wu"):
                R[f"l{i}.{n}"] = o.red(P[f"l{i}.{n}"] * g[None, :])
            R[f"l{i}.mlp_norm"] = np.array([1] * a.d, dtype=object) if o.exact else np.ones(a.d)
        a2, R2 = G.residual_orth(a, R, o, Q, fold=False)
        return a2, R2
    return _compare(ops, rng, arch, make)


@claim("H6o", "H", "With a tied head and a final norm weight equal to 1 (and the layer norms folded), an orthogonal Q applied to the stream leaves f unchanged.",
       {"tied": True}, True, "derived: H6 with the obstruction removed")
def h6o(ops, rng, arch):
    P = init_params(arch, ops, rng)
    P["norm_f"] = np.array([1] * arch.d, dtype=object) if ops.exact else np.ones(arch.d)
    a1, F1 = fold_norms(arch, P, ops)                  # same function; untied only because the final norm weight is multiplied into the head, which changes nothing here
    R = dict(F1)
    R["embed"] = P["embed"]
    del R["lm_head"]                                   # the head is the embedding again: with norm_f = 1 the folded head is the embedding
    a2, R2 = G.residual_orth(arch, R, ops, ops.rand_orthogonal(arch.d, rng), fold=False)
    x = _tokens(ops, rng, arch)
    return forward(arch, P, x, ops), forward(a2, R2, x, ops)


@claim("H7", "H", "With LayerNorm (the mean subtracted), an orthogonal Q applied to the stream (norms folded) changes f unless Q fixes the all-ones vector.",
       {"norm": "ln"}, False, "derived: the stabilizer of the mean direction (SliceGPT converts LayerNorm to RMSNorm first)")
def h7(ops, rng, arch):
    return _compare(ops, rng, arch, lambda a, P, o, r: G.residual_orth(a, P, o, o.rand_orthogonal(a.d, r), fold=True))


@claim("H7f", "H", "With LayerNorm, an orthogonal Q that fixes the all-ones vector (norms folded) leaves f unchanged.", {"norm": "ln"}, True, "derived: H7 restricted to the stabilizer of the all-ones vector")
def h7f(ops, rng, arch):
    return _compare(ops, rng, arch, lambda a, P, o, r: G.residual_orth(a, P, o, o.rand_orthogonal_fixing_ones(a.d, r), fold=True))


# ---- A: attention ---------------------------------------------------------------------------------------------------------------------------

@claim("A1", "A", "Per key/value group, wv -> A wv and the group's o columns -> o A^-1, A in GL(hd), leaves f unchanged (the gauge of the value/output circuit).", {}, True,
       "derived: pasted 2 (W_OV); the product W_O W_V is gauge invariant")
def a1(ops, rng, arch):
    return _compare(ops, rng, arch, lambda a, P, o, r: G.ov_gauge(a, P, o, _per_layer_per_group(o, a, r, lambda: o.rand_invertible(a.hd, r))))


@claim("A2", "A", "The same A applied to wv alone changes f.", {}, False, "derived: the necessity of the compensation in A1")
def a2(ops, rng, arch):
    return _compare(ops, rng, arch, lambda a, P, o, r: G.ov_value_only(a, P, o, _per_layer_per_group(o, a, r, lambda: o.rand_invertible(a.hd, r))))


@claim("A3", "A", "With RoPE, per group and rotary plane, k -> M k and q -> M^-T q with M = [[a, -b], [b, a]] (a complex scalar) leaves f unchanged (the gauge of the query/key circuit).", {}, True,
       "derived: pasted 2 (W_QK); the commutant of the rotations")
def a3(ops, rng, arch):
    def make(a, P, o, r):
        sc = [o.rand_plane_scalars((a.n_kv, a.hd // 2), r) for _ in range(a.n_layers)]
        return G.qk_gauge_rope(a, P, o, sc)
    return _compare(ops, rng, arch, make)


@claim("A4", "A", "With RoPE, k -> A k and q -> A^-T q with A a general element of GL(hd) changes f.", {}, False, "derived: pasted 2 ('W_Q W_K as a static operator') fails under RoPE")
def a4(ops, rng, arch):
    return _compare(ops, rng, arch, lambda a, P, o, r: G.qk_gauge_general(a, P, o, _per_layer_per_group(o, a, r, lambda: o.rand_invertible(a.hd, r))))


@claim("A5", "A", "Without positional information, k -> A k and q -> A^-T q with A in GL(hd) leaves f unchanged.", {"pos": "none"}, True, "derived: A4 without RoPE")
def a5(ops, rng, arch):
    return _compare(ops, rng, arch, lambda a, P, o, r: G.qk_gauge_general(a, P, o, _per_layer_per_group(o, a, r, lambda: o.rand_invertible(a.hd, r))))


@claim("A6", "A", "A permutation of the key/value groups and of the query heads inside each group (with the matching permutation of the o columns) leaves f unchanged.", {}, True, "derived")
def a6(ops, rng, arch):
    def make(a, P, o, r):
        return G.head_perm(a, P, o, [_perm(r, a.n_kv) for _ in range(a.n_layers)], [[_perm(r, a.rep) for _ in range(a.n_kv)] for _ in range(a.n_layers)])
    return _compare(ops, rng, arch, make)


@claim("A7", "A", "A permutation of the query heads (and the o columns) that moves a head to another key/value group, the key and value matrices staying, changes f.", {"n_heads": 4, "n_kv": 2}, False,
       "derived: the group structure of A6 is necessary")
def a7(ops, rng, arch):
    def make(a, P, o, r):
        R = dict(P)
        hd, d = a.hd, a.d
        for i in range(a.n_layers):
            while True:
                order = _perm(r, a.n_heads)
                if any(order[h] // a.rep != h // a.rep for h in range(a.n_heads)):
                    break
            wq = P[f"l{i}.wq"].reshape(a.n_heads, hd, d)
            wo = P[f"l{i}.wo"].reshape(d, a.n_heads, hd)
            R[f"l{i}.wq"] = wq[order].reshape(a.n_heads * hd, d)
            R[f"l{i}.wo"] = wo[:, order].reshape(d, a.n_heads * hd)
        return a, R
    return _compare(ops, rng, arch, make)


# ---- M, N: the feed-forward block and the norms -----------------------------------------------------------------------------------------------

@claim("M1", "M", "A permutation of the hidden units of the gated block, and a nonzero scale c on each up row with 1/c on the matching down column, leaves f unchanged.", {}, True, "derived")
def m1(ops, rng, arch):
    def make(a, P, o, r):
        return G.mlp_gauge(a, P, o, [np.array(_perm(r, a.d_ff)) for _ in range(a.n_layers)], [o.rand_nonzero(r, (a.d_ff,)) for _ in range(a.n_layers)])
    return _compare(ops, rng, arch, make)


@claim("M2", "M", "A scale on the gate rows (with the down columns compensating) changes f: the gate nonlinearity is not homogeneous.", {}, False, "derived: the limit of M1")
def m2(ops, rng, arch):
    def make(a, P, o, r):
        return G.gate_scale(a, P, o, [np.arange(a.d_ff) for _ in range(a.n_layers)], [o.rand_nonzero(r, (a.d_ff,)) for _ in range(a.n_layers)],
                            [o.rand_nonzero(r, (a.d_ff,)) for _ in range(a.n_layers)])
    return _compare(ops, rng, arch, make)


@claim("N1", "N", "A scale c_i on a norm weight and 1/c_i on column i of every matrix that reads it leaves f unchanged (the fold of H4 is the case c = 1/gamma).", {}, True, "derived")
def n1(ops, rng, arch):
    def make(a, P, o, r):
        cs = {f"l{i}.{n}": o.rand_nonzero(r, (a.d,)) for i in range(a.n_layers) for n in ("attn_norm", "mlp_norm")}
        cs["norm_f"] = o.rand_nonzero(r, (a.d,))
        return G.norm_gauge(a, P, o, cs)
    return _compare(ops, rng, arch, make)


@claim("N2", "N", "Folding every norm weight into the matrices that read it (RMSNorm or LayerNorm) leaves f unchanged.", {}, True, "derived")
def n2(ops, rng, arch):
    return _compare(ops, rng, arch, lambda a, P, o, r: fold_norms(a, P, o))


@claim("C1", "C", "The composition of the symmetries H1, A1, A3, A6, M1 and N1, with random elements, leaves f unchanged.", {}, True, "derived: the group generated by the exact symmetries")
def c1(ops, rng, arch):
    def make(a, P, o, r):
        a, P = G.residual_signed_perm(a, P, o, _perm(r, a.d), _signs(r, a.d))
        a, P = G.ov_gauge(a, P, o, _per_layer_per_group(o, a, r, lambda: o.rand_invertible(a.hd, r)))
        a, P = G.qk_gauge_rope(a, P, o, [o.rand_plane_scalars((a.n_kv, a.hd // 2), r) for _ in range(a.n_layers)])
        a, P = G.head_perm(a, P, o, [_perm(r, a.n_kv) for _ in range(a.n_layers)], [[_perm(r, a.rep) for _ in range(a.n_kv)] for _ in range(a.n_layers)])
        a, P = G.mlp_gauge(a, P, o, [np.array(_perm(r, a.d_ff)) for _ in range(a.n_layers)], [o.rand_nonzero(r, (a.d_ff,)) for _ in range(a.n_layers)])
        cs = {f"l{i}.{n}": o.rand_nonzero(r, (a.d,)) for i in range(a.n_layers) for n in ("attn_norm", "mlp_norm")}
        cs["norm_f"] = o.rand_nonzero(r, (a.d,))
        return G.norm_gauge(a, P, o, cs)
    return _compare(ops, rng, arch, make)


# ---- H8: the centering of the embedding --------------------------------------------------------------------------------------------------------

@claim("H8", "H", "With RMSNorm, subtracting from every embedding row its mean (what the fuse_layer_norms of QuaRot and SpinQuant do for every model type), the head kept, changes f.", {}, False,
       "derived: a published code practice; the all-ones component of the stream is part of the RMS")
def h8(ops, rng, arch):
    return _compare(ops, rng, arch, lambda a, P, o, r: G.embed_center(a, P, o))


@claim("H8n", "H", "With LayerNorm (the mean subtracted), subtracting from every embedding row its mean leaves f unchanged.", {"norm": "ln"}, True, "derived: the case in which the centering of H8 is exact")
def h8n(ops, rng, arch):
    return _compare(ops, rng, arch, lambda a, P, o, r: G.embed_center(a, P, o))


# ---- K: the KV cache --------------------------------------------------------------------------------------------------------------------------
# The cache of a layer is the keys (after the rotary embedding) and the values of its key/value groups, a function of the layer's normalized input and of the layer's own input projections only.
# So a transformation of the residual stream that is compensated in the projections leaves it unchanged; a permutation of the groups permutes it; the value/output gauge maps the values by A; the
# query/key gauge of RoPE maps each key plane by the complex scalar. Each claim compares the cache of the transformed model with the cache that the claim says it must be.

def _cache(a, P, x, ops):
    tr = {}
    forward(a, P, x, ops, trace=tr)
    return [(tr[("k", i)], tr[("v", i)]) for i in range(a.n_layers)]


def _flat(c):
    return np.concatenate([np.concatenate([k.reshape(-1), v.reshape(-1)]) for k, v in c])


def _cache_pair(ops, rng, arch, make, expected=None):
    """The flattened cache of the transformed model, and either the original's (expected is None) or `expected(original cache, info)`; make returns (arch, parameters, info)."""
    P = init_params(arch, ops, rng)
    x = _tokens(ops, rng, arch)
    a2, P2, info = make(arch, P, ops, rng)
    c1, c2 = _cache(arch, P, x, ops), _cache(a2, P2, x, ops)
    return (_flat(c1) if expected is None else _flat(expected(c1, info))), _flat(c2)


def _plain(make):
    return lambda a, P, o, r: (*make(a, P, o, r), None)


@claim("K1", "K", "The KV cache is unchanged by a signed permutation of the hidden coordinates (H1).", {}, True, "derived: the cache is a function of the normalized input and the input projections")
def k1(ops, rng, arch):
    return _cache_pair(ops, rng, arch, _plain(lambda a, P, o, r: G.residual_signed_perm(a, P, o, _perm(r, a.d), _signs(r, a.d))))


@claim("K1t", "K", "K1 with the head tied to the embedding.", {"tied": True}, True, "derived")
def k1t(ops, rng, arch):
    return k1(ops, rng, arch)


@claim("K2", "K", "The KV cache is unchanged by an orthogonal Q applied to the stream with the norm weights folded (H4).", {}, True, "derived: computational invariance does not touch the cache")
def k2(ops, rng, arch):
    return _cache_pair(ops, rng, arch, _plain(lambda a, P, o, r: G.residual_orth(a, P, o, o.rand_orthogonal(a.d, r), fold=True)))


@claim("K3", "K", "The KV cache is unchanged by the gauge of the feed-forward units (M1).", {}, True, "derived")
def k3(ops, rng, arch):
    return _cache_pair(ops, rng, arch, _plain(lambda a, P, o, r: G.mlp_gauge(a, P, o, [np.array(_perm(r, a.d_ff)) for _ in range(a.n_layers)],
                                                                           [o.rand_nonzero(r, (a.d_ff,)) for _ in range(a.n_layers)])))


@claim("K3n", "K", "The KV cache is unchanged by the gauge of the norm weights (N1).", {}, True, "derived")
def k3n(ops, rng, arch):
    def make(a, P, o, r):
        cs = {f"l{i}.{n}": o.rand_nonzero(r, (a.d,)) for i in range(a.n_layers) for n in ("attn_norm", "mlp_norm")}
        cs["norm_f"] = o.rand_nonzero(r, (a.d,))
        return G.norm_gauge(a, P, o, cs)
    return _cache_pair(ops, rng, arch, _plain(make))


def _group_perms(ops, rng, arch):
    gps = [_nontrivial_perm(rng, arch.n_kv) for _ in range(arch.n_layers)]
    return gps, [[_perm(rng, arch.rep) for _ in range(arch.n_kv)] for _ in range(arch.n_layers)]


@claim("K4", "K", "A permutation of the key/value groups (A6) permutes the groups of the KV cache: the new group g holds the old group gp[g].", {}, True, "derived")
def k4(ops, rng, arch):
    def make(a, P, o, r):
        gps, hps = _group_perms(o, r, a)
        return (*G.head_perm(a, P, o, gps, hps), gps)
    return _cache_pair(ops, rng, arch, make, lambda c, gps: [(k[list(gps[i])], v[list(gps[i])]) for i, (k, v) in enumerate(c)])


@claim("K4n", "K", "The KV cache is not unchanged by a non-trivial permutation of the key/value groups.", {}, False, "derived: the necessity of the permutation in K4")
def k4n(ops, rng, arch):
    def make(a, P, o, r):
        gps, hps = _group_perms(o, r, a)
        return (*G.head_perm(a, P, o, gps, hps), gps)
    return _cache_pair(ops, rng, arch, make)


def _ov_make(a, P, o, r):
    As = _per_layer_per_group(o, a, r, lambda: o.rand_invertible(a.hd, r))
    return (*G.ov_gauge(a, P, o, As), As)


@claim("K5", "K", "Under the value/output gauge (A1) the keys are unchanged and the values of group g are mapped by A_g: v -> A_g v.", {}, True, "derived")
def k5(ops, rng, arch):
    def expected(c, As):
        return [(k, np.stack([ops.mm(v[g], As[i][g].T) for g in range(arch.n_kv)])) for i, (k, v) in enumerate(c)]
    return _cache_pair(ops, rng, arch, _ov_make, expected)


@claim("K5n", "K", "The KV cache is not unchanged by the value/output gauge: the values move.", {}, False, "derived: what the gauge freedom costs a cache")
def k5n(ops, rng, arch):
    return _cache_pair(ops, rng, arch, _ov_make)


def _qk_make(a, P, o, r):
    sc = [o.rand_plane_scalars((a.n_kv, a.hd // 2), r) for _ in range(a.n_layers)]
    return (*G.qk_gauge_rope(a, P, o, sc), sc)


@claim("K6", "K", "Under the query/key gauge of RoPE (A3) the values are unchanged and each key plane of the cache is mapped by its complex scalar: (x1, x2) -> (a x1 - b x2, b x1 + a x2).", {}, True,
       "derived: a complex scalar commutes with the rotation by position, so the stored (rotated) key is mapped by the same scalar")
def k6(ops, rng, arch):
    def expected(c, sc):
        out = []
        h2 = arch.hd // 2
        for i, (k, v) in enumerate(c):
            a_, b_ = sc[i]
            e = k.copy()
            for g in range(arch.n_kv):
                for j in range(h2):
                    x1, x2 = k[g, :, j], k[g, :, j + h2]
                    e[g, :, j] = ops.red(a_[g, j] * x1 - b_[g, j] * x2)
                    e[g, :, j + h2] = ops.red(b_[g, j] * x1 + a_[g, j] * x2)
            out.append((e, v))
        return out
    return _cache_pair(ops, rng, arch, _qk_make, expected)


@claim("K6n", "K", "The KV cache is not unchanged by the query/key gauge of RoPE: the keys move.", {}, False, "derived")
def k6n(ops, rng, arch):
    return _cache_pair(ops, rng, arch, _qk_make)


@claim("K8", "K", "The cache of the model with the permuted body and the unpermuted embedding ('encrypting the body', H3) is not the original's: the body reads coordinates that the embedding does not write.", {}, False,
       "pasted 1, applied to the cache")
def k8(ops, rng, arch):
    return _cache_pair(ops, rng, arch, _plain(lambda a, P, o, r: G.body_only_perm(a, P, o, _perm(r, a.d), _signs(r, a.d))))


@claim("K9", "K", "The cache of the model with the permuted body and the key applied at the interface (H3k) is the original's: the cache is in the public gauge, so a permutation of the hidden dimension hides nothing in it.", {}, True,
       "pasted 1 ('perfectly decrypting'), applied to the cache; the weights are permuted, what the model computes with is not")
def k9(ops, rng, arch):
    def make(a, P, o, r):
        perm, signs = _perm(r, a.d), _signs(r, a.d)
        a2, R = G.body_only_perm(a, P, o, perm, signs)
        a3, full = G.residual_signed_perm(a, P, o, perm, signs)
        for k in ("embed", "lm_head"):
            if k in full:
                R[k] = full[k]
        return a2, R
    return _cache_pair(ops, rng, arch, _plain(make))


# ---- the runner -------------------------------------------------------------------------------------------------------------------------------

def run_claim(c, make_ops, trials, seed=0):
    """make_ops(trial) returns the arithmetic for one trial (a fresh set of random oracles for each trial, the same object for both sides of the comparison)."""
    arch = BASE.with_(**c.scope)
    outcomes = []
    for t in range(trials):
        rng = np.random.default_rng([seed, hash_id(c.id), t])
        for attempt in range(5):
            ops = make_ops(t * 10 + attempt)
            try:
                y1, y2 = c.build(ops, rng, arch)
                break
            except ZeroDivisionError:
                continue
        else:
            raise RuntimeError(f"{c.id}: five draws had a zero denominator")
        d = ops.diff(y1, y2)
        if ops.exact:
            outcomes.append({"differing_entries": d, "entries": int(np.size(y1)), "holds": d == 0})
        else:
            scale = max(1.0, float(np.max(np.abs(np.asarray(y1, dtype=np.float64)))))
            outcomes.append({"max_abs_difference": d, "scale": scale, "holds": d <= 1e-9 * scale, "differs": d > 1e-6 * scale})
    return outcomes


def hash_id(s):
    return sum((i + 1) * ord(ch) for i, ch in enumerate(s))


def judge(c, outcomes, exact):
    if c.expect:
        ok = all(o["holds"] for o in outcomes)
    else:
        ok = all((not o["holds"]) if exact else o["differs"] for o in outcomes)
    return ok


MODES = {"poly": lambda t: __import__("ops").FieldOps(rope_seed=1234 + t),
         "oracle": lambda t: __import__("ops").OracleFieldOps(seed=t, rope_seed=1234 + t),
         "float": lambda t: __import__("ops").FloatOps(eps=0.0)}


def main(argv):
    mode, trials = argv[1], int(argv[2])
    out = argv[3] if len(argv) > 3 else None
    make_ops = MODES[mode]
    name = make_ops(0).name
    res, bad = [], 0
    for c in CLAIMS:
        o = run_claim(c, make_ops, trials)
        ok = judge(c, o, make_ops(0).exact)
        bad += 0 if ok else 1
        res.append({"id": c.id, "family": c.family, "expect": c.expect, "as_expected": ok, "scope": c.scope, "trials": o})
        print(f"{c.id:4s} {'holds' if c.expect else 'fails':5s} {'as expected' if ok else 'UNEXPECTED'}  [{name}] {c.text[:110]}")
    print(f"{len(res)} claims, {bad} unexpected")
    if out:
        json.dump({"arithmetic": name, "mode": mode, "trials": trials, "claims": res}, open(out, "w"), indent=1)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
