# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The exact symmetries of a Llama-style decoder that are signed permutations and powers of two, acting on its weights. Each is an
# exact operation on any float dtype (a move of entries, a negation, a scaling by 2^k) and a symmetry of the function in real
# arithmetic; whether it is one in the arithmetic of a run is batch 0014's question. The group, with the tensors each element moves:
#   residual   a signed permutation of the hidden coordinates: embedding and head columns; the columns of q k v gate up; the rows
#              of o and down; the norm weights, permuted and not negated
#   units      per layer, a permutation of the feed-forward units, a sign and 2^k on each up row with 1/(sign 2^k) on its down column
#   heads      per layer, a permutation of the key/value groups and of the query heads inside each group (group-major order)
#   vo         per key/value group and value dimension, a sign and 2^k on the value row, the inverse on the o columns that read it
#   qk         per group and rotary plane (dimension p and p + hd/2), a sign and 2^k on the key plane, the inverse on the query planes
# The norm weights' own gauge (a power of two on a norm weight, the inverse on the columns that read it) is not in this file.
#   zsigns     (added after the first run) the sign bit of zeros: not a symmetry but a bit pattern, a no-op on values
import numpy as np
import forms as F


def _pow2(k, like):
    return np.ldexp(np.ones(k.shape, like.dtype), k.astype(int)).astype(like.dtype)


def residual(P, d, rng, tied):
    """A random signed permutation of the hidden coordinates. Returns (new parameters, (new index of each old coordinate, signs))."""
    pi, s = rng.permutation(d.H), np.where(rng.random(d.H) < 0.5, -1.0, 1.0)
    return apply_residual(P, d, pi, s), (pi, s)


def apply_residual(P, d, pi, s):
    inv = np.empty_like(pi)
    inv[pi] = np.arange(d.H)         # the old coordinate that lands at each new index
    R = dict(P)
    sg = s.astype(P[F.EMBED].dtype)
    cols = lambda W: W[:, inv] * sg[inv][None, :]
    rows = lambda W: W[inv, :] * sg[inv][:, None]
    R[F.EMBED] = cols(P[F.EMBED])
    if "lm_head.weight" in P:
        R["lm_head.weight"] = cols(P["lm_head.weight"])
    for i in range(d.L):
        for n in (F.Q, F.K, F.V, F.G, F.U):
            R[F.key(i, n)] = cols(P[F.key(i, n)])
        for n in (F.O, F.D):
            R[F.key(i, n)] = rows(P[F.key(i, n)])
        for n in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            R[F.key(i, n)] = P[F.key(i, n)][inv]
    R["model.norm.weight"] = P["model.norm.weight"][inv]
    return R


def units(P, d, rng, kmax=3, permute=True):
    R = dict(P)
    for i in range(d.L):
        p = rng.permutation(d.F) if permute else np.arange(d.F)
        t = np.where(rng.random(d.F) < 0.5, -1.0, 1.0)
        c = t * (2.0 ** rng.integers(-kmax, kmax + 1, d.F))
        g, u, w = P[F.key(i, F.G)], P[F.key(i, F.U)], P[F.key(i, F.D)]
        c = c.astype(g.dtype)
        R[F.key(i, F.G)], R[F.key(i, F.U)], R[F.key(i, F.D)] = g[p], (u * c[:, None])[p], (w / c[None, :])[:, p]
    return R


def heads(P, d, rng):
    R = dict(P)
    for i in range(d.L):
        q, k, v, o = (P[F.key(i, n)] for n in (F.Q, F.K, F.V, F.O))
        gp = rng.permutation(d.nkv)                                   # new group g' takes old group gp[g']
        qb, kb, vb = q.reshape(d.nh, d.hd, d.H), k.reshape(d.nkv, d.hd, d.H), v.reshape(d.nkv, d.hd, d.H)
        ob = o.reshape(d.H, d.nh, d.hd)
        order = []                                                    # the old head at each new head index
        for g in gp:
            order += [g * d.rep + r for r in rng.permutation(d.rep)]
        R[F.key(i, F.Q)] = qb[order].reshape(d.nh * d.hd, d.H)
        R[F.key(i, F.K)], R[F.key(i, F.V)] = kb[gp].reshape(-1, d.H), vb[gp].reshape(-1, d.H)
        R[F.key(i, F.O)] = ob[:, order].reshape(d.H, d.nh * d.hd)
    return R


def vo(P, d, rng, kmax=3):
    R = dict(P)
    for i in range(d.L):
        v = P[F.key(i, F.V)].reshape(d.nkv, d.hd, d.H).copy()
        o = P[F.key(i, F.O)].reshape(d.H, d.nh, d.hd).copy()
        c = (np.where(rng.random((d.nkv, d.hd)) < 0.5, -1.0, 1.0) * 2.0 ** rng.integers(-kmax, kmax + 1, (d.nkv, d.hd))).astype(v.dtype)
        v *= c[:, :, None]
        for h in range(d.nh):
            o[:, h, :] /= c[h // d.rep][None, :]
        R[F.key(i, F.V)], R[F.key(i, F.O)] = v.reshape(-1, d.H), o.reshape(d.H, -1)
    return R


def qk(P, d, rng, kmax=3):
    R, h2 = dict(P), d.hd // 2
    for i in range(d.L):
        q = P[F.key(i, F.Q)].reshape(d.nh, d.hd, d.H).copy()
        k = P[F.key(i, F.K)].reshape(d.nkv, d.hd, d.H).copy()
        c = (np.where(rng.random((d.nkv, h2)) < 0.5, -1.0, 1.0) * 2.0 ** rng.integers(-kmax, kmax + 1, (d.nkv, h2))).astype(q.dtype)
        for g in range(d.nkv):
            k[g, :h2] *= c[g][:, None]
            k[g, h2:] *= c[g][:, None]
            for h in range(g * d.rep, (g + 1) * d.rep):
                q[h, :h2] /= c[g][:, None]
                q[h, h2:] /= c[g][:, None]
        R[F.key(i, F.Q)], R[F.key(i, F.K)] = q.reshape(-1, d.H), k.reshape(-1, d.H)
    return R


def zsigns(P, d, rng):
    """The sign bit of a random half of the zeros of every tensor set (-0.0): not a symmetry of anything but a bit pattern, and a no-op on
    values. The units', value/output and query/key gauges do it as a side effect to the zeros that they negate."""
    R = dict(P)
    for k, W in P.items():
        idx = np.flatnonzero(W == 0)
        if len(idx):
            flip = idx[rng.random(len(idx)) < 0.5]
            W2 = W.copy()
            W2.reshape(-1)[flip] = W.dtype.type(-0.0)
            R[k] = W2
    return R


def act(P, d, rng, tied, parts=("residual", "units", "heads", "vo", "qk")):
    """A random element of the group, the parts in this order. "units-signed" is the units' signs and powers of two without the
    permutation (every operation then exact in any arithmetic with a fixed order of sums)."""
    R = P
    if "residual" in parts:
        R, _ = residual(R, d, rng, tied)
    if "units" in parts:
        R = units(R, d, rng)
    if "units-signed" in parts:
        R = units(R, d, rng, permute=False)
    if "heads" in parts:
        R = heads(R, d, rng)
    if "vo" in parts:
        R = vo(R, d, rng)
    if "qk" in parts:
        R = qk(R, d, rng)
    if "zsigns" in parts:
        R = zsigns(R, d, rng)
    return R
