# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# A canonical form of a Llama-style decoder's weights under the exact symmetries of symmetry.py (signed permutations and powers of
# two), for the case in which no two items of an index space are alike: the hidden coordinates are ordered by their columns of the
# embedding (the vocabulary is a fixed index, not a permutable one), and then every other index space by a digest of its items as
# vectors in that order. Nothing is searched. The steps, in order, each an exact operation:
#   A  the hidden coordinates: the sign that makes the first non-zero entry of each embedding column positive (of the head's column
#      when the embedding's is zero), then the order of the columns by digest (of the embedding column and, when the head is not
#      tied, of the head's column)
#   B  each feed-forward unit: 2^k and a sign on its up row, the inverse on its down column, so that the largest entry of the up row
#      is in [1, 2) and its first non-zero entry is positive; then the units ordered by the digest of (gate row, up row, down column)
#   C  each key/value group: the same normalization of each value row (with the o columns that read it) and of each rotary plane of the
#      key (with the query planes of the group's heads); then the heads of a group ordered by the digest of (query block, o block),
#      and the groups by the digest of (key block, value block, the digests of their heads)
# The gauge of the norm weights is not normalized.
# Amended after the first run (docs/batches/0013.md, "After the first run"), in three ways:
#   zeros      -0.0 is read as 0.0 in every digest, and every tensor of the result has +0.0. The gauge of a pair whose normalizing part is all
#              zero changes only the sign bit of its zeros (a key plane that is all zero, in the first run, TriLM 390M's layer 0), which no
#              item's content can fix; the form is canonical up to the sign of zeros, as the values are
#   pairs      when the part that a pair is normalized by is all zero and its partner is not, the partner fixes the gauge (one partner:
#              every unit, and the value and key pairs of a model with one query head per key/value head); with several partners
#              (grouped-query attention) the form raises
#   ties       it raises where two hidden coordinates have the same embedding and head columns, since the form would not be canonical
import hashlib
import numpy as np
import forms as F


class Degenerate(Exception):
    """The weights have an item that this canonical form cannot place canonically."""


def z0(x):
    """x with -0.0 as +0.0 (a new array)."""
    return x + x.dtype.type(0)


def dig(*arrays):
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        h.update(np.ascontiguousarray(z0(a)).tobytes())
    return h.digest()


def first_sign(rows):
    """The sign of the first non-zero entry of each row of a 2-D array (1 for a row of zeros)."""
    nz = rows != 0
    s = np.sign(rows[np.arange(len(rows)), nz.argmax(axis=1)])
    s[s == 0] = 1
    return s


def exp_scale(m):
    """2^-k with m * 2^-k in [1, 2) (exactly), for each non-zero m; 1 for zeros."""
    f, e = np.frexp(m)
    c = np.ldexp(np.ones_like(m), -(e - 1))
    return np.where(m > 0, c, np.ones_like(m))


def multiplier(a, b):
    """The signed power of two m for pairs (a row of `a`, the same row of `b`) that are scaled as a * m and b / m: the one that normalizes a
    (largest entry in [1, 2), first non-zero entry positive), or b where a row of a is all zero; 1 where both are. `a` and `b` are 2-D
    arrays with one pair per row; returns the multiplier and whether each pair has a non-zero partner only (a zero, b not)."""
    sa = exp_scale(np.abs(a).max(axis=1))
    ma = sa * first_sign(a * sa[:, None])
    sb = exp_scale(np.abs(b).max(axis=1))
    mb = sb * first_sign(b * sb[:, None])
    za, zb = ~a.any(axis=1), ~b.any(axis=1)
    m = np.where(za & ~zb, np.ones_like(ma) / mb, ma)
    return m.astype(a.dtype), za & ~zb


def canonical(P, d):
    dt = P[F.EMBED].dtype
    R = dict(P)
    # A: the hidden coordinates
    E = P[F.EMBED]
    ET = np.ascontiguousarray(E.T)
    HT = np.ascontiguousarray(P["lm_head.weight"].T) if "lm_head.weight" in P else None
    s = first_sign(ET).astype(dt)
    dead = ~ET.any(axis=1)
    if dead.any():
        if HT is None or (~HT[dead].any(axis=1)).any():
            raise Degenerate("a hidden coordinate is zero in the embedding (and in the head): its sign cannot be fixed")
        s[dead] = first_sign(HT[dead]).astype(dt)
    keys = []
    for j in range(d.H):
        keys.append(dig(ET[j] * s[j]) + (dig(HT[j] * s[j]) if HT is not None else b""))
    if len(set(keys)) < d.H:
        raise Degenerate("two hidden coordinates have the same embedding and head columns: the order of the coordinates is not fixed")
    order = np.array(sorted(range(d.H), key=lambda j: keys[j]))
    sg = s[order]
    cols = lambda W: W[:, order] * sg[None, :]
    rows = lambda W: W[order, :] * sg[:, None]
    R[F.EMBED] = cols(E)
    if HT is not None:
        R["lm_head.weight"] = cols(P["lm_head.weight"])
    for i in range(d.L):
        for n in (F.Q, F.K, F.V, F.G, F.U):
            R[F.key(i, n)] = cols(P[F.key(i, n)])
        for n in (F.O, F.D):
            R[F.key(i, n)] = rows(P[F.key(i, n)])
        for n in ("input_layernorm.weight", "post_attention_layernorm.weight"):
            R[F.key(i, n)] = P[F.key(i, n)][order]
    R["model.norm.weight"] = P["model.norm.weight"][order]
    del ET, HT
    for i in range(d.L):
        # B: the feed-forward units
        g, u, w = R[F.key(i, F.G)], R[F.key(i, F.U)], R[F.key(i, F.D)]
        c, _ = multiplier(u, np.ascontiguousarray(w.T))
        u, w = u * c[:, None], w / c[None, :]
        keys = [dig(g[j], u[j], w[:, j]) for j in range(d.F)]
        p = np.array(sorted(range(d.F), key=lambda j: keys[j]))
        R[F.key(i, F.G)], R[F.key(i, F.U)], R[F.key(i, F.D)] = g[p], u[p], w[:, p]
        # C: the attention heads
        qb = R[F.key(i, F.Q)].reshape(d.nh, d.hd, d.H).copy()
        kb = R[F.key(i, F.K)].reshape(d.nkv, d.hd, d.H).copy()
        vb = R[F.key(i, F.V)].reshape(d.nkv, d.hd, d.H).copy()
        ob = R[F.key(i, F.O)].reshape(d.H, d.nh, d.hd).copy()
        h2 = d.hd // 2
        # value rows against the o columns that read them
        o_cols = ob.transpose(1, 2, 0).reshape(d.nkv, d.rep, d.hd, d.H)         # (group, head in group, dim, H)
        for g_ in range(d.nkv):
            partner = o_cols[g_].transpose(1, 0, 2).reshape(d.hd, d.rep * d.H)  # (dim, the group's o columns side by side)
            cv, mixed = multiplier(vb[g_], partner)
            if mixed.any():
                if d.rep > 1:
                    raise Degenerate(f"layer {i}, group {g_}: a value row is zero and an o column that reads it is not, with {d.rep} heads in the group")
            vb[g_] *= cv[:, None]
            for r in range(d.rep):
                ob[:, g_ * d.rep + r, :] /= cv[None, :]
        # key planes against the query planes of the group's heads
        for g_ in range(d.nkv):
            plane = np.concatenate([kb[g_, :h2], kb[g_, h2:]], axis=1)             # (hd/2, 2H): the two rows of each rotary plane
            partner = np.concatenate([np.concatenate([qb[g_ * d.rep + r, :h2], qb[g_ * d.rep + r, h2:]], axis=1) for r in range(d.rep)], axis=1)
            ck, mixed = multiplier(plane, partner)
            if mixed.any() and d.rep > 1:
                raise Degenerate(f"layer {i}, group {g_}: a key plane is zero and a query plane of the group is not, with {d.rep} heads in the group")
            kb[g_, :h2] *= ck[:, None]
            kb[g_, h2:] *= ck[:, None]
            for h in range(g_ * d.rep, (g_ + 1) * d.rep):
                qb[h, :h2] /= ck[:, None]
                qb[h, h2:] /= ck[:, None]
        hk = [dig(qb[h], ob[:, h, :]) for h in range(d.nh)]
        gk, horder = [], []
        for g_ in range(d.nkv):
            mem = sorted(range(g_ * d.rep, (g_ + 1) * d.rep), key=lambda h: hk[h])
            horder.append(mem)
            gk.append(dig(kb[g_], vb[g_]) + b"".join(hk[h] for h in mem))
        gp = sorted(range(d.nkv), key=lambda g_: gk[g_])
        order_h = [h for g_ in gp for h in horder[g_]]
        R[F.key(i, F.Q)] = qb[order_h].reshape(-1, d.H)
        R[F.key(i, F.K)], R[F.key(i, F.V)] = kb[gp].reshape(-1, d.H), vb[gp].reshape(-1, d.H)
        R[F.key(i, F.O)] = ob[:, order_h, :].reshape(d.H, -1)
    for k in list(R):
        R[k] = z0(R[k])
    return R
