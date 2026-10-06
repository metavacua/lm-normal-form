# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# A canonical form of a Llama-style decoder's weights under the exact symmetries of symmetry.py (signed permutations and powers of
# two), for the case in which no two items of an index space are alike: the hidden coordinates are ordered by their columns of the
# embedding (the vocabulary is a fixed index, not a permutable one), and then every other index space by a digest of its items as
# vectors in that order. Nothing is searched. A tie between items that are not identical would end the construction; the check of
# that is in special.py. The steps, in order, each an exact operation:
#   A  the hidden coordinates: the sign that makes the first non-zero entry of each embedding column positive, then the order of the
#      columns by digest (of the embedding column and, when the head is not tied, of the head's column)
#   B  each feed-forward unit: 2^k and a sign on its up row, the inverse on its down column, so that the largest entry of the up row
#      is in [1, 2) and its first non-zero entry is positive; then the units ordered by the digest of (gate row, up row, down column)
#   C  each key/value group: the same normalization of each value row (with the o columns that read it) and of each rotary plane of the
#      key (with the query planes of the group's heads); then the heads of a group ordered by the digest of (query block, o block),
#      and the groups by the digest of (key block, value block, the digests of their heads)
# The gauge of the norm weights is not normalized.
import hashlib
import numpy as np
import forms as F


def dig(*arrays):
    h = hashlib.blake2b(digest_size=16)
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
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


def canonical(P, d):
    dt = P[F.EMBED].dtype
    R = dict(P)
    # A: the hidden coordinates
    E = P[F.EMBED]
    s = first_sign(np.ascontiguousarray(E.T)).astype(dt)
    keys = []
    ET = np.ascontiguousarray(E.T)
    HT = np.ascontiguousarray(P["lm_head.weight"].T) if "lm_head.weight" in P else None
    for j in range(d.H):
        keys.append(dig(ET[j] * s[j]) + (dig(HT[j] * s[j]) if HT is not None else b""))
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
    for i in range(d.L):
        # B: the feed-forward units
        g, u, w = R[F.key(i, F.G)], R[F.key(i, F.U)].copy(), R[F.key(i, F.D)].copy()
        c = exp_scale(np.abs(u).max(axis=1)) * first_sign(u * exp_scale(np.abs(u).max(axis=1))[:, None])
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
        cv = exp_scale(np.abs(vb).max(axis=2))
        cv = cv * np.stack([first_sign(vb[g_] * cv[g_][:, None]) for g_ in range(d.nkv)])
        vb *= cv[:, :, None]
        for h in range(d.nh):
            ob[:, h, :] /= cv[h // d.rep][None, :]
        for g_ in range(d.nkv):
            plane = np.concatenate([kb[g_, :h2], kb[g_, h2:]], axis=1)         # (hd/2, 2H): the two rows of each rotary plane
            ck = exp_scale(np.abs(plane).max(axis=1))
            ck = ck * first_sign(plane * ck[:, None])
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
    return R


def ties(P, d):
    """The items that the digests cannot tell apart although they are not identical: none, if the construction above is canonical.
    For each index space the number of items and of distinct digests of the normalized items; identical items count as one."""
    out = {}
    C = canonical(P, d)
    out["coordinates"] = (d.H, len({dig(np.ascontiguousarray(C[F.EMBED].T)[j]) for j in range(d.H)}))
    for i in range(d.L):
        g, u, w = C[F.key(i, F.G)], C[F.key(i, F.U)], C[F.key(i, F.D)]
        out[f"units {i}"] = (d.F, len({dig(g[j], u[j], w[:, j]) for j in range(d.F)}))
    return out
