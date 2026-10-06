# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The transformations of the parameters of model.py that the claims of batch 0015 are about: the candidate symmetries (each says what it
# changes and on which side of which matrix), and the wrong versions of them that serve as counterexamples. Every function takes
# (arch, params, ops, ...) and returns (arch, params); none changes its input. Matrices are (out, in), as in model.py, and a map that
# acts on the residual stream is written for row vectors, x' = x Q.
#
#   residual_signed_perm   a signed permutation of the hidden coordinates: every matrix that reads or writes the stream, the embedding, the
#                          head, and the norm weights (permuted, not negated)
#   residual_orth          an orthogonal Q on the stream, after the norm weights have been folded into the matrices that read them
#   ov_gauge               per key/value group an invertible A on the value dimensions: wv -> A wv, the o columns of the group's heads -> . A^-1
#   qk_gauge_rope          per group and rotary plane a complex scalar a + bi: the key plane by M, the query planes by M^-T
#   qk_gauge_general       per group an invertible A on the key and query dimensions: k -> A k, q -> A^-T q (an exact symmetry without RoPE only)
#   mlp_gauge              a permutation of the hidden units of the gated block and a scale c on each up row with 1/c on its down column
#   head_perm              a permutation of the key/value groups and of the query heads inside each group
#   norm_gauge             a scale c_i on a norm weight with 1/c_i on the column i of every matrix that reads it
# and the counterexamples: body_only_perm, pasted_rule, residual_orth_unfolded, gate_scale, ov_value_only.
import numpy as np
from model import fold_norms


def _inv_vec(ops, c):
    return ops.inv(c)


def signed_perm_matrix(ops, perm, signs):
    """Q with x' = x Q meaning x'[perm[j]] = signs[j] * x[j] (signs are +1 or -1)."""
    d = len(perm)
    Q = np.zeros((d, d), dtype=object if ops.exact else np.float64)
    for j in range(d):
        Q[j, perm[j]] = (signs[j] % ops.p) if ops.exact else float(signs[j])
    return Q


def _readers(arch):
    return ("wq", "wk", "wv", "wg", "wu")


def residual_signed_perm(arch, P, ops, perm, signs, norms=True):
    Q = signed_perm_matrix(ops, perm, signs)
    ab = np.zeros((arch.d, arch.d), dtype=object if ops.exact else np.float64)       # the unsigned permutation, for the norm weights
    for j in range(arch.d):
        ab[j, perm[j]] = 1
    R = dict(P)
    R["embed"] = ops.mm(P["embed"], Q)
    if not arch.tied:
        R["lm_head"] = ops.mm(P["lm_head"], Q)
    R["norm_f"] = ops.mm(P["norm_f"][None, :], ab)[0]
    for i in range(arch.n_layers):
        for n in ("wq", "wk", "wv", "wg", "wu"):
            R[f"l{i}.{n}"] = ops.mm(P[f"l{i}.{n}"], Q)
        for n in ("wo", "wd"):
            R[f"l{i}.{n}"] = ops.mm(Q.T, P[f"l{i}.{n}"])
        for n in ("attn_norm", "mlp_norm"):
            R[f"l{i}.{n}"] = ops.mm(P[f"l{i}.{n}"][None, :], ab)[0]
    return arch, R


def residual_orth(arch, P, ops, Q, fold=True):
    """x' = x Q with Q orthogonal. With fold=True the norm weights are folded first (the function is unchanged by that), which is what makes this
    a symmetry; fold=False applies Q to the matrices and leaves the norm weights, which is not."""
    if fold:
        arch, P = fold_norms(arch, P, ops)
    R = dict(P)
    R["embed"] = ops.mm(P["embed"], Q)
    if not arch.tied:
        R["lm_head"] = ops.mm(P["lm_head"], Q)
    for i in range(arch.n_layers):
        for n in ("wq", "wk", "wv", "wg", "wu"):
            R[f"l{i}.{n}"] = ops.mm(P[f"l{i}.{n}"], Q)
        for n in ("wo", "wd"):
            R[f"l{i}.{n}"] = ops.mm(Q.T, P[f"l{i}.{n}"])
    return arch, R


def residual_orth_conj(arch, P, ops, Q):
    """x' = x Q with the norm weights left as they are: the matrices that read the output of a norm with weight gamma get W -> W diag(gamma) Q diag(gamma)^-1 (the
    action of folding, rotating and unfolding), the matrices that write the stream and the embedding as in residual_orth. An exact symmetry for an untied head. With a
    tied head the final reader would have to be E diag(gamma_f) Q diag(gamma_f)^-1, which is not the rotated embedding E Q the tie forces, unless gamma_f is constant: the
    function below applies the rotated embedding to the tied head, so for a tied head with a learned gamma_f it is not a symmetry."""
    R = dict(P)
    R["embed"] = ops.mm(P["embed"], Q)

    def read(W, g):
        return ops.red(ops.mm(ops.red(W * g[None, :]), Q) * ops.inv(g)[None, :])
    for i in range(arch.n_layers):
        for n in ("wq", "wk", "wv"):
            R[f"l{i}.{n}"] = read(P[f"l{i}.{n}"], P[f"l{i}.attn_norm"])
        for n in ("wg", "wu"):
            R[f"l{i}.{n}"] = read(P[f"l{i}.{n}"], P[f"l{i}.mlp_norm"])
        for n in ("wo", "wd"):
            R[f"l{i}.{n}"] = ops.mm(Q.T, P[f"l{i}.{n}"])
    if not arch.tied:
        R["lm_head"] = read(P["lm_head"], P["norm_f"])
    return arch, R


def residual_orth_unfolded(arch, P, ops, Q):
    return residual_orth(arch, P, ops, Q, fold=False)


def ov_gauge(arch, P, ops, As, value_only=False):
    """As[i][g]: an invertible hd x hd matrix for layer i, group g."""
    R = dict(P)
    hd = arch.hd
    for i in range(arch.n_layers):
        wv, wo = P[f"l{i}.wv"].copy(), P[f"l{i}.wo"].copy()
        for g in range(arch.n_kv):
            A = As[i][g]
            wv[g * hd:(g + 1) * hd, :] = ops.mm(A, wv[g * hd:(g + 1) * hd, :])
            if not value_only:
                Ai = ops.matinv(A)
                for h in range(g * arch.rep, (g + 1) * arch.rep):
                    wo[:, h * hd:(h + 1) * hd] = ops.mm(wo[:, h * hd:(h + 1) * hd], Ai)
        R[f"l{i}.wv"], R[f"l{i}.wo"] = wv, wo
    return arch, R


def ov_value_only(arch, P, ops, As):
    return ov_gauge(arch, P, ops, As, value_only=True)


def qk_gauge_rope(arch, P, ops, scalars):
    """scalars[i]: (a, b), each (n_kv, hd/2). The key plane (rows j and j + hd/2 of a group's wk) goes to M times itself, M = [[a, -b], [b, a]];
    the same plane of each query head of the group to M^-T times itself, M^-T = M / (a^2 + b^2)."""
    R = dict(P)
    hd, h2 = arch.hd, arch.hd // 2
    for i in range(arch.n_layers):
        a, b = scalars[i]
        wk, wq = P[f"l{i}.wk"].copy(), P[f"l{i}.wq"].copy()
        for g in range(arch.n_kv):
            for j in range(h2):
                ra, rb = a[g, j], b[g, j]
                n2 = ops.red(ra * ra + rb * rb)
                M = ops.asarray([[ra, -rb], [rb, ra]])
                Mq = ops.red(M * ops.inv(n2))                       # M^-T
                rows = [g * hd + j, g * hd + j + h2]
                wk[rows, :] = ops.mm(M, wk[rows, :])
                for h in range(g * arch.rep, (g + 1) * arch.rep):
                    qr = [h * hd + j, h * hd + j + h2]
                    wq[qr, :] = ops.mm(Mq, wq[qr, :])
        R[f"l{i}.wk"], R[f"l{i}.wq"] = wk, wq
    return arch, R


def qk_gauge_general(arch, P, ops, As):
    """As[i][g]: invertible hd x hd. k -> A k, q -> A^-T q for the group's key and its query heads."""
    R = dict(P)
    hd = arch.hd
    for i in range(arch.n_layers):
        wk, wq = P[f"l{i}.wk"].copy(), P[f"l{i}.wq"].copy()
        for g in range(arch.n_kv):
            A = As[i][g]
            AiT = ops.matinv(A).T
            wk[g * hd:(g + 1) * hd, :] = ops.mm(A, wk[g * hd:(g + 1) * hd, :])
            for h in range(g * arch.rep, (g + 1) * arch.rep):
                wq[h * hd:(h + 1) * hd, :] = ops.mm(AiT, wq[h * hd:(h + 1) * hd, :])
        R[f"l{i}.wk"], R[f"l{i}.wq"] = wk, wq
    return arch, R


def mlp_gauge(arch, P, ops, perms, up_scales, gate_scales=None):
    """perms[i]: a permutation of range(d_ff); up_scales[i]: nonzero c per unit (before the permutation). If gate_scales is given, the gate rows are
    scaled as well and the down column is scaled by 1/(c_up c_gate): not a symmetry (the gate nonlinearity is not homogeneous)."""
    R = dict(P)
    for i in range(arch.n_layers):
        p, c = perms[i], up_scales[i]
        wg, wu, wd = P[f"l{i}.wg"], P[f"l{i}.wu"], P[f"l{i}.wd"]
        ci = ops.inv(c)
        if gate_scales is not None:
            cg = gate_scales[i]
            wg = ops.red(wg * cg[:, None])
            ci = ops.red(ci * ops.inv(cg))
        R[f"l{i}.wg"] = wg[p]
        R[f"l{i}.wu"] = ops.red(wu * c[:, None])[p]
        R[f"l{i}.wd"] = ops.red(wd * ci[None, :])[:, p]
    return arch, R


def gate_scale(arch, P, ops, perms, up_scales, gate_scales):
    return mlp_gauge(arch, P, ops, perms, up_scales, gate_scales)


def head_perm(arch, P, ops, gperms, hperms):
    """gperms[i]: a permutation of the key/value groups; hperms[i][g]: a permutation of the rep heads inside new group g."""
    R = dict(P)
    hd, d = arch.hd, arch.d
    for i in range(arch.n_layers):
        gp = gperms[i]
        wq = P[f"l{i}.wq"].reshape(arch.n_heads, hd, d)
        wk = P[f"l{i}.wk"].reshape(arch.n_kv, hd, d)
        wv = P[f"l{i}.wv"].reshape(arch.n_kv, hd, d)
        wo = P[f"l{i}.wo"].reshape(d, arch.n_heads, hd)
        order = [gp[g] * arch.rep + r for g in range(arch.n_kv) for r in hperms[i][g]]
        R[f"l{i}.wq"] = wq[order].reshape(arch.n_heads * hd, d)
        R[f"l{i}.wk"] = wk[list(gp)].reshape(arch.n_kv * hd, d)
        R[f"l{i}.wv"] = wv[list(gp)].reshape(arch.n_kv * hd, d)
        R[f"l{i}.wo"] = wo[:, order].reshape(d, arch.n_heads * hd)
    return arch, R


def norm_gauge(arch, P, ops, cs):
    """cs: dict norm-name -> nonzero vector c. gamma -> gamma * c, and every matrix that reads that norm has its column i multiplied by 1/c_i.
    Needs an untied head for the final norm."""
    R = dict(P)
    for i in range(arch.n_layers):
        for nm, readers in (("attn_norm", ("wq", "wk", "wv")), ("mlp_norm", ("wg", "wu"))):
            c = cs[f"l{i}.{nm}"]
            R[f"l{i}.{nm}"] = ops.red(P[f"l{i}.{nm}"] * c)
            ci = ops.inv(c)
            for n in readers:
                R[f"l{i}.{n}"] = ops.red(P[f"l{i}.{n}"] * ci[None, :])
    if not arch.tied:
        c = cs["norm_f"]
        R["norm_f"] = ops.red(P["norm_f"] * c)
        R["lm_head"] = ops.red(P["lm_head"] * ops.inv(c)[None, :])
    return arch, R


def body_only_perm(arch, P, ops, perm, signs):
    """The signed permutation of residual_signed_perm applied to the layers and the final norm, and NOT to the embedding or the head."""
    a2, R = residual_signed_perm(arch, P, ops, perm, signs)
    R["embed"] = P["embed"]
    if not arch.tied:
        R["lm_head"] = P["lm_head"]
    return a2, R


def pasted_rule(arch, P, ops, perm):
    """The rule as pasted, W_new = P W P^T, for every weight matrix that is square (q and o when n_heads * hd == d), the others left as they are."""
    Pm = np.zeros((arch.d, arch.d), dtype=object if ops.exact else np.float64)
    for j in range(arch.d):
        Pm[j, perm[j]] = 1
    R = dict(P)
    for k, W in P.items():
        if W.ndim == 2 and W.shape[0] == W.shape[1] == arch.d:
            R[k] = ops.mm(ops.mm(Pm, W), Pm.T)
    return arch, R


def embed_center(arch, P, ops):
    """The centering that the fuse_layer_norms of QuaRot and SpinQuant apply to the embedding whatever the norm of the model: every embedding row minus its mean. The head is kept as it was
    (untied first, if it was the embedding). Exact for a model whose norm subtracts the mean (LayerNorm), not for RMSNorm: the all-ones component of the stream is part of the root mean square."""
    R = dict(P)
    if arch.tied:
        R["lm_head"] = P["embed"]
        arch = arch.with_(tied=False)
    E = P["embed"]
    R["embed"] = ops.red(E - ops.red(E.sum(axis=1, keepdims=True) * ops.const_inv(arch.d)))
    return arch, R
