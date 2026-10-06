# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The normal forms of matrix algebra, and the exact symmetries of a Llama-style decoder, as transformations of the weights of a
# model. Every function takes the parameters as a dict of float64 numpy arrays keyed as in the Hugging Face state dict (the tied
# lm_head is not in it) and returns a new dict that shares the arrays it did not change. Nothing here touches a model or torch: the
# transformations are tested apart from the numbers they are run in.
#   exact symmetries      sign_flip, pow2, permute_units                      (a signed permutation; a power of two)
#   gauges, compensated   gauge_vo(make_A), gauge_qk(spread)                  (exact in real arithmetic, not in a stored encoding)
#   textbook forms        rref, rank_normal, truncate (SVD), truncate_vo      (the first two do not preserve the function)
import numpy as np

Q, K, V, O = (f"self_attn.{n}_proj.weight" for n in "qkvo")
G, U, D = "mlp.gate_proj.weight", "mlp.up_proj.weight", "mlp.down_proj.weight"
EMBED = "model.embed_tokens.weight"
CLASSES = {"q": Q, "k": K, "v": V, "o": O, "gate": G, "up": U, "down": D}


class Dims:
    def __init__(self, cfg):
        self.H, self.nh, self.nkv = cfg.hidden_size, cfg.num_attention_heads, cfg.num_key_value_heads
        self.hd = getattr(cfg, "head_dim", None) or self.H // self.nh
        self.F, self.L, self.rep = cfg.intermediate_size, cfg.num_hidden_layers, self.nh // self.nkv


def key(i, name):
    return f"model.layers.{i}.{name}"


def params(model):
    """float64 copies of every parameter; the head only when it is not the embedding."""
    tied = model.config.tie_word_embeddings
    return {k: v.detach().double().numpy().copy() for k, v in model.state_dict().items() if not (tied and k == "lm_head.weight")}


def kappa(A):
    s = np.linalg.svd(A, compute_uv=False)
    return float(s[0] / s[-1])


# ---- the exact symmetries: a signed permutation of the feed-forward units, a power of two -----------------------------------

def sign_flip(P, d, rng, frac=0.5):
    """Negate the up row and the down column of a random half of the units: exact in any IEEE arithmetic."""
    R = dict(P)
    for i in range(d.L):
        s = np.where(rng.random(d.F) < frac, -1.0, 1.0)
        R[key(i, U)], R[key(i, D)] = P[key(i, U)] * s[:, None], P[key(i, D)] * s[None, :]
    return R


def pow2(P, d, rng, kmax=3):
    """Scale the up row of each unit by 2^k and its down column by 2^-k, k in [-kmax, kmax]."""
    R = dict(P)
    for i in range(d.L):
        c = 2.0 ** rng.integers(-kmax, kmax + 1, d.F)
        R[key(i, U)], R[key(i, D)] = P[key(i, U)] * c[:, None], P[key(i, D)] / c[None, :]
    return R


def permute_units(P, d, rng):
    """The same permutation of the gate rows, the up rows and the down columns: exact in real arithmetic, not in the order of a sum."""
    R = dict(P)
    for i in range(d.L):
        p = rng.permutation(d.F)
        R[key(i, G)], R[key(i, U)], R[key(i, D)] = P[key(i, G)][p], P[key(i, U)][p], P[key(i, D)][:, p]
    return R


# ---- gauges: a change of basis in a space no operator sees, compensated in the neighbour --------------------------------------

def orthogonal(n, rng):
    q, r = np.linalg.qr(rng.standard_normal((n, n)))
    return q * np.sign(np.diag(r))


def A_orthogonal(rng):
    def make(Wv):
        q = orthogonal(Wv.shape[0], rng)
        return q, q.T
    return make


def A_cond(kappa_target, rng):
    """A random matrix of this condition number: Q1 diag(s) Q2, s from 1 to kappa on a log scale."""
    def make(Wv):
        n = Wv.shape[0]
        s = np.logspace(0, np.log10(kappa_target), n)
        q1, q2 = orthogonal(n, rng), orthogonal(n, rng)
        return q1 @ np.diag(s) @ q2, q2.T @ np.diag(1 / s) @ q1.T
    return make


def A_rref(Wv):
    """A such that A Wv is the reduced row echelon form of Wv, when its leading square block is invertible: that block's inverse."""
    lead = Wv[:, :Wv.shape[0]].copy()   # a copy: Wv is overwritten by A Wv, and a view of it would then be the identity
    return np.linalg.inv(lead), lead


def A_svd_orthonormal(Wv):
    """A Wv = Vt, the rows orthonormal."""
    u, s, _ = np.linalg.svd(Wv, full_matrices=False)
    return (u / s).T, u * s


def A_svd_balanced(Wv):
    """A Wv = S^(1/2) Vt."""
    u, s, _ = np.linalg.svd(Wv, full_matrices=False)
    return (u / np.sqrt(s)).T, u * np.sqrt(s)


def gauge_vo(P, d, make_A):
    """In each key/value group the value space (hd dimensions) is changed by A: Wv -> A Wv, and the o columns of every query head
    that reads the group by A^-1. Returns the new parameters and the condition number of A for every group."""
    R, kappas = dict(P), []
    for i in range(d.L):
        v, o = P[key(i, V)].copy(), P[key(i, O)].copy()
        for g in range(d.nkv):
            blk = slice(g * d.hd, (g + 1) * d.hd)
            A, Ainv = make_A(v[blk])
            kappas.append(kappa(A))
            v[blk] = A @ v[blk]
            for h in range(g * d.rep, (g + 1) * d.rep):
                o[:, h * d.hd:(h + 1) * d.hd] = o[:, h * d.hd:(h + 1) * d.hd] @ Ainv
        R[key(i, V)], R[key(i, O)] = v, o
    return R, kappas


def gauge_qk(P, d, rng, spread):
    """A rotation of each rotary plane (dimension p and p + hd/2 of a head, which the rotary embedding rotates together) by a
    random angle, and a scale r in [1/spread, spread] on the query side and 1/r on the key side. A rotation commutes with the rotary
    rotation, so the scores are unchanged: the group is a complex scalar per plane, not GL(hd). Every query head of a group gets the
    transformation of the group's key."""
    R, h2 = dict(P), d.hd // 2
    for i in range(d.L):
        q, k = P[key(i, Q)].copy(), P[key(i, K)].copy()
        for g in range(d.nkv):
            for p in range(h2):
                r, phi = np.exp(rng.uniform(-np.log(spread), np.log(spread))), rng.uniform(0, 2 * np.pi)
                rot = np.array([[np.cos(phi), -np.sin(phi)], [np.sin(phi), np.cos(phi)]])
                a, b = g * d.hd + p, g * d.hd + p + h2
                k[[a, b]] = (rot / r) @ k[[a, b]]
                for h in range(g * d.rep, (g + 1) * d.rep):
                    a, b = h * d.hd + p, h * d.hd + p + h2
                    q[[a, b]] = (rot * r) @ q[[a, b]]
        R[key(i, Q)], R[key(i, K)] = q, k
    return R


# ---- the textbook forms of one matrix ----------------------------------------------------------------------------------------

def rref(W, tol=1e-9):
    """Gauss-Jordan with partial pivoting: the reduced row echelon form and the pivot columns."""
    A, (m, n), r, piv = W.astype(float).copy(), W.shape, 0, []
    scale = max(1.0, np.abs(A).max())
    for c in range(n):
        if r == m:
            break
        p = r + int(np.argmax(np.abs(A[r:, c])))
        if abs(A[p, c]) <= tol * scale:
            continue
        A[[r, p]] = A[[p, r]]
        A[r] /= A[r, c]
        rows = np.arange(m) != r
        A[rows] -= np.outer(A[rows, c], A[r])
        piv.append(c)
        r += 1
    return A, piv


def rref_closed(W, rank):
    """The same for a matrix of full rank, without the elimination: [I; 0] for rank = columns <= rows; [I | lead^-1 rest] for rank =
    rows < columns when the leading square block is invertible (otherwise the elimination)."""
    m, n = W.shape
    if rank == n and m >= n:
        out = np.zeros_like(W)
        out[:n, :n] = np.eye(n)
        return out
    if rank == m and m < n and np.linalg.cond(W[:, :m]) < 1e12:
        return np.hstack([np.eye(m), np.linalg.solve(W[:, :m], W[:, m:])])
    return rref(W)[0]


def rank_normal(W, rank):
    """diag(I_r, 0), the normal form under equivalence W -> P W Q."""
    out = np.zeros_like(W)
    out[:rank, :rank] = np.eye(rank)
    return out


def numerical_rank(s, tol):
    return int((s > tol * s[0]).sum())


def svd_of(W):
    return np.linalg.svd(W, full_matrices=False)


def truncate(usv, r):
    u, s, vt = usv
    return (u[:, :r] * s[:r]) @ vt[:r]


def truncate_vo(P, d, r):
    """The value/output circuit of each key/value group cut to rank r, by the singular value decomposition of what it computes: the
    stacked o blocks of the group's query heads times its value matrix. Wv' = pinv(O) M_r, so that O Wv' = M_r exactly."""
    R = dict(P)
    for i in range(d.L):
        v, o = P[key(i, V)].copy(), P[key(i, O)]
        for g in range(d.nkv):
            blk = slice(g * d.hd, (g + 1) * d.hd)
            S = np.vstack([o[:, h * d.hd:(h + 1) * d.hd] for h in range(g * d.rep, (g + 1) * d.rep)])
            M = S @ v[blk]
            v[blk] = np.linalg.solve(S.T @ S, S.T @ truncate(svd_of(M), r))
        R[key(i, V)] = v
    return R
