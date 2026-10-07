# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Batch 0023, first steps toward a general covariant version of the theory. Measurements only (the verdicts are in grade23.py, which was committed before the run): no number here is compared
# with anything.
#   rank      the dimension of the group of the function, as the rank deficiency of the Jacobian of the function with respect to the parameters (batch 0015's method), of the finite instance
#             (finite23.py) and of four models that change what is absolute in its position structure:
#               base         the rotary table is fixed (the frequencies are constants, the position of token t is t): the special theory
#               theta        the frequencies of the rotary planes are parameters of each layer
#               delta        the position of token t is the proper time tau_t = sum over i < t of exp(b + w . x_i) (x_i the normalized input of the layer), with b and w parameters; the frequencies fixed
#               theta+delta  both are parameters: the angle of plane p at token t is theta_p tau_t
#               posdep       the queries and the keys have their own projection at every position and there is no rotary embedding: the covariant form of the attention of which the rotary table is a
#                            constraint (q_t = R_t W_q x_t is the point of this model at which W_t = R_t W)
#   constrained  at the point of `posdep` that is the image iota(theta) of a point theta of the special model (W_t = R_t W), the dimension of the fibre of the covariant model, and the dimension of its
#                intersection with the tangent space of the image of iota, which is the fibre of the special model: the group of the special theory as the stabilizer of the constraint
#   vocab        the same count for models whose vocabulary is at most, or larger than, the hidden size (the first layer of a model with at most d tokens has extra symmetries: found in the pilot of this
#                batch, see docs/batches/0023.md)
#   causal       the dependency structure of the computation: which input positions of a layer, and of the whole stack, the output at each position depends on, by the Jacobian of the stream
# Usage: cov23.py OUT.json [--smoke]
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from finite23 import (ARCH, H, make_spec, contexts, init_theta, jacobian, logits_of, ops_of, selftest, unpack, layer_pb, embed_pb)  # noqa: E402

SEEDS = [11, 12, 13, 14, 15]          # fresh draws: the pilot of this batch used the seed 5
VARIANTS = ("base", "theta", "delta", "theta+delta", "posdep")
VOCAB_CONFIGS = [(4, 3, 4), (6, 6, 3), (5, 6, 3), (4, 8, 3)]      # (hidden size d, vocabulary V, longest context)
TOL = 1e-8                                                          # relative to the largest singular value


def spectrum_report(J, n):
    s = np.linalg.svd(J, compute_uv=False)
    r = int((s > s[0] * TOL).sum())
    return {"n": int(n), "rank": r, "deficiency": int(n - r), "sv_at_cut": float(s[r - 1]), "sv_after_cut": float(s[r]) if r < len(s) else 0.0, "sigma_max": float(s[0])}


def formula(a, variant):
    """The dimension of the group by the derivation of batches 0015 and 0019 for the model with its norm weights frozen at 1 (no norm gauge), an untied head, epsilon > 0 (no scale), logits (no shift):
    per layer the value/output matrices of every group, the complex scalar of every group and plane, the scale of every unit; the rotations of the stream. For `theta+delta` one dilation per layer;
    for `posdep` the covariant attention has an invertible matrix per group in place of the complex scalar, and the queries' projection at position 0 is not a gauge but is not seen (the softmax
    over one position)."""
    per = a.n_kv * a.hd * a.hd + a.n_kv * a.hd + a.d_ff
    base = a.n_layers * per + a.d * (a.d - 1) // 2
    if variant == "theta+delta":
        return base + a.n_layers
    if variant == "posdep":
        return a.n_layers * (2 * a.n_kv * a.hd * a.hd + a.d_ff) + a.d * (a.d - 1) // 2 + a.n_layers * a.n_heads * a.hd * a.d
    return base


def rotation_matrices(a, ops, T):
    """R_t (hd x hd) of the rotate-half rotary embedding at positions 0..T-1: out1 = x1 C - x2 S, out2 = x2 C + x1 S."""
    C, S = ops.rope_tables(T, a.hd)
    h2 = a.hd // 2
    R = np.zeros((T, a.hd, a.hd))
    for t in range(T):
        R[t, :h2, :h2], R[t, :h2, h2:], R[t, h2:, :h2], R[t, h2:, h2:] = np.diag(C[t]), -np.diag(S[t]), np.diag(S[t]), np.diag(C[t])
    return R


def iota(a, ops, spec_s, spec_c, tmax):
    """The map from the parameters of the special model to those of the position-dependent model: W_t = R_t W for the queries of each head and the keys of each group. Linear."""
    R = rotation_matrices(a, ops, tmax)

    def f(theta):
        Ps = {k: v[0] for k, v in unpack(theta[None, :], spec_s).items()}
        parts = []
        for k, shp in spec_c.items():
            if k.endswith(".wq_t") or k.endswith(".wk_t"):
                W = Ps[k[:-2]]
                nb = W.shape[0] // a.hd
                blocks = W.reshape(nb, a.hd, a.d)
                parts.append(np.stack([np.concatenate([R[t] @ blocks[b] for b in range(nb)], axis=0) for t in range(tmax)]).reshape(-1))
            else:
                parts.append(Ps[k].reshape(-1))
        return np.concatenate(parts)
    return f


def rank_experiment(seeds):
    a, ops = ARCH, ops_of()
    groups = contexts(a)
    out = {"seeds": seeds, "variants": {}, "constrained": []}
    for variant in VARIANTS:
        spec = make_spec(a, variant)
        rows = []
        for seed in seeds:
            th = init_theta(spec, np.random.default_rng(seed), a, ops)
            rep = spectrum_report(jacobian(th, spec, groups, a, ops), th.size)
            rep.update({"seed": seed, "expected": formula(a, variant)})
            rows.append(rep)
            print(f"rank {variant:12s} seed {seed}: n {rep['n']:3d} deficiency {rep['deficiency']:3d} (derived {rep['expected']:3d})  gap {rep['sv_at_cut']:.2e} | {rep['sv_after_cut']:.2e}", flush=True)
        out["variants"][variant] = rows
    spec_s, spec_c = make_spec(a, "base"), make_spec(a, "posdep")
    f = iota(a, ops, spec_s, spec_c, groups[-1].shape[1])
    n_s = sum(int(np.prod(s)) for s in spec_s.values())
    D = np.array([f(np.eye(n_s)[j]) - f(np.zeros(n_s)) for j in range(n_s)]).T          # the linear map, as a matrix (n_c, n_s)
    for seed in seeds:
        th_s = init_theta(spec_s, np.random.default_rng(seed), a, ops)
        th_c = f(th_s)
        diff = float(np.max(np.abs(logits_of(th_s, spec_s, groups, a, ops) - logits_of(th_c, spec_c, groups, a, ops))))
        J_c = jacobian(th_c, spec_c, groups, a, ops)
        Uc, sc, Vt = np.linalg.svd(J_c, full_matrices=True)
        r = int((sc > sc[0] * TOL).sum())
        K = Vt[r:].T                                                                     # the fibre of the covariant model at the point: an orthonormal basis (n_c, k)
        rd = int(np.linalg.matrix_rank(D, tol=TOL * np.linalg.norm(D, 2)))
        rk = int(np.linalg.matrix_rank(np.hstack([K, D]), tol=TOL * np.linalg.norm(np.hstack([K, D]), 2)))
        inter = K.shape[1] + rd - rk
        special = spectrum_report(jacobian(th_s, spec_s, groups, a, ops), th_s.size)
        out["constrained"].append({"seed": seed, "function_difference": diff, "deficiency_covariant": int(K.shape[1]), "rank_of_iota": rd, "intersection": int(inter),
                                   "deficiency_special": special["deficiency"], "expected_covariant": formula(a, "posdep"), "expected_special": formula(a, "base")})
        print(f"constrained seed {seed}: function difference {diff:.1e}; covariant fibre {K.shape[1]} (derived {formula(a, 'posdep')}); intersection with the tangent of the image {inter}; special fibre {special['deficiency']}", flush=True)
    return out


def vocab_experiment(seeds):
    ops = ops_of()
    rows = {"V_at_most_d": [], "V_larger_than_d": []}
    for d, V, tm in VOCAB_CONFIGS:
        a = ARCH.with_(d=d, vocab=V)
        spec, groups = make_spec(a, "base", tm), contexts(a, tm)
        for seed in seeds[:3]:
            th = init_theta(spec, np.random.default_rng(seed), a, ops)
            rep = spectrum_report(jacobian(th, spec, groups, a, ops, chunk=24), th.size)
            rep.update({"d": d, "V": V, "longest_context": tm, "seed": seed, "formula": formula(a, "base")})
            rep["extra"] = rep["deficiency"] - rep["formula"]
            rows["V_at_most_d" if V <= d else "V_larger_than_d"].append(rep)
            print(f"vocab d {d} V {V} seed {seed}: deficiency {rep['deficiency']} formula {rep['formula']} extra {rep['extra']:+d}", flush=True)
    return rows


def stream_dependency(a, P1, h_in, i_from, i_to, ops):
    """M[t', t] = the Frobenius norm of d h_{i_to}[t'] / d h_{i_from}[t]: the stream after layers i_from .. i_to - 1 against the stream that enters layer i_from (complex step)."""
    T, d = h_in.shape[2], h_in.shape[3]
    B = T * d
    Hc = np.broadcast_to(h_in.astype(np.complex128), (B, 1, T, d)).copy()
    for b in range(B):
        Hc[b, 0, b // d, b % d] += 1j * H
    P = {k: np.broadcast_to(v, (B,) + v.shape[1:]) for k, v in P1.items()}
    h = Hc
    for i in range(i_from, i_to):
        h = layer_pb(a, P, i, h, ops)
    dh = h.imag / H
    M = np.zeros((T, T))
    for b in range(B):
        M[:, b // d] += np.sum(dh[b, 0] ** 2, axis=-1)
    return np.sqrt(M)


def causal_experiment(seeds):
    a, ops = ARCH, ops_of()
    spec = make_spec(a)
    rows = []
    for seed in seeds[:3]:
        rng = np.random.default_rng(seed)
        th = init_theta(spec, rng, a, ops)
        P1 = unpack(th[None, :].astype(np.float64), spec)
        T = 4
        toks = rng.integers(0, a.vocab, (1, T))
        hs = [embed_pb(P1, toks)]
        for i in range(a.n_layers):
            hs.append(layer_pb(a, P1, i, hs[-1], ops))
        for i_from, i_to in ((0, 1), (1, 2), (0, 2)):
            M = stream_dependency(a, P1, hs[i_from], i_from, i_to, ops)
            fut = max(float(M[tp, t]) for tp in range(T) for t in range(T) if t > tp)          # the output at t' against inputs after it
            past = min(float(M[tp, t]) for tp in range(T) for t in range(T) if t <= tp)         # against inputs at or before it
            rows.append({"seed": seed, "from_layer": i_from, "to_layer": i_to, "largest_dependence_on_a_later_position": fut, "smallest_dependence_on_an_earlier_or_equal_position": past})
            print(f"causal seed {seed} layers {i_from}->{i_to}: largest on a later position {fut:.2e}, smallest on an earlier or equal position {past:.2e}", flush=True)
    return rows


def main(out, smoke=False):
    selftest()
    seeds = [99] if smoke else SEEDS          # the smoke run uses a draw that is not registered
    res = {"seeds": seeds, "tolerance_relative_to_largest_singular_value": TOL}
    res["rank"] = rank_experiment(seeds)
    res["vocab"] = vocab_experiment(seeds)
    res["causal"] = causal_experiment(seeds)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], "--smoke" in sys.argv)
