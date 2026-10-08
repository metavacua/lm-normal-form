# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The dimension of the continuous symmetry group of the function of model.py at a generic parameter point, counted as the rank deficiency of the
# Jacobian of the function with respect to the parameters: if the map theta -> f(theta) has rank r at theta, its fibre through theta has dimension
# (number of parameters - r), and the fibre contains the orbit of every continuous symmetry; where nothing but those symmetries moves the parameters
# without moving the function, the two dimensions are equal. The function f is the logits (or log-probabilities) of every position of many random
# token sequences, so that every parameter that matters at all appears in some output.
# The Jacobian is computed by the complex step, d f / d theta_j = Im f(theta + i h e_j) / h, which has no subtraction and so no cancellation error
# (h = 1e-30), in numpy complex128 with the real nonlinearities; model.py's forward pass runs unchanged on complex arrays.
# The rank threshold (1e-8 times the largest singular value) is not derived; in the committed results it lies inside an empty gap of 9 orders of magnitude between the smallest counted and the largest
# uncounted singular value. One random point (seed 15) per configuration.
# Usage: symdim.py OUT.json CONFIG [CONFIG ...]    CONFIG is the name of an entry of CONFIGS. Exits non-zero if a deficiency differs from the formula.
import json, sys, time
import numpy as np
from ops import FloatOps
from model import Arch, shapes, init_params, forward, n_params

H = 1e-30


def expected(a, eps_zero, logprobs):
    """The dimension of the continuous symmetry group that the generators of gauge.py account for, at generic parameters."""
    per_layer = a.n_kv * a.hd * a.hd + (a.n_kv * a.hd if a.pos == "rope" else a.n_kv * a.hd * a.hd) + a.d_ff
    norm_gauge = (2 * a.n_layers + (0 if a.tied else 1)) * a.d
    rotation = 0 if a.tied else a.d * (a.d - 1) // 2
    scale = 1 if eps_zero else 0
    shift = a.d if (logprobs and not a.tied) else 0
    return a.n_layers * per_layer + norm_gauge + rotation + scale + shift


def jacobian(a, P, seqs, ops, logprobs):
    keys = list(shapes(a))
    Pc = {k: P[k].astype(np.complex128) for k in keys}
    cols = sum(int(np.prod(shapes(a)[k])) for k in keys)
    rows = sum(len(s) * a.vocab for s in seqs)
    J = np.empty((rows, cols))
    j = 0
    for k in keys:
        flat = Pc[k].reshape(-1)
        for i in range(flat.size):
            flat[i] += 1j * H
            out = []
            for s in seqs:
                L = forward(a, Pc, s, ops)
                if logprobs:
                    L = L - np.log(np.exp(L).sum(axis=-1, keepdims=True))
                out.append(L.reshape(-1))
            J[:, j] = np.concatenate(out).imag / H
            flat[i] -= 1j * H
            j += 1
    return J


def rank_report(J, tol_rel=1e-8):
    s = np.linalg.svd(J, compute_uv=False)
    n = J.shape[1]
    thr = s[0] * tol_rel
    r = int((s > thr).sum())
    return {"rows": J.shape[0], "cols": n, "rank": r, "deficiency": n - r, "sv_at_rank": float(s[r - 1]), "sv_after_rank": float(s[r]) if r < len(s) else None,
            "largest": float(s[0]), "gap_ratio": float(s[r - 1] / s[r]) if r < len(s) and s[r] > 0 else None}


CONFIGS = {
    # name: (arch, eps, logprobs)
    "untied-eps0-logits": (Arch(d=6, n_heads=4, n_kv=2, hd=4, d_ff=8, n_layers=2, vocab=7, tied=False), 0.0, False),
    "untied-eps0-logprobs": (Arch(d=6, n_heads=4, n_kv=2, hd=4, d_ff=8, n_layers=2, vocab=7, tied=False), 0.0, True),
    "untied-eps-logits": (Arch(d=6, n_heads=4, n_kv=2, hd=4, d_ff=8, n_layers=2, vocab=7, tied=False), 1e-5, False),
    "tied-eps0-logits": (Arch(d=6, n_heads=4, n_kv=2, hd=4, d_ff=8, n_layers=2, vocab=7, tied=True), 0.0, False),
    "tied-eps-logits": (Arch(d=6, n_heads=4, n_kv=2, hd=4, d_ff=8, n_layers=2, vocab=7, tied=True), 1e-5, False),
    "mha-untied-eps0-logits": (Arch(d=6, n_heads=2, n_kv=2, hd=4, d_ff=8, n_layers=1, vocab=7, tied=False), 0.0, False),
    "nope-untied-eps0-logits": (Arch(d=6, n_heads=4, n_kv=2, hd=4, d_ff=8, n_layers=2, vocab=7, tied=False, pos="none"), 0.0, False),
}


def run(name, n_seqs=None, T=6):
    a, eps, logprobs = CONFIGS[name]
    ops = FloatOps(eps=eps)
    rng = np.random.default_rng(15)
    P = init_params(a, ops, rng)
    need = n_params(a)
    n_seqs = n_seqs or int(np.ceil(1.6 * need / (T * a.vocab))) + a.vocab
    seqs = []
    for _ in range(n_seqs):
        s = [int(t) for t in rng.integers(0, a.vocab, T)]
        seqs.append(s)
    for t in range(a.vocab):                       # every token is an input somewhere: otherwise its embedding row moves nothing
        seqs.append([t] * 2 + [int(x) for x in rng.integers(0, a.vocab, T - 2)])
    t0 = time.time()
    J = jacobian(a, P, seqs, ops, logprobs)
    rep = rank_report(J)
    rep.update({"config": name, "arch": vars(a) if hasattr(a, "__dict__") else str(a), "eps": eps, "logprobs": logprobs, "parameters": need,
                "sequences": len(seqs), "expected_deficiency": expected(a, eps == 0.0, logprobs), "seconds": round(time.time() - t0, 1)})
    rep["matches"] = rep["deficiency"] == rep["expected_deficiency"]
    return rep


if __name__ == "__main__":
    out = [run(n) for n in sys.argv[2:]]
    for r in out:
        print(f"{r['config']:26s} parameters {r['parameters']:5d} deficiency {r['deficiency']:4d} expected {r['expected_deficiency']:4d} "
              f"{'MATCH' if r['matches'] else 'DIFFERS'}  singular values at the cut: {r['sv_at_rank']:.3g} | {r['sv_after_rank']:.3g}  ({r['seconds']}s)")
    json.dump(out, open(sys.argv[1], "w"), indent=1)
    sys.exit(0 if all(r["matches"] for r in out) else 1)
