# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Batch 0023, the hole argument of training, on the Llama equations (the finite instance of finite23.py: not a language model). Two constant vectors that an equivalence transformation g relates are the
# same function. A training rule is a map on constants; if it respects g, the runs from theta and from g.theta stay the same function, and the state of a run is the function; if it does not, the state is
# (function, form), and the rule's metric on the constants is a fixed structure that the function does not have. Two ways to ask:
#   field    the velocity that a rule gives the function. In continuous time a rule defines a dynamics on functions if and only if the velocity V(theta) = J(theta) v(theta) (J the Jacobian of the function,
#            v the direction in which the rule moves the constants) is the same at theta and at g.theta. It is measured at ten random points for each of seven equivalence transformations (orthogonal on the
#            constants: a permutation of the units, a signed permutation of the hidden coordinates, a swap of the two heads, a rotation of the stream; not orthogonal: a scaling of the units by +-2^k, an invertible
#            matrix on the value dimensions, a complex scalar on the rotary plane), as the relative difference |V(g.theta) - V(theta)| / |V(theta)| over the 780 x 5 centered scores (the scores of each context minus their mean: the predictive distributions' log-probabilities up to a constant; the least-squares problem that the natural-gradient rules solve fixes J v only up to a shift of the scores of each context, which no distribution sees), for the rules
#              sgd          v = grad L                       (the Euclidean metric of the constants)
#              adam         v = sign(grad L)                 (the first step of Adam: each coordinate divided by its own scale)
#              ngd-pinv     v = F^+ grad L                   natural gradient: F the Fisher metric of the predictive distributions summed over the 780 contexts, F^+ its pseudo-inverse (singular values
#                           of its square root below 1e-10 of the largest taken as zero: they are the directions along which the function does not move). The function-space velocity J F^+ J^T r does
#                           not depend on a metric of the constants: the rule is covariant.
#              ngd-euclid   v = (F + lambda I)^-1 grad L     the damping that natural-gradient codes add, which brings the Euclidean metric of the constants back (lambda = 1e-9, a fixed number)
#              ngd-covdamp  v = (F + lambda J^T J)^-1 grad L the damping by the Euclidean metric of the scores, a metric on functions (lambda = 1e-6 and 1e-2 times the mean nonzero eigenvalue of the Fisher blocks, a number of the predictions)
#            (the update is -v) and the size of each v in the constants, which is what an implementation would have to take.
#   sgd, adam   the runs themselves: 200 steps with lr 1 and with lr 0.01, from theta and from g.theta; the largest difference of the centered scores over the recorded steps.
# The natural-gradient step as an actual update (the pseudo-inverse step, and a proximal step that minimizes L + KL/eta) was tried before this registration and not used: the pseudo-inverse step has a
# norm of 1.8e4 at the first step, and the proximal step's inner problem did not converge in 40 iterations at eta = 1 or 0.1 (docs/batches/0023.md, pilots): at this size the covariant rule can be
# stated and measured as a velocity field, and cannot be integrated without moving the constants far along directions in which the function barely changes.
# The task of the runs is the distillation of a random teacher of the same equations on every context (the loss is a function of the function alone: it is invariant).
# Usage: opt23.py field OUT.json | opt23.py sgd OUT.json | opt23.py adam OUT.json
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "0015"))
from finite23 import ARCH, make_spec, contexts, init_theta, to_dict, from_dict, jacobian, logits_of, softmax, ops_of, selftest  # noqa: E402
import gauge as G  # noqa: E402

A, OPS, SPEC, GROUPS = ARCH, ops_of(), make_spec(ARCH), contexts(ARCH)
NCTX = sum(len(g) for g in GROUPS)
V = A.vocab
TEACHER_SEED, GAUGE_SEED, SHARPNESS = 101, 7, 6.0
STUDENT_SEED = int(os.environ.get("LMNF_STUDENT_SEED", "202"))          # the registered runs use 202 and the points 301 to 310; the smoke run sets other values
FIELD_SEEDS = [int(x) for x in os.environ.get("LMNF_FIELD_SEEDS", ",".join(str(s) for s in range(301, 311))).split(",")]          # ten points, not used in any pilot (the pilots used the student seed 202 and seed 5)
RCOND = 1e-10
RULES = ("sgd", "adam", "ngd-pinv", "ngd-euclid", "ngd-covdamp:1e-6", "ngd-covdamp:1e-2")


def teacher():
    z = logits_of(init_theta(SPEC, np.random.default_rng(TEACHER_SEED), A, OPS), SPEC, GROUPS, A, OPS) * SHARPNESS
    return softmax(z)


def make_gauges(seed=GAUGE_SEED):
    rng = np.random.default_rng(seed)
    perm = [int(i) for i in rng.permutation(A.d)]
    signs = [int(s) for s in rng.choice([1, -1], A.d)]
    Q = OPS.rand_orthogonal(A.d, rng)
    uperm = [np.array(rng.permutation(A.d_ff)) for _ in range(A.n_layers)]
    ident = [np.arange(A.d_ff) for _ in range(A.n_layers)]
    ones = [np.ones(A.d_ff) for _ in range(A.n_layers)]
    cs = [np.sign(rng.standard_normal(A.d_ff)) * 2.0 ** rng.integers(-3, 4, A.d_ff) for _ in range(A.n_layers)]
    As = [[OPS.rand_invertible(A.hd, rng, spread=0.7) for _ in range(A.n_kv)] for _ in range(A.n_layers)]
    scal = [OPS.rand_plane_scalars((A.n_kv, A.hd // 2), rng) for _ in range(A.n_layers)]
    gperms = [list(range(A.n_kv)) for _ in range(A.n_layers)]
    hperms = [[list(reversed(range(A.rep))) for _ in range(A.n_kv)] for _ in range(A.n_layers)]
    return {
        "unit permutation": (True, lambda P: G.mlp_gauge(A, P, OPS, uperm, ones)[1]),
        "signed permutation of the hidden coordinates": (True, lambda P: G.residual_signed_perm(A, P, OPS, perm, signs)[1]),
        "head swap": (True, lambda P: G.head_perm(A, P, OPS, gperms, hperms)[1]),
        "rotation of the stream": (True, lambda P: G.residual_orth(A, P, OPS, Q, fold=True)[1]),
        "unit scaling by +-2^k": (False, lambda P: G.mlp_gauge(A, P, OPS, ident, cs)[1]),
        "invertible matrix on the value dimensions": (False, lambda P: G.ov_gauge(A, P, OPS, As)[1]),
        "complex scalar on the rotary plane": (False, lambda P: G.qk_gauge_rope(A, P, OPS, scal)[1]),
    }


def apply(g, th):
    return from_dict(g(to_dict(th, SPEC, A)), SPEC)


def centered(z):
    """The scores of every context minus their mean over the symbols: the log-probabilities up to a constant per context, which is what the predictive distribution, the loss and the Fisher metric depend on (adding the same number to all the scores of a context changes none of them)."""
    z = np.asarray(z).reshape(NCTX, V)
    return z - z.mean(axis=1, keepdims=True)


def loss_of(z, ps):
    return float(-(ps * np.log(softmax(z))).sum(axis=1).mean())


def fisher_root(p):
    """M^1/2 and its pseudo-inverse for the blocks M_c = diag(p_c) - p_c p_c^T of the predictive distributions, (n_ctx, V, V)."""
    Mc = np.einsum("cv,vw->cvw", p, np.eye(V)) - np.einsum("cv,cw->cvw", p, p)
    lam, U = np.linalg.eigh(Mc)
    half = np.einsum("cvk,ck,cwk->cvw", U, np.where(lam > 1e-12, np.sqrt(np.clip(lam, 0, None)), 0.0), U)        # the zero eigenvalue of M (the shift of all scores of a context) is exactly zero, not its rounding error's root
    inv_half = np.einsum("cvk,ck,cwk->cvw", U, np.where(lam > 1e-12, 1.0 / np.sqrt(np.clip(lam, 1e-300, None)), 0.0), U)
    return half, inv_half


def direction(kind, J, z, ps):
    """The direction v (the update is -v) of a rule at a point with Jacobian J, scores z, teacher ps. With B = M^1/2 J / sqrt(n_ctx) and s = (M^1/2)^+ (p - p*) / sqrt(n_ctx): F = B^T B and
    grad L = B^T s, so that F^+ grad L = B^+ s and the squares of the singular values never form."""
    p = softmax(z)
    g = J.T @ ((p - ps) / NCTX).reshape(-1)
    if kind == "sgd":
        return g
    if kind == "adam":
        return np.sign(g)
    half, inv_half = fisher_root(p)
    n = J.shape[1]
    B = np.einsum("cvw,cwn->cvn", half, J.reshape(NCTX, V, n)).reshape(NCTX * V, n) / np.sqrt(NCTX)
    s = (np.einsum("cvw,cw->cv", inv_half, p - ps) / np.sqrt(NCTX)).reshape(-1)
    if kind == "ngd-pinv":
        return np.linalg.lstsq(B, s, rcond=RCOND)[0]
    if kind == "ngd-euclid":
        aug = np.vstack([B, np.sqrt(1e-9) * np.eye(n)])                  # lambda = 1e-9, an absolute number: the same rule at every point (the eigenvalues of F run from 1.7 to 1e-12)
    elif kind.startswith("ngd-covdamp"):
        mean_eig = float(np.sum(half * half)) / (NCTX * NCTX * (V - 1))   # the mean nonzero eigenvalue of M / n_ctx: a number of the function (it depends on the predictions only), the same at theta and at g.theta
        aug = np.vstack([B, np.sqrt(float(kind.split(":")[1]) * mean_eig) * J])
    else:
        raise ValueError(kind)
    return np.linalg.lstsq(aug, np.concatenate([s, np.zeros(aug.shape[0] - s.size)]), rcond=RCOND)[0]


def velocity(kind, J, z, ps):
    """The velocity of the predictive distributions (centered scores) under the rule: -J v with the shift of every context removed, and the direction v."""
    v = direction(kind, J, z, ps)
    return centered(-(J @ v)).reshape(-1), v


def field_experiment(out):
    selftest()
    ps = teacher()
    gauges = make_gauges()
    res = {"points": FIELD_SEEDS, "rcond": RCOND, "rules": {r: {n: {"relative_differences": []} for n in gauges} for r in RULES}, "step_norms": {r: [] for r in RULES}, "function_unchanged_at_the_transformed_point": []}
    t0 = time.time()
    for seed in FIELD_SEEDS:
        th = init_theta(SPEC, np.random.default_rng(seed), A, OPS)
        z0 = logits_of(th, SPEC, GROUPS, A, OPS)
        J0 = jacobian(th, SPEC, GROUPS, A, OPS)
        vel0 = {}
        for r in RULES:
            vel0[r], v = velocity(r, J0, z0, ps)
            res["step_norms"][r].append(float(np.linalg.norm(v)))
        for name, (orth, g) in gauges.items():
            th1 = apply(g, th)
            z1 = logits_of(th1, SPEC, GROUPS, A, OPS)
            res["function_unchanged_at_the_transformed_point"].append(float(np.max(np.abs(z1 - z0))))
            J1 = jacobian(th1, SPEC, GROUPS, A, OPS)
            for r in RULES:
                vel1, _ = velocity(r, J1, z1, ps)
                res["rules"][r][name]["relative_differences"].append(float(np.linalg.norm(vel1 - vel0[r]) / np.linalg.norm(vel0[r])))
        print(f"point {seed} done ({time.time() - t0:.0f}s)", flush=True)
    for r in RULES:
        for n in gauges:
            d = res["rules"][r][n]["relative_differences"]
            res["rules"][r][n].update({"largest": max(d), "smallest": min(d), "orthogonal_on_the_constants": gauges[n][0]})
        print(f"{r:18s} " + "  ".join(f"{n[:12]}: {res['rules'][r][n]['smallest']:.1e}..{res['rules'][r][n]['largest']:.1e}" for n in gauges), flush=True)
    res["median_step_norm_over_the_step_norm_of_sgd"] = {r: float(np.median(np.array(res["step_norms"][r]) / np.array(res["step_norms"]["sgd"]))) for r in RULES}
    res["functions_unchanged_largest_difference"] = max(res["function_unchanged_at_the_transformed_point"])
    json.dump(res, open(out, "w"), indent=1)


def run(kind, th0, steps, rec_every, ps, lr):
    th = th0.copy()
    recs = {}
    m, v2 = np.zeros_like(th), np.zeros_like(th)
    for k in range(steps + 1):
        z = logits_of(th, SPEC, GROUPS, A, OPS)
        if k % rec_every == 0:
            recs[k] = (z.copy(), loss_of(z, ps))
        if k == steps:
            break
        J = jacobian(th, SPEC, GROUPS, A, OPS)
        g = J.T @ ((softmax(z) - ps) / NCTX).reshape(-1)
        if kind == "sgd":
            th = th - lr * g
        else:
            m = 0.9 * m + 0.1 * g
            v2 = 0.999 * v2 + 0.001 * g * g
            th = th - lr * (m / (1 - 0.9 ** (k + 1))) / (np.sqrt(v2 / (1 - 0.999 ** (k + 1))) + 1e-8)
    return th, recs


TRAJ = {"sgd": (1.0, 200, 10), "adam": (0.01, 200, 10)}


def trajectory_experiment(kind, out):
    selftest()
    lr, steps, rec = TRAJ[kind]
    steps = int(os.environ.get("LMNF_STEPS", steps))
    ps = teacher()
    th0 = init_theta(SPEC, np.random.default_rng(STUDENT_SEED), A, OPS)
    res = {"kind": kind, "lr": lr, "steps": steps, "record_every": rec, "n_contexts": NCTX, "gauges": {}}
    t0 = time.time()
    _, base = run(kind, th0, steps, rec, ps, lr)
    res["loss_first"], res["loss_last"] = base[0][1], base[steps][1]
    print(f"{kind}: loss {res['loss_first']:.4f} -> {res['loss_last']:.4f} ({time.time() - t0:.0f}s)", flush=True)
    for gname, (orth, g) in make_gauges().items():
        th1 = apply(g, th0)
        start = float(np.max(np.abs(logits_of(th0, SPEC, GROUPS, A, OPS) - logits_of(th1, SPEC, GROUPS, A, OPS))))
        t1 = time.time()
        _, rec1 = run(kind, th1, steps, rec, ps, lr)
        d = {k: float(np.max(np.abs(centered(base[k][0]) - centered(rec1[k][0])))) for k in base}
        res["gauges"][gname] = {"orthogonal_on_the_constants": orth, "start_difference": start, "max_difference": max(d.values()), "difference_at_end": d[steps], "difference_by_step": d, "seconds": round(time.time() - t1)}
        print(f"  {gname:46s} orthogonal {str(orth):5s} start {start:.1e}  largest difference of the scores {max(d.values()):.3e}  ({time.time() - t1:.0f}s)", flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "field":
        field_experiment(sys.argv[2])
    else:
        trajectory_experiment(sys.argv[1], sys.argv[2])
