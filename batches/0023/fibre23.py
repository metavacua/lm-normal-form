# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Batch 0023, the group of the function at a point that training has reached. The count of the dimension of the group (cov23.py, batch 0015) is for a generic point; the point that training reaches
# is not generic, and the fibre of the map from parameters to the function there may be larger for reasons that are not symmetries (a unit that no context turns on, two heads that have become equal).
# Here the finite instance (finite23.py) is trained by Adam to a target that it cannot realize (a distribution over the next symbol, drawn at random for every context), and the rank deficiency of the
# Jacobian is measured at the initial point and at the point reached. Measurements only; the verdicts are in grade23.py.
# Usage: fibre23.py OUT.json [--smoke]
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from finite23 import ARCH, make_spec, contexts, init_theta, jacobian, logits_of, softmax, ops_of, selftest  # noqa: E402
from cov23 import formula, TOL  # noqa: E402

SEEDS = [21, 22, 23, 24, 25]
STEPS, LR, ALPHA = 300, 0.02, 0.3


def deficiency(th, spec, groups, a, ops):
    J = jacobian(th, spec, groups, a, ops)
    s = np.linalg.svd(J, compute_uv=False)
    r = int((s > s[0] * TOL).sum())
    return {"deficiency": int(th.size - r), "sv_at_cut": float(s[r - 1]), "sv_after_cut": float(s[r]) if r < len(s) else 0.0, "soft_directions_below_1e-3": int(((s <= s[0] * 1e-3) & (s > s[0] * TOL)).sum()),
            "soft_directions_below_1e-5": int(((s <= s[0] * 1e-5) & (s > s[0] * TOL)).sum())}


def train(th0, spec, groups, ps, a, ops, steps, lr):
    th, m, v = th0.copy(), np.zeros_like(th0), np.zeros_like(th0)
    n = sum(len(g) for g in groups)
    losses = []
    for k in range(steps):
        z = logits_of(th, spec, groups, a, ops)
        p = softmax(z)
        losses.append(float(-(ps * np.log(p)).sum(axis=1).mean()))
        g = jacobian(th, spec, groups, a, ops).T @ ((p - ps) / n).reshape(-1)
        m, v = 0.9 * m + 0.1 * g, 0.999 * v + 0.001 * g * g
        th = th - lr * (m / (1 - 0.9 ** (k + 1))) / (np.sqrt(v / (1 - 0.999 ** (k + 1))) + 1e-8)
    z = logits_of(th, spec, groups, a, ops)
    losses.append(float(-(ps * np.log(softmax(z))).sum(axis=1).mean()))
    return th, losses


def main(out, smoke):
    selftest()
    a, ops = ARCH, ops_of()
    spec, groups = make_spec(a), contexts(a)
    n = sum(len(g) for g in groups)
    seeds = [98] if smoke else SEEDS
    steps = 20 if smoke else STEPS
    rows = []
    for seed in seeds:
        rng = np.random.default_rng(seed)
        th0 = init_theta(spec, rng, a, ops)
        ps = np.random.default_rng(1000 + seed).dirichlet(ALPHA * np.ones(a.vocab), size=n)
        th, losses = train(th0, spec, groups, ps, a, ops, steps, LR)
        row = {"seed": seed, "steps": steps, "loss_first": losses[0], "loss_last": losses[-1], "entropy_floor_of_the_target": float(-(ps * np.log(ps)).sum(axis=1).mean()),
               "expected": formula(a, "base"), "initial": deficiency(th0, spec, groups, a, ops), "trained": deficiency(th, spec, groups, a, ops)}
        rows.append(row)
        print(f"seed {seed}: loss {losses[0]:.3f} -> {losses[-1]:.3f}; deficiency at the start {row['initial']['deficiency']}, at the trained point {row['trained']['deficiency']} (derived {row['expected']}); "
              f"soft directions (below 1e-3 of the largest): {row['initial']['soft_directions_below_1e-3']} -> {row['trained']['soft_directions_below_1e-3']}", flush=True)
    json.dump({"smoke": smoke, "learning_rate": LR, "alpha_of_the_target": ALPHA, "runs": rows}, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], "--smoke" in sys.argv)
