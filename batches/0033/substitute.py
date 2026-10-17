# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cells 03 and 04: sequence-wise and graph-wise substitution (docs/batches/0033.md).

usage: substitute.py MODEL_DIR TEXT_FILE OUT_DIR [--only sequence|G11|G11perm|G189] [--support 30000]
Sequence-wise: one IO sequence (France), the minimum-L2 change by Newton, the minimum-L1 change with a box (a fractional
knapsack), and a random change of the same support and norm as a control. Graph-wise: a linear program per round
(HiGHS), install constraints, preserve constraints, elastic slack that names the facts in conflict, re-linearised at
the bf16-rounded weights each round."""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402

TRUST = [1.0, 1.0, 0.5, 0.5, 0.25, 0.25]


def solve_lp(J, m, e, box, M=100.0):
    from scipy.optimize import linprog
    F, N = J.shape
    c = np.concatenate([np.ones(2 * N), M * np.ones(F)])
    A = np.hstack([-J, J, -np.eye(F)])
    b = m - e
    bounds = np.concatenate([np.stack([np.zeros(N), box], 1), np.stack([np.zeros(N), box], 1), np.stack([np.zeros(F), np.full(F, np.inf)], 1)])
    r = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs", options={"time_limit": 900})
    if r.x is None:
        return None, None, r
    return r.x[:N] - r.x[N:2 * N], r.x[2 * N:], r


def change_stats(model, idx, w0):
    cur = core.get_vals(model, idx).double(); d = cur - w0.double()
    return {"weights_changed": int((d != 0).sum()), "l1": float(d.abs().sum()), "l2": float(d.norm()), "relative_l2_to_support": float(d.norm() / w0.double().norm())}


def sequence(model, tok, atoms_, targs, text_ids, the_id, N):
    fi = [x for x, _ in atoms_].index("France"); X, t = "France", targs[fi]
    base = core.state(model, tok, atoms_, targs, text_ids, the_id)
    idx = core.select_support(model, tok, [(X, t)], N)
    w0 = core.get_vals(model, idx).clone()
    out = {"base": {k: v for k, v in base.items() if k != "margins"}, "support": N}
    for variant in ("L2", "L1box"):
        core.set_vals(model, idx, w0); traj = []; reached = None
        for rnd in range(8):
            m = core.margin_and_grad(model, tok, X, t); gv = core.grad_vec(model, idx); traj.append({"round": rnd, "margin": m})
            if m >= core.EPS:
                reached = rnd; break
            need = core.EPS - m
            if variant == "L2":
                d = need * gv / (gv @ gv)
            else:
                cur = core.get_vals(model, idx).double(); box = np.maximum(0.25 * cur.abs().numpy(), 0.01)
                d = torch.zeros_like(gv); rem = need
                for k in torch.argsort(gv.abs(), descending=True).tolist():
                    gk = abs(float(gv[k])); cap = box[k] * gk
                    if cap >= rem:
                        d[k] = float(np.sign(float(gv[k]))) * rem / gk; rem = 0; break
                    d[k] = float(np.sign(float(gv[k]))) * box[k]; rem -= cap
            core.apply_delta(model, idx, d)
        mf = float(core.all_margins(model, tok, [atoms_[fi]], [t])[0][0])
        st = core.state(model, tok, atoms_, targs, text_ids, the_id)
        others_before = (base["margins"] > 0).copy(); others_after = st["margins"] > 0; others_before[fi] = False; others_after[fi] = False
        out[variant] = {"rounds_to_reach": reached, "trajectory": traj, "final_margin": mf, **change_stats(model, idx, w0), "count_after": st["count"], "bits_per_token_after": st["bits_per_token"],
                        "the_answers_after": st["the_answers"], "other_facts_changed_to_wrong": int((others_before & ~others_after).sum()), "other_facts_changed_to_right": int((~others_before & others_after).sum()),
                        "T12_correct_after": st["T12_correct"]}
        print("sequence", variant, {k: v for k, v in out[variant].items() if k not in ("trajectory", "T12_correct_after")}, flush=True)
        if variant == "L2":
            # control: a random change with the same support and the same L2 norm
            d_sol = core.get_vals(model, idx).double() - w0.double(); supp = (d_sol != 0).nonzero().squeeze(1); nrm = float(d_sol.norm())
            gen = torch.Generator().manual_seed(33); rd = torch.zeros_like(d_sol); rd[supp] = torch.randn(len(supp), generator=gen, dtype=torch.float64); rd = rd * (nrm / float(rd.norm()))
            core.set_vals(model, idx, w0); core.apply_delta(model, idx, rd)
            mr = float(core.all_margins(model, tok, [atoms_[fi]], [t])[0][0])
            out["random_control"] = {"solver_change_in_margin": mf - base["margins"][fi], "random_change_in_margin": mr - float(base["margins"][fi]), "ratio": abs(mr - float(base["margins"][fi])) / max(abs(mf - float(base["margins"][fi])), 1e-12),
                                     "support_size": int(len(supp))}
    core.set_vals(model, idx, w0)
    return out


def graph(model, tok, atoms_, targs, text_ids, the_id, name, install, preserve, N, rounds=6):
    """install: {fact index: target token}; preserve: fact indices (their own targets)."""
    base = core.state(model, tok, atoms_, targs, text_ids, the_id)
    facts = [(k, atoms_[k][0], tg, "install") for k, tg in install.items()] + [(k, atoms_[k][0], targs[k], "preserve") for k in preserve]
    idx = core.select_support(model, tok, [(X, tg) for _, X, tg, _ in facts], N)
    w0 = core.get_vals(model, idx).clone(); hist = []
    for r in range(rounds):
        rows, m = [], []
        for k, X, tg, kind in facts:
            m.append(core.margin_and_grad(model, tok, X, tg)); rows.append(core.grad_vec(model, idx).numpy())
        J = np.stack(rows); m = np.array(m)
        e = np.array([core.EPS if kind == "install" else min(core.EPS, mm) for (_, _, _, kind), mm in zip(facts, m)])
        cur = core.get_vals(model, idx).double().numpy(); box = TRUST[r] * np.maximum(0.25 * np.abs(cur), 0.01)
        d, s, res = solve_lp(J, m, e, box)
        if d is None:
            hist.append({"round": r, "lp_status": int(res.status), "message": res.message}); break
        core.apply_delta(model, idx, torch.tensor(d))
        ms_c = np.array([float(core.all_margins(model, tok, [atoms_[k]], [tg])[0][0]) for k, _, tg, _ in facts])
        st = core.state(model, tok, atoms_, targs, text_ids, the_id)
        inst = np.array([kind == "install" for _, _, _, kind in facts])
        rec = {"round": r, "lp_status": int(res.status), "lp_objective": float(res.fun), "slack_facts": [facts[q][1] for q in np.nonzero(s > 1e-6)[0]], "nnz_lp": int((np.abs(d) > 1e-12).sum()),
               "installs_satisfied": int((ms_c[inst] > 0).sum()), "installs": int(inst.sum()), "preserved_kept": int((ms_c[~inst] > 0).sum()), "preserves": int((~inst).sum()),
               "count_all": st["count"], "bits_per_token": st["bits_per_token"], "the_answers": st["the_answers"], "T12_correct": st["T12_correct"], **change_stats(model, idx, w0), "seconds": 0}
        hist.append(rec)
        print(name, {k: v for k, v in rec.items() if k not in ("T12_correct", "slack_facts")}, "slack:", rec["slack_facts"][:6], flush=True)
        if rec["installs_satisfied"] == rec["installs"] and rec["preserved_kept"] == rec["preserves"]:
            break
    final = change_stats(model, idx, w0)
    diff = (core.get_vals(model, idx).double() - w0.double()).numpy()
    ps = core.params(model); off = 0; arrs = {}
    for ti in sorted(idx):
        n_ = len(idx[ti]); dd = diff[off:off + n_]; nz = np.nonzero(dd)[0]
        if len(nz):
            arrs[ps[ti][0] + ".flat"] = idx[ti].numpy()[nz]; arrs[ps[ti][0] + ".delta"] = dd[nz]
        off += n_
    by_sort = {}
    for key in arrs:
        if key.endswith(".delta"):
            pn = key[:-6]; k_ = (core.tensor_sort(pn), core.tensor_layer(pn)); by_sort[str(k_)] = by_sort.get(str(k_), 0) + int(len(arrs[key]))
    core.set_vals(model, idx, w0)
    return {"name": name, "install": len(install), "preserve": len(preserve), "rounds": hist, "final": final, "changed_by_sort_and_layer": by_sort, "base_count": base["count"]}, arrs


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("model_dir"); ap.add_argument("text_file"); ap.add_argument("out_dir"); ap.add_argument("--only", default="all"); ap.add_argument("--support", type=int, default=30000)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    clock = core.Clock()
    model, tok = core.load(a.model_dir)
    atoms_ = core.atoms(); targs = core.target_ids(tok, atoms_); text_ids = core.english_ids(tok, a.text_file); the_id = tok(" the", add_special_tokens=False)["input_ids"][0]
    base = core.state(model, tok, atoms_, targs, text_ids, the_id)
    res = {"gate_count": base["count"], "base_bits_per_token": base["bits_per_token"], "base_the_answers": base["the_answers"]}
    print("base", {k: v for k, v in base.items() if k != "margins"}, flush=True)
    wrong = [k for k in range(len(atoms_)) if base["margins"][k] <= 0]; right = [k for k in range(len(atoms_)) if base["margins"][k] > 0]
    t11 = [k for k, (X, _) in enumerate(atoms_) if X in core.T12]
    runs = {"G11": ({k: targs[k] for k in t11}, [k for k in right if k not in t11]),
            "G11perm": ({k: targs[t11[(i + 1) % len(t11)]] for i, k in enumerate(t11)}, [k for k in right if k not in t11]),
            "G189": ({k: targs[k] for k in wrong}, right)}
    if a.only in ("all", "sequence"):
        res["sequence"] = sequence(model, tok, atoms_, targs, text_ids, the_id, a.support); core.dump(os.path.join(a.out_dir, "substitute.json"), res)
    for name in ("G11", "G11perm", "G189"):
        if a.only in ("all", name):
            r, arrs = graph(model, tok, atoms_, targs, text_ids, the_id, name, runs[name][0], runs[name][1], a.support, rounds=(5 if name == "G189" else 6))
            res[name] = r; np.savez_compressed(os.path.join(a.out_dir, f"diff_{name}.npz"), **arrs); core.dump(os.path.join(a.out_dir, "substitute.json"), res)
    res["seconds"] = clock.s(); core.dump(os.path.join(a.out_dir, "substitute.json"), res)


if __name__ == "__main__":
    main()
