# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cell 0V: verify that the isolated variables are the correct ones (docs/batches/0033.md, amendment 1).
usage: verify.py MODEL_DIR TEXT_FILE OUT_JSON"""
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402


def capture(model, tok, X, t):
    """Hooks: the embedding term, every layer's attention and feed-forward output, the final norm's input, the down
    projection's input in the last layer; all at the last position. Returns the dict and the logits."""
    c = {"attn": [], "mlp": []}; hs = []
    hs.append(model.model.embed_tokens.register_forward_hook(lambda m, i, o: c.__setitem__("emb", o[0, -1].detach().clone())))
    for L in model.model.layers:
        hs.append(L.self_attn.register_forward_hook(lambda m, i, o: c["attn"].append((o[0] if isinstance(o, tuple) else o)[0, -1].detach().clone())))
        hs.append(L.mlp.register_forward_hook(lambda m, i, o: c["mlp"].append(o[0, -1].detach().clone())))
    hs.append(model.model.norm.register_forward_pre_hook(lambda m, i: c.__setitem__("xL", i[0][0, -1].detach().clone())))
    hs.append(model.model.layers[-1].mlp.down_proj.register_forward_pre_hook(lambda m, i: c.__setitem__("a", i[0][0, -1].detach().clone())))
    with torch.no_grad():
        z = model(input_ids=core.prompt_ids(tok, X), use_cache=False).logits[0, -1].detach().clone()
    for h in hs:
        h.remove()
    return c, z


def decomposition(model, c, z, t, eps_n):
    E = model.model.embed_tokens.weight.data; g = model.model.norm.weight.data
    xL = c["xL"]; r = torch.sqrt((xL * xL).mean() + eps_n)
    recon = c["emb"] + sum(c["attn"]) + sum(c["mlp"])
    zc = E @ (g * xL / r)
    o = z.clone(); o[t] = float("-inf"); j = int(o.argmax())
    d = (E[t] - E[j]) * g / r
    comp = {"embedding": float(d @ c["emb"])}
    for k, (a, m) in enumerate(zip(c["attn"], c["mlp"])):
        comp[f"attn.{k}"] = float(d @ a); comp[f"ffn.{k}"] = float(d @ m)
    margin = float(z[t] - z[j])
    return {"competitor": j, "margin": margin, "sum_of_components": sum(comp.values()),
            "residual_reconstruction_max_diff": float((recon - xL).abs().max()), "logits_recomputed_max_diff": float((zc - z).abs().max()),
            "components": comp}, r, j


def main():
    model_dir, text_path, out = sys.argv[1:4]
    clock = core.Clock()
    model, tok = core.load(model_dir)
    atoms_ = core.atoms(); targs = core.target_ids(tok, atoms_)
    ms, _ = core.all_margins(model, tok, atoms_, targs)
    res = {"gate_count": int((ms > 0).sum()), "n": len(atoms_)}
    eps_n = model.config.rms_norm_eps
    fi = [X for X, _ in atoms_].index("France"); si = [X for X, _ in atoms_].index("Spain")
    # V1
    v1 = {}
    for name, k in (("France", fi), ("Spain", si)):
        c, z = capture(model, tok, name, targs[k])
        dec, r, j = decomposition(model, c, z, targs[k], eps_n); v1[name] = dec
    res["V1"] = v1
    # V2 and V3 on France
    X, t = "France", targs[fi]
    c, z = capture(model, tok, X, t); dec, r, j = decomposition(model, c, z, t, eps_n)
    m0 = core.margin_and_grad(model, tok, X, t)
    E = model.model.embed_tokens.weight.data; g = model.model.norm.weight.data; xL = c["xL"]; a = c["a"]; d = xL.numel()
    n = xL / r
    ge = model.model.embed_tokens.weight.grad.clone(); gg = model.model.norm.weight.grad.clone(); gd = model.model.layers[-1].mlp.down_proj.weight.grad.clone()
    prompt_tokens = set(core.prompt_ids(tok, X)[0].tolist())
    # closed forms
    a_gain = (E[t] - E[j]) * n
    rel = lambda x, y: float((x - y).abs().max() / max(float(y.abs().max()), 1e-12))
    # head rows
    a_t = g * n; a_j = -g * n
    # down projection: dz_v/dx_i = E[v,i] g_i / r - x_i / (d r^3) * sum_k E[v,k] g_k x_k
    def dz(v):
        s = float((E[v] * g * xL).sum())
        return E[v] * g / r - xL / (d * r ** 3) * s
    D = dz(t) - dz(j)
    Gd = torch.outer(D, a)
    v2 = {"final_norm_gain_rel_err": rel(gg, a_gain), "head_row_target_rel_err": rel(ge[t], a_t) if t not in prompt_tokens else None,
          "head_row_competitor_rel_err": rel(ge[j], a_j) if j not in prompt_tokens else None, "last_down_rel_err": rel(gd, Gd),
          "target_in_prompt": t in prompt_tokens, "competitor_in_prompt": j in prompt_tokens, "entries_down": int(gd.numel())}
    res["V2"] = v2
    # V3: exact zero off-path, and the conditional nature of "off-path" (the margin is a maximum)
    mask = torch.ones(ge.shape[0], dtype=torch.bool); mask[[t, j]] = False
    for p_ in prompt_tokens:
        mask[p_] = False
    off_nonzero = int((ge[mask] != 0).sum())
    gen = torch.Generator().manual_seed(33)
    cand_rows = torch.nonzero(mask).squeeze(1)
    rows = cand_rows[torch.randperm(len(cand_rows), generator=gen)[:1000]]            # 1,000 distinct off-path rows, one entry each
    cols = torch.randint(0, E.shape[1], (len(rows),), generator=gen)
    delta = (torch.rand(len(rows), generator=gen) - 0.5) * 0.8
    shift = delta * (g * n)[cols]                                                       # the exact change of that row's logit
    keep = (z[rows] + shift) < (z[j] - 1e-3)                                            # the registered condition: stay below the competitor
    ps = dict(core.params(model)); Emb = ps["model.embed_tokens.weight"]
    saved = Emb.data[rows[keep], cols[keep]].clone()
    with torch.no_grad():
        Emb.data[rows[keep], cols[keep]] += delta[keep]
        z2 = model(input_ids=core.prompt_ids(tok, X), use_cache=False).logits[0, -1]
        m2, j2 = core.margin_of(z2, t); m2 = float(m2)
        Emb.data[rows[keep], cols[keep]] = saved
        # the crossing: raise one off-path row's logit to the competitor's plus 1; the new margin is predicted exactly
        v_star = int(rows[0]); ci = int(cols[0]); need = float(z[j] + 1.0 - z[v_star]); dstar = need / float((g * n)[ci])
        old = float(Emb.data[v_star, ci]); Emb.data[v_star, ci] = old + dstar
        z3 = model(input_ids=core.prompt_ids(tok, X), use_cache=False).logits[0, -1]
        m3, j3 = core.margin_of(z3, t); Emb.data[v_star, ci] = old
    res["V3"] = {"offpath_rows": int(mask.sum()), "nonzero_gradient_entries_offpath": off_nonzero, "perturbed_entries": int(keep.sum()),
                 "rejected_for_crossing": int((~keep).sum()), "margin_before": m0, "margin_after": m2, "abs_change": abs(m2 - m0), "competitor_same": int(j2) == j,
                 "crossing": {"row": v_star, "predicted_margin": float(z[t] - (z[j] + 1.0)), "actual_margin": float(m3), "new_competitor": int(j3),
                              "abs_error": abs(float(m3) - float(z[t] - (z[j] + 1.0)))}}
    core.dump(out, res)                                     # V1 to V3 are kept even if a later part fails
    if os.environ.get("VERIFY_ONLY") == "v123":
        print("V3", res["V3"]); return
    # V4: finite perturbations against first order
    ps_list = core.params(model)
    g_all = [p.grad.clone() for _, p in ps_list]
    score = [(gt.abs() * torch.clamp(0.05 * p.data.abs(), min=0.002)).view(-1) for gt, (_, p) in zip(g_all, ps_list)]
    top = []
    for ti, s in enumerate(score):
        v, i = torch.topk(s, min(1000, s.numel())); top.append((v, torch.full_like(i, ti), i))
    V = torch.cat([x[0] for x in top]); T = torch.cat([x[1] for x in top]); I = torch.cat([x[2] for x in top])
    order = torch.argsort(V, descending=True)
    A = [(int(T[o]), int(I[o])) for o in order[:1000]]
    nxt = [(int(T[o]), int(I[o])) for o in order[1000:min(len(order), 6000)]]
    rng = np.random.default_rng(33)
    C = [nxt[k] for k in rng.choice(len(nxt), 1000, replace=False)]
    sizes = np.array([p.numel() for _, p in ps_list]); probs = sizes / sizes.sum()
    B = []
    for ti in rng.choice(len(ps_list), 1000, p=probs):
        B.append((int(ti), int(rng.integers(0, sizes[ti]))))
    coords = [("A", *c_) for c_ in A] + [("C", *c_) for c_ in C] + [("B", *c_) for c_ in B]
    pred, act, strat, switches = [], [], [], 0
    with torch.no_grad():
        for s_, ti, fi_ in coords:
            p = ps_list[ti][1]; flat = p.data.view(-1); w = float(flat[fi_]); h = max(0.05 * abs(w), 0.002); gi = float(g_all[ti].view(-1)[fi_])
            flat[fi_] = w + h; zp = model(input_ids=core.prompt_ids(tok, X), use_cache=False).logits[0, -1]; mp, jp = core.margin_of(zp, t)
            flat[fi_] = w - h; zm = model(input_ids=core.prompt_ids(tok, X), use_cache=False).logits[0, -1]; mm, jm = core.margin_of(zm, t)
            flat[fi_] = w
            pred.append(gi * h); act.append(float(mp - mm) / 2.0); strat.append(s_); switches += int(jp != j or jm != j)
    pred, act, strat = np.array(pred), np.array(act), np.array(strat)
    from scipy.stats import spearmanr
    sp = float(spearmanr(np.abs(pred), np.abs(act)).correlation)
    top_pred = set(np.argsort(-np.abs(pred))[:100]); top_act = set(np.argsort(-np.abs(act))[:100])
    mA = strat == "A"
    relerr = np.abs(pred[mA] - act[mA]) / np.maximum(np.abs(act[mA]), 1e-6)
    res["V4"] = {"sampled": len(coords), "spearman_abs": sp, "precision_at_100": len(top_pred & top_act) / 100.0, "median_rel_err_top_stratum": float(np.median(relerr)),
                 "competitor_switches": switches, "median_abs_actual_by_stratum": {k: float(np.median(np.abs(act[strat == k]))) for k in "ABC"},
                 "median_abs_pred_by_stratum": {k: float(np.median(np.abs(pred[strat == k]))) for k in "ABC"}}
    # V5: support energy
    total = float(sum((gt.double() ** 2).sum() for gt in g_all))
    idx = core.select_support(model, tok, [(X, t)], 30000)
    core.margin_and_grad(model, tok, X, t); gS = core.grad_vec(model, idx)
    res["V5"] = {"support": 30000, "energy_fraction": float((gS ** 2).sum() / total), "grad_norm_total": total ** 0.5, "grad_norm_support": float((gS ** 2).sum() ** 0.5),
                 "min_change_full_support": (core.EPS - m0) / total ** 0.5, "min_change_support": (core.EPS - m0) / float((gS ** 2).sum() ** 0.5)}
    res["seconds"] = clock.s()
    core.dump(out, res)
    print("V1", {k: (round(v["residual_reconstruction_max_diff"], 6), round(v["logits_recomputed_max_diff"], 6), round(v["sum_of_components"] - v["margin"], 6)) for k, v in v1.items()})
    print("V2", v2); print("V3", res["V3"]); print("V4", res["V4"]); print("V5", res["V5"], "gate", res["gate_count"], f"{clock.s()} s")


if __name__ == "__main__":
    main()
