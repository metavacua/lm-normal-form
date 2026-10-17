# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cell 06: isolate the IO path of a prompt and verify the isolation by sufficiency (docs/batches/0033.md, amendment 3).

usage: path.py MODEL_DIR NAME OUT_DIR [COUNTRY ...]
For each prompt: one forward pass with hooks on every position; per-(layer, position) scores of the feed-forward units by
four criteria; per-layer scores of the heads; sufficiency curves of the masked forward (selection recomputed inside the
masked run for the activation and gate criteria, as LARQL's walk does for the gate); the smallest sufficient K and H;
the path graph with exact edge weights; the wall-clock time of the isolation."""
import math
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402

KS = [1, 2, 5, 10, 20, 50, 100, 200, 400, 800]
SUFFICIENT = {"kl": 0.05, "margin": 0.1}


def capture_all(model, ids):
    """Every position: embedding, per-layer gate/up/act, attention input to o_proj, attention and mlp outputs, the residual
    entering the feed-forward norm; and the residual entering the final norm."""
    L = model.config.num_hidden_layers
    c = {k: [None] * L for k in ("gate", "up", "act", "o_in", "attn", "mlp", "xmid")}
    hs = [model.model.embed_tokens.register_forward_hook(lambda m, i, o: c.__setitem__("emb", o[0].detach().clone()))]
    for k, Ly in enumerate(model.model.layers):
        hs.append(Ly.mlp.gate_proj.register_forward_hook(lambda m, i, o, k=k: c["gate"].__setitem__(k, o[0].detach().clone())))
        hs.append(Ly.mlp.up_proj.register_forward_hook(lambda m, i, o, k=k: c["up"].__setitem__(k, o[0].detach().clone())))
        hs.append(Ly.mlp.down_proj.register_forward_pre_hook(lambda m, i, k=k: c["act"].__setitem__(k, i[0][0].detach().clone())))
        hs.append(Ly.self_attn.o_proj.register_forward_pre_hook(lambda m, i, k=k: c["o_in"].__setitem__(k, i[0][0].detach().clone())))
        hs.append(Ly.self_attn.register_forward_hook(lambda m, i, o, k=k: c["attn"].__setitem__(k, (o[0] if isinstance(o, tuple) else o)[0].detach().clone())))
        hs.append(Ly.mlp.register_forward_hook(lambda m, i, o, k=k: c["mlp"].__setitem__(k, o[0].detach().clone())))
        hs.append(Ly.post_attention_layernorm.register_forward_pre_hook(lambda m, i, k=k: c["xmid"].__setitem__(k, i[0][0].detach().clone())))
    hs.append(model.model.norm.register_forward_pre_hook(lambda m, i: c.__setitem__("xL", i[0][0].detach().clone())))
    with torch.no_grad():
        z = model(input_ids=ids, use_cache=False).logits[0, -1].detach().clone()
    for h in hs:
        h.remove()
    return c, z


def masked_logits(model, ids, K=None, unit_mode="act", static=None, H=None, head_scores=None, head_mode="static", seed=0):
    """The last-position logits with only K units per (layer, position) and H heads per layer kept."""
    L = model.config.num_hidden_layers; nh = model.config.num_attention_heads; hd = model.config.hidden_size // nh
    gate_now = {}; hs = []
    gen = torch.Generator().manual_seed(seed)
    for k, Ly in enumerate(model.model.layers):
        if K is not None:
            hs.append(Ly.mlp.gate_proj.register_forward_hook(lambda m, i, o, k=k: gate_now.__setitem__(k, o.detach())))
            def pre(m, i, k=k):
                a = i[0]
                if unit_mode == "act":
                    sc = a.abs()
                elif unit_mode == "gate":
                    sc = gate_now[k].abs()
                elif unit_mode == "static":
                    sc = static[k][None]
                else:
                    sc = torch.rand(a.shape, generator=gen)
                idx = sc.topk(min(K, a.shape[-1]), dim=-1).indices
                mask = torch.zeros_like(a); mask.scatter_(-1, idx, 1.0)
                return (a * mask,)
            hs.append(Ly.mlp.down_proj.register_forward_pre_hook(pre))
        if H is not None:
            def prh(m, i, k=k):
                a = i[0]; T = a.shape[1]
                if head_mode == "static":
                    sc = head_scores[k]
                else:
                    sc = torch.rand(nh, generator=gen)
                mask = torch.zeros(nh); mask[sc.topk(min(H, nh)).indices] = 1.0
                return ((a.view(1, T, nh, hd) * mask[None, None, :, None]).view(1, T, nh * hd),)
            hs.append(Ly.self_attn.o_proj.register_forward_pre_hook(prh))
    with torch.no_grad():
        z = model(input_ids=ids, use_cache=False).logits[0, -1].detach().clone()
    for h in hs:
        h.remove()
    return z


def fidelity(z, zfull, t):
    pf = torch.softmax(zfull.double(), -1); pm = torch.softmax(z.double(), -1)
    kl = float((pf * (torch.log(pf + 1e-30) - torch.log(pm + 1e-30))).sum())
    o = z.clone(); o[t] = float("-inf"); of = zfull.clone(); of[t] = float("-inf")
    return {"top1_same": bool(z.argmax() == zfull.argmax()), "kl": kl, "margin": float(z[t] - o.max()), "margin_full": float(zfull[t] - of.max()),
            "margin_abs_diff": abs(float(z[t] - o.max()) - float(zfull[t] - of.max())), "target_rank": int((z > z[t]).sum()) + 1}


def sufficient(f):
    return f["top1_same"] and f["kl"] <= SUFFICIENT["kl"] and f["margin_abs_diff"] <= SUFFICIENT["margin"]


def isolate(model, tok, name, X, C, clock):
    t0 = time.time()
    L = model.config.num_hidden_layers; nh = model.config.num_attention_heads; nkv = model.config.num_key_value_heads; hd = model.config.hidden_size // nh
    ffn = model.config.intermediate_size; eps_n = model.config.rms_norm_eps
    t = core.target_ids(tok, [(X, C)])[0]; ids = core.prompt_ids(tok, X); T = ids.shape[1]
    c, zfull = capture_all(model, ids)
    E = model.model.embed_tokens.weight.data; g = model.model.norm.weight.data; xL = c["xL"][-1]; r = torch.sqrt((xL ** 2).mean() + eps_n)
    o = zfull.clone(); o[t] = float("-inf"); j = int(o.argmax())
    dvec = (E[t] - E[j]) * g / r
    # scores
    direct = []
    static = []
    for k in range(L):
        Wd = model.model.layers[k].mlp.down_proj.weight.data
        sc = c["act"][k].abs().clone()
        sc[-1] = (c["act"][k][-1] * (dvec @ Wd)).abs()
        static.append(sc)
        direct.append((c["act"][k][-1] * (dvec @ Wd)))
    head_scores = []
    for k in range(L):
        Wo = model.model.layers[k].self_attn.o_proj.weight.data
        head_scores.append(torch.stack([(Wo[:, h * hd:(h + 1) * hd] @ c["o_in"][k][-1][h * hd:(h + 1) * hd]).norm() for h in range(nh)]))
    # sufficiency: units
    out = {"prompt": X, "target": t, "competitor": j, "positions": T}
    out["sanity_all_kept"] = fidelity(masked_logits(model, ids, K=ffn, unit_mode="act"), zfull, t)
    curves = {}
    for crit in ("act", "gate", "static", "random"):
        curves[crit] = {}
        for K in KS + [ffn]:
            if K > ffn:
                continue
            curves[crit][K] = fidelity(masked_logits(model, ids, K=K, unit_mode=crit, static=static, seed=7), zfull, t)
    out["unit_curves"] = curves
    kstar = {crit: next((K for K in sorted(curves[crit]) if sufficient(curves[crit][K])), None) for crit in curves}
    out["K_star"] = kstar
    # sufficiency: heads
    hcurves = {}
    for hm in ("static", "random"):
        hcurves[hm] = {}
        for Hn in [1, 2, 3, 5, nh]:
            if Hn > nh:
                continue
            hcurves[hm][Hn] = fidelity(masked_logits(model, ids, H=Hn, head_scores=head_scores, head_mode=hm, seed=7), zfull, t)
    out["head_curves"] = hcurves
    out["H_star"] = {hm: next((Hn for Hn in sorted(hcurves[hm]) if sufficient(hcurves[hm][Hn])), None) for hm in hcurves}
    # joint
    Kj = kstar["act"] or ffn; Hj = out["H_star"]["static"] or nh
    out["joint"] = {"K": Kj, "H": Hj, **fidelity(masked_logits(model, ids, K=Kj, unit_mode="act", H=Hj, head_scores=head_scores, head_mode="static"), zfull, t)}
    out["joint_fraction_of_units"] = Kj / ffn; out["joint_fraction_of_heads"] = Hj / nh
    # the path graph: nodes at the last position; the exact edges into the gate and up pre-activations of the top units
    comps = [("embedding", c["emb"][-1])] + [(n, v) for k in range(L) for n, v in ((f"attn.{k}", c["attn"][k][-1]), (f"ffn.{k}", c["mlp"][k][-1]))]
    order = {n: i for i, (n, _) in enumerate(comps)}
    flat = torch.cat([d.abs() for d in direct]); top = torch.topk(flat, 50).indices.tolist()
    units = []; sum_err = 0.0
    for f in top:
        k, i = divmod(f, ffn); Ly = model.model.layers[k]
        xmid = c["xmid"][k][-1]; rr = torch.sqrt((xmid ** 2).mean() + eps_n); gp = Ly.post_attention_layernorm.weight.data
        wg = Ly.mlp.gate_proj.weight.data[i] * gp / rr; wu = Ly.mlp.up_proj.weight.data[i] * gp / rr
        srcs = [(n, v) for n, v in comps if n == "embedding" or (n.startswith("attn.") and int(n.split(".")[1]) <= k) or (n.startswith("ffn.") and int(n.split(".")[1]) < k)]
        eg = {n: float(wg @ v) for n, v in srcs}; eu = {n: float(wu @ v) for n, v in srcs}
        sum_err = max(sum_err, abs(sum(eg.values()) - float(c["gate"][k][-1][i])), abs(sum(eu.values()) - float(c["up"][k][-1][i])))
        tot = sum(abs(v) for v in eg.values()); topn = sorted(eg.items(), key=lambda kv: -abs(kv[1]))
        Wd = Ly.mlp.down_proj.weight.data
        rd = torch.topk(E @ (g * Wd[:, i]), 5)
        units.append({"layer": k, "relative_depth": k / L, "index": i, "contribution": float(direct[k][i]), "gate_pre": float(c["gate"][k][-1][i]), "up_pre": float(c["up"][k][-1][i]),
                      "edges_gate": [{"from": n, "weight": v} for n, v in topn[:8]], "top5_source_share": float(sum(abs(v) for _, v in topn[:5]) / tot) if tot > 0 else None,
                      "value_readout": [tok.convert_ids_to_tokens(int(a)) for a in rd.indices]})
    out["path_units"] = units; out["edge_decomposition_max_abs_err"] = sum_err
    out["path_heads"] = [{"layer": k, "head": int(h), "relative_depth": k / L, "write_norm": float(head_scores[k][h]), "kv_group": int(h) // (nh // nkv)}
                         for k in range(L) for h in head_scores[k].topk(min(3, nh)).indices.tolist()]
    out["layer_profile"] = [{"layer": k, "relative_depth": k / L, "attn": float(dvec @ c["attn"][k][-1]), "ffn": float(dvec @ c["mlp"][k][-1])} for k in range(L)]
    out["seconds_isolation"] = round(time.time() - t0, 2); out["margin_full"] = out["sanity_all_kept"]["margin_full"]
    out["model_shape"] = {"layers": L, "hidden": model.config.hidden_size, "heads": nh, "kv_heads": nkv, "ffn": ffn}
    return out


def main():
    model_dir, name, out_dir = sys.argv[1:4]; prompts = sys.argv[4:] or ["France", "Spain"]
    os.makedirs(out_dir, exist_ok=True); clock = core.Clock()
    model, tok = core.load(model_dir)
    for p in model.parameters():
        p.requires_grad_(False)
    caps = dict(core.atoms())
    for X in prompts:
        res = isolate(model, tok, name, X, caps[X], clock)
        core.dump(os.path.join(out_dir, f"path_{name}_{X}.json"), res)
        print(f"{name} {X}: margin {res['margin_full']:.3f} | K* {res['K_star']} of {res['model_shape']['ffn']} | H* {res['H_star']} of {res['model_shape']['heads']} | joint K={res['joint']['K']} H={res['joint']['H']} "
              f"top1 {res['joint']['top1_same']} kl {res['joint']['kl']:.4f} | sanity kl {res['sanity_all_kept']['kl']:.1e} | edge err {res['edge_decomposition_max_abs_err']:.1e} | "
              f"top5-source share median {float(np.median([u['top5_source_share'] for u in res['path_units'] if u['top5_source_share'] is not None])):.2f} | {res['seconds_isolation']} s", flush=True)


if __name__ == "__main__":
    main()
