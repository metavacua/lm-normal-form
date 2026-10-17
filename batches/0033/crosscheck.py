# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cell 10 (docs/batches/0033.md, amendment 7). Read-only.
usage: crosscheck.py v6 MODEL_DIR OUT_JSON      per-layer attention and feed-forward terms for Paris and ' the' against LARQL's two trace tables
       crosscheck.py rope MODEL_DIR LABEL THETA OUT_JSON   f for four prompts with the rotary base set to THETA (config edit in a copy)"""
import json
import os
import re
import shutil
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402
from path import capture_all  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def parse_larql(path):
    rows = {}
    for ln in open(path, encoding="utf-8"):
        m = re.match(r"\s+(emb|L(\d+))\s+(\d+)\s+([\d.]+)\s+([+-][\d.]+)\s+([+-][\d.]+)", ln)
        if m:
            k = -1 if m.group(1) == "emb" else int(m.group(2))
            rows[k] = {"rank": int(m.group(3)), "prob": float(m.group(4)), "attn": float(m.group(5)), "ffn": float(m.group(6))}
    return rows


def v6(model_dir, out):
    model, tok = core.load(model_dir)
    for p in model.parameters():
        p.requires_grad_(False)
    L = model.config.num_hidden_layers; eps_n = model.config.rms_norm_eps
    ids = core.prompt_ids(tok, "France"); t = core.target_ids(tok, [("France", dict(core.atoms())["France"])])[0]; j = tok(" the", add_special_tokens=False)["input_ids"][0]
    with torch.no_grad():
        c, z = capture_all(model, ids)
    E = model.model.embed_tokens.weight.data; g = model.model.norm.weight.data; xL = c["xL"][-1]; r = float(torch.sqrt((xL ** 2).mean() + eps_n))
    res = {"r": r, "f": float(z[t] - z[j]), "tokens": {}}
    for name, u in (("Paris", t), ("the", j)):
        tab = parse_larql(os.path.join(HERE, "larql", "08_trace_for_paris.txt" if name == "Paris" else "09_trace_for_the.txt"))
        exact_a = [float((E[u] * g / r) @ c["attn"][k][-1]) for k in range(L)]; exact_f = [float((E[u] * g / r) @ c["mlp"][k][-1]) for k in range(L)]
        ra = [tab[k]["attn"] / exact_a[k] if abs(exact_a[k]) > 0.05 else None for k in range(L)]; rf = [tab[k]["ffn"] / exact_f[k] if abs(exact_f[k]) > 0.05 else None for k in range(L)]
        ok_a = sum(1 for k in range(L) if abs(tab[k]["attn"] - r * exact_a[k]) <= max(0.06, 0.02 * abs(r * exact_a[k])))
        ok_f = sum(1 for k in range(L) if abs(tab[k]["ffn"] - r * exact_f[k]) <= max(0.06, 0.02 * abs(r * exact_f[k])))
        # V6d: the running-residual candidates (A) and (B)
        x = c["emb"][-1].clone(); candA_a, candA_f, candB_a, candB_f = [], [], [], []
        lens = lambda v: float((E[u] * g / torch.sqrt((v ** 2).mean() + eps_n)) @ v)
        for k in range(L):
            for kind, store_a, store_b in (("attn", candA_a, candB_a), ("mlp", candA_f, candB_f)):
                d_ = c[kind][k][-1]; x1 = x + d_
                store_a.append(float((E[u] * g / torch.sqrt((x1 ** 2).mean() + eps_n)) @ d_)); store_b.append(lens(x1) - lens(x)); x = x1
        candC_a = [float((E[u] * g / torch.sqrt((c["attn"][k][-1] ** 2).mean() + eps_n)) @ c["attn"][k][-1]) for k in range(L)]
        candC_f = [float((E[u] * g / torch.sqrt((c["mlp"][k][-1] ** 2).mean() + eps_n)) @ c["mlp"][k][-1]) for k in range(L)]
        close = lambda a_, b_: sum(1 for k in range(L) if abs(a_[k] - b_[k]) <= max(0.06, 0.02 * abs(b_[k])))
        res["tokens"][name + "_V6d"] = {"A_attn": close(candA_a, tab_attn := [tab[k]["attn"] for k in range(L)]), "A_ffn": close(candA_f, [tab[k]["ffn"] for k in range(L)]),
                                        "B_attn": close(candB_a, tab_attn), "B_ffn": close(candB_f, [tab[k]["ffn"] for k in range(L)]), "C_attn": close(candC_a, tab_attn), "C_ffn": close(candC_f, [tab[k]["ffn"] for k in range(L)]), "C_attn_values": candC_a, "C_ffn_values": candC_f, "A_attn_values": candA_a, "B_attn_values": candB_a,
                                        "A_ffn_values": candA_f, "B_ffn_values": candB_f}
        print(f"V6d {name}: layers within 2% of LARQL: (A) attn {res['tokens'][name + '_V6d']['A_attn']} ffn {res['tokens'][name + '_V6d']['A_ffn']} | (B) attn {res['tokens'][name + '_V6d']['B_attn']} ffn {res['tokens'][name + '_V6d']['B_ffn']} | "
              f"(C, own-rms) attn {res['tokens'][name + '_V6d']['C_attn']} ffn {res['tokens'][name + '_V6d']['C_ffn']} | layer 23/29 attn LARQL {tab[23]['attn']}/{tab[29]['attn']} A {candA_a[23]:.2f}/{candA_a[29]:.2f} B {candB_a[23]:.2f}/{candB_a[29]:.2f}", flush=True)
        e0 = c["emb"][-1]
        rank_normed = int((((E @ (g * e0 / torch.sqrt((e0 ** 2).mean() + eps_n))) > (E[u] @ (g * e0 / torch.sqrt((e0 ** 2).mean() + eps_n)))).sum()) + 1)
        rank_raw = int(((E @ e0) > (E[u] @ e0)).sum()) + 1
        res["tokens"][name] = {"larql_attn": [tab[k]["attn"] for k in range(L)], "larql_ffn": [tab[k]["ffn"] for k in range(L)], "exact_attn": exact_a, "exact_ffn": exact_f,
                               "ratio_attn": ra, "ratio_ffn": rf, "layers_attn_equal_r_times_exact": ok_a, "layers_ffn_equal_r_times_exact": ok_f,
                               "rank_at_embedding": {"larql": tab[-1]["rank"], "ours_final_norm_applied": rank_normed, "ours_raw_dot": rank_raw}}
        print(f"V6 {name}: r={r:.3f} | attention layers equal r*exact: {ok_a}/{L} | feed-forward: {ok_f}/{L} | ratios attn {[round(x, 2) if x else None for x in ra[20:]]} ffn {[round(x, 2) if x else None for x in rf[20:]]} | "
              f"rank at embedding LARQL {tab[-1]['rank']} ours normed {rank_normed} raw {rank_raw}", flush=True)
    core.dump(out, res)


def rope(model_dir, label, theta, out):
    d = os.path.join(os.path.dirname(os.path.abspath(out)), f"cfg_{label}_{theta}"); os.makedirs(d, exist_ok=True)
    for fn in os.listdir(model_dir):
        if fn != "config.json" and not os.path.exists(os.path.join(d, fn)):
            os.symlink(os.path.abspath(os.path.join(model_dir, fn)), os.path.join(d, fn))
    cfg = json.load(open(os.path.join(model_dir, "config.json"))); cfg["rope_theta"] = float(theta)
    if isinstance(cfg.get("rope_parameters"), dict):
        cfg["rope_parameters"]["rope_theta"] = float(theta)
    json.dump(cfg, open(os.path.join(d, "config.json"), "w"))
    model, tok = core.load(d)
    inv = model.model.rotary_emb.inv_freq
    jid = tok(" the", add_special_tokens=False)["input_ids"][0]; res = {"label": label, "theta_requested": float(theta), "inv_freq_1": float(inv[1]), "inv_freq_last": float(inv[-1]), "prompts": {}}
    expected1 = float(theta) ** (-2.0 / (2 * inv.numel()))
    assert abs(float(inv[1]) - expected1) / expected1 < 1e-3, f"the rotary edit did not take: inv_freq[1]={float(inv[1])} expected {expected1}"
    caps = dict(core.atoms())
    for X in ("France", "Spain", "Japan", "Egypt"):
        t = core.target_ids(tok, [(X, caps[X])])[0]
        with torch.no_grad():
            z = model(input_ids=core.prompt_ids(tok, X), use_cache=False).logits[0, -1]
        res["prompts"][X] = {"f": float(z[t] - z[jid]), "top1": tok.convert_ids_to_tokens(int(z.argmax()))}
    print(f"rope {label} theta {theta}: inv_freq[1]={float(inv[1]):.5f}", {X: (round(v["f"], 2), v["top1"]) for X, v in res["prompts"].items()}, flush=True)
    core.dump(out, res)




def route(model_dir, out):
    model, tok = core.load(model_dir)
    for p in model.parameters():
        p.requires_grad_(False)
    L = model.config.num_hidden_layers; ffn = model.config.intermediate_size; eps_n = model.config.rms_norm_eps
    ids = core.prompt_ids(tok, "France"); t = core.target_ids(tok, [("France", dict(core.atoms())["France"])])[0]; j = tok(" the", add_special_tokens=False)["input_ids"][0]
    with torch.no_grad():
        c, z = capture_all(model, ids)
    E = model.model.embed_tokens.weight.data; g = model.model.norm.weight.data; xL = c["xL"][-1]; r = torch.sqrt((xL ** 2).mean() + eps_n); dvec = (E[t] - E[j]) * g / r
    contrib = torch.stack([(dvec @ model.model.layers[k].mlp.down_proj.weight.data) * c["act"][k][-1] for k in range(L)])
    gate = torch.stack([c["gate"][k][-1] for k in range(L)]); silu = torch.nn.functional.silu(gate)
    listed = []
    for ln in open(os.path.join(HERE, "larql", "05_explain_infer.txt"), encoding="utf-8"):
        m = re.match(r"\s+L\s*(\d+):\s+F(\d+)\s+gate=([+-][\d.]+)", ln)
        if m:
            listed.append((int(m.group(1)), int(m.group(2)), float(m.group(3))))
    assert len(listed) == 150, len(listed)
    gate_ok = sum(1 for k, i, gs in listed if abs(float(gate[k, i]) - gs) <= max(0.05, 0.01 * abs(gs)))
    flat = contrib.abs().view(-1); top150 = set(torch.topk(flat, 150).indices.tolist())
    lid = [k * ffn + i for k, i, _ in listed]
    overlap = sum(1 for q in lid if q in top150)
    share = float(sum(float(flat[q]) for q in lid) / float(flat.sum()))
    near0 = sum(1 for k, i, _ in listed if abs(float(silu[k, i])) < 0.1)
    signed = float(sum(float(contrib.view(-1)[q]) for q in lid))
    best_listed = sorted(((float(contrib[k, i]), k, i, gs, float(silu[k, i])) for k, i, gs in listed), key=lambda x: -abs(x[0]))[:8]
    res = {"gate_equal": gate_ok, "listed": 150, "overlap_with_top150_abs": overlap, "share_of_total_abs": share, "near_zero_activation": near0, "signed_sum_of_listed": signed, "f": float(z[t] - z[j]),
           "best_listed": best_listed, "top_true_units": [(int(q // ffn), int(q % ffn), float(contrib.view(-1)[q]), float(gate[q // ffn, q % ffn])) for q in torch.topk(flat, 10).indices.tolist()]}
    print(f"V7 route: gate scores equal for {gate_ok}/150 | overlap with the 150 largest |contribution| units: {overlap} | share of total |contribution|: {share:.3%} | |silu(gate)|<0.1 for {near0}/150 | signed sum {signed:+.3f} of f {res['f']:+.3f}", flush=True)
    print("V7 best listed:", [(k, i, round(c_, 3), gs, round(s_, 3)) for c_, k, i, gs, s_ in best_listed], flush=True)
    print("V7 true top units (layer, index, contribution, gate):", [(a, b, round(c_, 3), round(d_, 2)) for a, b, c_, d_ in res["top_true_units"]], flush=True)
    core.dump(out, res)


if __name__ == "__main__":
    if sys.argv[1] == "output":
        output_path(*sys.argv[2:5])
    elif sys.argv[1] == "v6":
        v6(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "route":
        route(sys.argv[2], sys.argv[3])
    else:
        rope(*sys.argv[2:6])


def output_path(model_dir, name, out):
    model, tok = core.load(model_dir)
    for p in model.parameters():
        p.requires_grad_(False)
    L = model.config.num_hidden_layers; ffn = model.config.intermediate_size; eps_n = model.config.rms_norm_eps; d_model = model.config.hidden_size
    ids = core.prompt_ids(tok, "France")
    with torch.no_grad():
        c, z = capture_all(model, ids)
    E = model.model.embed_tokens.weight.data; g = model.model.norm.weight.data; xL = c["xL"][-1]; r = torch.sqrt((xL ** 2).mean() + eps_n)
    res = {"model": name, "layers": L}
    paris = core.target_ids(tok, [("France", dict(core.atoms())["France"])])[0]
    for label, o in (("output", int(z.argmax())), ("paris", paris)):
        w = E[o] * g / r
        comp = {"embedding": float(w @ c["emb"][-1])}
        for k in range(L):
            comp[f"attn.{k}"] = float(w @ c["attn"][k][-1]); comp[f"ffn.{k}"] = float(w @ c["mlp"][k][-1])
        pos = sum(v for v in comp.values() if v > 0)
        shares = {k: v / pos for k, v in comp.items()}
        contrib = torch.stack([(w @ model.model.layers[k].mlp.down_proj.weight.data) * c["act"][k][-1] for k in range(L)])
        top = torch.topk(contrib.view(-1), 10).indices.tolist()
        units = [{"layer": q // ffn, "unit": q % ffn, "contribution": float(contrib.view(-1)[q]), "gate_pre": float(c["gate"][q // ffn][-1][q % ffn]),
                  "gate_row": f"model.layers.{q // ffn}.mlp.gate_proj.weight[{q % ffn},:]", "up_row": f"model.layers.{q // ffn}.mlp.up_proj.weight[{q % ffn},:]",
                  "down_col": f"model.layers.{q // ffn}.mlp.down_proj.weight[:,{q % ffn}]"} for q in top]
        biggest = sorted(shares.items(), key=lambda kv: -kv[1])[:6]
        res[label] = {"token": tok.convert_ids_to_tokens(o), "logit": float(z[o]), "sum_components": sum(comp.values()), "components": comp, "positive_mass": pos, "largest_shares": biggest, "top_units": units}
        print(f"O {name} {label} {tok.convert_ids_to_tokens(o)!r}: logit {float(z[o]):.2f} (sum of components {sum(comp.values()):.2f}) | embedding share {shares['embedding']:.1%} | largest shares {[(k, round(v, 3)) for k, v in biggest]} | "
              f"top units {[(u['layer'], u['unit'], round(u['contribution'], 2)) for u in units[:6]]}", flush=True)
    a = {(u["layer"], u["unit"]) for u in res["output"]["top_units"]}; b = {(u["layer"], u["unit"]) for u in res["paris"]["top_units"]}
    res["top10_common_output_vs_paris"] = len(a & b); print(f"O {name}: units common to the top 10 of the output and of Paris: {len(a & b)}", flush=True)
    core.dump(out, res)
