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


if __name__ == "__main__":
    if sys.argv[1] == "v6":
        v6(sys.argv[2], sys.argv[3])
    else:
        rope(*sys.argv[2:6])
