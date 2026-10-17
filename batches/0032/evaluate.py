# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cells 01 and 02: a model and every sub-term of its lattice under batch 0004's readout, against T189 and T12.

usage: evaluate.py MODEL_DIR OUT_DIR [--only full|prefixes|blocks|groups|thin|controls] [--save-best DIR]
Per sub-term: the assertion (greedy next token) for every determined country, whether it is the first token of the
canonical capital (batch 0004's soundness count), soundness (all), the margin, the mass on the correct token, the
dependence over subjects (mutual information between X and the next-token distribution, in bits), and for the
twelve countries of T12 the greedy continuation verbatim.
"""
import argparse
import hashlib
import json
import math
import os
import sys
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import subterms  # noqa: E402

T12 = ["France", "Germany", "Italy", "Spain", "Japan", "China", "Russia", "Egypt", "Canada", "Brazil", "India", "Australia"]
THIN = [0.25, 0.5, 0.75]


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def theory():
    d = json.load(open(os.path.join(HERE, "countries.json")))
    return [(x["countryLabel"], x["canonical"][0]["capitalLabel"]) for x in d if x["determined"]]


def evaluate(model, tok, atoms, targets, gen12=True, prefix_tag=""):
    V = model.config.vocab_size
    mean_row = np.zeros(V); ent_sum = 0.0; h = hashlib.sha256()
    rows = {}; correct = 0; finite = True; margins = []
    with torch.inference_mode():
        for X, C in atoms:
            ids = tok(f"The capital of {X} is", return_tensors="pt")["input_ids"]
            lg = model(input_ids=ids, use_cache=False).logits[0, -1].double()
            if not torch.isfinite(lg).all():
                finite = False
            p = torch.softmax(lg, -1).numpy(); mean_row += p; ent_sum += entropy(p)
            h.update(np.ascontiguousarray(lg.numpy().astype(np.float32)).tobytes())
            top = torch.topk(lg, 2); top1 = int(top.indices[0]); margin = float(top.values[0] - top.values[1])
            ok = top1 == targets[C]; correct += int(ok); margins.append(margin)
            rows[X] = {"capital": C, "top1": tok.decode([top1]), "correct": bool(ok), "p_correct": float(p[targets[C]]), "margin": margin}
    n = len(atoms)
    mi = entropy(mean_row / n) - ent_sum / n
    out = {"count": correct, "n": n, "sound": correct == n, "finite": finite, "dependence_bits": mi, "logits_sha256": h.hexdigest(),
           "min_margin": min(margins), "rows": rows}
    if gen12:
        cont = {}
        with torch.inference_mode():
            for X in T12:
                if X not in dict(atoms):
                    continue
                ids = tok(f"The capital of {X} is", return_tensors="pt")["input_ids"]
                g = model.generate(ids, max_new_tokens=8, do_sample=False, use_cache=False, pad_token_id=tok.eos_token_id)
                cont[X] = tok.decode(g[0][ids.shape[1]:])
        out["continuations_T12"] = cont
        out["count_T12"] = sum(rows[X]["correct"] for X in T12 if X in rows)
        out["sound_T12"] = out["count_T12"] == len([X for X in T12 if X in rows])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("out_dir"); ap.add_argument("--only", default="all"); ap.add_argument("--save-best", default=None)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = subterms.load(a.model_dir)
    atoms = theory()
    targets = {C: tok(" " + C, add_special_tokens=False)["input_ids"][0] for _, C in atoms}
    L = model.config.num_hidden_layers; nkv = model.config.num_key_value_heads
    t0 = time.time(); results = {}

    def run(view, key):
        with view as m:
            r = evaluate(m, tok, atoms, targets)
        r["seconds"] = round(time.time() - t0, 1); results[key] = r
        print(f"{key:8s} count {r['count']:3d}/{r['n']}  T12 {r['count_T12']:2d}/12  dep {r['dependence_bits']:.3f}  margin {r['min_margin']:.3f}  {r['seconds']:7.1f}s", flush=True)

    want = a.only
    if want in ("all", "full"):
        run(subterms.prefix(model, L), "full")
    if want in ("all", "prefixes"):
        for k in range(0, L):
            run(subterms.prefix(model, k), f"P{k}")
    if want in ("all", "blocks"):
        for j in range(1, L + 1):
            run(subterms.delete_block(model, j), f"D{j}")
    if want in ("all", "groups"):
        for j in range(1, L + 1):
            for g in range(nkv):
                run(subterms.delete_group(model, j, g), f"G{j}.{g}")
    if want in ("all", "thin"):
        for j in range(1, L + 1):
            for f in THIN:
                run(subterms.thin_ffn(model, j, f), f"U{j}.{f}")
    if want in ("all", "controls"):
        full = results.get("full") or evaluate(model, tok, atoms, targets)
        # (a) the permuted authority: the same assertions scored against the wrong capitals
        rng = np.random.default_rng(32); caps = [C for _, C in atoms]; perm = list(rng.permutation(caps))
        pc = sum(1 for (X, C), Cp in zip(atoms, perm) if tok.decode([tok(" " + Cp, add_special_tokens=False)["input_ids"][0]]) == full["rows"][X]["top1"])
        # (b) the null variant: one ULP on one gain of the final norm
        g = model.model.norm.weight; v = g[0].item(); g[0] = torch.nextafter(torch.tensor(v), torch.tensor(float("inf"))).item()
        nullr = evaluate(model, tok, atoms, targets, gen12=False); g[0] = v
        # (c) NaN
        w = model.model.layers[0].mlp.down_proj.weight; wv = w[0, 0].item(); w[0, 0] = float("nan")
        nanr = evaluate(model, tok, atoms[:5], targets, gen12=False); w[0, 0] = wv
        results["controls"] = {"permuted_authority_count": pc, "null": {"count": nullr["count"], "hash_changed": nullr["logits_sha256"] != full["logits_sha256"]},
                               "nan_finite": nanr["finite"]}
        print("controls:", results["controls"])
    json.dump({"model_dir": a.model_dir, "layers": L, "kv_heads": nkv, "targets_first_token": {C: tok.decode([t]) for C, t in targets.items()}, "results": results},
              open(os.path.join(a.out_dir, f"eval_{want}.json"), "w"), indent=1)
    if a.save_best and want in ("all", "prefixes", "blocks"):
        if "full" not in results:                       # the full model was evaluated by another invocation; read it or redo it
            fp = os.path.join(a.out_dir, "eval_full.json")
            results["full"] = json.load(open(fp))["results"]["full"] if os.path.exists(fp) else evaluate(model, tok, atoms, targets)
        pp = os.path.join(a.out_dir, "eval_prefixes.json")
        if os.path.exists(pp):
            results.update({k: v for k, v in json.load(open(pp))["results"].items() if k.startswith("P")})
        cand = {k: v for k, v in results.items() if k[0] in "PD" or k == "full"}
        best = max(cand, key=lambda k: (cand[k]["count"], -len(k)))
        shallow = min((k for k in cand if k.startswith("P") and cand[k]["count"] >= max(1, cand["full"]["count"] // 2)), key=lambda k: int(k[1:]), default=None)
        for tag in {best, shallow} - {None}:
            view = subterms.prefix(model, int(tag[1:])) if tag.startswith("P") else (subterms.delete_block(model, int(tag[1:])) if tag.startswith("D") else subterms.prefix(model, L))
            d = os.path.join(a.save_best, tag); os.makedirs(d, exist_ok=True)
            subterms.save_homogeneous(model, view, d); tok.save_pretrained(d)
            print("saved", tag, "->", d)
        json.dump({"best_by_count": best, "shallowest_half": shallow}, open(os.path.join(a.out_dir, "saved.json"), "w"))


if __name__ == "__main__":
    main()
