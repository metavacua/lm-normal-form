# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cell 09: the differential characterisation (docs/batches/0033.md, amendment 6). Read-only: two published models, their gradients
and their difference. f = z_target - z_competitor, the competitor fixed. The change f(B) - f(A) is attributed to weights by the
trapezoid 0.5 * (grad f at A + grad f at B) * (w_B - w_A), coordinate by coordinate.

usage: diff.py OUT_JSON LABEL_A DIR_A LABEL_B DIR_B [COUNTRY ...]"""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402
from path import capture_all  # noqa: E402


def f_and_grad(model, tok, X, t, j):
    for p in model.parameters():
        p.grad = None
    z = model(input_ids=core.prompt_ids(tok, X), use_cache=False).logits[0, -1]
    f = z[t] - z[j]
    f.backward()
    return float(f.detach()), int(z.detach().argmax()), [p.grad.detach().clone() for p in model.parameters()]


def components(model, tok, X, t, j):
    ids = core.prompt_ids(tok, X); L = model.config.num_hidden_layers; eps_n = model.config.rms_norm_eps
    with torch.no_grad():
        c, z = capture_all(model, ids)
    E = model.model.embed_tokens.weight.data; g = model.model.norm.weight.data; xL = c["xL"][-1]; r = torch.sqrt((xL ** 2).mean() + eps_n); d = (E[t] - E[j]) * g / r
    out = {"embedding": float(d @ c["emb"][-1])}
    for k in range(L):
        out[f"attn.{k}"] = float(d @ c["attn"][k][-1]); out[f"ffn.{k}"] = float(d @ c["mlp"][k][-1])
    return out, float(z[t] - z[j]), int(z.argmax())


def main():
    out_json, la, da, lb, db = sys.argv[1:6]; prompts = sys.argv[6:] or ["France", "Spain", "Japan", "Egypt"]
    A, tok = core.load(da); B, _ = core.load(db)
    capsd = dict(core.atoms())
    names = [n for n, _ in core.params(A)]
    assert names == [n for n, _ in core.params(B)], "parameter lists differ"
    pa = [p for _, p in core.params(A)]; pb = [p for _, p in core.params(B)]
    dw = [(b.data - a.data) for a, b in zip(pa, pb)]
    cfg = {k: (getattr(A.config, k, None), getattr(B.config, k, None)) for k in ("rope_theta", "max_position_embeddings", "rms_norm_eps")}
    cfg["rope_theta"] = ((getattr(A.config, "rope_parameters", None) or {}).get("rope_theta", cfg["rope_theta"][0]), (getattr(B.config, "rope_parameters", None) or {}).get("rope_theta", cfg["rope_theta"][1]))
    res = {"A": la, "B": lb, "config_A_B": {k: list(v) for k, v in cfg.items()}, "weights_changed_fraction": float(sum(int((d != 0).sum()) for d in dw) / sum(d.numel() for d in dw)),
           "delta_l2": float(sum(float((d.double() ** 2).sum()) for d in dw) ** 0.5), "prompts": {}}
    jid = tok(" the", add_special_tokens=False)["input_ids"][0]
    for X in prompts:
        t = core.target_ids(tok, [(X, capsd[X])])[0]
        fa, ta, ga = f_and_grad(A, tok, X, t, jid); fb, tb, gb = f_and_grad(B, tok, X, t, jid)
        per_tensor = []; tops = []; allabs = []; total = 0.0
        for n, g1, g2, d in zip(names, ga, gb, dw):
            a = 0.5 * (g1 + g2) * d; s = float(a.double().sum()); total += s
            per_tensor.append({"tensor": n, "sum": s, "abs_sum": float(a.abs().double().sum()), "delta_l2": float(d.double().norm())})
            v, i = torch.topk(a.abs().view(-1), min(30, a.numel())); tops.append((v, i, n, a.view(-1)[i]))
            allabs.append(a.abs().view(-1))
        flat = torch.cat(allabs); tot_abs = float(flat.double().sum())
        conc = {str(k): float(torch.topk(flat, k).values.double().sum() / tot_abs) for k in (10, 100, 1000, 10000, 100000, 1000000)}
        del flat, allabs
        topc = sorted(((float(v_), n_, int(i_), float(s_)) for vv, ii, n_, ss in tops for v_, i_, s_ in zip(vv, ii, ss)), key=lambda x: -x[0])[:30]
        coordinates = []
        pmap = dict(core.params(A))
        for v, n, i, sv in topc:
            shp = list(pmap[n].shape)
            coordinates.append({"tensor": n, "flat": i, "row": (i // shp[1] if len(shp) == 2 else None), "col": (i % shp[1] if len(shp) == 2 else i), "attribution": sv,
                                "value_A": float(pmap[n].data.view(-1)[i]), "value_B": float(dict(core.params(B))[n].data.view(-1)[i])})
        by_group = {}
        for r_ in per_tensor:
            k = f"{core.tensor_sort(r_['tensor'])}@{core.tensor_layer(r_['tensor'])}"; by_group[k] = by_group.get(k, 0.0) + r_["sum"]
        ca, _, _ = components(A, tok, X, t, jid); cb, _, _ = components(B, tok, X, t, jid)
        dcomp = sorted(((k, cb[k] - ca[k], ca[k], cb[k]) for k in ca), key=lambda x: -abs(x[1]))[:12]
        res["prompts"][X] = {"target": tok.convert_ids_to_tokens(t), "f_A": fa, "f_B": fb, "delta_f": fb - fa, "top1_A": tok.convert_ids_to_tokens(ta), "top1_B": tok.convert_ids_to_tokens(tb),
                             "sum_attribution": total, "completeness_ratio": (total / (fb - fa)) if abs(fb - fa) > 1e-9 else None, "residual": (fb - fa) - total,
                             "abs_attribution_total": tot_abs, "concentration_top_n_share": conc,
                             "top_tensors": sorted(per_tensor, key=lambda r_: -abs(r_["sum"]))[:12], "top_groups": sorted(by_group.items(), key=lambda kv: -abs(kv[1]))[:12],
                             "top_coordinates": coordinates, "component_changes": [{"component": k, "change": c_, "A": a_, "B": b_} for k, c_, a_, b_ in dcomp]}
        print(f"{la} -> {lb} {X:7s} f {fa:+.3f} -> {fb:+.3f} (d {fb - fa:+.3f}) sum attribution {total:+.3f} completeness {res['prompts'][X]['completeness_ratio']} | top groups {[(k, round(v, 2)) for k, v in res['prompts'][X]['top_groups'][:3]]} | "
              f"top components {[(d_['component'], round(d_['change'], 2)) for d_ in res['prompts'][X]['component_changes'][:3]]} | top-1 {res['prompts'][X]['top1_A']!r} -> {res['prompts'][X]['top1_B']!r}", flush=True)
    core.dump(out_json, res)


if __name__ == "__main__":
    main()
