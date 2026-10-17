# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The same prompt's isolated path across models of different depth and width.  usage: pathcompare.py ROOT (ROOT/{name}/path_{name}_{X}.json)"""
import json
import os
import sys

import numpy as np


def main():
    root = sys.argv[1]; names = [n for n in ("base", "instruct", "360M") if os.path.isdir(os.path.join(root, n))]
    res = {}
    for X in ("France", "Spain"):
        rows = {}
        for n in names:
            p = os.path.join(root, n, f"path_{n}_{X}.json")
            if not os.path.exists(p):
                continue
            d = json.load(open(p)); sh = d["model_shape"]
            prof = np.array([l["attn"] + l["ffn"] for l in d["layer_profile"]])
            rd = np.linspace(0, 1, 21); interp = np.interp(rd, [l["relative_depth"] for l in d["layer_profile"]], prof)
            rows[n] = {"shape": sh, "margin": d["margin_full"], "K_star": d["K_star"], "K_star_fraction_act": (d["K_star"]["act"] / sh["ffn"]) if d["K_star"]["act"] else None,
                       "K_star_fraction_gate": (d["K_star"]["gate"] / sh["ffn"]) if d["K_star"]["gate"] else None, "H_star": d["H_star"], "joint": {k: d["joint"][k] for k in ("K", "H", "top1_same", "kl", "margin_abs_diff")},
                       "random_kl_at_Kstar": (d["unit_curves"]["random"][str(d["K_star"]["act"])]["kl"] if d["K_star"]["act"] and str(d["K_star"]["act"]) in d["unit_curves"]["random"] else None),
                       "seconds": d["seconds_isolation"], "top5_source_share_median": float(np.median([u["top5_source_share"] for u in d["path_units"] if u["top5_source_share"] is not None])),
                       "path_unit_relative_depths": sorted(round(u["relative_depth"], 2) for u in d["path_units"])[:50], "layer_profile_interp": interp.tolist(),
                       "top_unit_readouts": [(u["layer"], u["index"], u["value_readout"][:3]) for u in d["path_units"][:6]]}
        if len(rows) > 1:
            ks = list(rows); a = np.array(rows[ks[0]]["layer_profile_interp"])
            for k in ks[1:]:
                b = np.array(rows[k]["layer_profile_interp"]); rows[k]["profile_corr_with_" + ks[0]] = float(np.corrcoef(a, b)[0, 1])
        res[X] = rows
    json.dump(res, open(os.path.join(root, "pathcompare.json"), "w"), indent=1)
    for X, rows in res.items():
        print("=====", X)
        for n, r in rows.items():
            print(f"  {n:9s} shape {r['shape']} margin {r['margin']:.2f} K*(act) {r['K_star']['act']} ({r['K_star_fraction_act']}) K*(gate) {r['K_star']['gate']} K*(rand) {r['K_star']['random']} H*(static) {r['H_star']['static']} "
                  f"joint {r['joint']} top5share {r['top5_source_share_median']:.2f} {r['seconds']} s " + " ".join(f"{k}={v:.2f}" for k, v in r.items() if k.startswith("profile_corr")))


if __name__ == "__main__":
    main()
