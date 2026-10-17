# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The same prompt across two models, from their IO records.  usage: compare.py OUT_DIR  (reads OUT_DIR/{base,instruct}/record_*.json)"""
import json
import os
import sys


def main():
    root = sys.argv[1]
    res = {}
    for X in ("France", "Spain"):
        a = json.load(open(os.path.join(root, "base", f"record_base_{X}.json"))); b = json.load(open(os.path.join(root, "instruct", f"record_instruct_{X}.json")))
        ua = {(u["layer"], u["index"]): u["contribution"] for u in a["units"]}; ub = {(u["layer"], u["index"]): u["contribution"] for u in b["units"]}
        wa = {(w["tensor"], w["flat"]): w for w in a["weights"]}; wb = {(w["tensor"], w["flat"]): w for w in b["weights"]}
        shared_w = set(wa) & set(wb)
        res[X] = {
            "margin_logit": [a["output"]["margin_logit"], b["output"]["margin_logit"]], "margin_bits": [a["output"]["margin_bits"], b["output"]["margin_bits"]],
            "target_rank": [a["output"]["target_rank"], b["output"]["target_rank"]], "object_string_logprob": [a["output"]["object_string_logprob"], b["output"]["object_string_logprob"]],
            "entropy_bits": [a["output"]["entropy_bits"], b["output"]["entropy_bits"]], "greedy": [a["output"]["greedy_continuation"], b["output"]["greedy_continuation"]],
            "competitor": [a["output"]["competitor_token"], b["output"]["competitor_token"]],
            "first_layer_stays_rank1": [a["layers"]["first_layer_target_stays_rank1"], b["layers"]["first_layer_target_stays_rank1"]],
            "top50_units_shared": len(set(ua) & set(ub)), "top50_units_jaccard": len(set(ua) & set(ub)) / len(set(ua) | set(ub)),
            "top_weights_shared": len(shared_w), "top_weights_union": len(set(wa) | set(wb)),
            "shared_weight_values": [{"tensor": k[0], "flat": k[1], "base": wa[k]["value"], "instruct": wb[k]["value"], "delta": wb[k]["value"] - wa[k]["value"]} for k in sorted(shared_w)[:20]],
            "attribution_concentration": [a["comparison"]["attribution_concentration"], b["comparison"]["attribution_concentration"]],
            "largest_layer_components_base": sorted(a["layers"]["components"].items(), key=lambda kv: -abs(kv[1]))[:6],
            "largest_layer_components_instruct": sorted(b["layers"]["components"].items(), key=lambda kv: -abs(kv[1]))[:6],
        }
    json.dump(res, open(os.path.join(root, "compare.json"), "w"), indent=1)
    print(json.dumps({X: {k: v for k, v in r.items() if k not in ("shared_weight_values",)} for X, r in res.items()}, indent=1)[:4000])


if __name__ == "__main__":
    main()
