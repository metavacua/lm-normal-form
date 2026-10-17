# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The graded table of batch 0033 against docs/batches/0033.md.
usage: grade33.py ROOT  (ROOT holds verify/verify.json, record-base/base/..., record-instruct/instruct/..., element/element.json, substitute/substitute.json)"""
import json
import os
import sys


def rd(*p):
    p = os.path.join(*p)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    root = sys.argv[1]
    V = rd(root, "verify", "verify.json"); E = rd(root, "element", "element.json"); S = rd(root, "substitute", "substitute.json")
    rows = []
    line = lambda a, b, c, d: rows.append((a, b, c, d))
    if V:
        v1 = V["V1"]; ok = all(abs(x["sum_of_components"] - x["margin"]) < 1e-3 and x["residual_reconstruction_max_diff"] < 1e-3 and x["logits_recomputed_max_diff"] < 1e-3 for x in v1.values())
        line("V1", "as predicted" if ok else "refuted", {k: (round(x["sum_of_components"] - x["margin"], 7), x["residual_reconstruction_max_diff"]) for k, x in v1.items()}, "components sum to the margin; reconstruction within 1e-3")
        v2 = V["V2"]; ok = all(v2[k] is not None and v2[k] < 1e-3 for k in ("final_norm_gain_rel_err", "head_row_target_rel_err", "head_row_competitor_rel_err", "last_down_rel_err"))
        line("V2", "as predicted" if ok else "refuted", {k: v2[k] for k in ("final_norm_gain_rel_err", "head_row_target_rel_err", "head_row_competitor_rel_err", "last_down_rel_err")}, "closed form equals autograd to 1e-3 relative")
        v3 = V["V3"]; ok = v3["nonzero_gradient_entries_offpath"] == 0 and v3["abs_change"] <= 1e-6 and v3["crossing"]["abs_error"] < 1e-3
        line("V3", "as predicted" if ok else "refuted", {k: v3[k] for k in ("offpath_rows", "nonzero_gradient_entries_offpath", "perturbed_entries", "abs_change")} | {"crossing_error": v3["crossing"]["abs_error"]}, "off-path gradient exactly 0; margin unchanged to 1e-6; crossing predicted")
        v4 = V["V4"]; ok = v4["spearman_abs"] >= 0.95 and v4["precision_at_100"] >= 0.8 and v4["median_rel_err_top_stratum"] <= 0.05
        line("V4", "as predicted" if ok else "refuted", {k: v4[k] for k in ("spearman_abs", "precision_at_100", "median_rel_err_top_stratum", "competitor_switches")}, "Spearman >= 0.95; precision@100 >= 0.8; median rel err <= 5%")
        v5 = V["V5"]; line("V5", "as predicted" if v5["energy_fraction"] >= 0.5 else "refuted", {k: v5[k] for k in ("energy_fraction", "min_change_full_support", "min_change_support")}, "support carries >= 50% of the gradient energy")
        line("T1", "as predicted" if V["gate_count"] == 22 else "refuted", V["gate_count"], "the count before any change is 22")
    for name in ("base", "instruct"):
        ck = rd(root, f"record-{name}", name, f"record_{name}_check.json")
        if ck:
            ok = all(r["sum_of_components_diff"] < 1e-3 and r["heads_sum_max_diff"] < 1e-3 and r["units_sum_max_diff"] < 1e-3 and r["lens_last_diff"] < 1e-3 and not r["blank_nodes"] and r["triples"] == r["triples_loaded"] for r in ck["records"].values())
            line(f"R1 ({name})", "as predicted" if ok else "refuted", {X: (round(r["sum_of_components_diff"], 7), r["triples"], r["blank_nodes"]) for X, r in ck["records"].items()}, "sums within 1e-3; ground; loads")
            ok2 = all(r["tensors_located"]["onnx"] == r["tensors_located"]["of"] and r["value_checks"]["values_equal"] == r["value_checks"]["value_checks"] for r in ck["records"].values())
            line(f"R2/R3 ({name})", "as predicted" if ok2 else "refuted", {X: (r["tensors_located"], r["value_checks"]) for X, r in ck["records"].items()}, "tensors located in ONNX and GGUF; listed values equal")
    cmp_ = rd(root, "compare", "compare.json")
    if cmp_:
        f = cmp_["France"]; ok = f["top50_units_shared"] >= 10 and f["margin_logit"][1] > 0 > f["margin_logit"][0]
        line("R4", "as predicted" if ok else "refuted", {"shared_units": f["top50_units_shared"], "margins": f["margin_logit"]}, "top-50 units share >= 10; Instruct positive, base negative")
    if E:
        ex = E["exact"]; chk_ok = all(e["checks"]["at_current_abs_diff"] < 1e-3 and all(o["abs_diff"] < 1e-3 for o in e["checks"]["other_values"]) for e in ex)
        line("T2", "as predicted" if chk_ok else "refuted", {"weights": len(ex), "worst_at_current": max(e["checks"]["at_current_abs_diff"] for e in ex), "worst_other": max(o["abs_diff"] for e in ex for o in e["checks"]["other_values"])}, "enumeration equals the base margin and full forwards to 1e-3")
        nflip = sum(1 for e in ex if e["feasible_count_eps"] > 0)
        line("T3", "as predicted" if (E["france_margin_start"] < 0 and nflip >= 5) else "refuted", {"start_margin": E["france_margin_start"], "exact_weights": len(ex), "weights_with_a_feasible_value": nflip,
             "grid_weights_with_feasible": sum(1 for g in E["grid"] if g["feasible_count_eps"] > 0)}, "France starts negative; >= 5 of the exact weights can flip it alone")
    if S:
        sq = S.get("sequence")
        if sq:
            L2 = sq["L2"]; ok = L2["rounds_to_reach"] is not None and L2["rounds_to_reach"] <= 8 and L2["relative_l2_to_support"] < 0.05 and (L2["other_facts_changed_to_wrong"] + L2["other_facts_changed_to_right"]) <= 3 and abs(L2["bits_per_token_after"] - S["base_bits_per_token"]) < 0.01
            line("T4", "as predicted" if ok else "refuted", {k: L2[k] for k in ("rounds_to_reach", "final_margin", "weights_changed", "relative_l2_to_support", "other_facts_changed_to_wrong", "other_facts_changed_to_right", "bits_per_token_after")} | {"bits_before": S["base_bits_per_token"]}, "reaches 0.5 in <= 8 rounds, relative norm < 0.05, <= 3 other facts move, bits/token within 0.01")
            K = sq["L1box"]; line("T5", "as predicted" if (K["rounds_to_reach"] is not None and K["weights_changed"] <= 200) else "refuted", {k: K[k] for k in ("rounds_to_reach", "final_margin", "weights_changed", "l1")}, "the knapsack reaches 0.5 changing <= 200 weights")
            rc = sq.get("random_control"); line("T9", "as predicted" if (rc and rc["ratio"] < 0.25) else "refuted", rc, "a random change of the same support and norm moves France's margin < 25% as much")
        g11 = S.get("G11")
        if g11:
            last = g11["rounds"][-1]; ok = all(not r.get("slack_facts") for r in g11["rounds"]) and last["installs_satisfied"] == last["installs"] and last["preserved_kept"] >= last["preserves"] - 1 and last["weights_changed"] < 10000 and last["count_all"] >= 40
            line("T6", "as predicted" if ok else "refuted", {k: last[k] for k in ("round", "installs_satisfied", "installs", "preserved_kept", "preserves", "weights_changed", "count_all", "bits_per_token")} | {"slack_rounds": [len(r.get("slack_facts", [])) for r in g11["rounds"]]},
                 "feasible without slack, installs all 11, keeps >= 17/18, < 10,000 weights, count >= 40")
        g189 = S.get("G189")
        if g189:
            last = g189["rounds"][-1]; ok = last["count_all"] >= 150 and last["bits_per_token"] - S["base_bits_per_token"] > 0.05
            line("T7", "as predicted" if ok else "refuted", {k: last[k] for k in ("round", "count_all", "installs_satisfied", "installs", "preserved_kept", "preserves", "weights_changed", "bits_per_token")} | {"bits_before": S["base_bits_per_token"], "slack_facts_last": len(last.get("slack_facts", []))},
                 ">= 150 of 189 correct and bits/token up by > 0.05")
        gp = S.get("G11perm")
        if g11 and gp:
            a = g11["final"]["l1"]; b = gp["final"]["l1"]
            line("T8", "as predicted" if b >= 2 * a else "refuted", {"G11_l1": a, "G11perm_l1": b, "ratio": b / a if a else None, "G11perm_installs": (gp["rounds"][-1]["installs_satisfied"], gp["rounds"][-1]["installs"])}, "the permuted install costs >= 2x the genuine one")
    md = ["# Batch 0033, graded", "", "| line | verdict | measured | criterion |", "|---|---|---|---|"] + [f"| {a} | {b} | {str(c).replace('|', '/')} | {d} |" for a, b, c, d in rows]
    if E:
        md += ["", "## Element-wise (France)", ""]
        for e in E["exact"]:
            md.append(f"- {e['kind']} {e['tensor'].replace('model.', '')} [{e['row']},{e['col']}] w={e['value']:+.4f} g={e['gradient']:+.3f}: feasible values {e['feasible_count_eps']} of {E['finite_bf16_values']}, "
                      f"current ok={e['current_feasible']}, nearest {e.get('nearest_feasible')}, count after {e.get('collateral', {}).get('count')}, France margin {e.get('collateral', {}).get('france_margin_full_forward')}")
    if S and S.get("G189"):
        md += ["", "## Graph-wise, where the changes landed (sort, layer: weights changed)", "", "- G11: " + str(S.get("G11", {}).get("changed_by_sort_and_layer")), "- G189: " + str(S["G189"].get("changed_by_sort_and_layer"))]
    open(os.path.join(root, "graded.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md)[:12000])


if __name__ == "__main__":
    main()
