# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The graded table of batch 0031 against docs/batches/0031.md.  usage: grade31.py OUT_ROOT"""
import json
import os
import sys

LADDER = [52, 40, 32, 24, 16, 12, 8]


def rd(*p):
    p = os.path.join(*p)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    root = sys.argv[1]
    cons = {c["model"]: c for c in (rd(root, "models", "constructions.json") or [])}
    c1 = rd(root, "check_M1.json"); c2 = rd(root, "check_M2_d52.json")
    ladder = {d: rd(root, f"check_M2_d{d}.json") for d in LADDER}
    h1 = rd(root, "harness_M1.json"); h2 = rd(root, "harness_M2_d52.json")
    car = rd(root, "carriers.json") or {}
    swap = rd(root, "check_M2_d52_swap.json"); nan = rd(root, "check_M2_d52_nan.json"); null = rd(root, "check_M2_d52_null.json")
    st = rd(root, "structural_M2", "structural_summary.json")
    rows = []
    line = lambda a, b, c, d: rows.append((a, b, c, d))
    # R1
    ok = bool(c1 and c1["sound"] and c1["complete"])
    line("R1", "as predicted" if ok else "refuted", f"M1 assertion {c1 and c1['assertions_on_T']}; over-generation {c1 and c1['over_generation']}",
         "M1 sound and complete for T1: greedy emits paris then </s>")
    # R2
    m1 = cons.get("M1", {})
    ok = bool(m1) and m1.get("rank_S", 99) <= 12 and m1.get("parameters", 10**9) < 700
    line("R2", "as predicted" if ok else "refuted", f"rank S {m1.get('rank_S')}, width {m1.get('width')}, parameters {m1.get('parameters')}", "rank S <= 12 and fewer than 700 parameters")
    # R3
    ok = bool(c2 and c2["sound"] and c2["complete"] and c2["min_margin_on_T"] >= 3)
    line("R3", "as predicted" if ok else "refuted", f"sound {c2 and c2['sound']} complete {c2 and c2['complete']} failures {c2 and c2['failures_on_T']} min margin {c2 and round(c2['min_margin_on_T'], 3)}",
         "M2 (d=52) sound and complete for T2 with margin >= 3")
    # R4
    ok = bool(c2 and c2["input_dependence_bits_T_subjects"] > 3 and c1 and abs(c1["input_dependence_bits_T_subjects"]) < 1e-6 and abs(c1["input_dependence_bits_all_candidates"]) < 1e-6)
    line("R4", "as predicted" if ok else "refuted", f"M2 dependence over T2 subjects {c2 and round(c2['input_dependence_bits_T_subjects'], 3)} bits (log2 12 = 3.585), over all 50 {c2 and round(c2['input_dependence_bits_all_candidates'], 3)}; "
         f"M1 over the 12 countries {c1 and c1['input_dependence_bits_all_candidates']}", "M2 context use above 3 bits; M1 exactly 0")
    # R5
    passing = [d for d in LADDER if ladder[d] and ladder[d]["sound"] and ladder[d]["complete"]]
    ok = (24 in passing) and (8 not in passing) and all(ladder[d] is not None for d in LADDER)
    line("R5", "as predicted" if ok else "refuted", f"passing widths {passing}; min margins {{d: round(ladder[d]['min_margin_on_T'], 3) for d in LADDER if ladder[d]}}", "24 passes, 8 fails")
    # R6
    legs = {"onnx_M1": bool(h1 and h1.get("onnxruntime") and h1["onnxruntime"]["max_abs_diff_logits"] < 1e-4),
            "onnx_M2": bool(h2 and h2.get("onnxruntime") and h2["onnxruntime"]["max_abs_diff_logits"] < 1e-4),
            "gguf_M2_accepted": bool(car.get("M2_d52", {}).get("gguf")), "gguf_M1_refused": not bool(car.get("M1", {}).get("gguf")),
            "llama_M2_equal": bool(h2 and h2.get("llamacpp") and all(v.get("equals_pytorch") for v in h2["llamacpp"].values()))}
    line("R6", "as predicted" if all(legs.values()) else "refuted", f"{legs}; errors M1 {h1 and h1['errors']} M2 {h2 and h2['errors']}; gguf notes {car}",
         "transformers and ONNX Runtime agree for both; llama.cpp accepts M2 and refuses M1")
    # controls
    exp = sorted(["france", "germany"])
    line("control a", "as predicted" if (swap and sorted(swap["failures_on_T"]) == exp) else "refuted", f"failures {swap and swap['failures_on_T']}; assertions {swap and swap['assertions_on_T']}",
         "swapping paris and berlin in W_o fails exactly france and germany")
    line("control b", "as predicted" if (nan and nan["finite"] is False) else "refuted", f"finite {nan and nan['finite']}", "the NaN copy is reported non-finite")
    line("control c", "as predicted" if (null and c2 and null["logits_sha256"] != c2["logits_sha256"] and null["assertions_on_T"] == c2["assertions_on_T"]) else "refuted",
         f"hash changed {null and c2 and null['logits_sha256'] != c2['logits_sha256']}; assertions equal {null and c2 and null['assertions_on_T'] == c2['assertions_on_T']}", "hash changes, no assertion changes")
    md = ["# Batch 0031, graded", "", "| line | verdict | measured | criterion |", "|---|---|---|---|"] + [f"| {a} | {b} | {str(c).replace('|', '/')} | {d} |" for a, b, c, d in rows]
    md += ["", "## Observed", ""]
    for name, c in cons.items():
        md.append(f"- {name}: blocks {c['blocks']}, width {c['width']}, parameters {c['parameters']}" + (f", rank S {c['rank_S']}, eigenvalues {[round(x, 3) for x in c['eigenvalues_kept']]}" if "rank_S" in c else f", coherence {c['max_code_coherence']:.3f}, association residual {c['association_residual']:.2e}"))
    for d in LADDER:
        if ladder[d]:
            md.append(f"- ladder d={d}: sound {ladder[d]['sound']}, complete {ladder[d]['complete']}, failures {ladder[d]['failures_on_T']}, min margin {ladder[d]['min_margin_on_T']:.3f}, "
                      f"dependence over T {ladder[d]['input_dependence_bits_T_subjects']:.3f} bits, over all {ladder[d]['input_dependence_bits_all_candidates']:.3f}")
    if c1:
        md.append(f"- M1 over-generation (the eleven other prompts): {c1['over_generation']}")
    if c2:
        md.append(f"- M2 over-generation (the 38 other X): {c2['over_generation']}")
    if st:
        md.append(f"- structural bounds on M2 over the prompt domain: contained {st['all_contained']}, logit bound {st['logit_bound_max']:.3f}, resid bound {st['resid_after_last_block_bound_max']:.3f}")
    if h1:
        md.append(f"- M1 in the harnesses: pytorch {[v['text'] for v in h1['pytorch'].values()]}, onnx {h1['onnxruntime']}, llama {h1['llamacpp'] or h1['errors'].get('llamacpp')}")
    if h2:
        md.append(f"- M2 in the harnesses: pytorch {[v['text'] for v in h2['pytorch'].values()]}, onnx {h2['onnxruntime']}, llama {[(k, v.get('continuation'), v.get('llama_ids')) for k, v in h2['llamacpp'].items()]}")
    open(os.path.join(root, "graded.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
