# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The graded table of batch 0030 against docs/batches/0030.md.

usage: grade30.py OUT_ROOT  (expects build/, carriers.json, working.json, tables/, tables_null/, tables_nan/, structural_T{1..4}/, rdf/)
"""
import json
import math
import os
import sys


def rd(*p):
    p = os.path.join(*p)
    return json.load(open(p)) if os.path.exists(p) else None


def main():
    root = sys.argv[1]
    build = rd(root, "build", "build.json"); car = rd(root, "carriers.json"); wk = rd(root, "working.json")
    T = {t: rd(root, "tables", f"tables_T{t}.json") for t in range(1, 5)}
    Tn = rd(root, "tables_null", "tables_T1.json"); Tnan = rd(root, "tables_nan", "tables_T1.json")
    S = {t: rd(root, f"structural_T{t}", "structural_summary.json") for t in range(1, 5)}
    dec = rd(root, "rdf", "decompile.json")
    rows = []
    line = lambda a, b, c, d: rows.append((a, b, c, d))
    # Q1
    legs = {}
    legs["onnx_export"] = car is not None and car.get("onnx_initializers", 0) > 0
    legs["onnxruntime_1e-4"] = bool(wk and wk.get("onnxruntime") and wk["onnxruntime"]["max_abs_diff_logits"] < 1e-4)
    legs["gguf_convert"] = car is not None and car.get("gguf_tensors", 0) > 0
    legs["llamacpp_greedy_equal"] = bool(wk and wk.get("llamacpp") and all(v.get("equals_pytorch") for v in wk["llamacpp"].values()))
    legs["transformers_chat"] = bool(wk and wk.get("chat"))
    line("Q1", "as predicted" if all(legs.values()) else "refuted", f"{legs}; onnx diff {wk and wk.get('onnxruntime', {}).get('max_abs_diff_logits')}; "
         f"errors {wk and wk.get('errors')}; gguf not byte-identical: {car and car.get('gguf_not_identical')}",
         "three carriers, three harnesses, ONNX Runtime within 1e-4, llama.cpp greedy text equals PyTorch")
    # Q2
    fin = {t: (T[t] or {}).get("finite") for t in T}
    cont = {t: (S[t] or {}).get("all_contained") for t in S}
    viol = {t: max((v["max_violation"] for v in (S[t] or {"checks": {}})["checks"].values()), default=None) for t in S}
    q2 = all(fin.values()) and all(cont.values()) and all(v is not None for v in fin.values()) and all(v is not None for v in cont.values())
    line("Q2", "as predicted" if q2 else "refuted", f"finite {fin}; structural bounds contain exact extrema {cont}; max violation {viol}",
         "finite at lengths 1-4; structural bounds contain every exact extremum at every length")
    # Q3
    b = wk and wk["L1_english"]["bits_per_char_pytorch"]; floor = math.log2(50)
    lp = wk and wk["L1_english"].get("llama_perplexity", {}).get("bits_per_char")
    w3 = wk and wk["L2_capitals"]["within_factor_3_of_uniform"]
    q3 = b is not None and (floor - 0.5) <= b <= (floor + 0.5) and (w3 or 0) >= 10
    line("Q3", "as predicted" if q3 else "refuted", f"L1 bits/char pytorch {b}, llama-perplexity {lp}, floor {floor:.3f}; L2 within x3 of uniform {w3}/12",
         "bits/char within 0.5 of log2(50); at least 10 of 12 countries within a factor 3 of 1/50")
    # Q4
    t1 = T[1] or {}
    q4 = t1.get("distinct_rows") == 50 and t1.get("distinct_argmax", 0) >= 6
    line("Q4", "as predicted" if q4 else "refuted", f"length-1 distinct rows {t1.get('distinct_rows')}/50, distinct argmax {t1.get('distinct_argmax')}",
         "50 distinct rows and at least 6 distinct argmax values (passes the vocabulary-relative rejection rule)")
    # Q5
    q5 = dec is not None and dec["blank_nodes"] is False and dec["tensor_sha256_in_graph_equal_bytes"] and dec["triples_loaded"] == dec["triples_written"] == dec["triples_after_round_trip"]
    line("Q5", "as predicted" if q5 else "refuted", f"{dec}", "ground, byte references exact, counts equal through the round trip; size ratio recorded")
    # controls
    if Tn and T[1]:
        A = [l.split("\t")[1] for l in open(os.path.join(root, "tables", "rows_T1.tsv")).read().splitlines()[1:]]
        B = [l.split("\t")[1] for l in open(os.path.join(root, "tables_null", "rows_T1.tsv")).read().splitlines()[1:]]
        ch = sum(x != y for x, y in zip(A, B))
        line("control a", "as predicted" if (Tn["table_sha256"] != T[1]["table_sha256"] and ch == 0) else "refuted",
             f"hash changed {Tn['table_sha256'] != T[1]['table_sha256']}; argmax rows changed {ch}", "hash changes, no argmax changes at length 1")
    else:
        line("control a", "not run", "", "")
    line("control b", "as predicted" if (Tnan and Tnan["finite"] is False) else ("refuted" if Tnan else "not run"), f"finite reported {Tnan and Tnan['finite']}", "the NaN copy is reported non-finite")
    cd = (S[1] or {}).get("control_d", {})
    line("control c", "as predicted" if (cd.get("ratio_is_exact") and cd.get("other_blocks_unchanged")) else "refuted", f"{cd}", "block 1's FFN box scales by exactly 2^8")
    md = ["# Batch 0030, graded", "", "| line | verdict | measured | criterion |", "|---|---|---|---|"] + [f"| {a} | {b} | {str(c).replace('|', '/')} | {d} |" for a, b, c, d in rows]
    md += ["", "## Observed", ""]
    if build:
        md.append(f"- instance: {build['parameters']} parameters, {len(build['tensors'])} tensors, safetensors sha256 `{build['sha256']['model.safetensors']}`; pieces {build['pieces']}")
    for t in range(1, 5):
        if T[t]:
            md.append(f"- length {t}: {T[t]['sequences']} sequences, distinct rows {T[t]['distinct_rows']}, distinct argmax {T[t]['distinct_argmax']}, argmax top {T[t]['argmax_top'][:4]}, "
                      f"entropy mean {T[t]['entropy_bits']['mean']:.3f} bits, resid |max| {T[t]['resid_abs_max']:.4g}, logits |max| {T[t]['logit_abs_max']:.4g}, table sha256 `{T[t]['table_sha256']}`, {T[t]['seconds']} s")
        if S[t]:
            ck = S[t]["checks"]
            md.append(f"  structural at length {t}: resid bound after last block {S[t]['resid_after_last_block_bound_max']:.4g}, logit bound {S[t]['logit_bound_max']:.4g}, "
                      f"looseness min at last resid {ck.get('resid.3', {}).get('looseness_min')}, at logits {ck.get('logits', {}).get('looseness_min')}")
    if wk:
        md.append(f"- chat (transformers, greedy, 32 tokens): prompt {wk.get('chat', {}).get('prompt_text')!r} -> {wk.get('chat', {}).get('output_text')!r}")
        md.append("- greedy on three prompts, PyTorch / llama.cpp: " + "; ".join(f"{p!r}: {wk['pytorch_greedy'][p]['text']!r} / {wk.get('llamacpp', {}).get(p, {}).get('continuation')!r}" for p in wk["pytorch_greedy"]))
        md.append(f"- L2 per country (p_correct x 50): " + ", ".join(f"{c} {v['ratio_to_uniform']:.2f}" for c, v in wk["L2_capitals"]["countries"].items()))
    if car:
        md.append(f"- carriers: onnx {car['onnx']}; gguf kv keys {len(car['gguf_kv'])}; byte-identical to safetensors: onnx {car['onnx_identical']}/{car['safetensors_tensors']}, gguf {car['gguf_identical']}/{car['safetensors_tensors']} (not: {car['gguf_not_identical']})")
    open(os.path.join(root, "graded.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
