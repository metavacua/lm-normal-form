# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The graded table of batch 0032 against docs/batches/0032.md.  usage: grade32.py OUT_ROOT (with base/ and instruct/)"""
import glob
import json
import os
import sys


def load_model(root, name):
    R = {}
    for p in glob.glob(os.path.join(root, name, "eval_*.json")):
        R.update(json.load(open(p))["results"])
    return R


def main():
    root = sys.argv[1]
    B = load_model(root, "base"); I = load_model(root, "instruct")
    rows = []
    line = lambda a, b, c, d: rows.append((a, b, c, d))
    fb, fi = B.get("full"), I.get("full")
    # P1
    ok = fb and fi and abs(fb["count"] - 22) <= 2 and abs(fi["count"] - 62) <= 2
    line("P1", "as predicted" if ok else "refuted", f"base {fb and fb['count']}/189 (0004: 22); instruct {fi and fi['count']}/189 (0004: 62)", "22 +- 2 and 62 +- 2")
    # P2
    ok = fb and fi and not fb["sound_T12"] and not fi["sound_T12"] and fi["count_T12"] >= 6 and fb["count_T12"] <= 3
    line("P2", "as predicted" if ok else "refuted", f"T12: base {fb and fb['count_T12']}/12 (France -> {fb and fb['rows']['France']['top1']!r}), instruct {fi and fi['count_T12']}/12", "neither sound; instruct >= 6, base <= 3")
    # P3
    sound_any = [(m, k) for m, R in (("base", B), ("instruct", I)) for k, v in R.items() if k != "controls" and v.get("sound_T12")]
    line("P3", "as predicted" if not sound_any else "refuted", f"sub-terms sound for T12: {sound_any}", "no sub-term is sound for T12")
    # P4
    pre = {int(k[1:]): v["count"] for k, v in B.items() if k.startswith("P")}
    early = max((c for k, c in pre.items() if k <= 11), default=None); best_k = max(pre, key=pre.get) if pre else None
    ok = pre and early is not None and early <= 2 and fb and max(pre.values()) <= fb["count"]
    line("P4", "as predicted" if ok else "refuted", f"base prefix counts {dict(sorted(pre.items()))}; max before block 12: {early}; best prefix {best_k}; full {fb and fb['count']}",
         "count <= 2 for k <= 11; no prefix above the full model")
    # P5
    d12 = B.get("D12", {}).get("count"); late = {j: B[f'D{j}']['count'] for j in range(20, 31) if f'D{j}' in B}
    ok = fb and d12 is not None and d12 < 5 and late and all(abs(c - fb["count"]) <= 5 for c in late.values())
    line("P5", "as predicted" if ok else "refuted", f"delete block 12: {d12}; deletions of blocks 20-30: {late}; full {fb and fb['count']}", "D12 < 5; D20..D30 within 5 of the full model")
    # P6
    better_g = [k for k, v in B.items() if k.startswith("G") and fb and v["count"] > fb["count"]]
    better_u = [k for k, v in B.items() if k.startswith("U") and fb and v["count"] > fb["count"]]
    ok = bool(better_g) and bool(better_u)
    line("P6", "as predicted" if ok else "refuted", f"group deletions above full: {better_g[:8]} ({len(better_g)}); thinnings above full: {better_u[:8]} ({len(better_u)})", "at least one of each improves the count")
    # P7
    zero = [v["dependence_bits"] for k, v in B.items() if k.startswith("P") and v["count"] == 0]
    ok = (not zero or max(zero) < 0.5) and fi and fi["dependence_bits"] > 2
    line("P7", "as predicted" if ok else "refuted", f"max dependence among zero-count base prefixes {max(zero) if zero else None}; full instruct {fi and round(fi['dependence_bits'], 3)}", "< 0.5 bits for zero-count prefixes; > 2 bits for full instruct")
    for name, R in (("base", B), ("instruct", I)):
        c = R.get("controls", {})
        line(f"control a ({name})", "as predicted" if c and c["permuted_authority_count"] <= 3 else "refuted", f"{c.get('permuted_authority_count')}", "permuted authority: count at chance (<= 3)")
        line(f"control b ({name})", "as predicted" if c and c["null"]["hash_changed"] and c["null"]["count"] == R["full"]["count"] else "refuted", f"{c.get('null')}", "null variant: hash changes, count unchanged")
        line(f"control c ({name})", "as predicted" if c and c["nan_finite"] is False else "refuted", f"finite {c.get('nan_finite')}", "NaN copy reported non-finite")
    md = ["# Batch 0032, graded", "", "| line | verdict | measured | criterion |", "|---|---|---|---|"] + [f"| {a} | {b} | {str(c).replace('|', '/')} | {d} |" for a, b, c, d in rows]
    md += ["", "## Observed", ""]
    for name, R in (("base", B), ("instruct", I)):
        if "full" in R:
            md.append(f"- {name} full: count {R['full']['count']}/189, T12 {R['full']['count_T12']}/12, dependence {R['full']['dependence_bits']:.3f} bits; continuations: {R['full'].get('continuations_T12')}")
            pre = {int(k[1:]): v['count'] for k, v in R.items() if k.startswith('P')}
            md.append(f"- {name} prefix counts by depth: {dict(sorted(pre.items()))}")
            md.append(f"- {name} block-deletion counts: { {int(k[1:]): v['count'] for k, v in R.items() if k.startswith('D')} }")
            gs = sorted(((v['count'], k) for k, v in R.items() if k.startswith('G')), reverse=True)[:6]; us = sorted(((v['count'], k) for k, v in R.items() if k.startswith('U')), reverse=True)[:6]
            md.append(f"- {name} best group deletions {gs}; best thinnings {us}")
            first = {}
            for k in sorted((k for k in R if k.startswith('P')), key=lambda k: int(k[1:])):
                for X, r in R[k]['rows'].items():
                    if r['correct'] and X not in first: first[X] = int(k[1:])
            md.append(f"- {name} depth at which each T12 country first becomes correct: { {X: first.get(X) for X in ['France','Germany','Italy','Spain','Japan','China','Russia','Egypt','Canada','Brazil','India','Australia']} }")
    open(os.path.join(root, "graded.md"), "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
