#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Grades batch 0022: the arrays that the Lean definition printed (lean_out.txt) against the arrays that the exact arithmetic of model.py and gauge.py gave (expected.txt), and the table of
# batch 0021 against what Lean computed. An array that Lean did not print (the run stopped) is `not run`, not a mismatch.
#   check22.py LEAN_OUT EXPECTED OUTDIR
import json, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from gen22 import TABLE, GAUGES


def parse(path):
    out, cur = {}, None
    for line in open(path):
        line = line.rstrip("\n")
        if line.startswith("## "):
            cur = line[3:]
            out[cur] = []
        elif cur is not None and line.strip() and not line.startswith("# "):
            out[cur].append([int(x) for x in line.split()])
    return out


def main(argv):
    lean_path, exp_path, outdir = argv
    lean, exp = parse(lean_path), parse(exp_path)
    os.makedirs(outdir, exist_ok=True)
    cats = {}

    def add(cat, label, status):
        cats.setdefault(cat, []).append((label, status))

    def cat_of(label):
        if label in ("sigma", "sigma_heads"):
            return "R6 permutations"
        pref, rest = label.split(".", 1)
        kind = "weights" if rest.split(".")[0] in ("emb", "head", "gf", "l0", "l1") else "run"
        if pref == "orig1":
            return "R2 the definition of the logits (one layer)" if rest in ("logits", "logits_def") else "R1 the original model"
        if pref in ("orig", "fold"):
            return "R1 the original model"
        return "R3 the weights that the Lean gauges make (flat layout, against gauge.py)" if kind == "weights" else "R4 what the transformed model computes (against model.py)"

    for label, rows in exp.items():
        if label not in lean:
            add(cat_of(label), label, "not run")
        else:
            add(cat_of(label), label, "equal" if lean[label] == rows else "MISMATCH")
    # Lean against Lean: the logits of each transformed model are the original's (R5), and the table (R7)
    base = {"h1": "orig", "h4": "fold", "ov": "orig", "qk": "orig", "hp": "orig", "m1": "orig", "n1": "orig"}
    for g in GAUGES:
        b = base[g]
        lab = f"{g}.logits"
        if lab in lean and f"{b}.logits" in lean:
            add("R5 the logits of the transformed model are the original's", g, "equal" if lean[lab] == lean[f"{b}.logits"] else "MISMATCH")
        else:
            add("R5 the logits of the transformed model are the original's", g, "not run")
        for st, differs in TABLE[g].items():
            a_, b_ = f"{g}.{st}", f"{b}.{st}"
            if a_ in lean and b_ in lean:
                got = lean[a_] != lean[b_]
                add("R7 the table: which arrays move (Lean against Lean)", f"{g} {st}", "as the table" if got == differs else "NOT AS THE TABLE")
            else:
                add("R7 the table: which arrays move (Lean against Lean)", f"{g} {st}", "not run")
    if "orig1.logits_def" in lean and "orig1.logits" in lean:
        add("R2 the definition of the logits (one layer)", "definition against tables", "equal" if lean["orig1.logits_def"] == lean["orig1.logits"] else "MISMATCH")
    ok = True
    lines = ["| category | arrays | equal / as expected | mismatches | not run |", "|---|---|---|---|---|"]
    res = {}
    for cat in sorted(cats):
        items = cats[cat]
        good = sum(1 for _, s in items if s in ("equal", "as the table"))
        bad = [lab for lab, s in items if s in ("MISMATCH", "NOT AS THE TABLE")]
        nr = sum(1 for _, s in items if s == "not run")
        ok = ok and not bad and nr == 0
        lines.append(f"| {cat} | {len(items)} | {good} | {len(bad)} | {nr} |")
        res[cat] = {"arrays": len(items), "good": good, "mismatches": bad, "not_run": nr}
    open(os.path.join(outdir, "results22.md"), "w").write("\n".join(lines) + "\n")
    json.dump({"ok": ok, "categories": res}, open(os.path.join(outdir, "results22.json"), "w"), indent=1)
    print("\n".join(lines))
    for cat, r in res.items():
        if r["mismatches"]:
            print("MISMATCH in", cat, ":", ", ".join(r["mismatches"][:20]))
    print("all as predicted:", ok)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
