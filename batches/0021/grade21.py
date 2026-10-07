#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Grades batch 0021 from the output of `#print axioms` of every theorem of the new Lean modules (axioms21.txt) and the registered claims (claims21.tsv): a claim is accepted if its theorem is
# listed and depends on no axiom outside propext, Classical.choice, Quot.sound (in particular not sorryAx). With --measured it also compares the table that the theorems state (for each
# generator and for the states of the stream, the queries, the keys and the values) with the classes measured by batch 0019 (toy.json); the table was written with the 0019 table in view, so
# the agreement is a consistency check, not an independent prediction.
#   grade21.py AXIOMS_TXT CLAIMS_TSV OUTDIR [--measured TOY_JSON]
import json, os, re, sys

OK = {"propext", "Classical.choice", "Quot.sound"}

# (generator, state) -> class that the Lean theorems state, and the theorem that states it
LEAN = {
    ("H1", "h"): ("signed_perm", "logits_signedPerm/table_stream"), ("H1", "q"): ("invariant", "qv_stream"), ("H1", "k"): ("invariant", "kv_stream"), ("H1", "v"): ("invariant", "vv_stream"),
    ("H4", "h"): ("orthogonal", "logits_orthogonal/table_stream"), ("H4", "q"): ("invariant", "qv_stream"), ("H4", "k"): ("invariant", "kv_stream"), ("H4", "v"): ("invariant", "vv_stream"),
    ("N1", "h"): ("invariant", "table_attnNorm/table_mlpNorm"), ("N1", "q"): ("invariant", "qv_attnNorm"), ("N1", "k"): ("invariant", "kv_attnNorm"), ("N1", "v"): ("invariant", "vv_attnNorm"),
    ("M1", "h"): ("invariant", "table_mlpScale/table_unitPerm"), ("M1", "q"): ("invariant", "table_mlpScale"), ("M1", "k"): ("invariant", "table_mlpScale"), ("M1", "v"): ("invariant", "table_mlpScale"),
    ("A1", "h"): ("invariant", "layerFn_ov"), ("A1", "q"): ("invariant", "qv_ov"), ("A1", "k"): ("invariant", "kv_ov"), ("A1", "v"): ("gl", "vv_ov"),
    ("A3", "h"): ("invariant", "layerFn_qk"), ("A3", "q"): ("complex_plane", "qv_qk"), ("A3", "k"): ("complex_plane", "kv_qk"), ("A3", "v"): ("invariant", "vv_qk"),
    ("A6", "h"): ("invariant", "layerFn_headPerm"), ("A6", "q"): ("perm", "qv_head"), ("A6", "k"): ("perm", "kv_head"), ("A6", "v"): ("perm", "vv_head"),
}


def parse_axioms(path):
    out = {}
    for line in open(path):
        m = re.match(r"'Lmnf\.(\S+)' depends on axioms: \[(.*)\]", line)
        if m:
            out[m.group(1)] = [a.strip() for a in m.group(2).split(",") if a.strip()]
            continue
        m = re.match(r"'Lmnf\.(\S+)' does not depend on any axioms", line)
        if m:
            out[m.group(1)] = []
    return out


def main(argv):
    measured = None
    if "--measured" in argv:
        i = argv.index("--measured")
        measured = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    axioms_path, claims_path, outdir = argv
    os.makedirs(outdir, exist_ok=True)
    ax = parse_axioms(axioms_path)
    rows, ok = [], True
    for line in open(claims_path):
        if line.startswith("#") or not line.strip():
            continue
        cid, name, kind, what = line.rstrip("\n").split("\t")
        if name not in ax:
            status = "missing"
        elif "sorryAx" in ax[name]:
            status = "sorry"
        elif set(ax[name]) - OK:
            status = "other axiom: " + ", ".join(sorted(set(ax[name]) - OK))
        else:
            status = "accepted"
        ok = ok and status == "accepted"
        rows.append({"id": cid, "theorem": name, "kind": kind, "status": status, "axioms": ax.get(name)})
    out = {"theorems_listed": len(ax), "claims": rows, "all_claims_accepted": ok}
    if measured:
        t = json.load(open(measured))
        n = bad = 0
        diffs = []
        for cfg, c in t.items():
            for (g, st), (cls, thm) in LEAN.items():
                s = c["generators"].get(g, {}).get("states", {}).get(st)
                if s is None:
                    continue
                for layer, m in s["measured"].items():
                    n += 1
                    if m != cls:
                        bad += 1
                        diffs.append({"config": cfg, "generator": g, "state": st, "layer": layer, "lean": cls, "measured": m})
        out["crosscheck"] = {"comparisons": n, "differences": bad, "diffs": diffs}
    json.dump(out, open(os.path.join(outdir, "grade21.json"), "w"), indent=1)
    with open(os.path.join(outdir, "grade21.md"), "w") as f:
        f.write("| id | theorem | kind | status |\n|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['id']} | `{r['theorem']}` | {r['kind']} | {r['status']} |\n")
        if measured:
            f.write(f"\nCross-check with the classes measured by batch 0019: {out['crosscheck']['comparisons']} comparisons, {out['crosscheck']['differences']} differences.\n")
    print(open(os.path.join(outdir, "grade21.md")).read())
    print(f"{len(ax)} theorems listed; all registered claims accepted: {ok}")
    return 0 if ok and (not measured or out["crosscheck"]["differences"] == 0) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
