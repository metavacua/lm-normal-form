#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Grades batch 0021 from the output of `#print axioms` of every declaration of the new Lean modules (axioms21.txt) and the registered claims (claims21.tsv): a claim is accepted if a theorem
# of that NAME is listed and depends on no axiom outside propext, Classical.choice, Quot.sound (in particular not sorryAx). The grader does not compare the statement of the theorem with the
# sentence of the claim.
#   grade21.py AXIOMS_TXT CLAIMS_TSV OUTDIR
import json, os, re, sys

OK = {"propext", "Classical.choice", "Quot.sound"}


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
    out = {"declarations_listed": len(ax), "claims": rows, "all_claims_accepted": ok}
    json.dump(out, open(os.path.join(outdir, "grade21.json"), "w"), indent=1)
    with open(os.path.join(outdir, "grade21.md"), "w") as f:
        f.write("| id | theorem | kind | status |\n|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['id']} | `{r['theorem']}` | {r['kind']} | {r['status']} |\n")
    print(open(os.path.join(outdir, "grade21.md")).read())
    print(f"{len(ax)} declarations listed; {len(rows)} registered claims; all accepted: {ok}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
