#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Batch 0027: the grader. Every check has a verdict in Kleene's strong logic: supported (the prediction held), refuted (it did not) or unknown
(it could not be observed). A subcommand exits 1 if any verdict is refuted or unknown, and writes its checks as JSON.

  compare OUT.json EXPECTED.json CHECKS.json
      the results of ref.py or ternary.py against the committed results, key by key, ignoring the keys named `seconds`
  axioms AXIOMS.txt CLAIMS.tsv OUTDIR
      the output of `#print axioms` against the registered claims: a theorem is supported when it is listed and depends on no axiom beyond propext,
      Classical.choice and Quot.sound; writes OUTDIR/proved.facts (the obligations of nf.dl that are supported) and OUTDIR/lean-claims.json
  controls ESCAPE_DIR OPEN_DIR CHECKS.json
      the output of closure.dl on the two controls (a module with an import and a start function; a module whose memory has no maximum)
  plan PLAN.json CHECKS.json
      the plan that nf.dl derived for each configuration against the registered predictions
  report DIR OUT_MD OUT_JSON
      every *checks.json, lean-claims.json and observed*.json under DIR, counted, with the failed steps of every outcomes.json
"""

import csv
import glob
import json
import os
import re
import sys

STANDARD_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}


def verdict(predicted, observed):
    if observed is None or predicted is None:
        return "unknown"
    return "supported" if predicted == observed else "refuted"


class Checks:
    def __init__(self, name):
        self.name = name
        self.rows = []

    def add(self, cid, claim, predicted, observed):
        v = verdict(predicted, observed)
        self.rows.append({"id": cid, "claim": claim, "predicted": predicted, "observed": observed, "verdict": v})
        print(f"{v:9s} {cid}: predicted {predicted!r}, observed {observed!r}")

    def counts(self):
        return {v: sum(1 for r in self.rows if r["verdict"] == v) for v in ("supported", "refuted", "unknown")}

    def write(self, path):
        with open(path, "w") as f:
            json.dump({"name": self.name, "counts": self.counts(), "checks": self.rows}, f, indent=1, sort_keys=True)
        c = self.counts()
        print(f"{self.name}: {c}")
        return 0 if c["refuted"] == 0 and c["unknown"] == 0 else 1


def strip_seconds(x):
    if isinstance(x, dict):
        return {k: strip_seconds(v) for k, v in x.items() if k != "seconds"}
    if isinstance(x, list):
        return [strip_seconds(v) for v in x]
    return x


def diff_paths(a, b, path=""):
    """The paths at which two JSON values differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}")
            else:
                out += diff_paths(a[k], b[k], f"{path}/{k}")
        return out
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        out = []
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            out += diff_paths(x, y, f"{path}[{i}]")
        return out
    return [] if a == b else [path or "/"]


def cmd_compare(out_path, expected_path, checks_path):
    with open(out_path) as f:
        got = strip_seconds(json.load(f))
    with open(expected_path) as f:
        want = strip_seconds(json.load(f))
    diffs = diff_paths(got, want)
    c = Checks(os.path.basename(expected_path))
    c.add("equal_to_committed", f"{os.path.basename(out_path)} equals {expected_path} (the keys named seconds ignored)", 0, len(diffs))
    for d in diffs[:20]:
        print("  differs at", d)
    return c.write(checks_path)


def cmd_axioms(axioms_path, claims_path, outdir):
    with open(axioms_path) as f:
        text = f.read()
    listed = {}
    for m in re.finditer(r"'([^']+)' depends on axioms: \[([^\]]*)\]", text):
        listed[m.group(1)] = {a.strip() for a in m.group(2).split(",") if a.strip()}
    for m in re.finditer(r"'([^']+)' does not depend on any axioms", text):
        listed[m.group(1)] = set()
    with open(claims_path) as f:
        claims = list(csv.DictReader(f, delimiter="\t"))
    c = Checks("lean-claims")
    proved = []
    for row in claims:
        name = row["theorem"]
        axioms = listed.get(name)
        if axioms is None:
            c.add(row["id"], f"{name} is listed by #print axioms", True, None)
            continue
        clean = axioms <= STANDARD_AXIOMS
        c.add(row["id"], f"{name} depends on no axiom beyond propext, Classical.choice and Quot.sound ({row['what']})", True, clean)
        if clean and row["kind"] == "obligation":
            proved.append(name.removeprefix("LmnfNormal."))
    c.add("no_sorry", "no registered theorem depends on sorryAx", False, "sorryAx" in text)
    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "proved.facts"), "w") as f:
        f.write("".join(p + "\n" for p in sorted(proved)))
    return c.write(os.path.join(outdir, "lean-claims.json"))


def read_predict(d):
    pred = {}
    with open(os.path.join(d, "predict.csv")) as f:
        for k, v in csv.reader(f, delimiter="\t"):
            pred.setdefault(k, []).append(int(v))
    sets = {}
    for name in ("escape", "may_grow", "fixed_memory"):
        p = os.path.join(d, name + ".csv")
        if os.path.exists(p):
            with open(p) as f:
                sets[name] = [line.rstrip("\n").split("\t") for line in f]
        else:
            sets[name] = []
    return pred, sets


def cmd_controls(escape_dir, open_dir, checks_path):
    c = Checks("closure-controls")
    pred, sets = read_predict(escape_dir)
    c.add("escape_module_not_closed", "a module with an import and a start function is not closed", [0], pred.get("closed"))
    c.add("escape_module_escapes", "closure.dl derives two escapes: the import and the start function", [2], pred.get("escapes"))
    c.add("escape_kinds", "the escapes are an import and a start function", [["import", "env.host_log:func"], ["start", "1"]], sorted(sets["escape"]))
    pred, sets = read_predict(open_dir)
    c.add("open_module_closed", "a module with no import and no start function is closed", [1], pred.get("closed"))
    c.add("open_module_no_declared_max", "the memory of the open module declares no maximum", [-1], pred.get("mem_declared_max_kib"))
    c.add("open_module_bound_is_the_host", "its bound is the host's limit alone", [16384], pred.get("mem_bound_kib"))
    c.add("open_module_may_grow", "the export grow may grow it", [["grow"]], sets["may_grow"])
    c.add("open_module_fixed", "the export size may not", [["size"]], sets["fixed_memory"])
    return c.write(checks_path)


THEOREMS_TERNARY = ["canon_Perm_sort", "canon_SP_sign_sort"]


def cmd_plan(plan_path, checks_path):
    with open(plan_path) as f:
        plans = {p["config"]: p for p in json.load(f)}
    c = Checks("plan")
    llama, bitnet = plans["llama"], plans["bitnet"]
    c.add(
        "llama_needs",
        "the plan of the Llama-style model needs the sort, the scale, and the reduced row echelon form",
        ["canon_GL_rref", "canon_Mono_scale_sort", "canon_Perm_sort"],
        llama["theorems_needed"],
    )
    c.add("llama_open", "its residual stream (O) and its query/key spaces (Cplx) have no method", ["qk:Cplx", "res:O"], llama["spaces_without_method"])
    c.add(
        "bitnet_needs",
        "the plan with the sub-layer norms needs the sort, the scale and the sign",
        ["canon_Mono_scale_sort", "canon_Perm_sort", "canon_SP_sign_sort"],
        bitnet["theorems_needed"],
    )
    c.add("bitnet_open", "its value/output space (O) and its query/key spaces (Cplx) have no method", ["ov:O", "qk:Cplx"], bitnet["spaces_without_method"])
    for name in ("bitnet_ternary", "bitnet_2b4t"):
        p = plans[name]
        groups = sorted({g for gs in p["groups_by_kind"].values() for g in gs})
        c.add(f"{name}_groups", "every space of the ternary model has a group in Triv, Perm, SP", ["Perm", "SP", "Triv"], groups)
        c.add(f"{name}_needs", "the plan needs only the sort and the sign", THEOREMS_TERNARY, p["theorems_needed"])
        c.add(f"{name}_open", "no space is without a method", [], p["spaces_without_method"])
    for name, p in plans.items():
        c.add(f"{name}_anchored", "every space is joined to the vocabulary by parameter tensors, norm weights and gated products", [], p["unanchored"])
        c.add(f"{name}_consistent", "no theorem is both proved and refuted", [], p["inconsistent"])
        c.add(f"{name}_library", "the checks of the lattice and the library are empty", [], p["library_checks"])
    for name in ("bitnet_ternary", "bitnet_2b4t"):
        c.add(f"{name}_spaces_value", "with both theorems proved, every space is supported (1)", [1], plans[name]["spaces_value"])
        c.add(
            f"{name}_model_value",
            "the plan as a whole is unknown (0): the theorem of the stratified composition is not proved",
            [0],
            plans[name]["model_value"],
        )
    c.add("llama_spaces_value", "the Llama-style plan is unknown (0): two spaces have no method", [0], llama["spaces_value"])
    c.add("bitnet_spaces_value", "the plan with the sub-layer norms is unknown (0): two spaces have no method", [0], bitnet["spaces_value"])
    return c.write(checks_path)


def cmd_report(root, out_md, out_json):
    files = sorted(glob.glob(os.path.join(root, "**", "*checks.json"), recursive=True)) + sorted(
        glob.glob(os.path.join(root, "**", "lean-claims.json"), recursive=True)
    )
    files += sorted(glob.glob(os.path.join(root, "**", "observed*.json"), recursive=True))
    total = {"supported": 0, "refuted": 0, "unknown": 0}
    lines = ["# Batch 0027: the checks", "", "| file | supported | refuted | unknown |", "|---|---:|---:|---:|"]
    bad = []
    for path in files:
        with open(path) as f:
            d = json.load(f)
        counts = d.get("counts")
        if not counts:
            continue
        for k in total:
            total[k] += counts.get(k, 0)
        lines.append(f"| {os.path.relpath(path, root)} | {counts.get('supported', 0)} | {counts.get('refuted', 0)} | {counts.get('unknown', 0)} |")
        rows = d.get("checks", [])
        bad += [(os.path.relpath(path, root), r) for r in rows if r.get("verdict") != "supported"]
    lines += ["", f"total: {total}", ""]
    if bad:
        lines += ["## not supported", ""]
        lines += [f"- {p}: {r['id']}: {r['verdict']} (predicted {r.get('predicted')!r}, observed {r.get('observed')!r})" for p, r in bad]
    failed = []
    for path in sorted(glob.glob(os.path.join(root, "**", "outcomes.json"), recursive=True)):
        with open(path) as f:
            steps = json.load(f)
        failed += [f"{os.path.relpath(path, root)}: {k}" for k, v in steps.items() if v.get("outcome") == "failure"]
    if failed:
        lines += ["", "## failed steps", ""] + [f"- {x}" for x in failed]
    with open(out_md, "w") as f:
        f.write("\n".join(lines) + "\n")
    with open(out_json, "w") as f:
        json.dump({"totals": total, "not_supported": [{"file": p, **r} for p, r in bad], "failed_steps": failed}, f, indent=1, sort_keys=True)
    print("\n".join(lines))
    return 0 if not bad and not failed else 1


def main(argv):
    cmd, args = argv[0], argv[1:]
    table = {"compare": cmd_compare, "axioms": cmd_axioms, "controls": cmd_controls, "plan": cmd_plan, "report": cmd_report}
    return table[cmd](*args)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
