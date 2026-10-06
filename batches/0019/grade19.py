# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Grades the predictions S1 to S7 of docs/batches/0019.md from the files that the jobs kept. Written before the real-checkpoint cell was read.
# Usage: grade19.py DERIVE_DIR REAL_DIR [OUT.tsv]    DERIVE_DIR: dims.json, toy.json, ckpt.json; REAL_DIR: real.json
import json, os, sys

SYMDIM = {"untied-eps0-logits": 142, "untied-eps0-logprobs": 148, "untied-eps-logits": 141, "tied-eps0-logits": 121, "tied-eps-logits": 120, "mha-untied-eps0-logits": 82, "nope-untied-eps0-logits": 190}
DIMENSIONS = {"smol-instruct": 455040, "smol-base": 455040, "floatlm-99m": 700672, "trilm-99m": 700672}
rows = []


def say(q, subject, ok, seen):
    rows.append((q, subject, {True: "as predicted", False: "REFUTED", None: "not graded"}[ok], seen))


def load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def pairs(res):
    """(agreeing, total, disagreeing list) over the allowed generators of one measurement."""
    ok, n, bad = 0, 0, []
    for g, r in res["generators"].items():
        for st, v in r["states"].items():
            n += 1
            ok += bool(v["agree"])
            if not v["agree"]:
                bad.append(f"{g}/{st}: derived {v['derived']}, measured {v['measured']}")
    return ok, n, bad


def check_generators(res, tol_sym, tol_bad):
    bad = []
    for g, r in res["generators"].items():
        if r["allowed"] and r["logit_change"] > tol_sym:
            bad.append(f"{g} changes the logits by {r['logit_change']:.2e}")
        if not r["allowed"] and r["logit_change"] < tol_bad:
            bad.append(f"{g} is rejected by the derivation but changes the logits by only {r['logit_change']:.2e}")
    return bad


def main(dd, rd, tsv=None):
    d = load(dd, "dims.json")
    if d is None:
        say("S2", "dimensions", None, "no file")
    else:
        bad = [r["config"] for r in d if r["derived"] != SYMDIM[r["config"]]]
        say("S1", "the library and the derivations", all(r["library_checks_empty"] for r in d), "every pair of groups has one greatest lower bound; every space has one group (7 configurations)")
        say("S2", "dimension of the gauge group, 7 configurations", not bad and len(d) == 7, ", ".join(f"{r['config']} {r['derived']}" for r in d) + (f"; different from the measured rank deficiency: {bad}" if bad else ""))
    t = load(dd, "toy.json")
    if t is None:
        say("S3", "toy models", None, "no file")
    else:
        tot_ok, tot, bad = 0, 0, []
        for name, res in t.items():
            ok, n, b = pairs(res)
            tot_ok, tot = tot_ok + ok, tot + n
            bad += [f"{name}: {x}" for x in b] + [f"{name}: {x}" for x in check_generators(res, 1e-12, 1.0)]
            if not res["library_checks_empty"]:
                bad.append(f"{name}: library checks")
        say("S3", "toy models, derived against measured", not bad, f"{tot_ok} of {tot} (generator, state) pairs agree in {len(t)} models" + (f"; {bad[:4]}" if bad else ""))
    c = load(dd, "ckpt.json")
    if c is None:
        say("S4", "checkpoints", None, "no file")
    else:
        bad = [k for k, v in c.items() if v["derived_dimension"] != DIMENSIONS[k] or v["mismatched"] or v["parameters_in_header"] != v["parameters_in_program"] or v["tensors_in_header"] != v["tensors_in_program"]]
        say("S4", "gauge dimension and conformance of four checkpoints", not bad, ", ".join(f"{k} {v['derived_dimension']:,} ({v['parameters_in_program']:,} parameters, {v['tensors_in_program']} tensors)" for k, v in c.items()) + (f"; outside: {bad}" if bad else ""))
    r = load(rd, "real.json")
    if r is None:
        for q in ("S5", "S6", "S7"):
            say(q, "SmolLM2-135M-Instruct", None, "no file")
    else:
        ok, n, bad = pairs(r)
        bad += check_generators(r, 1e-8, 1e-3)
        say("S5", "SmolLM2-135M-Instruct, derived against measured", not bad and r["library_checks_empty"], f"{ok} of {n} (generator, state) pairs agree; logit changes " + ", ".join(f"{g} {v['logit_change']:.1e}" for g, v in r["generators"].items()) + (f"; {bad[:4]}" if bad else ""))
        moved = {}
        for g, v in r["generators"].items():
            for st, s in v["states"].items():
                if any(x != "invariant" for x in s["measured"].values()):
                    moved.setdefault(st, set()).add(g)
        h, k, v_ = moved.get("h", set()), moved.get("k", set()), moved.get("v", set())
        want = (h == {"H1", "H4"} or h == {"H1", "H4", "H4t"}) and k == {"A3", "A6"} and v_ == {"A1", "A6"}
        say("S6", "which generators move the residual stream and the KV cache (measured)", want and not (h & (k | v_)), f"h: {sorted(h)}; k: {sorted(k)}; v: {sorted(v_)}; common to h and k or v: {sorted(h & (k | v_))}")
        t4 = r["generators"].get("H4t")
        say("S7", "the rotation of the stream, norm weights kept, with the head tied", None if t4 is None else (not t4["allowed"] and t4["logit_change"] >= 1e-3), "no record" if t4 is None else f"allowed by the derivation {t4['allowed']}, logits change {t4['logit_change']:.3g}")
    print("| prediction | subject | result | seen |\n|---|---|---|---|")
    for q, s, res, w in rows:
        print(f"| {q} | {s} | {res} | {w} |")
    print(f"\n{len(rows)} lines; as predicted {sum(1 for x in rows if x[2] == 'as predicted')}, refuted {sum(1 for x in rows if x[2] == 'REFUTED')}, not graded {sum(1 for x in rows if x[2] == 'not graded')}")
    if tsv:
        with open(tsv, "w") as out:
            out.write("prediction\tsubject\tresult\tseen\n")
            for q, s, res, w in rows:
                out.write(f"{q}\t{s}\t{res}\t{w.replace(chr(9), ' ').replace(chr(10), ' ')}\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
