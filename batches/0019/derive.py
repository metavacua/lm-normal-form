# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Runs states.dl (Souffle, interpreter mode: nothing is compiled) on the facts that ir.py makes of an architecture, and returns what the Datalog program derived: the group of each
# space, the dimension of the gauge group, and for each test generator whether it is inside the derived group and what class of transformation each state undergoes.
#   derive.py CONFIG [CONFIG ...]    the derived dimension of the named configurations of batch 0015's symdim.py, against the formula of that file
import csv, os, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "0015"))
import ir

# the test generators: name -> (the kinds of space it acts on, with the group element it applies to each; the program it is applied to: u unfolded (norm weights as parameters), f folded)
GENERATORS = {
    "H1": ([("res", "SP"), ("normed", "SP")], "u", []),             # a signed permutation of the hidden coordinates (the norm weights follow)
    "H4": ([("res", "O")], "f", []),                                # an orthogonal matrix on the stream, norm weights folded
    "H4t": ([("res", "O")], "u", [("normed", "gl")]),               # the same with the norm weights kept: the readers are conjugated with the weights (a symmetry unless the head is tied)
    "N1": ([("normed", "Mono")], "u", []),                          # scalings of the norm weights, compensated in the readers
    "M1": ([("unit", "Mono"), ("ugate", "Perm")], "u", []),         # units permuted, scaled on the up branch
    "A1": ([("ov", "GL")], "u", []),                                # the value/output gauge
    "A3": ([("qk", "Cplx")], "u", []),                              # complex scalars on the rotary planes of queries and keys
    "A6": ([("grp", "Perm"), ("hin", "Perm")], "u", []),            # key/value groups and heads inside a group permuted
    "A4": ([("qk", "GL")], "u", []),                                # a general invertible matrix on queries and keys (not a symmetry with rotary embeddings)
    "M2": ([("ugate", "Mono")], "u", []),                           # a scale on the gate branch (not a symmetry)
}


def souffle(p, gens, workdir):
    facts, out = os.path.join(workdir, "facts"), os.path.join(workdir, "out")
    os.makedirs(out, exist_ok=True)
    ir.write_facts(p, facts)
    with open(os.path.join(facts, "gen_space.facts"), "w") as f, open(os.path.join(facts, "gen_induced.facts"), "w") as f2:
        for g in gens:
            for kind, e in GENERATORS[g][0]:
                f.write(f"{g}\t{kind}\t{e}\n")
            for kind, c in GENERATORS[g][2]:
                f2.write(f"{g}\t{kind}\t{c}\n")
    subprocess.run(["souffle", "-w", "-F", facts, "-D", out, os.path.join(HERE, "states.dl")], check=True)
    res = {}
    for name in ("grp_of", "sdim", "total", "allowed", "bad", "expect", "no_glb", "many_glb", "ungrouped", "two_groups"):
        with open(os.path.join(out, name + ".csv")) as f:
            res[name] = [tuple(r) for r in csv.reader(f, delimiter="\t")]
    return res


def derive(a, eps_zero=False, logprobs=False, gens=("H1", "H4", "H4t", "N1", "M1", "A1", "A3", "A6", "A4", "M2"), fold=False, workdir=None):
    wd = workdir or tempfile.mkdtemp(prefix="lmnf_derive_")
    p = ir.build(a, eps_zero=eps_zero, logprobs=logprobs, fold=fold)
    r = souffle(p, [g for g in gens if GENERATORS[g][1] == ("f" if fold else "u")], wd)
    r["total"] = int(r["total"][0][0])
    r["groups"] = {s: g for s, g in r["grp_of"]}
    r["expect"] = {(g, st): c for g, st, c in r["expect"]}
    r["allowed"] = {g for (g,) in r["allowed"]}
    r["bad"] = {g for (g,) in r["bad"]}
    return r, p


def main(argv):
    import json
    import symdim
    out = None
    if argv and argv[0] == "--json":
        out, argv = argv[1], argv[2:]
    names = argv or list(symdim.CONFIGS)
    rows, ok = [], True
    for n in names:
        a, eps, lp = symdim.CONFIGS[n]
        r, _ = derive(a, eps_zero=eps == 0.0, logprobs=lp)
        want = symdim.expected(a, eps == 0.0, lp)
        ok = ok and r["total"] == want
        rows.append({"config": n, "derived": r["total"], "formula": want, "groups": {k: v for k, v in sorted({(s.rstrip("0123456789"), g) for s, g in r["groups"].items()})},
                     "library_checks_empty": not (r["no_glb"] or r["many_glb"] or r["ungrouped"] or r["two_groups"])})
        print(f"{n:26s} derived {r['total']:4d}  formula of symdim.py {want:4d}  {'equal' if r['total'] == want else 'DIFFERENT'}   groups: " + ", ".join(f"{k}={v}" for k, v in rows[-1]["groups"].items()))
    if out:
        json.dump(rows, open(out, "w"), indent=1)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
