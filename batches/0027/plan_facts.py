#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Batch 0027: the facts of the normal-form plan, and the run of nf.dl on them.

Three programs over the dataflow of batch 0015's model, reified by batch 0019's ir.py:
  llama           the program as batch 0019 has it (RMSNorm, SwiGLU, rotary embedding, grouped-query attention, norm weights kept).
  bitnet          llama plus the sub-layer norms of BitNet b1.58 2B4T (Hugging Face modeling_bitnet.py): an RMSNorm with a weight over the units before the down
                  projection, and one over the concatenated head outputs before the output projection. The weights are new spaces (the output of the weight), as
                  batch 0019 models the norm weights of the stream.
  bitnet_ternary  bitnet with the role `ternary` on every space on which a ternary projection has the role `lin` (its input and output coordinates): the
                  seven projections of every layer (q, k, v, o, gate, up, down). The embedding and the head stay real, as in BitNet and TriLM.
Nothing here says which transformations are symmetries; only what exists and what each operation requires. nf.dl derives the group of each space
(batch 0019's lattice), the order in which the spaces can be canonicalized (the distance from the vocabulary through the parameter tensors, the
diagonal links of the norm weights and the elementwise products), the method and the theorem that each space needs, and the verdict of the whole
plan in Kleene's strong logic (supported 1, unknown 0, refuted -1; the conjunction is the minimum).

Usage: plan_facts.py OUTDIR [--proved FILE]    OUTDIR/<config>/{facts,out}, OUTDIR/plan.json
"""

import csv
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "0019"))
sys.path.insert(0, os.path.join(HERE, "..", "0015"))
import ir  # noqa: E402
from model import Arch  # noqa: E402

CONFIGS = {
    # name: (variant, arch). The toy sizes are those of batch 0015's symdim.py; bitnet_2b4t has the sizes of the Hugging Face configuration of
    # microsoft/bitnet-b1.58-2B-4T (hidden 2560, intermediate 6912, 30 layers, 20 heads over 5 key/value heads, vocabulary 128256, tied head).
    "llama": ("llama", Arch(d=6, n_heads=4, n_kv=2, hd=4, d_ff=8, n_layers=2, vocab=7, tied=False)),
    "bitnet": ("bitnet", Arch(d=6, n_heads=4, n_kv=2, hd=4, d_ff=8, n_layers=2, vocab=7, tied=True)),
    "bitnet_ternary": ("bitnet_ternary", Arch(d=6, n_heads=4, n_kv=2, hd=4, d_ff=8, n_layers=2, vocab=7, tied=True)),
    "bitnet_2b4t": (
        "bitnet_ternary",
        Arch(d=2560, n_heads=20, n_kv=5, hd=128, d_ff=6912, n_layers=30, vocab=128256, tied=True),
    ),
}
PROJ = ("proj_q", "proj_k", "proj_v", "proj_o", "proj_g", "proj_u", "proj_d")


def add_sublayer_norms(p):
    """The sub-layer norms of BitNet: RMSNorm over the units, and over the concatenated heads, each followed by a weight."""
    a = p.arch
    for l in range(a.n_layers):
        unit, ov = f"unit{l}", f"ov{l}"
        unitn, ovn = f"unitn{l}", f"ovn{l}"
        # feed-forward: the product of the gate and up branches is normalized over the units, scaled by a weight, and the down projection reads the result
        p.spaces.append((unitn, a.d_ff, 1, "normed"))
        p.tensors.append(
            (
                f"gam_u{l}",
                "param",
                "gam_u",
                l,
            )
        )
        p.axes.append((f"gam_u{l}", 0, unitn))
        p.tensors.append((f"xt_u{l}", "state", "xt_u", l))
        p.axes += [(f"xt_u{l}", 0, "tok"), (f"xt_u{l}", 1, unitn)]
        p.ops.append((f"norm_u{l}", "rms_norm"))
        p.cons.append((f"norm_u{l}", unit, "rmsnorm"))
        p.ops.append((f"scale_u{l}", "diag_mul"))
        p.diag.append((f"scale_u{l}", unit, unitn))
        p.axes = [(t, i, unitn if (t == f"Wd{l}" and s == unit) else s) for t, i, s in p.axes]
        p.cons = [(o, unitn if (o == f"proj_d{l}" and s == unit) else s, r) for o, s, r in p.cons]
        # attention: the heads' outputs are normalized over the concatenation, scaled by a weight, and the output projection reads the result
        p.spaces.append((ovn, a.hd, a.n_kv, "normed"))
        p.tensors.append((f"gam_o{l}", "param", "gam_o", l))
        p.axes += [(f"gam_o{l}", 0, f"grp{l}"), (f"gam_o{l}", 1, f"hin{l}"), (f"gam_o{l}", 2, ovn)]
        p.ops.append((f"norm_o{l}", "rms_norm"))
        p.cons.append((f"norm_o{l}", ov, "rmsnorm"))
        p.ops.append((f"scale_o{l}", "diag_mul"))
        p.diag.append((f"scale_o{l}", ov, ovn))
        p.axes = [(t, i, ovn if (t == f"Wo{l}" and s == ov) else s) for t, i, s in p.axes]
        p.cons = [(o, ovn if (o == f"proj_o{l}" and s == ov) else s, r) for o, s, r in p.cons]


def add_ternary(p):
    """Every space on which a ternary projection has the role `lin` (its input and output coordinates) also carries the role `ternary`."""
    extra = []
    for o, s, r in p.cons:
        if r == "lin" and any(o.startswith(x) and o[len(x) :].isdigit() for x in PROJ):
            extra.append((o, s, "ternary"))
    p.cons += extra


def build(name):
    variant, arch = CONFIGS[name]
    p = ir.build(arch, eps_zero=False, logprobs=False, fold=False)
    if variant in ("bitnet", "bitnet_ternary"):
        add_sublayer_norms(p)
    if variant == "bitnet_ternary":
        add_ternary(p)
    return p


def run_souffle(name, workdir, proved):
    p = build(name)
    facts, out = os.path.join(workdir, "facts"), os.path.join(workdir, "out")
    os.makedirs(out, exist_ok=True)
    ir.write_facts(p, facts)
    for fname, items in (("proved", proved), ("refuted", [])):  # nothing in this batch refutes a theorem
        with open(os.path.join(facts, fname + ".facts"), "w") as f:
            for t in items:
                f.write(t + "\n")
    run = subprocess.run(["souffle", "-F", facts, "-D", out, os.path.join(HERE, "nf.dl")], check=True, capture_output=True, text=True)
    if run.stderr.strip() or run.stdout.strip():
        sys.stderr.write(run.stderr + run.stdout)
        raise SystemExit("souffle printed a warning")
    res = {}
    for rel in (
        "grp_of",
        "plan",
        "noplan",
        "unanchored",
        "need",
        "value",
        "spaces_value",
        "model_value",
        "inconsistent",
        "library_checks",
    ):
        path = os.path.join(out, rel + ".csv")
        if os.path.exists(path):
            with open(path, newline="") as f:
                res[rel] = [tuple(r) for r in csv.reader(f, delimiter="\t")]
        else:
            res[rel] = []
    return p, res


def strip_digits(s):
    return s.rstrip("0123456789")


def summarize(name, p, r):
    groups = {}
    for s, g in r["grp_of"]:
        groups.setdefault(strip_digits(s), set()).add(g)
    plan = {}
    for s, d, g, m, t in r["plan"]:
        plan.setdefault(strip_digits(s), set()).add((int(d), g, m, t))
    return {
        "config": name,
        "variant": CONFIGS[name][0],
        "spaces": len(p.spaces),
        "groups_by_kind": {k: sorted(v) for k, v in sorted(groups.items())},
        "plan_by_kind": {k: sorted(v) for k, v in sorted(plan.items())},
        "theorems_needed": sorted(t for (t,) in r["need"]),
        "spaces_without_method": sorted({strip_digits(s) + ":" + g for s, g in r["noplan"]}),
        "unanchored": sorted({strip_digits(s) for (s,) in r["unanchored"]}),
        "spaces_value": [int(v) for (v,) in r["spaces_value"]],
        "model_value": [int(v) for (v,) in r["model_value"]],
        "inconsistent": [t for (t,) in r["inconsistent"]],
        "library_checks": r["library_checks"],
    }


def main(argv):
    out = argv[0]
    opt = dict(zip(argv[1::2], argv[2::2], strict=False))

    def lines(path):
        return [l.strip() for l in open(path) if l.strip()] if path else []

    proved = lines(opt.get("--proved"))
    result = []
    for name in CONFIGS:
        p, r = run_souffle(name, os.path.join(out, name), proved)
        result.append(summarize(name, p, r))
        s = result[-1]
        print(
            f"{name:15s} spaces {s['spaces']:5d}  needs {s['theorems_needed']}  without a method {s['spaces_without_method']}  "
            f"spaces value {s['spaces_value']}  model value {s['model_value']}"
        )
    with open(os.path.join(out, "plan.json"), "w") as f:
        json.dump(result, f, indent=1, sort_keys=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
