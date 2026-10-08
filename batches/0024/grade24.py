#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Grades batch 0024 (docs/batches/0024.md): how much the noise of a quantized run varies. The measurements are the
# summaries that batch 0014's driver (xrt.py) writes for one cell, one file per cell: for every variant, the NMSE of its logits against the logits of the original in the same runtime and cell. A claim is
# "as predicted", "REFUTED", or "not run" (a number it needs is missing, or a validity condition failed); a NaN never satisfies a threshold; controls are listed apart and never counted among the claims.
# The exit status is 0 only if every claim is as predicted and every control holds.
#   grade24.py RESULTS_DIR OUT_DIR | grade24.py --table | grade24.py --selftest
import copy, json, math, os, statistics, sys


class NotRun(Exception):
    pass


def resolve(tree, path):
    cur = tree
    for tok in path.split("/"):
        if tok == "*":
            raise NotRun(f"{path}: no wildcard here")
        if isinstance(cur, dict) and tok in cur:
            cur = cur[tok]
        else:
            raise NotRun(f"{path}: nothing at '{tok}'")
    if cur is None:
        raise NotRun(f"{path}: null")
    return cur


def put(tree, path, value):
    node = tree
    toks = path.split("/")
    for i, tok in enumerate(toks):
        if i == len(toks) - 1:
            node[tok] = value
        else:
            node = node.setdefault(tok, {})


def num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool) and not math.isnan(v)


class Fn:
    """A comparison of numbers at several paths: fn(values) -> (holds, detail). `ok` and `bad` give, for the self-test, numbers that satisfy it and numbers that violate it."""
    def __init__(self, paths, fn, ok, bad, text):
        self.paths, self.fn, self.ok, self.bad, self.t = list(paths), fn, ok, bad, text

    def check(self, t):
        vals = [resolve(t, p) for p in self.paths]
        if not all(num(v) for v in vals):
            return False, f"{self.t}: a value is not a number: {vals}"
        ok, detail = self.fn(vals)
        return ok, f"{self.t}: {detail}"

    def fixture(self, t, ok):
        for p, v in (self.ok if ok else self.bad).items():
            put(t, p, v)

    def text(self):
        return self.t


class Eq:
    def __init__(self, path, val):
        self.path, self.val = path, val

    def check(self, t):
        v = resolve(t, self.path)
        return v == self.val and type(v) is type(self.val), f"{self.path} = {v!r}, {self.val!r} was predicted"

    def fixture(self, t, ok):
        put(t, self.path, self.val if ok else (not self.val))

    def text(self):
        return f"{self.path} = {self.val!r}"


class Claim:
    def __init__(self, cid, severity, subject, text, preds, valid=()):
        self.id, self.severity, self.subject, self.text, self.preds, self.valid = cid, severity, subject, text, list(preds), list(valid)

    def grade(self, t):
        try:
            for v in self.valid:
                ok, d = v.check(t)
                if not ok:
                    return "not run", "validity condition failed: " + d
            res = [p.check(t) for p in self.preds]
            return ("as predicted" if all(ok for ok, _ in res) else "REFUTED"), "; ".join(("ok: " if ok else "FAILS: ") + d for ok, d in res)
        except NotRun as e:
            return "not run", str(e)


EXACT9 = ("scale", "perm", "all", "canon", "heads", "units", "units_blk", "resid", "resid_blk")
NULLS = ("nullall", "nullm", "nullr1", "nullr2", "nullr3", "nullr4", "nullr5", "nullr6")
SAME_W, DIFF_W = ("heads", "units_blk", "resid_blk"), ("units", "resid", "scale", "perm", "all", "canon")
SAME_C, DIFF_C = ("heads", "perm", "resid", "resid_blk", "units", "units_blk"), ("scale", "all", "canon")
CELLS = [("summary-llamacpp-q8_0", "llama.cpp, Q8_0 weights", SAME_W, DIFF_W), ("summary-llamacpp-q4_0", "llama.cpp, Q4_0 weights", SAME_W, DIFF_W),
         ("summary-llamacpp-float32-kvf16", "llama.cpp, f16 cache", EXACT9, ()), ("summary-llamacpp-float32-kvq8_0", "llama.cpp, Q8_0 cache, rotated", SAME_C, DIFF_C),
         ("summary-llamacpp-float32-kvq4_0", "llama.cpp, Q4_0 cache, rotated", SAME_C, DIFF_C), ("summary-llamacpp-float32-kvq8_0-norot", "llama.cpp, Q8_0 cache, not rotated", SAME_C, DIFF_C),
         ("summary-llamacpp-float32-kvq4_0-norot", "llama.cpp, Q4_0 cache, not rotated", SAME_C, DIFF_C), ("summary-ctranslate2-int8", "CTranslate2, int8", EXACT9, ())]
# The classes are those of batch 0014 (P7, P17), except CTranslate2 int8, where all nine variants are registered as "same" (N2 of this batch); P7 of batch 0014 has `scale`, `all`, `canon` as different.
OK_NULL, OK_SAME, OK_DIFF, OK_BROKEN = 1.0, 1.0, 10.0, 1000.0


def nm(cell, v):
    return f"{cell}/variants/{v}/vs_same_runtime_original/nmse"


def claims():
    C = []
    for cell, label, same, diff in CELLS:
        nulls = [nm(cell, v) for v in NULLS]
        sames = [nm(cell, v) for v in same]
        diffs = [nm(cell, v) for v in diff]
        base = {p: OK_NULL for p in nulls}
        base.update({p: OK_SAME for p in sames})
        base.update({p: OK_DIFF for p in diffs})
        subj = f"SmolLM2-135M-Instruct @12fd25f, {label}; the original, nine exact variants, the broken variant and eight null variants, in one job on one machine"
        spread = Fn(nulls, lambda v: (min(v) > 0 and max(v) / min(v) <= 3.0, f"the largest of the eight nulls over the smallest {max(v) / min(v) if min(v) > 0 else float('inf'):.3g} (at most 3 predicted; a null of zero is a failure)"),
                    base, {**base, nulls[0]: 100.0}, "spread of the eight nulls")
        C.append(Claim(f"N1 {label}", "high", subj, "the noise that one unit in the last place of the norm weights makes in this cell varies by at most a factor 3 among eight perturbations", [spread]))
        k = len(nulls)
        same_vs_null = Fn(nulls + sames, lambda v, k=k: (max(v[k:]) <= 4.0 * statistics.median(v[:k]), f"largest variant claimed the same {max(v[k:]):.3g} against 4 x the median null {4.0 * statistics.median(v[:k]):.3g}"),
                          base, {**base, sames[0]: 100.0}, "same class against the median null")
        C.append(Claim(f"N2 {label}", "high", subj, "every exact variant that commutes with the quantizer of this cell differs from the original by at most 4 times the median NMSE of the eight nulls", [same_vs_null]))
        if diff:
            diff_vs_null = Fn(nulls + diffs, lambda v, k=k: (min(v[k:]) >= 1.5 * max(v[:k]), f"smallest variant claimed different {min(v[k:]):.3g} against 1.5 x the largest null {1.5 * max(v[:k]):.3g}"),
                              base, {**base, diffs[0]: 1.0}, "different class against the largest null")
            C.append(Claim(f"N3 {label}", "high", subj, "every exact variant that does not commute with the quantizer of this cell is at least 1.5 times the largest of the eight nulls", [diff_vs_null]))
    return C


def controls():
    K = []
    for cell, label, same, diff in CELLS:
        nulls = [nm(cell, v) for v in NULLS]
        base = {p: OK_NULL for p in nulls}
        base[nm(cell, "broken")] = OK_BROKEN
        K.append((f"control {label}: the broken variant (the permutation applied to the layers and not to the embedding) is at least 10 times the largest null",
                  [Fn(nulls + [nm(cell, "broken")], lambda v: (v[-1] >= 10.0 * max(v[:-1]), f"broken {v[-1]:.3g} against 10 x the largest null {10.0 * max(v[:-1]):.3g}"), base, {**base, nm(cell, "broken"): 1.0}, "broken")]))
        K += [(f"control {label}: the original run twice gives the same {k}", [Eq(f"{cell}/determinism/{k}", True)]) for k in ("generations_equal", "logits_bit_identical", "ppl_equal")]
    return K


def load_tree(d):
    t = {}
    for f in sorted(os.listdir(d)) if os.path.isdir(d) else []:
        if f.endswith(".json") and f.startswith("summary-"):
            try:
                t[f[:-5]] = json.load(open(os.path.join(d, f)))
            except ValueError as e:
                t[f[:-5]] = {"__unreadable__": str(e)}
    return t


def grade(d, out):
    tree = load_tree(d)
    rows = [(c,) + c.grade(tree) for c in claims()]
    ctl = []
    for text, preds in controls():
        try:
            ctl.append((text, "holds" if all(p.check(tree)[0] for p in preds) else "FAILS"))
        except NotRun as e:
            ctl.append((text, f"not run ({e})"))
    tot = {s: sum(1 for r in rows if r[1] == s) for s in ("as predicted", "REFUTED", "not run")}
    os.makedirs(out, exist_ok=True)
    lines = [f"# Batch 0024, graded (claims: {len(rows)}; as predicted {tot['as predicted']}, REFUTED {tot['REFUTED']}, not run {tot['not run']}; controls: {sum(1 for _, s in ctl if s == 'holds')} of {len(ctl)} hold)", "",
             "| id | severity | status | detail |", "|---|---|---|---|"]
    lines += [f"| {c.id} | {c.severity} | {st} | {det[:500].replace('|', '/')} |" for c, st, det in rows]
    lines += ["", "## Controls (not claims)", "", "| control | status |", "|---|---|"] + [f"| {t} | {s} |" for t, s in ctl]
    open(os.path.join(out, "graded24.md"), "w").write("\n".join(lines) + "\n")
    with open(os.path.join(out, "graded24.tsv"), "w") as f:
        f.write("id\tseverity\tstatus\tdetail\n" + "".join(f"{c.id}\t{c.severity}\t{st}\t{det}\n" for c, st, det in rows))
    print("\n".join(lines[:3]))
    for c, st, det in rows:
        print(f"{st}: {c.id}: {det[:300]}")
    for t, s in ctl:
        if s != "holds":
            print(f"control {s}: {t}")
    return 0 if tot["REFUTED"] == 0 and tot["not run"] == 0 and all(s == "holds" for _, s in ctl) else 1


def table():
    print("| id | severity | claim | what is compared |\n|---|---|---|---|")
    for c in claims():
        print(f"| {c.id} | {c.severity} | {c.text} | " + "; ".join(p.text() for p in c.preds) + " |")


def selftest():
    cs = claims()
    ok = {}
    for c in cs:
        for p in c.preds + c.valid:
            p.fixture(ok, True)
    for text, preds in controls():
        for p in preds:
            p.fixture(ok, True)
    for c in cs:
        st, det = c.grade(ok)
        assert st == "as predicted", (c.id, st, det)
        t = copy.deepcopy(ok)
        for p in c.preds:
            p.fixture(t, False)
        st, det = c.grade(t)
        assert st == "REFUTED", (c.id, st, det)
        assert c.grade({})[0] == "not run", c.id
    for text, preds in controls():
        assert all(p.check(ok)[0] for p in preds), text
        t = copy.deepcopy(ok)
        for p in preds:
            p.fixture(t, False)
        assert not all(p.check(t)[0] for p in preds), text
    t = copy.deepcopy(ok)
    put(t, nm(CELLS[0][0], "nullall"), float("nan"))
    assert cs[0].grade(t)[0] == "REFUTED", "a NaN must not satisfy a claim"
    ids = [c.id for c in cs]
    assert len(ids) == len(set(ids))
    print(f"selftest: {len(cs)} claims and {len(controls())} controls; each holds on its fixture, is refuted when its own inputs are violated, is not run without data; a NaN refutes")
    return 0


if __name__ == "__main__":
    if sys.argv[1] == "--selftest":
        sys.exit(selftest())
    if sys.argv[1] == "--table":
        table()
        sys.exit(0)
    sys.exit(grade(sys.argv[1], sys.argv[2]))
