# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Batch 0023, the power of the conformance test. Batch 0022 compared the Lean definition of the forward pass with the exact arithmetic of model.py and gauge.py (batch 0015) over F_p: 314 arrays in
# each of four configurations, no mismatch. A test that has never been shown to detect a fault proves little: here faults are put in on purpose. Each MUTANT is a small textual change to
# model.py, gauge.py, ops.py or gen22.py (the Python side of that comparison); the generator is run on the mutated copy with the seed of the stored run, and the arrays it then expects are compared
# with the arrays that Lean printed in the stored run (docs/batches/0022/results/*/lean_out.txt, which are not recomputed). A mutant is DETECTED if some array differs or the generator stops with an
# error, and SURVIVES if every array is equal. The list of mutants, and for each one whether it should be detected, was written before any of them was run (docs/batches/0023.md): the mutants that
# are predicted to survive are the changes whose effect needs a larger architecture than the one that Lean ran (one rotary plane, two positions, an untied head, the RMS norm, the causal mask)
# or a path that the run does not take; they are the blind spots that the test is predicted to have. A mutant that is detected only through the caches and not the logits is detected.
# Usage: mutate23.py OUT.json [ID ...]
import json, os, re, shutil, subprocess, sys, tempfile
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CONFIGS = [("matrix-d3", "d3"), ("matrix-d2", "d2"), ("matrix-d2-s100", "d2"), ("matrix-d2-s200", "d2")]
RES = os.path.join(ROOT, "docs", "batches", "0022", "results")
M15, M22 = "batches/0015/", "batches/0022/"

# id: (file, [(old, new), ...], predicted ("detected" or "survives"), why)
MUTANTS = {
    "M01": (M15 + "model.py", [("scale = ops.const(ops.attn_scale(a.hd))", "scale = ops.const(1)")], "detected", "the scale of the scores"),
    "M02": (M15 + "model.py", [("mask = np.tril(np.ones((T, T), dtype=np.int64)) if a.causal else", "mask = np.triu(np.ones((T, T), dtype=np.int64)) if a.causal else")], "detected", "the causal mask points the other way"),
    "M03": (M15 + "model.py", [("mask = np.tril(np.ones((T, T), dtype=np.int64)) if a.causal else np.ones((T, T), dtype=np.int64)", "mask = np.ones((T, T), dtype=np.int64)")], "detected", "no causal mask: position 0 reads position 1"),
    "M04": (M15 + "model.py", [("mask = np.tril(np.ones((T, T), dtype=np.int64)) if a.causal else", "mask = np.tril(np.ones((T, T), dtype=np.int64), -1) if a.causal else")], "detected", "a position may not read itself: the first position reads nothing and the normalization divides by zero"),
    "M05": (M15 + "model.py", [("q, k = _rope(ops, q, C, S), _rope(ops, k, C, S)", "q, k = _rope(ops, q, C, S), k")], "detected", "the keys are not rotated"),
    "M06": (M15 + "model.py", [("ops.red(x1 * C - x2 * S), ops.red(x2 * C + x1 * S)", "ops.red(x1 * C + x2 * S), ops.red(x2 * C + x1 * S)")], "detected", "a sign of the rotation"),
    "M07": (M15 + "model.py", [("    h2 = x.shape[-1] // 2\n    x1, x2 = x[..., :h2], x[..., h2:]\n    return np.concatenate([ops.red(x1 * C - x2 * S), ops.red(x2 * C + x1 * S)], axis=-1)",
                                  "    x1, x2 = x[..., 0::2], x[..., 1::2]\n    out = np.empty_like(x)\n    out[..., 0::2], out[..., 1::2] = ops.red(x1 * C - x2 * S), ops.red(x2 * C + x1 * S)\n    return out")],
             "survives", "the pairing of the coordinates of a rotary plane, half-split or interleaved, is the same with one plane of two coordinates"),
    "M08": (M15 + "model.py", [("k, v = np.repeat(k, a.rep, axis=0), np.repeat(v, a.rep, axis=0)", "k, v = np.tile(k, (a.rep, 1, 1)), np.tile(v, (a.rep, 1, 1))")], "detected", "the heads of a group read the wrong group"),
    "M09": (M15 + "model.py", [("w = ops.red(w * ops.inv(ops.red(w.sum(axis=-1, keepdims=True))))", "w = ops.red(w * ops.inv(ops.red(w.sum(axis=-2, keepdims=True))))")], "detected", "the weights are normalized over the wrong axis"),
    "M10": (M15 + "model.py", [('h = ops.red(h + ops.mm(o, P[f"l{i}.wo"].T))', 'h = ops.red(ops.mm(o, P[f"l{i}.wo"].T))')], "detected", "no residual connection around the attention"),
    "M11": (M15 + "model.py", [('h = ops.red(h + ops.mm(m, P[f"l{i}.wd"].T))', 'h = ops.red(ops.mm(m, P[f"l{i}.wd"].T))')], "detected", "no residual connection around the gated block"),
    "M12": (M15 + "model.py", [('m = ops.red(ops.act_silu(ops.mm(x, P[f"l{i}.wg"].T)) * ops.mm(x, P[f"l{i}.wu"].T))', 'm = ops.red(ops.act_silu(ops.mm(x, P[f"l{i}.wu"].T)) * ops.mm(x, P[f"l{i}.wg"].T))')], "detected", "the gate and the up matrices exchanged"),
    "M13": (M15 + "model.py", [("return ops.red(ops.red(x * ops.norm_scale(msq)) * gamma)", "return ops.red(x * ops.norm_scale(msq))")], "detected", "the norm weight is not applied"),
    "M14": (M15 + "model.py", [("d_inv = ops.const_inv(a.d)", "d_inv = ops.const_inv(a.d + 1)")], "detected", "the mean of the squares is divided by the wrong number"),
    "M15": (M15 + "model.py", [('return ops.mm(_norm(a, ops, h, P["norm_f"]), head_matrix(a, P).T)', "return ops.mm(h, head_matrix(a, P).T)")], "detected", "no final norm"),
    "M16": (M15 + "model.py", [('return P["embed"] if a.tied else P["lm_head"]', 'return P["embed"]')], "detected", "the head is the embedding although the head is untied"),
    "M17": (M15 + "model.py", [("q, k = _rope(ops, q, C, S), _rope(ops, k, C, S)", "k_pre = k\n            q, k = _rope(ops, q, C, S), _rope(ops, k, C, S)"), ('trace[("k", i)], trace[("v", i)] = k, v', 'trace[("k", i)], trace[("v", i)] = k_pre, v')],
             "detected", "the cache keeps the keys before the rotation (the logits are right)"),
    "M18": (M15 + "model.py", [('Q["lm_head"] = ops.red(head_matrix(a, P) * P["norm_f"][None, :])', 'Q["lm_head"] = head_matrix(a, P)')], "detected", "the folded model forgets the final norm weight"),
    "M19": (M15 + "model.py", [('        g = P[f"l{i}.mlp_norm"]\n        for n in ("wg", "wu"):\n            Q[f"l{i}.{n}"] = ops.red(P[f"l{i}.{n}"] * g[None, :])', '        g = P[f"l{i}.mlp_norm"]\n        for n in ("wg", "wu"):\n            Q[f"l{i}.{n}"] = P[f"l{i}.{n}"]')], "detected", "the fold forgets the norm weight of the gated block"),
    "M20": (M15 + "ops.py", [("return self.red(s * s + s + 1)", "return self.red(s * s + s + 2)")], "detected", "the surrogate of the exponential"),
    "M21": (M15 + "ops.py", [("return self.red(s * s + 2 * s)", "return self.red(s * s + 3 * s)")], "detected", "the surrogate of the gate function"),
    "M22": (M15 + "ops.py", [("return self.inv(self.red(msq * msq + 3))", "return self.inv(self.red(msq * msq + 4))")], "detected", "the surrogate of the normalization"),
    "M23": (M15 + "ops.py", [("    def attn_scale(self, hd):\n        return 7", "    def attn_scale(self, hd):\n        return 8")], "detected", "the scale of the scores (the surrogate)"),
    "G01": (M15 + "gauge.py", [('R["norm_f"] = ops.mm(P["norm_f"][None, :], ab)[0]', 'R["norm_f"] = P["norm_f"]')], "detected", "H1: the final norm weight is not permuted"),
    "G02": (M15 + "gauge.py", [('        for n in ("attn_norm", "mlp_norm"):\n            R[f"l{i}.{n}"] = ops.mm(P[f"l{i}.{n}"][None, :], ab)[0]', '        for n in ("attn_norm", "mlp_norm"):\n            R[f"l{i}.{n}"] = P[f"l{i}.{n}"]')], "detected", "H1: the layer norm weights are not permuted"),
    "G03": (M15 + "gauge.py", [('            R[f"l{i}.{n}"] = ops.mm(P[f"l{i}.{n}"], Q)\n        for n in ("wo", "wd"):\n            R[f"l{i}.{n}"] = ops.mm(Q.T, P[f"l{i}.{n}"])\n    return arch, R\n\n\ndef residual_orth_conj',
                                  '            R[f"l{i}.{n}"] = ops.mm(P[f"l{i}.{n}"], Q)\n        for n in ("wo", "wd"):\n            R[f"l{i}.{n}"] = ops.mm(Q, P[f"l{i}.{n}"])\n    return arch, R\n\n\ndef residual_orth_conj')], "detected", "H4: the writers are rotated by Q and not by its transpose"),
    "G04": (M15 + "gauge.py", [("Ai = ops.matinv(A)", "Ai = A.T")], "detected", "A1: the transpose in place of the inverse"),
    "G05": (M15 + "gauge.py", [("Mq = ops.red(M * ops.inv(n2))", "Mq = M")], "detected", "A3: the queries are scaled by the same complex number as the keys"),
    "G06": (M15 + "gauge.py", [("wk[rows, :] = ops.mm(M, wk[rows, :])", "wk[rows, :] = ops.mm(Mq, wk[rows, :])")], "detected", "A3: keys and queries exchanged"),
    "G07": (M15 + "gauge.py", [("ci = ops.inv(c)\n        if gate_scales is not None:", "ci = c\n        if gate_scales is not None:")], "detected", "M1: the down column is scaled by c and not by 1/c"),
    "G08": (M15 + "gauge.py", [('R[f"l{i}.wu"] = ops.red(wu * c[:, None])[p]', 'R[f"l{i}.wu"] = ops.red(wu[p] * c[:, None])')], "detected", "M1: the scale is applied after the permutation instead of before"),
    "G09": (M15 + "gauge.py", [("order = [gp[g] * arch.rep + r for g in range(arch.n_kv) for r in hperms[i][g]]", "order = [g * arch.rep + r for g in range(arch.n_kv) for r in hperms[i][g]]")], "detected", "A6: the group permutation is not applied to the heads"),
    "G10": (M15 + "gauge.py", [('R[f"l{i}.{n}"] = ops.red(P[f"l{i}.{n}"] * ci[None, :])', 'R[f"l{i}.{n}"] = ops.red(P[f"l{i}.{n}"] * c[None, :])')], "detected", "N1: the readers are scaled by c and not by 1/c"),
    "G11": (M15 + "gauge.py", [('R["lm_head"] = ops.red(P["lm_head"] * ops.inv(c)[None, :])', 'R["lm_head"] = ops.red(P["lm_head"] * c[None, :])')], "survives", "N1 on the final norm: the generator takes the scale 1 there, so the direction of the scale is not tested"),
    "G12": (M15 + "gauge.py", [("rows = [g * hd + j, g * hd + j + h2]", "rows = [g * hd + j, g * hd + j + 1]")], "survives", "A3: the second row of a rotary plane is j + 1 or j + hd/2: the same with one plane"),
    "H01": (M22 + "gen22.py", [("cn, sn = 1, 0\n            for n in range(T):", "cn, sn = c, s\n            for n in range(T):")], "detected", "the positions start at 1: the logits are the same, the cache is not"),
    "H02": (M22 + "gen22.py", [("cn, sn = (c * cn - s * sn) % P, (s * cn + c * sn) % P", "cn, sn = (c * cn - s * sn) % P, (s * cn - c * sn) % P")], "survives", "the recurrence of the rotation is wrong from the third position on: only two positions are run"),
    "B01": (M15 + "model.py", [('return P["embed"] if a.tied else P["lm_head"]', 'return P["embed"].T if a.tied else P["lm_head"]')], "survives", "the tied head is wrong: the run has an untied head"),
    "B02": (M15 + "model.py", [("x = ops.red(x - ops.red(x.sum(axis=-1, keepdims=True) * d_inv))", "x = x")], "survives", "LayerNorm does not subtract the mean: the run uses the RMS norm"),
    "B03": (M15 + "model.py", [("if a.causal else np.ones((T, T), dtype=np.int64)", "if a.causal else np.zeros((T, T), dtype=np.int64)")], "survives", "the mask of a non-causal model: the run is causal"),
    "B04": (M15 + "model.py", [("return a.with_(tied=False), Q", "return a, Q")], "survives", "the fold does not untie the head: the head of the run is untied already"),
}


SMOKE_MUTANT = {"SMOKE": (M15 + "model.py", [("scale = ops.const(ops.attn_scale(a.hd))", "scale = ops.const(3)")], "detected", "smoke only: not one of the registered mutants")}


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


def seed_of(cfg):
    m = re.search(r"seed (\d+):", open(os.path.join(RES, cfg, "gen.txt")).read())
    return int(m.group(1))


def prepare(mutant_id, workdir):
    """A copy of batches/0015 and batches/0022 (the Python files) in workdir with the mutation applied; the number of places where each `old` occurs must be exactly one."""
    for d in ("batches/0015", "batches/0022"):
        os.makedirs(os.path.join(workdir, d), exist_ok=True)
        for f in os.listdir(os.path.join(ROOT, d)):
            if f.endswith(".py"):
                shutil.copy(os.path.join(ROOT, d, f), os.path.join(workdir, d, f))
    if mutant_id is None:
        return
    path, edits, _, _ = {**MUTANTS, **SMOKE_MUTANT}[mutant_id]
    src = open(os.path.join(workdir, path)).read()
    for old, new in edits:
        n = src.count(old)
        if n != 1:
            raise SystemExit(f"mutant {mutant_id}: the text to replace occurs {n} times in {path}, not once")
        src = src.replace(old, new)
    open(os.path.join(workdir, path), "w").write(src)


def run_one(mutant_id, cfg, arch, timeout=900):
    """(status, detail): status is 'equal' (every array that Lean printed is equal), 'mismatch', 'error' (the generator stopped), or 'not run' (the generator printed an array that Lean did not)."""
    work = tempfile.mkdtemp(prefix="mut23_")
    try:
        prepare(mutant_id, work)
        env = dict(os.environ, LMNF22_ARCH=arch)
        r = subprocess.run([sys.executable, os.path.join(work, "batches/0022/gen22.py"), os.path.join(work, "out"), "--seed", str(seed_of(cfg))], capture_output=True, text=True, env=env, timeout=timeout)
        if r.returncode != 0:
            return "error", (r.stderr.strip().splitlines() or ["?"])[-1][:200]
        lean, exp = parse(os.path.join(RES, cfg, "lean_out.txt")), parse(os.path.join(work, "out", "expected.txt"))
        bad = [lab for lab, rows in exp.items() if lab in lean and lean[lab] != rows]
        missing = [lab for lab in exp if lab not in lean]
        if bad:
            return "mismatch", f"{len(bad)} arrays differ, e.g. {', '.join(bad[:4])}"
        if missing:
            return "not run", f"{len(missing)} arrays not printed by Lean"
        return "equal", f"{len(exp)} arrays equal"
    finally:
        shutil.rmtree(work, ignore_errors=True)


def check():
    """Every text to replace occurs exactly once in its file (no run)."""
    bad = 0
    for mid, (path, edits, predicted, why) in MUTANTS.items():
        src = open(os.path.join(ROOT, path)).read()
        for old, _ in edits:
            n = src.count(old)
            if n != 1:
                bad += 1
                print(f"{mid}: the text occurs {n} times in {path}")
    print(f"{len(MUTANTS)} mutants, {sum(1 for m in MUTANTS.values() if m[2] == 'detected')} predicted detected, {sum(1 for m in MUTANTS.values() if m[2] == 'survives')} predicted to survive; texts not found exactly once: {bad}")
    return bad


def main(out, ids):
    res = {"configurations": [c for c, _ in CONFIGS], "mutants": {}}
    base = {cfg: run_one(None, cfg, arch) for cfg, arch in CONFIGS}
    res["unmutated"] = {cfg: {"status": s, "detail": d} for cfg, (s, d) in base.items()}
    print("unmutated:", {cfg: s for cfg, (s, d) in base.items()}, flush=True)
    for mid in (ids or list(MUTANTS)):
        path, edits, predicted, why = {**MUTANTS, **SMOKE_MUTANT}[mid]
        per = {cfg: run_one(mid, cfg, arch) for cfg, arch in CONFIGS}
        statuses = [s for s, _ in per.values()]
        detected = any(s in ("mismatch", "error") for s in statuses)
        res["mutants"][mid] = {"file": path, "predicted": predicted, "why": why, "detected": detected, "per_configuration": {cfg: {"status": s, "detail": d} for cfg, (s, d) in per.items()}}
        print(f"{mid} predicted {predicted:9s} -> {'detected' if detected else 'survives':9s} {statuses}  ({why})", flush=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "--check":
        sys.exit(1 if check() else 0)
    main(sys.argv[1], sys.argv[2:])
