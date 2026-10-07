#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Batch 0022: the Lean definition of the forward pass (batch 0021's Model.lean) run over F_p, against the exact arithmetic of model.py and gauge.py (batch 0015) over F_p (ops.FieldOps).
#   gen22.py OUTDIR [--seed N]   writes OUTDIR/ConformGen.lean (the data of a random model in the layout of model.py, and the models that the Lean definitions of the gauges make from it, with a
#                                main that prints their weights, caches, streams and logits) and OUTDIR/expected.txt (what every printed array is, according to model.py and gauge.py)
# Why F_p with a small p and not the rationals: the nonlinearities of FieldOps (a square, a reciprocal of a square plus three) double the degree of the rational function at every
# application, so exact rationals have thousands of digits after two layers; in F_p the numbers stay below p. p = 1000003 is prime (Lean proves it by norm_num). A wrong wiring that
# agrees in every one of the ~10^2 printed entries at a random draw has probability about p^-100.
import os, sys, random
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "0015"))
from model import Arch, shapes, forward, fold_norms, head_matrix, _norm, _rope
import gauge as G
from ops import FieldOps

P = 1000003
ARCH = Arch(d=3, n_heads=4, n_kv=2, hd=2, d_ff=3, n_layers=2, vocab=3, tied=False, pos="rope", causal=True, norm="rms")
TOKENS = [2, 0]
TS = [2]                                  # the rotary planes: plane j rotates by the point of the circle of t = TS[j], (1 - t^2) / (1 + t^2), 2 t / (1 + t^2)
FLAT = (("wq", "wq"), ("wk", "wk"), ("wv", "wv"), ("wo", "wo"), ("wg", "wg"), ("wu", "wu"), ("wd", "wd"), ("attn_norm", "an"), ("mlp_norm", "mn"))


class Ops(FieldOps):
    """FieldOps with the rotation of the planes given by TS (FieldOps draws them from a seed)."""

    def __init__(self, ts):
        super().__init__(P)
        self.ts = ts

    def rope_tables(self, T, hd, seed=None):
        C = np.empty((T, hd // 2), dtype=object)
        S = np.empty((T, hd // 2), dtype=object)
        for j in range(hd // 2):
            t = self.ts[j]
            den = pow(1 + t * t, P - 2, P)
            c, s = (1 - t * t) * den % P, 2 * t * den % P
            cn, sn = 1, 0
            for n in range(T):
                C[n, j], S[n, j] = cn, sn
                cn, sn = (c * cn - s * sn) % P, (s * cn + c * sn) % P
        return C, S


def rows(x):
    x = np.asarray(x, dtype=object)
    if x.ndim == 1:
        return [[int(v) % P for v in x]]
    return [[int(v) % P for v in r] for r in x]


def obj(shape, draw):
    n = int(np.prod(shape))
    out = np.empty(n, dtype=object)
    for i in range(n):
        out[i] = draw()
    return out.reshape(shape)


def rand_params(a, rng):
    out = {}
    for k, shp in shapes(a).items():
        out[k] = obj(shp, (lambda: rng.randrange(1, P)) if (k.endswith("norm") or k == "norm_f") else (lambda: rng.randrange(P)))
    return out


def qcache(a, Pm, tokens, ops, h, i):
    """The queries of layer i, (n_heads, T, hd), after the rotary embedding: the lines of model.forward that make them (the trace of model.py keeps the keys and the values only)."""
    T = len(tokens)
    x = _norm(a, ops, h, Pm[f"l{i}.attn_norm"])
    q = ops.mm(x, Pm[f"l{i}.wq"].T).reshape(T, a.n_heads, a.hd).transpose(1, 0, 2)
    C, S = ops.rope_tables(T, a.hd)
    return _rope(ops, q, C, S)


def expected_run(prefix, a, Pm, ops, tokens):
    out, tr = {}, {}
    logits = forward(a, Pm, tokens, ops, trace=tr)
    h = Pm["embed"][np.asarray(tokens)]
    for i in range(a.n_layers):
        out[f"{prefix}.q{i}"] = rows(qcache(a, Pm, tokens, ops, h, i).reshape(-1, a.hd))
        out[f"{prefix}.k{i}"] = rows(tr[("k", i)].reshape(-1, a.hd))
        out[f"{prefix}.v{i}"] = rows(tr[("v", i)].reshape(-1, a.hd))
        out[f"{prefix}.h{i}"] = rows(tr[i])
        h = tr[i]
    out[f"{prefix}.logits"] = rows(logits)
    return out


def expected_weights(prefix, a, Pm):
    out = {f"{prefix}.emb": rows(Pm["embed"]), f"{prefix}.head": rows(head_matrix(a, Pm)), f"{prefix}.gf": rows(Pm["norm_f"])}
    for i in range(a.n_layers):
        for py, ln in FLAT:
            out[f"{prefix}.l{i}.{ln}"] = rows(Pm[f"l{i}.{py}"])
    return out


# ---- Lean literals ---------------------------------------------------------------------------------------------------------------------------------------------------------------------
def lrows(r):
    return "[" + ", ".join("[" + ", ".join(f"({int(v) % P} : Fp)" for v in row) + "]" for row in r) + "]"


def lvec(v):
    return "[" + ", ".join(f"({int(x) % P} : Fp)" for x in v) + "]"


def swaps_of(sigma):
    """Transpositions (a, b) such that Lean's `swap a1 b1 * swap a2 b2 * ... * swap ak bk` (the rightmost applied first) is the permutation i -> sigma[i]."""
    arr, applied = list(sigma), []
    for i in range(len(arr)):
        if arr[i] != i:
            j = arr.index(i)
            arr[i], arr[j] = arr[j], arr[i]
            applied.append((i, j))
    return list(reversed(applied))


def apply_swaps(swaps, i):
    for a, b in reversed(swaps):
        i = b if i == a else a if i == b else i
    return i


def lperm(sigma, n):
    sw = swaps_of(sigma)
    assert all(apply_swaps(sw, i) == sigma[i] for i in range(n)), (sigma, sw)
    if not sw:
        return f"(1 : Equiv.Perm (Fin {n}))"
    return "(" + " * ".join(f"Equiv.swap ({x} : Fin {n}) {y}" for x, y in sw) + f" : Equiv.Perm (Fin {n}))"


def inverse(perm):
    inv = [0] * len(perm)
    for i, p in enumerate(perm):
        inv[p] = i
    return inv


def householder(v):
    n = len(v)
    vv = sum(x * x for x in v) % P
    assert vv != 0
    iv = pow(vv, P - 2, P)
    return np.array([[(int(i == j) - 2 * v[i] * v[j] * iv) % P for j in range(n)] for i in range(n)], dtype=object)


GAUGES = ("h1", "h4", "ov", "qk", "hp", "m1", "n1")
# what each gauge does to the stream after the first layer (h0) and to the queries, keys, values of the first layer (q0, k0, v0): True = the array differs from the original's
TABLE = {"h1": {"h0": True, "q0": False, "k0": False, "v0": False}, "h4": {"h0": True, "q0": False, "k0": False, "v0": False},
         "ov": {"h0": False, "q0": False, "k0": False, "v0": True}, "qk": {"h0": False, "q0": True, "k0": True, "v0": False},
         "hp": {"h0": False, "q0": True, "k0": True, "v0": True}, "m1": {"h0": False, "q0": False, "k0": False, "v0": False},
         "n1": {"h0": False, "q0": False, "k0": False, "v0": False}}


class Case:
    def __init__(self, seed):
        a, ops = ARCH, Ops(TS)
        self.a, self.ops, self.seed = a, ops, seed
        rng = random.Random(seed)
        self.P = rand_params(a, rng)
        self.a_fold, self.P_fold = fold_norms(a, self.P, ops)
        d, hd, nkv, rep, dff, h2 = a.d, a.hd, a.n_kv, a.rep, a.d_ff, a.hd // 2
        # H1: a permutation of the hidden coordinates (Lean's sigma, i -> sigma[i]) and signs
        self.sigma = rng.sample(range(d), d)
        while self.sigma == list(range(d)):
            self.sigma = rng.sample(range(d), d)
        self.signs = [rng.choice([1, P - 1]) for _ in range(d)]
        if all(s == 1 for s in self.signs):
            self.signs[0] = P - 1
        # H4: an orthogonal matrix, a product of two reflections (not symmetric), in the convention of gauge.py, x' = x Q
        self.Q = ops.mm(householder([rng.randrange(1, P) for _ in range(d)]), householder([rng.randrange(1, P) for _ in range(d)]))
        assert all(int(v) % P == 0 for v in np.ravel(ops.mm(self.Q.T, self.Q) - np.identity(d, dtype=object)))
        # A1: per layer and group an invertible hd x hd matrix
        self.As = []
        for _ in range(a.n_layers):
            row = []
            for _ in range(nkv):
                while True:
                    A = obj((hd, hd), lambda: rng.randrange(P))
                    if (A[0, 0] * A[1, 1] - A[0, 1] * A[1, 0]) % P != 0:
                        break
                row.append(A)
            self.As.append(row)
        # A3: per layer, group and plane a complex scalar
        self.scal = []
        for _ in range(a.n_layers):
            while True:
                x, y = obj((nkv, h2), lambda: rng.randrange(P)), obj((nkv, h2), lambda: rng.randrange(P))
                if all((x[g, j] ** 2 + y[g, j] ** 2) % P != 0 for g in range(nkv) for j in range(h2)):
                    break
            self.scal.append((x, y))
        # A6: per layer a permutation of the groups and of the heads inside each new group
        self.gperms, self.hperms = [], []
        for _ in range(a.n_layers):
            self.gperms.append(rng.sample(range(nkv), nkv))
            self.hperms.append([rng.sample(range(rep), rep) for _ in range(nkv)])
        if all(self.gperms[i] == list(range(nkv)) for i in range(a.n_layers)):
            self.gperms[0] = list(reversed(range(nkv)))
        # M1: per layer a permutation of the units and nonzero up scales
        self.uperms = [rng.sample(range(dff), dff) for _ in range(a.n_layers)]
        self.uscales = [obj((dff,), lambda: rng.randrange(1, P)) for _ in range(a.n_layers)]
        # N1: per layer scalings of the two norm weights (the final norm is left alone)
        self.cs = {}
        for i in range(a.n_layers):
            self.cs[f"l{i}.attn_norm"] = obj((d,), lambda: rng.randrange(1, P))
            self.cs[f"l{i}.mlp_norm"] = obj((d,), lambda: rng.randrange(1, P))
        self.cs["norm_f"] = obj((d,), lambda: 1)
        # the transformed parameters, by gauge.py
        perm = inverse(self.sigma)
        gs = [self.signs[perm[j]] if self.signs[perm[j]] == 1 else -1 for j in range(d)]
        self.gauged = {
            "h1": (a, G.residual_signed_perm(a, self.P, ops, perm, gs)[1]),
            "h4": (self.a_fold, G.residual_orth(self.a_fold, self.P_fold, ops, self.Q)[1]),
            "ov": (a, G.ov_gauge(a, self.P, ops, self.As)[1]),
            "qk": (a, G.qk_gauge_rope(a, self.P, ops, self.scal)[1]),
            "hp": (a, G.head_perm(a, self.P, ops, self.gperms, self.hperms)[1]),
            "m1": (a, G.mlp_gauge(a, self.P, ops, self.uperms, self.uscales)[1]),
            "n1": (a, G.norm_gauge(a, self.P, ops, self.cs)[1]),
        }
        self.base = {"h1": "orig", "h4": "fold", "ov": "orig", "qk": "orig", "hp": "orig", "m1": "orig", "n1": "orig"}
        # the permutation of the heads in Lean's terms: sigma_heads[i][a] = order[a]
        self.sig_heads = [[gp[g] * rep + r for g in range(nkv) for r in hp[g]] for gp, hp in zip(self.gperms, self.hperms)]

    def expected(self):
        out = {}
        out.update(expected_weights("orig", self.a, self.P))
        out.update(expected_run("orig", self.a, self.P, self.ops, TOKENS))
        out.update(expected_weights("fold", self.a_fold, self.P_fold))
        out.update(expected_run("fold", self.a_fold, self.P_fold, self.ops, TOKENS))
        a1 = self.a.with_(n_layers=1)
        out.update(expected_run("orig1", a1, self.P, self.ops, TOKENS))
        out["orig1.logits_def"] = out["orig1.logits"]
        for g, (ag, Pg) in self.gauged.items():
            out.update(expected_weights(g, ag, Pg))
            out.update(expected_run(g, ag, Pg, self.ops, TOKENS))
        out["sigma"] = [list(self.sigma)]
        out["sigma_heads"] = [list(s) for s in self.sig_heads]
        return out

    def generic(self, out):
        """The arrays that a gauge moves differ from the original's, those that it leaves are equal (the table of batch 0021), and the logits are equal, for these parameters."""
        for g, w in TABLE.items():
            for st, differs in w.items():
                if (out[f"{g}.{st}"] != out[f"{self.base[g]}.{st}"]) != differs:
                    return False
            if out[f"{g}.logits"] != out[f"{self.base[g]}.logits"]:
                return False
        return True

    # ---- the Lean file ---------------------------------------------------------------------------------------------------------------------------------------------------------------
    def lean(self):
        a, d, hd, nh, nkv, dff, nv, h2, T = self.a, self.a.d, self.a.hd, self.a.n_heads, self.a.n_kv, self.a.d_ff, self.a.vocab, self.a.hd // 2, len(TOKENS)
        L = []
        w = L.append
        w("-- SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean\n-- SPDX-License-Identifier: AGPL-3.0-or-later\n-- Written by batches/0022/gen22.py; do not edit.")
        w("import LmnfProofs.Runtime\nopen Lmnf Lmnf.Runtime Matrix\n")
        ty = f"Layer Fp {d} {hd} {dff} (Fin {nh}) (Fin {nkv}) (Fin {h2})"
        mty = f"Model Fp {d} {hd} {dff} (Fin {nh}) (Fin {nkv}) (Fin {h2}) (Fin {nv})"

        def flat_def(name, Pm, i):
            for py, ln in FLAT:
                w(f"def {name}_{ln} : List (List Fp) := {lrows(rows(Pm[f'l{i}.{py}']))}")
            w(f"def {name} : Flat := {{ wq := tab {name}_wq, wk := tab {name}_wk, wv := tab {name}_wv, wo := tab {name}_wo, wg := tab {name}_wg, wu := tab {name}_wu, wd := tab {name}_wd, an := tab1 ({name}_an.getD 0 []), mn := tab1 ({name}_mn.getD 0 []) }}")
            w(f"def {name}L : {ty} := toLayer {d} {hd} {h2} {nh} {nkv} {dff} {name}")

        def model_def(name, Pm, nl):
            w(f"def {name}_emb : List (List Fp) := {lrows(rows(Pm['embed']))}")
            w(f"def {name}_head : List (List Fp) := {lrows(rows(head_matrix(a, Pm)))}")
            w(f"def {name}_gf : List Fp := {lvec(list(Pm['norm_f']))}")
            for i in range(nl):
                flat_def(f"{name}_{i}", Pm, i)
            ls = ", ".join(f"{name}_{i}L" for i in range(nl))
            w(f"def {name} : {mty} := mkModel {d} {hd} {h2} {nh} {nkv} {dff} {nv} (tab {name}_emb) (tab {name}_head) (tab1 {name}_gf) [{ls}]")

        w(f"def E : Env Fp {T} (Fin {nh}) (Fin {nkv}) (Fin {h2}) := mkEnv {T} {nh} {nkv} {h2} {a.rep} (by intro a; have := a.isLt; omega) {lvec(TS)}")
        w(f"def toks : Fin {T} → Fin {nv} := ![{', '.join(str(t) for t in TOKENS)}]")
        model_def("M0", self.P, a.n_layers)
        model_def("Mf", self.P_fold, a.n_layers)
        w(f"def M0one : {mty} := mkModel {d} {hd} {h2} {nh} {nkv} {dff} {nv} (tab M0_emb) (tab M0_head) (tab1 M0_gf) [M0_0L]")
        # H1
        w(f"def sg1 : Equiv.Perm (Fin {d}) := {lperm(self.sigma, d)}")
        w(f"def sgn1 : Fin {d} → Fp := fun i => tab1 {lvec(self.signs)} i.val")
        w("def Mh1 := M0.streamGauge (sp sg1 sgn1) (sp sg1.symm (fun k => sgn1 (sg1.symm k))) (fun γ i => γ (sg1 i))")
        # H4
        w(f"def tQ : List (List Fp) := {lrows(rows(self.Q))}")
        w(f"def Qpy : Matrix (Fin {d}) (Fin {d}) Fp := Matrix.of fun i j => tab tQ i.val j.val")
        w("def Mh4 := Mf.streamGauge Qpyᵀ Qpy id")
        names = [f"M0_{i}L" for i in range(a.n_layers)]

        def vec_of_mats(prefix, mats):
            items = []
            for k, m in enumerate(mats):
                w(f"def {prefix}_{k} : Matrix (Fin {hd}) (Fin {hd}) Fp := Matrix.of fun i j => tab {lrows(rows(m))} i.val j.val")
                items.append(f"{prefix}_{k}")
            return "![" + ", ".join(items) + "]"

        ov_layers, qk_layers, hp_layers, m1_layers, n1_layers = [], [], [], [], []
        for i in range(a.n_layers):
            Aq = vec_of_mats(f"tA{i}", self.As[i])
            Bq = vec_of_mats(f"tB{i}", [self.ops.matinv(A) for A in self.As[i]])
            ov_layers.append(f"ovGauge E {names[i]} {Aq} {Bq}")
            qa, qb = self.scal[i]
            qk_layers.append(f"qkGauge E {names[i]} (fun g p => tab {lrows(rows(qa))} g.val p.val) (fun g p => tab {lrows(rows(qb))} g.val p.val)")
            hp_layers.append(f"headPerm {names[i]} {lperm(self.sig_heads[i], nh)} {lperm(self.gperms[i], nkv)}")
            m1_layers.append(f"unitPerm (mlpScale {names[i]} (fun u => tab1 {lvec(list(self.uscales[i]))} u.val)) {lperm(self.uperms[i], dff)}")
            n1_layers.append(f"mlpNorm (attnNorm {names[i]} (fun j => tab1 {lvec(list(self.cs[f'l{i}.attn_norm']))} j.val)) (fun j => tab1 {lvec(list(self.cs[f'l{i}.mlp_norm']))} j.val)")
        for tag, ls in (("ov", ov_layers), ("qk", qk_layers), ("hp", hp_layers), ("m1", m1_layers), ("n1", n1_layers)):
            w(f"def M_{tag} : {mty} := {{ M0 with layers := [{', '.join(ls)}] }}")
        w("def main : IO Unit := do")
        w(f'  emitNat "sigma" [(List.finRange {d}).map fun i => (sg1 i).val]')
        w('  emitNat "sigma_heads" [' + ", ".join(f"(List.finRange {nh}).map fun i => (({lperm(self.sig_heads[i], nh)}) i).val" for i in range(a.n_layers)) + "]")
        models = (("orig", "M0"), ("fold", "Mf"), ("h1", "Mh1"), ("h4", "Mh4"), ("ov", "M_ov"), ("qk", "M_qk"), ("hp", "M_hp"), ("m1", "M_m1"), ("n1", "M_n1"))
        for lab, mname in models:
            w(f'  emitWeights "{lab}" {mname}')
        w('  emitRun "orig1" E M0one toks')
        w('  emitLogitsDef "orig1" E M0one toks')
        for lab, mname in models:
            w(f'  emitRun "{lab}" E {mname} toks')
        return "\n".join(L) + "\n"


def write_expected(path, out):
    with open(path, "w") as f:
        for lab, rs in out.items():
            f.write(f"## {lab}\n")
            for r in rs:
                f.write(" ".join(str(int(v)) for v in r) + "\n")


def main(argv):
    outdir = argv[0]
    seed = int(argv[argv.index("--seed") + 1]) if "--seed" in argv else None
    os.makedirs(outdir, exist_ok=True)
    chosen = None
    for s in ([seed] if seed is not None else range(1000)):
        try:
            case = Case(s)
            exp = case.expected()
        except (ZeroDivisionError, AssertionError):
            continue
        if seed is not None or case.generic(exp):
            chosen = s
            break
    if chosen is None:
        raise SystemExit("no seed makes the parameters generic")
    open(os.path.join(outdir, "ConformGen.lean"), "w").write(case.lean())
    write_expected(os.path.join(outdir, "expected.txt"), exp)
    print(f"seed {chosen}: {len(exp)} arrays expected; generic: {case.generic(exp)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
