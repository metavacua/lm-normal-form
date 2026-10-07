# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# A Python port of Model.lean and Runtime.lean (structured layers, per head and plane, and the gauge definitions of batch 0021), run on the data of gen22.py: a check of the conventions that
# does not need Lean. It was run before the output of the Lean run was read; it is not the registered test (that is the Lean run, compared by check22.py).
#   port22.py [--start N]    LMNF22_ARCH=d3 (default) or d2
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "0015"))
import numpy as np
import gen22 as g
P = g.P

def mat(rows): return [list(r) for r in rows]
def mv(M, x): return [sum(M[i][j] * x[j] for j in range(len(x))) % P for i in range(len(M))]
def inv(x): return pow(x % P, P - 2, P)

class Lay:
    pass

def to_layer(d, hv, h2, nh, nkv, dff, F):
    L = Lay()
    # F: dict of numpy object arrays wq wk wv wo wg wu wd an mn (flat layout)
    L.Wq = [[[[int(F["wq"][a * hv + p + r * h2][j]) for j in range(d)] for r in range(2)] for p in range(h2)] for a in range(nh)]
    L.Wk = [[[[int(F["wk"][gi * hv + p + r * h2][j]) for j in range(d)] for r in range(2)] for p in range(h2)] for gi in range(nkv)]
    L.Wv = [[[int(F["wv"][gi * hv + r][j]) for j in range(d)] for r in range(hv)] for gi in range(nkv)]
    L.Wo = [[[int(F["wo"][i][a * hv + r]) for r in range(hv)] for i in range(d)] for a in range(nh)]
    L.Wg = [[int(F["wg"][i][j]) for j in range(d)] for i in range(dff)]
    L.Wu = [[int(F["wu"][i][j]) for j in range(d)] for i in range(dff)]
    L.Wd = [[int(F["wd"][i][j]) for j in range(dff)] for i in range(d)]
    L.ga = [int(x) for x in F["an"]]
    L.gm = [int(x) for x in F["mn"]]
    return L

def circle(t):
    den = inv(1 + t * t)
    return (1 - t * t) * den % P, 2 * t * den % P

def rot(ts, t, p):
    c, s = circle(ts[p])
    cn, sn = 1, 0
    for _ in range(t):
        cn, sn = (c * cn - s * sn) % P, (s * cn + c * sn) % P
    return [[cn, (-sn) % P], [sn, cn]]

class Env:
    def __init__(self, T, nh, nkv, h2, rep, ts):
        self.T, self.nh, self.nkv, self.h2, self.rep, self.ts = T, nh, nkv, h2, rep, ts
    def gr(self, a): return a // self.rep

def nrm(gamma, x):
    d = len(x)
    msq = sum(v * v for v in x) * inv(d) % P
    rs = inv(msq * msq + 3)
    return [gamma[i] * (rs * x[i] % P) % P for i in range(d)]

def layer_fn(E, L, h):
    T, nh, nkv, h2 = E.T, E.nh, E.nkv, E.h2
    out = []
    for t in range(T):
        n_t = [nrm(L.ga, h[s]) for s in range(T)]
        def q(t_, a, p): return mv(rot(E.ts, t_, p), mv(L.Wq[a][p], n_t[t_]))
        def k(s, gi, p): return mv(rot(E.ts, s, p), mv(L.Wk[gi][p], n_t[s]))
        def v(s, gi): return mv(L.Wv[gi], n_t[s])
        def score(t_, s, a):
            return 7 * sum(sum(x * y for x, y in zip(q(t_, a, p), k(s, E.gr(a), p))) for p in range(h2)) % P
        def wt(t_, s, a):
            if s <= t_:
                sc = score(t_, s, a); return (sc * sc + sc + 1) % P
            return 0
        att_out = [0] * len(h[0])
        for a in range(nh):
            ws = [wt(t, s, a) for s in range(T)]
            den = inv(sum(ws) % P)
            attn = [sum(ws[s] * den % P * v(s, E.gr(a))[r] for s in range(T)) % P for r in range(len(L.Wv[0]))]
            contrib = mv(L.Wo[a], attn)
            att_out = [(x + y) % P for x, y in zip(att_out, contrib)]
        after = [(h[t][i] + att_out[i]) % P for i in range(len(h[0]))]
        x = nrm(L.gm, after)
        gate, up = mv(L.Wg, x), mv(L.Wu, x)
        m = [((gt * gt + 2 * gt) % P) * u % P for gt, u in zip(gate, up)]
        down = mv(L.Wd, m)
        out.append([(after[i] + down[i]) % P for i in range(len(after))])
    return out

def run(E, layers, emb_rows, head, gf, toks):
    h = [list(map(int, emb_rows[t])) for t in toks]
    states = []
    for L in layers:
        h = layer_fn(E, L, h)
        states.append(h)
    logits = [mv(head, nrm(gf, h[t])) for t in range(len(toks))]
    return states, logits

case, _exp0, _seed = g.choose(None, int(sys.argv[sys.argv.index('--start') + 1]) if '--start' in sys.argv else 0)
print('seed', _seed, g.ARCH)
a = case.a
E = Env(len(g.TOKENS), a.n_heads, a.n_kv, a.hd // 2, a.rep, g.TS)
def flat_layers(Pm, ar):
    out = []
    for i in range(ar.n_layers):
        F = {"wq": Pm[f"l{i}.wq"], "wk": Pm[f"l{i}.wk"], "wv": Pm[f"l{i}.wv"], "wo": Pm[f"l{i}.wo"], "wg": Pm[f"l{i}.wg"], "wu": Pm[f"l{i}.wu"], "wd": Pm[f"l{i}.wd"], "an": Pm[f"l{i}.attn_norm"], "mn": Pm[f"l{i}.mlp_norm"]}
        out.append(to_layer(ar.d, ar.hd, ar.hd // 2, ar.n_heads, ar.n_kv, ar.d_ff, F))
    return out
exp = case.expected()
def check(label, ar, Pm):
    layers = flat_layers(Pm, ar)
    states, logits = run(E, layers, Pm["embed"], [[int(x) for x in r] for r in Pm["lm_head"]] if "lm_head" in Pm else None, [int(x) for x in Pm["norm_f"]], g.TOKENS)
    ok = True
    for i, st in enumerate(states):
        ok &= [[int(v) for v in r] for r in st] == exp[f"{label}.h{i}"]
    ok &= logits == exp[f"{label}.logits"]
    return ok
print("orig", check("orig", a, case.P))
print("fold", check("fold", case.a_fold, case.P_fold))
for gname, (ag, Pg) in case.gauged.items():
    print(gname, check(gname, ag, Pg))

# ---- the gauges of the Lean files, applied to the structured layers, flattened again and compared with gauge.py ----
import copy
def mm(A, B): return [[sum(A[i][k] * B[k][j] for k in range(len(B))) % P for j in range(len(B[0]))] for i in range(len(A))]
def T_(A): return [list(r) for r in zip(*A)]
def diag(v): return [[v[i] if i == j else 0 for j in range(len(v))] for i in range(len(v))]
def cplx(a, b): return [[a % P, (-b) % P], [b % P, a % P]]
def scal(c, A): return [[c * x % P for x in r] for r in A]

def flatten(L, d, hv, h2, nh, nkv, dff):
    out = {}
    out["wq"] = [L.Wq[a][p][r] for a in range(nh) for r in range(2) for p in range(h2)]
    out["wk"] = [L.Wk[gi][p][r] for gi in range(nkv) for r in range(2) for p in range(h2)]
    out["wv"] = [L.Wv[gi][r] for gi in range(nkv) for r in range(hv)]
    out["wo"] = [[L.Wo[a][i][r] for a in range(nh) for r in range(hv)] for i in range(d)]
    out["wg"], out["wu"], out["wd"] = L.Wg, L.Wu, L.Wd
    out["an"], out["mn"] = [L.ga], [L.gm]
    return out

def ar_dims(ar): return ar.d, ar.hd, ar.hd // 2, ar.n_heads, ar.n_kv, ar.d_ff

def compare(label, layers, ar, Pg):
    d, hv, h2, nh, nkv, dff = ar_dims(ar)
    ok = True
    for i, L in enumerate(layers):
        fl = flatten(L, d, hv, h2, nh, nkv, dff)
        for py, ln in g.FLAT:
            want = g.rows(Pg[f"l{i}.{py}"])
            got = [[int(x) % P for x in r] for r in fl[ln]]
            if got != want:
                ok = False
                print("   mismatch", label, i, ln)
    return ok

base = flat_layers(case.P, a)
d, hv, h2, nh, nkv, dff = ar_dims(a)
rep = a.rep
# ov
lay = []
for i, L in enumerate(base):
    L2 = copy.deepcopy(L)
    A = [[[int(x) for x in r] for r in case.As[i][gi]] for gi in range(nkv)]
    B = [[[int(x) for x in r] for r in case.ops.matinv(case.As[i][gi])] for gi in range(nkv)]
    L2.Wv = [mm(A[gi], L.Wv[gi]) for gi in range(nkv)]
    L2.Wo = [mm(L.Wo[ah], B[ah // rep]) for ah in range(nh)]
    lay.append(L2)
print("ov structured vs gauge.py:", compare("ov", lay, a, case.gauged["ov"][1]))
# qk
lay = []
for i, L in enumerate(base):
    L2 = copy.deepcopy(L)
    qa, qb = case.scal[i]
    L2.Wk = [[mm(cplx(int(qa[gi, p]), int(qb[gi, p])), L.Wk[gi][p]) for p in range(h2)] for gi in range(nkv)]
    def qmat(ah, p):
        gi = ah // rep
        aa, bb = int(qa[gi, p]), int(qb[gi, p])
        s = inv(aa * aa + bb * bb)
        return T_(scal(s, cplx(aa, -bb)))
    L2.Wq = [[mm(qmat(ah, p), L.Wq[ah][p]) for p in range(h2)] for ah in range(nh)]
    lay.append(L2)
print("qk structured vs gauge.py:", compare("qk", lay, a, case.gauged["qk"][1]))
# head perm
lay = []
for i, L in enumerate(base):
    L2 = copy.deepcopy(L)
    sg, tau = case.sig_heads[i], case.gperms[i]
    L2.Wq = [L.Wq[sg[ah]] for ah in range(nh)]
    L2.Wo = [L.Wo[sg[ah]] for ah in range(nh)]
    L2.Wk = [L.Wk[tau[gi]] for gi in range(nkv)]
    L2.Wv = [L.Wv[tau[gi]] for gi in range(nkv)]
    lay.append(L2)
print("headperm structured vs gauge.py:", compare("hp", lay, a, case.gauged["hp"][1]))
# units: scale then permute
lay = []
for i, L in enumerate(base):
    L2 = copy.deepcopy(L)
    c = [int(x) for x in case.uscales[i]]
    Wu1 = [[c[r] * x % P for x in L.Wu[r]] for r in range(dff)]
    Wd1 = [[x * inv(c[j]) % P for j, x in enumerate(row)] for row in L.Wd]
    sg = case.uperms[i]
    L2.Wg = [L.Wg[sg[r]] for r in range(dff)]
    L2.Wu = [Wu1[sg[r]] for r in range(dff)]
    L2.Wd = [[row[sg[j]] for j in range(dff)] for row in Wd1]
    lay.append(L2)
print("units structured vs gauge.py:", compare("m1", lay, a, case.gauged["m1"][1]))
# norms
lay = []
for i, L in enumerate(base):
    L2 = copy.deepcopy(L)
    ca = [int(x) for x in case.cs[f"l{i}.attn_norm"]]; cm = [int(x) for x in case.cs[f"l{i}.mlp_norm"]]
    L2.ga = [ca[j] * L.ga[j] % P for j in range(d)]
    L2.gm = [cm[j] * L.gm[j] % P for j in range(d)]
    dinv = lambda cc: diag([inv(x) for x in cc])
    L2.Wq = [[mm(L.Wq[ah][p], dinv(ca)) for p in range(h2)] for ah in range(nh)]
    L2.Wk = [[mm(L.Wk[gi][p], dinv(ca)) for p in range(h2)] for gi in range(nkv)]
    L2.Wv = [mm(L.Wv[gi], dinv(ca)) for gi in range(nkv)]
    L2.Wg, L2.Wu = mm(L.Wg, dinv(cm)), mm(L.Wu, dinv(cm))
    lay.append(L2)
print("norms structured vs gauge.py:", compare("n1", lay, a, case.gauged["n1"][1]))
# stream gauge H1 (signed permutation): Q = sp sigma s, Qi = sp sigma^-1 (s o sigma^-1)
sg = case.sigma; s = case.signs
def sp(sigma, sgn): return [[sgn[i] if sigma[i] == j else 0 for j in range(len(sigma))] for i in range(len(sigma))]
Qm = sp(sg, s)
sinv = inverse = [0] * d
for i, x in enumerate(sg): sinv[x] = i
Qi = sp(sinv, [s[sinv[k]] for k in range(d)])
lay = []
for i, L in enumerate(base):
    L2 = copy.deepcopy(L)
    L2.Wq = [[mm(L.Wq[ah][p], Qi) for p in range(h2)] for ah in range(nh)]
    L2.Wk = [[mm(L.Wk[gi][p], Qi) for p in range(h2)] for gi in range(nkv)]
    L2.Wv = [mm(L.Wv[gi], Qi) for gi in range(nkv)]
    L2.Wo = [mm(Qm, L.Wo[ah]) for ah in range(nh)]
    L2.Wg, L2.Wu = mm(L.Wg, Qi), mm(L.Wu, Qi)
    L2.Wd = mm(Qm, L.Wd)
    L2.ga = [L.ga[sg[j]] for j in range(d)]
    L2.gm = [L.gm[sg[j]] for j in range(d)]
    lay.append(L2)
print("h1 layers structured vs gauge.py:", compare("h1", lay, a, case.gauged["h1"][1]))
emb1 = [mv(Qm, [int(x) for x in case.P["embed"][v]]) for v in range(a.vocab)]
head1 = mm([[int(x) for x in r] for r in case.P["lm_head"]], Qi)
gf1 = [int(case.P["norm_f"][sg[j]]) for j in range(d)]
Pg = case.gauged["h1"][1]
print("h1 emb/head/gf:", emb1 == g.rows(Pg["embed"]), head1 == g.rows(Pg["lm_head"]), [gf1] == g.rows(Pg["norm_f"]))
