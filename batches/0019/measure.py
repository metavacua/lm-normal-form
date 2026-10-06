# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The table that states.dl derives, measured. For each generator (a random element of a gauge, applied to the weights by batch 0015's gauge.py) and each state of the forward pass (the
# residual stream, the normalized inputs, queries, keys and values, scores and patterns, head outputs, the pieces of the feed-forward block, the logits), the state of the transformed
# model is compared with the original's and the transformation is classified, from the most specific class to the most general:
#   invariant       the same numbers;
#   perm            the channels permuted; signed_perm: permuted and some signs changed; monomial: permuted and scaled (matching of the columns by the largest cosine, a bijection, and the
#                   residual of the fit of each column);
#   complex_plane   for queries and keys: each rotary plane (c, c + head_dim/2) multiplied by [[a, b], [-b, a]] (a least-squares fit of every plane);
#   orthogonal      the Gram matrix of the positions is the same (so the channels were rotated);
#   gl              the column spaces are the same (so the channels were mixed by an invertible matrix);
#   none            none of these.
# The program that is measured is batch 0015's model.py, traced (forward_states: its logits are the same as model.forward's, to the last bit, which is checked).
#   measure.py toy OUT.json                  the 7 configurations of symdim.py and a grouped-query model, random weights
#   measure.py real MODEL_DIR OUT.json       a Hugging Face Llama directory (all positions of two windows of WikiText-2, layers 0, middle, last)
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "0015"))
import derive
import gauge as G
from model import Arch, forward, fold_norms, init_params
from ops import FloatOps

STATES = ["emb", "h", "xhat_a", "xt_a", "q", "k", "v", "score", "pat", "oh", "ao", "hm", "xhat_m", "xt_m", "gate", "gact", "up", "m", "mo", "xhat_f", "xt_f", "logits"]
GENS = ["H1", "H4", "H4t", "N1", "M1", "A1", "A3", "A6"]
NEGATIVE = ["A4", "M2"]
ORDER = ["invariant", "perm", "signed_perm", "monomial", "complex_plane", "orthogonal", "gl", "none"]
TOL = 1e-8


def forward_states(a, P, tokens, ops, layers):
    """model.forward, with the states of the layers in `layers` (and of the end) kept, each as (positions, channels...) float64."""
    assert a.norm == "rms" and a.causal
    T = len(tokens)
    S = {}
    h = P["embed"][np.asarray(tokens)]
    S[("emb", -1)] = h
    C = Sn = None
    if a.pos == "rope":
        C, Sn = ops.rope_tables(T, a.hd)
    mask = np.tril(np.ones((T, T))).astype(np.float64)
    scale = ops.const(ops.attn_scale(a.hd))
    d_inv = ops.const_inv(a.d)

    def norm(x, gamma):
        msq = ops.red(ops.red(x * x).sum(axis=-1, keepdims=True) * d_inv)
        xh = ops.red(x * ops.norm_scale(msq))
        return xh, ops.red(xh * gamma)

    def rope(x):
        h2 = x.shape[-1] // 2
        x1, x2 = x[..., :h2], x[..., h2:]
        return np.concatenate([ops.red(x1 * C - x2 * Sn), ops.red(x2 * C + x1 * Sn)], axis=-1)
    for i in range(a.n_layers):
        keep = i in layers
        xh, x = norm(h, P[f"l{i}.attn_norm"])
        q = ops.mm(x, P[f"l{i}.wq"].T).reshape(T, a.n_heads, a.hd).transpose(1, 0, 2)
        k = ops.mm(x, P[f"l{i}.wk"].T).reshape(T, a.n_kv, a.hd).transpose(1, 0, 2)
        v = ops.mm(x, P[f"l{i}.wv"].T).reshape(T, a.n_kv, a.hd).transpose(1, 0, 2)
        if a.pos == "rope":
            q, k = rope(q), rope(k)
        kr, vr = np.repeat(k, a.rep, axis=0), np.repeat(v, a.rep, axis=0)
        s = ops.red(ops.mm(q, kr.transpose(0, 2, 1)) * scale)
        w = ops.red(ops.act_exp(s) * mask)
        w = ops.red(w * ops.inv(ops.red(w.sum(axis=-1, keepdims=True))))
        oh = ops.mm(w, vr).transpose(1, 0, 2)                                   # (T, heads, hd)
        ao = ops.mm(oh.reshape(T, a.n_heads * a.hd), P[f"l{i}.wo"].T)
        hm = ops.red(h + ao)
        xhm, xm = norm(hm, P[f"l{i}.mlp_norm"])
        gate = ops.mm(xm, P[f"l{i}.wg"].T)
        gact = ops.act_silu(gate)
        up = ops.mm(xm, P[f"l{i}.wu"].T)
        m = ops.red(gact * up)
        mo = ops.mm(m, P[f"l{i}.wd"].T)
        h = ops.red(hm + mo)
        if keep:
            S.update({("xhat_a", i): xh, ("xt_a", i): x, ("q", i): q.transpose(1, 0, 2), ("k", i): k.transpose(1, 0, 2), ("v", i): v.transpose(1, 0, 2), ("score", i): s, ("pat", i): w,
                      ("oh", i): oh, ("ao", i): ao, ("hm", i): hm, ("xhat_m", i): xhm, ("xt_m", i): xm, ("gate", i): gate, ("gact", i): gact, ("up", i): up, ("m", i): m, ("mo", i): mo, ("h", i): h})
    xhf, xf = norm(h, P["norm_f"])
    head = P["embed"] if a.tied else P["lm_head"]
    logits = ops.mm(xf, head.T)
    S.update({("xhat_f", a.n_layers): xhf, ("xt_f", a.n_layers): xf})
    return logits, S


def stack(stype, arrays):
    """The states of several sequences as one (samples, channels) matrix."""
    if stype in ("score", "pat"):
        return np.concatenate([x.reshape(x.shape[0], -1).T for x in arrays], axis=0)             # (positions x positions, heads)
    return np.concatenate([x.reshape(x.shape[0], -1) for x in arrays], axis=0)


def plane_layout(stype, a):
    return {"q": (a.n_heads, a.hd), "k": (a.n_kv, a.hd)}.get(stype)


def classify(stype, A, B, a):
    """The most specific class that relates B to A (the same state, as (samples, channels) matrices, of the transformed and of the original model), and the evidence."""
    plane = plane_layout(stype, a)
    nrm = np.linalg.norm(A)
    if np.linalg.norm(B - A) <= TOL * nrm:
        return "invariant", {}
    N, C = A.shape
    ev = {}
    if C <= 2048:
        na, nb = np.linalg.norm(A, axis=0), np.linalg.norm(B, axis=0)
        ok = (na > 1e-12 * na.max()) & (nb > 1e-12 * nb.max())
        if ok.sum() == C:
            An, Bn = A / na, B / nb
            Gm = np.abs(Bn.T @ An)
            j = Gm.argmax(1)
            if len(set(j.tolist())) == C and Gm[np.arange(C), j].min() >= 1 - 1e-9:
                s = (B * A[:, j]).sum(0) / (A[:, j] ** 2).sum(0)
                resid = np.linalg.norm(B - A[:, j] * s, axis=0) / nb
                if resid.max() <= TOL:
                    ev = {"scales_log2_range": [float(np.log2(np.abs(s)).min()), float(np.log2(np.abs(s)).max())]}
                    if np.all(np.abs(s - 1) <= 1e-8):
                        return "perm", ev
                    if np.all(np.abs(np.abs(s) - 1) <= 1e-8):
                        return "signed_perm", ev
                    return "monomial", ev
    if plane is not None:
        units, hd = plane
        h2 = hd // 2
        good = 0
        for u in range(units):
            for c in range(h2):
                cols = [u * hd + c, u * hd + c + h2]
                Ap, Bp = A[:, cols], B[:, cols]
                M, *_ = np.linalg.lstsq(Ap, Bp, rcond=None)
                if np.linalg.norm(Bp - Ap @ M) <= TOL * max(np.linalg.norm(Bp), 1e-300) and abs(M[0, 0] - M[1, 1]) <= 1e-7 * (abs(M[0, 0]) + 1e-300) and abs(M[0, 1] + M[1, 0]) <= 1e-7 * (abs(M[0, 1]) + abs(M[0, 0]) + 1e-300):
                    good += 1
        if good == units * h2:
            return "complex_plane", {}
    if N <= 4000:
        G0, G1 = A @ A.T, B @ B.T
        if np.linalg.norm(G1 - G0) <= 1e-7 * np.linalg.norm(G0):
            return "orthogonal", {}
    if C <= 64:
        M, *_ = np.linalg.lstsq(A, B, rcond=None)
        if np.linalg.norm(B - A @ M) <= TOL * np.linalg.norm(B):
            return "gl", {}
    elif N <= 4000:
        U0, s0, _ = np.linalg.svd(A, full_matrices=False)
        U1, s1, _ = np.linalg.svd(B, full_matrices=False)
        r0, r1 = int((s0 > 1e-10 * s0[0]).sum()), int((s1 > 1e-10 * s1[0]).sum())
        if r0 == r1 and np.linalg.norm(U1[:, :r1] @ U1[:, :r1].T - U0[:, :r0] @ U0[:, :r0].T) <= 1e-6 * np.sqrt(r0):
            return "gl", {"rank": r0}
    return "none", ev


def with_kappa(n, kappa, rng):
    u, _ = np.linalg.qr(rng.standard_normal((n, n)))
    v, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return u @ np.diag(np.geomspace(1.0, kappa, n)) @ v


def nontrivial_perm(rng, n):
    while True:
        p = [int(i) for i in rng.permutation(n)]
        if p != list(range(n)):
            return p


def apply(name, a, P, ops, rng):
    """(arch, parameters) of the transformed model, and (arch, parameters) of the model that it is compared with."""
    nl, nk, hd, d, F = a.n_layers, a.n_kv, a.hd, a.d, a.d_ff
    if name == "H1":
        perm = nontrivial_perm(rng, d)
        signs = [int(s) for s in rng.choice([1, -1], d)]
        signs[0] = -1
        return (a, P), G.residual_signed_perm(a, P, ops, perm, signs)
    if name == "H4":
        base = fold_norms(a, P, ops)
        return base, G.residual_orth(base[0], base[1], ops, ops.rand_orthogonal(d, rng), fold=False)
    if name == "H4t":
        return (a, P), G.residual_orth_conj(a, P, ops, ops.rand_orthogonal(d, rng))
    if name == "N1":
        cs = {f"l{i}.{n}": ops.rand_nonzero(rng, (d,)) for i in range(nl) for n in ("attn_norm", "mlp_norm")}
        cs["norm_f"] = ops.rand_nonzero(rng, (d,))
        return (a, P), G.norm_gauge(a, P, ops, cs)
    if name == "M1":
        return (a, P), G.mlp_gauge(a, P, ops, [np.array(nontrivial_perm(rng, F)) for _ in range(nl)], [ops.rand_nonzero(rng, (F,)) for _ in range(nl)])
    if name == "A1":
        return (a, P), G.ov_gauge(a, P, ops, [[with_kappa(hd, 3.0, rng) for _ in range(nk)] for _ in range(nl)])
    if name == "A3":
        return (a, P), G.qk_gauge_rope(a, P, ops, [ops.rand_plane_scalars((nk, hd // 2), rng) for _ in range(nl)])
    if name == "A6":
        return (a, P), G.head_perm(a, P, ops, [nontrivial_perm(rng, nk) for _ in range(nl)], [[nontrivial_perm(rng, a.rep) if a.rep > 1 else [0] for _ in range(nk)] for _ in range(nl)])
    if name == "A4":
        return (a, P), G.qk_gauge_general(a, P, ops, [[with_kappa(hd, 3.0, rng) for _ in range(nk)] for _ in range(nl)])
    if name == "M2":
        perms = [np.arange(F) for _ in range(nl)]
        return (a, P), G.gate_scale(a, P, ops, perms, [ops.rand_nonzero(rng, (F,)) for _ in range(nl)], [ops.rand_nonzero(rng, (F,)) for _ in range(nl)])
    raise ValueError(name)


def measure(a, P, seqs, ops, layers, eps_zero=False, logprobs=False, gens=GENS + NEGATIVE, seed=19, log=print):
    """The derived and the measured table for one architecture: {generator: {state type: (derived class, measured classes by layer)}}, with the logit change of each generator."""
    rng = np.random.default_rng(seed)
    der_u, _ = derive.derive(a, eps_zero=eps_zero, logprobs=logprobs, fold=False)
    der_f, _ = derive.derive(a.with_(tied=False), eps_zero=eps_zero, logprobs=logprobs, fold=True)
    out = {"derived_dimension_unfolded": der_u["total"], "derived_groups": der_u["groups"], "generators": {},
           "library_checks_empty": all(not der[k] for der in (der_u, der_f) for k in ("no_glb", "many_glb", "ungrouped", "two_groups"))}
    cache = {}
    for g in gens:
        t0 = time.time()
        (a0, P0), (a1, P1) = apply(g, a, P, ops, rng)
        key = (id(P0),)
        if key not in cache:
            cache.clear()
            cache[key] = [forward_states(a0, P0, s, ops, layers) for s in seqs]
        base = cache[key]
        new = [forward_states(a1, P1, s, ops, layers) for s in seqs]
        der = der_f if g == "H4" else der_u
        logit_change = max(float(np.abs(n[0] - b[0]).max()) for n, b in zip(new, base))
        scale0 = max(float(np.abs(b[0]).max()) for b in base)
        rec = {"program": "folded, untied" if g == "H4" else "unfolded", "allowed": g in der["allowed"], "logit_change": logit_change, "states": {}}
        if g in der["allowed"]:
            for st in STATES:
                want = der["expect"].get((g, st))
                if want is None:
                    continue
                if st == "logits":                                          # the logits are not stored with the states: invariance is the logit change
                    rec["states"][st] = {"derived": want, "measured": {"-": "invariant" if logit_change <= TOL * max(1.0, scale0) else "none"}, "agree": None}
                    rec["states"][st]["agree"] = rec["states"][st]["measured"]["-"] == want
                    continue
                kinds = sorted({k for k in base[0][1] if k[0] == st})
                got = {}
                for k in kinds:
                    got[k[1]] = classify(st, stack(st, [b[1][k] for b in base]), stack(st, [n[1][k] for n in new]), a0)[0]
                rec["states"][st] = {"derived": want, "measured": got, "agree": all(c == want for c in got.values())}
        out["generators"][g] = rec
        log(f"{g}: allowed by the derivation {rec['allowed']}, logits change {logit_change:.2e}, states agreeing {sum(1 for v in rec['states'].values() if v['agree'])}/{len(rec['states'])}  ({time.time() - t0:.1f}s)")
    return out


def toy_models():
    base = Arch(d=8, n_heads=4, n_kv=2, hd=4, d_ff=12, n_layers=2, vocab=7)
    yield "gqa-untied", base, False, False
    yield "gqa-tied", base.with_(tied=True), False, False
    yield "mha-untied", base.with_(n_kv=4), False, False
    yield "gqa-untied-eps0", base, True, False


def main(argv):
    ops_f = lambda eps, base=10000.0: FloatOps(eps=eps, base=base)
    if argv[0] == "toy":
        res = {}
        for name, a, e0, lp in toy_models():
            ops = ops_f(0.0 if e0 else 1e-6)
            rng = np.random.default_rng(3)
            P = init_params(a, ops, rng)
            seqs = [[int(t) for t in rng.integers(0, a.vocab, 24)] for _ in range(8)]
            lg = forward(a, P, seqs[0], ops)
            assert np.array_equal(lg, forward_states(a, P, seqs[0], ops, set(range(a.n_layers)))[0]), "the traced forward is not model.forward"
            print("==", name)
            res[name] = measure(a, P, seqs, ops, set(range(a.n_layers)), eps_zero=e0)
        json.dump(res, open(argv[1], "w"), indent=1)
    else:
        import hf_adapter as H
        from common import wikitext_windows
        from transformers import AutoTokenizer
        model_dir, out = argv[1], argv[2]
        a, P, cfg = H.load(model_dir)
        ops = ops_f(cfg["rms_norm_eps"], cfg.get("rope_theta", 10000.0))
        tok = AutoTokenizer.from_pretrained(model_dir)
        wins, _ = wikitext_windows(tok, n=2, size=512)
        layers = {0, a.n_layers // 2, a.n_layers - 1}
        print("traced forward equals model.forward:", np.array_equal(forward(a, P, wins[0][:48], ops), forward_states(a, P, wins[0][:48], ops, layers)[0]), flush=True)
        res = measure(a, P, wins, ops, layers)
        res["model_dir"] = model_dir
        res["arch"] = {k: getattr(a, k) for k in ("d", "n_heads", "n_kv", "hd", "d_ff", "n_layers", "vocab", "tied")}
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1:])
