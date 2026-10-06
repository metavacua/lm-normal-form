# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The symmetries on a real checkpoint, run as a language model. P is the checkpoint's weights as float64 (hf_adapter.load). For each gauge, P' = g(P) is computed (gauge.py, in
# float64, with a random element of the gauge), and then
#   (a) the formal forward pass (model.py, float64, the real nonlinearities) is run on P and on P': the largest difference of the logits of 6 sequences (the claim of claims.py on real
#       numbers), and the KV cache of every layer of P' against the cache that the claims K1 to K6 say it must be (unchanged; the groups permuted; the values mapped by A; the key planes
#       mapped by their complex scalars): the largest difference relative to the largest entry;
#   (b) P' is written as an ordinary Hugging Face directory in float32 and run by Transformers on the CPU: the largest difference of its logits from the original float32 model's,
#       the top-1 agreement, and whether the greedy generations of 32 tokens from 4 prompts are the original's;
#   (c) the same with the weights rounded to bfloat16, against the original in bfloat16.
# The gauges: the signed permutation of the hidden coordinates (H1), the heads (A6), the units and their scales (M1), the value/output gauge at condition numbers 1, 10, 100, 1,000 (A1),
# the query/key complex scalars (A3), the norm gauge (N1), a random orthogonal matrix after folding the norms and untying the head (H4), the composition of these (C1), and, as a control that
# must change the function, the centering of the embedding that the fuse_layer_norms of QuaRot and SpinQuant apply to every model type (H8).
# Usage: lm_gauge.py MODEL_DIR OUT.json
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hf_adapter as H
import gauge as G
from model import forward
from ops import FloatOps

PLAIN = ["The capital of France is", "Water boils at one hundred degrees Celsius at sea level, and", "In 1969, the first human to walk on the Moon was", "To make a cup of tea, first"]


def with_kappa(n, kappa, rng):
    """A random invertible matrix with singular values from 1 to kappa (geometrically spaced)."""
    u, _ = np.linalg.qr(rng.standard_normal((n, n)))
    v, _ = np.linalg.qr(rng.standard_normal((n, n)))
    return u @ np.diag(np.geomspace(1.0, kappa, n)) @ v


def perm(rng, n):
    return [int(i) for i in rng.permutation(n)]


def cache(a, P, seq, ops):
    tr = {}
    forward(a, P, seq, ops, trace=tr)
    return [(tr[("k", i)], tr[("v", i)]) for i in range(a.n_layers)]


def build_gauges(a, ops):
    """name -> g(arch, P, rng) = (arch2, P2, expected), where expected(cache of P) is the cache that P2 must have (None: the claims say nothing, or the control must change it)."""
    nl, nk, hd, d, F = a.n_layers, a.n_kv, a.hd, a.d, a.d_ff
    same = lambda c: c

    def h1(arch, P, rng):
        return (*G.residual_signed_perm(arch, P, ops, perm(rng, d), [int(s) for s in rng.choice([1, -1], d)]), same)

    def a6(arch, P, rng):
        gps = [perm(rng, nk) for _ in range(nl)]
        a2, P2 = G.head_perm(arch, P, ops, gps, [[perm(rng, a.rep) for _ in range(nk)] for _ in range(nl)])
        return a2, P2, lambda c: [(k[gps[i]], v[gps[i]]) for i, (k, v) in enumerate(c)]

    def m1(arch, P, rng):
        return (*G.mlp_gauge(arch, P, ops, [np.array(perm(rng, F)) for _ in range(nl)], [ops.rand_nonzero(rng, (F,)) for _ in range(nl)]), same)

    def ov(kappa):
        def g(arch, P, rng):
            As = [[with_kappa(hd, kappa, rng) for _ in range(nk)] for _ in range(nl)]
            a2, P2 = G.ov_gauge(arch, P, ops, As)
            return a2, P2, lambda c: [(k, np.stack([v[gi] @ As[i][gi].T for gi in range(nk)])) for i, (k, v) in enumerate(c)]
        return g

    def a3(arch, P, rng):
        sc = [ops.rand_plane_scalars((nk, hd // 2), rng) for _ in range(nl)]
        a2, P2 = G.qk_gauge_rope(arch, P, ops, sc)

        def expected(c):
            out, h2 = [], hd // 2
            for i, (k, v) in enumerate(c):
                ra, rb = sc[i]
                e = k.copy()
                for gi in range(nk):
                    for j in range(h2):
                        x1, x2 = k[gi, :, j], k[gi, :, j + h2]
                        e[gi, :, j], e[gi, :, j + h2] = ra[gi, j] * x1 - rb[gi, j] * x2, rb[gi, j] * x1 + ra[gi, j] * x2
                out.append((e, v))
            return out
        return a2, P2, expected

    def n1(arch, P, rng):
        cs = {f"l{i}.{n}": ops.rand_nonzero(rng, (d,)) for i in range(nl) for n in ("attn_norm", "mlp_norm")}
        cs["norm_f"] = ops.rand_nonzero(rng, (d,))
        return (*G.norm_gauge(arch, P, ops, cs), same)

    def h4(arch, P, rng):
        return (*G.residual_orth(arch, P, ops, ops.rand_orthogonal(d, rng), fold=True), same)

    def c1(arch, P, rng):
        arch, P = G.residual_signed_perm(arch, P, ops, perm(rng, d), [int(s) for s in rng.choice([1, -1], d)])
        arch, P = G.ov_gauge(arch, P, ops, [[with_kappa(hd, 10.0, rng) for _ in range(nk)] for _ in range(nl)])
        arch, P = G.qk_gauge_rope(arch, P, ops, [ops.rand_plane_scalars((nk, hd // 2), rng) for _ in range(nl)])
        arch, P = G.head_perm(arch, P, ops, [perm(rng, nk) for _ in range(nl)], [[perm(rng, a.rep) for _ in range(nk)] for _ in range(nl)])
        arch, P = G.mlp_gauge(arch, P, ops, [np.array(perm(rng, F)) for _ in range(nl)], [ops.rand_nonzero(rng, (F,)) for _ in range(nl)])
        cs = {f"l{i}.{n}": ops.rand_nonzero(rng, (d,)) for i in range(nl) for n in ("attn_norm", "mlp_norm")}
        cs["norm_f"] = ops.rand_nonzero(rng, (d,))
        return (*G.norm_gauge(arch, P, ops, cs), None)

    def h8(arch, P, rng):
        return (*G.embed_center(arch, P, ops), None)

    return {
        "H1 signed permutation of the hidden coordinates": h1,
        "A6 heads and key/value groups permuted": a6,
        "M1 units permuted and scaled": m1,
        "A1 value/output gauge, kappa 1": ov(1.0),
        "A1 value/output gauge, kappa 10": ov(10.0),
        "A1 value/output gauge, kappa 100": ov(100.0),
        "A1 value/output gauge, kappa 1000": ov(1000.0),
        "A3 query/key complex scalars": a3,
        "N1 norm gauge": n1,
        "H4 random orthogonal Q, norms folded, head untied": h4,
        "C1 composition": c1,
        "H8 control: the embedding centered (QuaRot, SpinQuant)": h8,
    }


def run_hf(dir_, seqs, prompts, dtype):
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(int(os.environ.get("LMNF_THREADS", "4")))
    dt = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    model = AutoModelForCausalLM.from_pretrained(dir_, dtype=dt).eval()
    with torch.no_grad():
        L = [model(input_ids=torch.tensor([s])).logits[0].float().numpy() for s in seqs]
        G_ = [model.generate(torch.tensor([p]), max_new_tokens=32, do_sample=False, pad_token_id=0)[0][len(p):].tolist() for p in prompts]
    return L, G_


def cache_error(c_expected, c_actual):
    scale = max(float(np.abs(x).max()) for kv in c_expected for x in kv)
    return max(float(np.abs(e - x).max()) for ce, ca in zip(c_expected, c_actual) for e, x in zip(ce, ca)) / scale


def main(model_dir, out):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_dir)
    texts = open(os.path.join(HERE, "..", "0010", "prompts.txt"), encoding="utf-8").read().splitlines() + open(os.path.join(HERE, "..", "0013", "sentences.txt"), encoding="utf-8").read().splitlines()[:3]
    seqs = [tok(t)["input_ids"] for t in texts]
    prompts = [tok(t)["input_ids"] for t in PLAIN]
    a, P, cfg = H.load(model_dir)
    ops = FloatOps(eps=cfg["rms_norm_eps"], base=cfg.get("rope_theta", 10000.0))
    base = [forward(a, P, s, ops) for s in seqs]
    base_cache = [cache(a, P, s, ops) for s in seqs]
    ref32, gen32 = run_hf(model_dir, seqs, prompts, "float32")
    ref16, gen16 = run_hf(model_dir, seqs, prompts, "bfloat16")
    res = {"model_dir": model_dir, "tied": a.tied, "gauges": {}}
    work = os.environ.get("LMNF_WORK", "work-gauge")
    for name, g in build_gauges(a, ops).items():
        t = time.time()
        rng = np.random.default_rng(15)
        a2, P2, expected = g(a, P, rng)
        mine = max(float(np.abs(forward(a2, P2, s, ops) - b).max()) for s, b in zip(seqs, base))
        c2 = [cache(a2, P2, s, ops) for s in seqs]
        r = {"formal_float64_max_abs_difference": mine, "untied_by_the_gauge": a2.tied != a.tied,
             "cache_difference_from_the_original": max(cache_error(b, c) for b, c in zip(base_cache, c2))}
        if expected is not None:
            r["cache_difference_from_the_predicted"] = max(cache_error(expected(b), c) for b, c in zip(base_cache, c2))
        for dtype, ref, gen in (("float32", ref32, gen32), ("bfloat16", ref16, gen16)):
            d = os.path.join(work, "g")
            H.save(a2, P2, model_dir, d, dtype)
            L, Gn = run_hf(d, seqs, prompts, dtype)
            r[dtype] = {"max_abs_difference": max(float(np.abs(x - y).max()) for x, y in zip(L, ref)), "top1_agreement": float(np.mean(np.concatenate([x.argmax(-1) == y.argmax(-1) for x, y in zip(L, ref)]))),
                        "generations_identical": sum(1 for x, y in zip(Gn, gen) if x == y), "of": len(gen)}
        r["seconds"] = round(time.time() - t, 1)
        res["gauges"][name] = r
        pred = r.get("cache_difference_from_the_predicted")
        print(f"{name:55s} formal {mine:.2e} | cache vs predicted {'n/a' if pred is None else format(pred, '.2e')}, vs original {r['cache_difference_from_the_original']:.2e} | "
              f"float32 {r['float32']['max_abs_difference']:.2e}, gen {r['float32']['generations_identical']}/{r['float32']['of']} | bf16 {r['bfloat16']['max_abs_difference']:.2e}, gen {r['bfloat16']['generations_identical']}/{r['bfloat16']['of']}", flush=True)
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
