# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Cell 08 of batch 0014: the KV cache under the transformations. A runtime stores, for every layer and every token, the keys (after the rotary embedding) and the values of
# every key/value group. The cache is computed from the layer's normalized input and the layer's own input projections only, so a change of basis of the residual stream
# that is compensated in the projections leaves it unchanged; a permutation of heads and groups permutes it; the diagonal value/output scalings and the per-plane
# key/query scalings of batch 0013 scale its channels. In exact arithmetic the cache of a variant is therefore the original's multiplied by a monomial matrix (a
# permutation times a diagonal), and for the symmetries of the residual stream and of the feed-forward units the identity. This cell measures that.
#   per layer and per variant, for keys and values separately: the channels (columns of the T x (n_kv head_dim) matrix of all cached vectors of the 33 texts) of the variant are matched to
#   the original's by the largest absolute cosine; min_cos is the smallest of those cosines; for each channel the factor s of the least-squares fit B_j = s A_match(j) and the
#   relative residual of the fit (max_resid); bijection: the matching uses every channel once; identity: it is the identity; unit: every s is 1; pow2: every |s| is a power of 2.
# The cache is taken from the float64 reference (ref64.py: tolerance 1e-9) and, with --hf, from Transformers' own DynamicCache in float32 (tolerance 1e-4; a real runtime's cache).
# Usage: cache.py WORK OUT.json [--hf]
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import dump


def monomial(A, B, tol):
    """Is B (T x C) = A (T x C) times a monomial matrix? Returns the measurements of the module docstring."""
    na, nb = np.linalg.norm(A, axis=0), np.linalg.norm(B, axis=0)
    la, lb = na > 1e-9 * max(na.max(), 1e-300), nb > 1e-9 * max(nb.max(), 1e-300)
    r = {"channels": int(A.shape[1]), "dead_original": int((~la).sum()), "dead_variant": int((~lb).sum())}
    if la.sum() != lb.sum() or la.sum() == 0:
        r.update({"min_cos": 0.0, "max_resid": 1.0, "bijection": False, "identity": False, "unit": False, "pow2": False, "matched": False})
        return r
    ia, ib = np.flatnonzero(la), np.flatnonzero(lb)
    An, Bn = A[:, ia] / na[ia], B[:, ib] / nb[ib]
    G = np.abs(Bn.T @ An)
    j = G.argmax(1)
    cos = G[np.arange(len(j)), j]
    Am = A[:, ia][:, j]
    s = (B[:, ib] * Am).sum(0) / (Am * Am).sum(0)
    resid = np.linalg.norm(B[:, ib] - Am * s, axis=0) / nb[ib]
    lg = np.log2(np.abs(s))
    r.update({"min_cos": float(cos.min()), "max_resid": float(resid.max()), "bijection": bool(len(set(j.tolist())) == len(j)), "identity": bool(np.array_equal(ia[j], ib)),
              "unit": bool(np.all(np.abs(s - 1.0) <= tol)), "pow2": bool(np.all(np.abs(lg - np.round(lg)) <= tol)), "negative": int((s < 0).sum()),
              "log2_scale_range": [float(lg.min()), float(lg.max())], "matched": bool(resid.max() <= tol)})
    return r


def flat(c):
    """list over sequences of list over layers of (K, V) each (n_kv, T, hd) -> per layer (K, V) as (sum T, n_kv hd) matrices."""
    L = len(c[0])
    out = []
    for l in range(L):
        K = np.concatenate([s[l][0].transpose(1, 0, 2).reshape(s[l][0].shape[1], -1) for s in c])
        V = np.concatenate([s[l][1].transpose(1, 0, 2).reshape(s[l][1].shape[1], -1) for s in c])
        out.append((K, V))
    return out


def summarize(per_layer):
    """per_layer: list over layers of {'K': r, 'V': r}. Aggregates over layers and over keys and values."""
    rs = [x[k] for x in per_layer for k in ("K", "V")]
    return {"layers": len(per_layer), "min_cos": min(r["min_cos"] for r in rs), "max_resid": max(r["max_resid"] for r in rs),
            "all_matched": all(r["matched"] for r in rs), "all_bijection": all(r["bijection"] for r in rs), "all_identity": all(r["identity"] for r in rs),
            "all_unit": all(r["unit"] for r in rs), "all_pow2": all(r["pow2"] for r in rs),
            "layers_not_matched": [i for i, x in enumerate(per_layer) if not (x["K"]["matched"] and x["V"]["matched"])],
            "keys_matched_layers": sum(1 for x in per_layer if x["K"]["matched"]), "values_matched_layers": sum(1 for x in per_layer if x["V"]["matched"])}


def caches_reference(model_dir, seqs):
    import ref64
    ref = ref64.Ref(model_dir)
    return [ref.forward(s, kv=True)[1] for s in seqs]


def caches_hf(model_dir, seqs):
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(int(os.environ.get("LMNF_THREADS", "4")))
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=torch.float32).eval()
    out = []
    with torch.no_grad():
        for s in seqs:
            pkv = model(input_ids=torch.tensor([s]), use_cache=True).past_key_values
            try:
                pairs = [(x.keys, x.values) for x in pkv.layers]
            except AttributeError:
                try:
                    pairs = list(zip(pkv.key_cache, pkv.value_cache))
                except AttributeError:
                    pairs = list(pkv.to_legacy_cache())
            out.append([(k[0].double().numpy(), v[0].double().numpy()) for k, v in pairs])
    return out


def main(work, out, hf=False):
    seqs = json.load(open(os.path.join(work, "texts.json")))["ids"]
    variants = sorted(v for v in os.listdir(os.path.join(work, "variants")) if os.path.isdir(os.path.join(work, "variants", v)))
    res = {"texts": len(seqs), "positions": int(sum(len(s) for s in seqs)), "views": {}}
    for view, getter, tol in (("float64_reference", caches_reference, 1e-9),) + ((("transformers_float32", caches_hf, 1e-4),) if hf else ()):
        base = flat(getter(os.path.join(work, "variants", "orig"), seqs))
        res["views"][view] = {"tolerance": tol, "variants": {}}
        for v in variants:
            if v == "orig":
                continue
            cur = flat(getter(os.path.join(work, "variants", v), seqs))
            per = [{"K": monomial(b[0], c[0], tol), "V": monomial(b[1], c[1], tol)} for b, c in zip(base, cur)]
            res["views"][view]["variants"][v] = {"summary": summarize(per), "layers": per}
            s = res["views"][view]["variants"][v]["summary"]
            print(f"{view:22s} {v:10s} matched {s['all_matched']} bijection {s['all_bijection']} identity {s['all_identity']} unit {s['all_unit']} pow2 {s['all_pow2']} "
                  f"min cos {s['min_cos']:.12f} max resid {s['max_resid']:.2e}", flush=True)
    dump(res, out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], "--hf" in sys.argv[3:])
