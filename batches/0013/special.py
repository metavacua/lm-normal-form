# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Is a language model's weights in the special case of permutation equivalence, and if so the canonical form for it. The special
# case (batch 0013): every item of every permutable index space can be told from every other by a signature, so that sorting by the
# signature is a canonical form and nothing is searched. The ladder checked, in order:
#   L0  one matrix at a time, rows and columns permuted independently (A -> PAQ): the items are the rows and the columns; the
#       signature of a row is its sorted entries. "Sort the rows, then the columns" is not a canonical form; ordering the rows by
#       the signature and then sorting the columns as vectors is, when the signatures differ
#   L1  the network's own group: the feed-forward units, the heads and the hidden coordinates are index spaces shared by several
#       matrices. (a) a signature that does not look at the order of any other index space (a unit's triples over the coordinates,
#       made invariant to its sign and power of two); (b) the vocabulary as a fixed index: the hidden coordinates ordered by their
#       embedding columns, and then every unit and head as a vector in that order
#   L2  where (a) ties, rounds of colour refinement on the unit-coordinate graph
# Usage: special.py check REPO OUT.json      L0 and L1 for the model REPO on the Hub
#        special.py canon REPO OUT.json      the canonical form against random elements of the group, and against non-symmetries
#        special.py naive REPO OUT.json      "sort rows then columns" and the signature procedure on real matrices
#        special.py function REPO OUT.json   the group's action run as a language model
#        special.py selftest                 the above on a random toy Llama, in float64
import json, os, sys, time, types
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "0011"))
import identity
import forms as F
import symmetry as S
import canon as C

PARTS = ("residual", "units", "heads", "vo", "qk")
M1, M2 = np.uint64(0x9E3779B97F4A7C15), np.uint64(0xBF58476D1CE4E5B9)


def weights(repo):
    """The weights of a Hub repository at the revision in HF_REVISION (the head of main if it is unset), float16 and bfloat16 widened."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    rev = os.environ.get("HF_REVISION") or None
    cfg = json.load(open(hf_hub_download(repo, "config.json", revision=rev)))
    P = {}
    with safe_open(hf_hub_download(repo, "model.safetensors", revision=rev), framework="np") as f:
        for k in f.keys():
            P[k] = f.get_tensor(k).astype(np.float32)        # bfloat16 and float16 widened: exact
    return P, F.Dims(types.SimpleNamespace(**cfg)), cfg


def n_distinct(a):
    """The number of distinct rows of a 2-D array."""
    v = np.ascontiguousarray(a)
    return len(np.unique(v.view(np.dtype((np.void, v.dtype.itemsize * v.shape[1])))))


# ---- L0: one matrix --------------------------------------------------------------------------------------------------------

def l0(W):
    """For the rows and for the columns: how many, how many distinct as vectors, how many distinct by signature (the sorted entries,
    -0.0 read as 0.0: the order of two zeros that compare equal is not fixed by the sort)."""
    out = {}
    for name, M in (("rows", W), ("columns", np.ascontiguousarray(W.T))):
        out[name] = (len(M), n_distinct(M), n_distinct(np.sort(M + M.dtype.type(0), axis=1)))
    return out


def zeros(P):
    """What the weights have of zeros: the entries that are 0.0 and the entries that are -0.0, over all tensors."""
    z = sum(int(np.count_nonzero(v == 0)) for v in P.values())
    n = sum(int(np.count_nonzero((v == 0) & np.signbit(v))) for v in P.values())
    return {"zeros": z, "negative_zeros": n, "entries": sum(int(v.size) for v in P.values())}


def l0_line(r):
    """One short string per class: for rows and for columns, the matrices that have a signature tie between unlike items, and the unlike items lost."""
    return {cls: " | ".join(f"{ax}: {x[ax]['matrices_where_signature_ties_unlike_items']} of {x['matrices']} matrices, {x[ax]['unlike_items_lost_to_signature_ties']} lost"
                            for ax in ("rows", "columns")) for cls, x in r.items()}


def l0_model(P, d):
    classes = {"q": F.Q, "k": F.K, "v": F.V, "o": F.O, "gate": F.G, "up": F.U, "down": F.D}
    res = {}
    for cls, name in list(classes.items()) + [("embed", None)]:
        keys = [F.EMBED] if name is None else [F.key(i, name) for i in range(d.L)]
        rows = [l0(P[k]) for k in keys]
        r = {"matrices": len(keys)}
        for axis in ("rows", "columns"):
            n = [x[axis][0] for x in rows]
            v = [x[axis][1] for x in rows]
            s = [x[axis][2] for x in rows]
            r[axis] = {"items_per_matrix": n[0], "matrices_with_identical_items": sum(1 for a, b in zip(n, v) if b < a),
                       "matrices_where_signature_ties_unlike_items": sum(1 for a, b in zip(v, s) if b < a),
                       "unlike_items_lost_to_signature_ties": int(sum(a - b for a, b in zip(v, s))), "worst_matrix": int(max(a - b for a, b in zip(v, s)))}
        res[cls] = r
    return res


# ---- L1 -------------------------------------------------------------------------------------------------------------------

def mix(x):
    x = x * M1
    x ^= x >> np.uint64(32)
    x = x * M2
    x ^= x >> np.uint64(29)
    return x


def triples(g, u, w):
    """The invariants of a unit that do not move with its sign or its power of two: the gate entry, the product of the up entry and
    the down entry, and the up entry's magnitude over the largest. One row per unit, one column per hidden coordinate. A zero is
    +0.0 here: a product of a zero and a negative number is -0.0, which a sort does not tell from 0.0."""
    m = np.abs(u).max(axis=1, keepdims=True)
    m[m == 0] = 1
    z = g.dtype.type(0)
    return g + z, u * w.T + z, np.abs(u) / m


def joint_signature(g, p, a):
    """Per unit, a digest of its sorted triples over the coordinates: nothing in it depends on the order of the coordinates."""
    out = []
    for j in range(len(g)):
        o = np.lexsort((a[j], p[j], g[j]))
        out.append(C.dig(g[j][o], p[j][o], a[j][o]))
    return out


def bits(x):
    """The bit patterns of a float32 or float64 array as uint64."""
    x = np.ascontiguousarray(x)
    return x.view(np.uint32 if x.dtype.itemsize == 4 else np.uint64).astype(np.uint64)


def refine(g, p, a, rounds=8):
    """Colour refinement on the graph of units and coordinates, edges labelled by the triple: the number of distinct unit colours
    after each round."""
    E = mix(mix(bits(g)) ^ (bits(p) * M1))
    E = mix(E ^ mix(bits(a)))
    uc, cc, hist = np.zeros(len(g), np.uint64), np.zeros(g.shape[1], np.uint64), []
    for _ in range(rounds):
        uc2 = mix(np.add.reduce(mix(E ^ cc[None, :]), axis=1))
        cc2 = mix(np.add.reduce(mix(E ^ uc[:, None]), axis=0))
        uc, cc = uc2, cc2
        hist.append(int(len(np.unique(uc))))
        if hist[-1] == len(g) or (len(hist) > 1 and hist[-1] == hist[-2]):
            break
    return hist


def l1(P, d):
    res = {"units": [], "heads": []}
    Cn = C.canonical(P, d)
    E = np.ascontiguousarray(Cn[F.EMBED].T)
    res["coordinates"] = {"items": d.H, "distinct_embedding_columns_up_to_sign": n_distinct(E), "distinct_norm_values": len(np.unique(P["model.norm.weight"]))}
    for i in range(d.L):
        g, u, w = Cn[F.key(i, F.G)], Cn[F.key(i, F.U)], Cn[F.key(i, F.D)]
        gp, p, a = triples(g, u, w)
        joint = len(set(joint_signature(gp, p, a)))
        sep = len({(C.dig(np.sort(gp[j])), C.dig(np.sort(p[j])), C.dig(np.sort(a[j]))) for j in range(d.F)})
        anchored = n_distinct(np.concatenate([g, u, w.T], axis=1))
        row = {"layer": i, "units": d.F, "distinct_by_separate_signatures": sep, "distinct_by_joint_signature": joint, "distinct_anchored": anchored}
        if joint < d.F:
            row["refinement_unit_colours_by_round"] = refine(gp, p, a)
        res["units"].append(row)
        qb = Cn[F.key(i, F.Q)].reshape(d.nh, -1)
        kv = np.concatenate([Cn[F.key(i, F.K)].reshape(d.nkv, -1), Cn[F.key(i, F.V)].reshape(d.nkv, -1)], axis=1)
        ob = np.ascontiguousarray(Cn[F.key(i, F.O)].reshape(d.H, d.nh, d.hd).transpose(1, 0, 2)).reshape(d.nh, -1)
        qo = np.concatenate([qb, ob], axis=1)
        res["heads"].append({"layer": i, "heads": d.nh, "distinct_anchored": n_distinct(qo), "distinct_index_free": n_distinct(np.sort(np.abs(qo), axis=1)),
                             "groups": d.nkv, "groups_distinct_anchored": n_distinct(kv)})
    return res


def summary_l1(r, d):
    u = r["units"]
    return {"coordinates_anchored": r["coordinates"]["distinct_embedding_columns_up_to_sign"] == d.H,
            "layers_units_discrete_by_separate_signatures": sum(1 for x in u if x["distinct_by_separate_signatures"] == d.F),
            "layers_units_discrete_by_joint_signature": sum(1 for x in u if x["distinct_by_joint_signature"] == d.F),
            "layers_units_discrete_anchored": sum(1 for x in u if x["distinct_anchored"] == d.F),
            "layers": d.L, "units_not_told_apart_by_joint_signature": sum(d.F - x["distinct_by_joint_signature"] for x in u),
            "heads_discrete_anchored_layers": sum(1 for x in r["heads"] if x["distinct_anchored"] == d.nh and x["groups_distinct_anchored"] == d.nkv),
            "heads_discrete_index_free_layers": sum(1 for x in r["heads"] if x["distinct_index_free"] == d.nh)}


# ---- the canonical form against the group -----------------------------------------------------------------------------------

def digests(P):
    return {k: identity.digest(v)[1] for k, v in P.items()}


def compare(base, other):
    return sum(1 for k in base if other[k] == base[k])


def roots(P):
    """Batch 0011's roots over these tensors, labelled as in a safetensors file."""
    ents = [identity.entry(v, {"safetensors": k}) for k, v in P.items()]
    return {"bag_root": identity.core.bag_root(ents), "labelled_root": identity.core.labelled_root(ents, "safetensors")}


def canon_cell(P, d, tied, n_all, n_part, seed=13):
    cbase = C.canonical(P, d)
    base = digests(cbase)
    raw = digests(P)
    res = {"tensors": len(base), "all_parts": [], "single_parts": {}, "negative_controls": {}, "raw_roots": roots(P), "canonical_roots": roots(cbase),
           "canonical_form_idempotent": all(np.array_equal(C.canonical(cbase, d)[k], cbase[k]) for k in cbase)}
    del cbase
    for t in range(n_all):
        g = S.act(P, d, np.random.default_rng(seed + t), tied)
        cg = C.canonical(g, d)
        c = digests(cg)
        gd = digests(g)
        row = {"canonical_tensors_equal": compare(base, c), "raw_tensors_changed": sum(1 for k in raw if gd[k] != raw[k])}
        if t == 0:
            row["raw_roots_of_the_image"], row["canonical_roots_of_the_image"] = roots(g), roots(cg)
        res["all_parts"].append(row)
        del cg
    for part in PARTS + ("units-signed",):
        rows = []
        for t in range(n_part):
            g = S.act(P, d, np.random.default_rng(seed + 100 + t), tied, (part,))
            c = digests(C.canonical(g, d))
            gd = digests(g)
            rows.append({"canonical_tensors_equal": compare(base, c), "raw_tensors_changed": sum(1 for k in raw if gd[k] != raw[k])})
        res["single_parts"][part] = rows
    L = min(3, d.L - 1)
    neg = {}
    g = dict(P); k = F.key(L, F.G); g[k] = P[k].copy(); g[k][5] *= -1
    neg["one gate row negated"] = g
    g = dict(P); a, b = F.key(L, F.G), F.key(L, F.U); g[a], g[b] = P[a].copy(), P[b].copy(); g[a][5], g[b][5] = P[b][5].copy(), P[a][5].copy()
    neg["one unit's gate and up rows exchanged"] = g
    g = dict(P); k = F.key(min(5, d.L - 1), F.Q); g[k] = P[k].copy(); g[k][0, 0] = np.nextafter(g[k][0, 0], np.float32(np.inf))
    neg["one float32 ulp in one query weight"] = g
    g = dict(P); E = P[F.EMBED].copy(); E[:, [0, 1]] = E[:, [1, 0]]; g[F.EMBED] = E
    neg["two embedding columns exchanged, nothing else"] = g
    for name, g in neg.items():
        c = digests(C.canonical(g, d))
        diff = sorted(k for k in base if c[k] != base[k])
        res["negative_controls"][name] = {"canonical_tensors_equal": compare(base, c), "differing_count": len(diff),
                                          "differing": diff if len(diff) <= 12 else diff[:6] + ["..."] + diff[-3:],
                                          "layers_touched": sorted({int(k.split(".layers.")[1].split(".")[0]) for k in diff if ".layers." in k})}
    return res


def canon_line(r):
    ap = r["all_parts"]
    cs = lambda xs: " ".join(str(x) for x in xs)
    return {"tensors": r["tensors"], "whole group, canonical tensors equal": cs(x["canonical_tensors_equal"] for x in ap),
            "whole group, raw tensors changed": cs(x["raw_tensors_changed"] for x in ap),
            "each part, canonical tensors equal": {k: cs(x["canonical_tensors_equal"] for x in v) for k, v in r["single_parts"].items()},
            "each part, raw tensors changed": {k: cs(x["raw_tensors_changed"] for x in v) for k, v in r["single_parts"].items()},
            "non-symmetries, canonical tensors equal": {k: v["canonical_tensors_equal"] for k, v in r["negative_controls"].items()},
            "non-symmetries, tensors that differ": {k: [v["differing"], "layers " + cs(v["layers_touched"])] for k, v in r["negative_controls"].items()},
            "canonical roots of model and of an image equal": r["canonical_roots"] == ap[0]["canonical_roots_of_the_image"],
            "raw roots of model and of an image equal": r["raw_roots"] == ap[0]["raw_roots_of_the_image"], "raw_roots": r["raw_roots"],
            "canonical_roots": r["canonical_roots"], "idempotent": r["canonical_form_idempotent"]}


# ---- "sort rows then columns" against the signature procedure, on real matrices -----------------------------------------------

def sort_rows(A):
    return A[np.lexsort(A.T[::-1])]


def naive(A):
    return sort_rows(sort_rows(A).T).T


def by_signature(A):
    return sort_rows(A[np.lexsort(np.sort(A, axis=1).T[::-1])].T).T


def tie_matrix(seed=0):
    """A 12 x 20 matrix of 0, 1, 2 whose rows 0 and 1 are the same entries in another order: the signatures of those two rows tie, the rows differ."""
    r = np.random.default_rng(seed)
    m = r.integers(0, 3, (12, 20)).astype(np.float32)
    m[1] = m[0][r.permutation(20)]
    return m


def pair_trials(kind, n=1500, seed=11):
    """n random matrices of 12 x 20; for each, two random images (P A Q): how often each procedure gives two different matrices. A canonical form
    gives the same one."""
    r = np.random.default_rng(seed)
    fail = {"sort rows then columns": 0, "signature, then columns": 0}
    for _ in range(n):
        A = r.random((12, 20)) if kind == "real" else r.integers(0, 3, (12, 20)).astype(np.float64)
        B1, B2 = (A[r.permutation(12)][:, r.permutation(20)] for _ in range(2))
        fail["sort rows then columns"] += not np.array_equal(naive(B1), naive(B2))
        fail["signature, then columns"] += not np.array_equal(by_signature(B1), by_signature(B2))
    return fail


def naive_cell(P, d, trials=12, seed=13):
    rng = np.random.default_rng(seed)
    out = {}
    for name in (F.key(0, F.Q), F.key(0, F.K), F.key(0, F.G)):
        W = P[name]
        imgs = [W[rng.permutation(W.shape[0])][:, rng.permutation(W.shape[1])] for _ in range(trials)]
        out[name] = {"shape": list(W.shape), "images": trials,
                     "distinct_outputs_sort_rows_then_columns": len({naive(m).tobytes() for m in imgs}),
                     "distinct_outputs_signature_then_columns": len({by_signature(m).tobytes() for m in imgs})}
    r = np.random.default_rng(0)
    small = tie_matrix()
    imgs = [small[r.permutation(12)][:, r.permutation(20)] for _ in range(40)]
    out["12x20 matrix of 0, 1, 2, two rows alike in signature"] = {"images": 40, "distinct_outputs_signature_then_columns": len({by_signature(m).tobytes() for m in imgs}),
                                                       "distinct_outputs_sort_rows_then_columns": len({naive(m).tobytes() for m in imgs})}
    return out


# ---- the function --------------------------------------------------------------------------------------------------------------

def function_cell(repo, out_dir):
    from harness import load, sequences, run, evaluate
    model, tok = load(repo)
    d = F.Dims(model.config)
    P = {k: v.astype(np.float32) for k, v in F.params(model).items()}
    tied = bool(model.config.tie_word_embeddings)
    chunks, _ = sequences(tok)
    ref = run(model, chunks)
    res = {}
    for name, parts in (("identity", ()), ("units-signed", ("units-signed",)), ("vo", ("vo",)), ("qk", ("qk",)), ("units", ("units",)), ("heads", ("heads",)),
                        ("residual", ("residual",)), ("all", PARTS)):
        g = S.act(P, d, np.random.default_rng(13), tied, parts) if parts else dict(P)
        res[name] = evaluate(model, {k: v.astype(np.float64) for k, v in g.items()}, "fp32", chunks, ref, 3)
        print(f"function {name:13s} max|d| {res[name]['max_abs']:.3g}  KL {res[name]['mean_kl']:.3g}  top1 {res[name]['top1']:.4f}", flush=True)
    return res


# ---- the self-test ---------------------------------------------------------------------------------------------------------------

def selftest():
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    from transformers.models.llama import modeling_llama
    modeling_llama.LlamaRMSNorm.forward = lambda self, h: self.weight * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
    torch.manual_seed(0)
    bad = 0

    def check(what, ok):
        nonlocal bad
        print(("ok     " if ok else "FAILED ") + what)
        bad += 0 if ok else 1
    for tied in (True, False):
        cfg = LlamaConfig(hidden_size=72, intermediate_size=96, num_hidden_layers=2, num_attention_heads=6, num_key_value_heads=3, vocab_size=120,
                          max_position_embeddings=64, tie_word_embeddings=tied)
        model = LlamaForCausalLM(cfg).double().eval()
        d = F.Dims(model.config)
        P = F.params(model)
        ids = torch.randint(0, 120, (3, 12))

        def logits(R):
            model.load_state_dict({k: torch.from_numpy(v) for k, v in R.items()}, strict=False)
            with torch.no_grad():
                return model(input_ids=ids, use_cache=False).logits
        base = logits(P)
        tag = "tied head" if tied else "untied head"
        for part in PARTS + ("units-signed",):
            g = S.act(P, d, np.random.default_rng(3), tied, (part,))
            diff = float((logits(g) - base).abs().max())
            check(f"[{tag}] the action of '{part}' is a symmetry of the function: max |difference of logits| {diff:.3g}", diff < 1e-12 and not all(np.array_equal(g[k], P[k]) for k in P))
        g = S.act(P, d, np.random.default_rng(4), tied)
        diff = float((logits(g) - base).abs().max())
        check(f"[{tag}] a random element of the whole group: max |difference of logits| {diff:.3g}", diff < 1e-12)
        for dt in (np.float64, np.float32):
            Q = {k: v.astype(dt) for k, v in P.items()}
            c0 = C.canonical(Q, d)
            ok = all(np.array_equal(C.canonical(S.act(Q, d, np.random.default_rng(s), tied), d)[k], c0[k]) for s in range(3) for k in c0)
            check(f"[{tag}, {dt.__name__}] the canonical form of three random images of the model is the canonical form of the model, bit for bit", ok)
            ok = all(np.array_equal(C.canonical(c0, d)[k], c0[k]) for k in c0)
            check(f"[{tag}, {dt.__name__}] the canonical form of a canonical form is itself", ok)
        c0 = C.canonical(P, d)
        diff = float((logits(c0) - base).abs().max())
        check(f"[{tag}] the canonical form computes the same function: max |difference of logits| {diff:.3g}", diff < 1e-12)
        gg = S.act(P, d, np.random.default_rng(7), tied)
        check(f"[{tag}] the roots over the raw tensors of an image differ from the model's, and over the canonical tensors they are equal",
              roots(P)["labelled_root"] != roots(gg)["labelled_root"] and roots(P)["bag_root"] != roots(gg)["bag_root"] and roots(c0) == roots(C.canonical(gg, d)))
        g = dict(P); g[F.key(1, F.G)] = P[F.key(1, F.G)].copy(); g[F.key(1, F.G)][2] *= -1
        c1 = C.canonical(g, d)
        check(f"[{tag}] a gate row negated is not a symmetry, and the canonical form shows it", any(not np.array_equal(c0[k], c1[k]) for k in c0))
        r = l1(P, d)
        s = summary_l1(r, d)
        check(f"[{tag}] the special case is found on a random toy: coordinates anchored, every unit told apart ({s})", s["coordinates_anchored"] and s["layers_units_discrete_by_joint_signature"] == d.L)
    # colour refinement: a unit that is another unit with its coordinates permuted ties by the joint signature and is told apart by the refinement; a
    # duplicate is never told apart
    r0 = np.random.default_rng(5)
    for dt in (np.float32, np.float64):
        g, u, w = (r0.standard_normal(sh).astype(dt) for sh in ((20, 16), (20, 16), (16, 20)))
        sigma = r0.permutation(16)
        g[1], u[1], w[:, 1] = g[0][sigma], u[0][sigma], w[:, 0][sigma]
        gp, p, a = triples(g, u, w)
        sig = joint_signature(gp, p, a)
        check(f"[{dt.__name__}] a unit that is another with its coordinates permuted ties in the joint signature ({len(set(sig))} of 20)", len(set(sig)) == 19 and sig[0] == sig[1])
        h = refine(gp, p, a)
        check(f"[{dt.__name__}] the refinement tells them apart: unit colours by round {h}", h[-1] == 20 and h[0] == 19)
        g[2], u[2], w[:, 2] = g[3], u[3], w[:, 3]
        gp, p, a = triples(g, u, w)
        h = refine(gp, p, a)
        check(f"[{dt.__name__}] and a duplicate of a unit stays tied to it: unit colours by round {h}", h[-1] == 19)
    r = np.random.default_rng(0)
    A = np.array([[1, 5], [2, 3], [2, 4]], np.float32)
    check("sort rows then columns of [[1,5],[2,3],[2,4]] and of the same with its columns exchanged are not equal", not np.array_equal(naive(A), naive(A[:, ::-1])))
    check("the signature procedure gives the same matrix for both", np.array_equal(by_signature(A), by_signature(A[:, ::-1])))
    small = tie_matrix()
    imgs = [small[r.permutation(12)][:, r.permutation(20)] for _ in range(40)]
    check("on a matrix with two rows that tie in signature the procedure is not canonical (the special case ends here)", len({by_signature(m).tobytes() for m in imgs}) > 1)
    real, small = pair_trials("real"), pair_trials("small")
    print(f"1500 random 12 x 20 matrices of distinct real entries, two random images of each: different results from {real}")
    print(f"1500 random 12 x 20 matrices of the entries 0, 1, 2, two random images of each: different results from {small}")
    check("on matrices of distinct real entries the signature procedure is canonical and sorting rows then columns is not (fails in 90% or more)",
          real["signature, then columns"] == 0 and real["sort rows then columns"] >= 1350)
    check("on matrices of 0, 1, 2 the signature procedure fails, because signatures tie", small["signature, then columns"] > 0)
    print("selftest:", "all ok" if not bad else f"{bad} failed")
    return not bad


def main(argv):
    if argv[1] == "selftest":
        return 0 if selftest() else 1
    cmd, repo, out = argv[1:4]
    t = time.time()
    if cmd == "function":
        res = {"model": repo, "variants": function_cell(repo, os.path.dirname(out))}
    else:
        P, d, cfg = weights(repo)
        tied = "lm_head.weight" not in P
        res = {"model": repo, "dims": vars(d), "tied": tied, "tensors": len(P), "config_torch_dtype": cfg.get("torch_dtype")}
        if cmd == "check":
            res["zeros"] = zeros(P)
            res["l0"] = l0_model(P, d)
            r = l1(P, d)
            res["l1"], res["l1_summary"] = r, summary_l1(r, d)
        elif cmd == "canon":
            big = d.H * d.F * d.L > 3e7
            res["canon"] = canon_cell(P, d, tied, 3 if big else 8, 1 if big else 2)
        elif cmd == "naive":
            res["naive"] = naive_cell(P, d)
    res["seconds"] = time.time() - t
    json.dump(res, open(out, "w"), indent=1, sort_keys=True)
    shown = {k: v for k, v in res.items() if k not in ("l1", "canon", "l0")}
    if "canon" in res:
        shown["canon"] = canon_line(res["canon"])
    if "l0" in res:
        shown["l0"] = l0_line(res["l0"])
    print(json.dumps(shown, indent=1, sort_keys=True)[:6000])
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
