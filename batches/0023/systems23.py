# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Batch 0023, the kinematics of the special theory measured on real checkpoints of the Llama family (float64, Hugging Face Transformers on the CPU, the float32 islands of the norm and of the rotary
# tables replaced by float64). Measurements only; the verdicts are in grade23.py, committed before the run. For each checkpoint, on six sentences (cut to 24 tokens):
#   translation  the logits with the positions shifted by c = 1, 16, 128 against the logits with the positions 0, 1, 2, ...  (the rotary embedding makes the scores depend on t - s only)
#   phase        the weight-level phase element: the planes of the queries and keys rotated by c * theta_p (theta_p the frequency of plane p), c = 16, at the original positions: its logits against the
#                original's; its cache of keys (after the rotary embedding) against the original's at the shifted positions; and how far the shift moves the cache of the original (the control)
#   dilation     the logits with the angles of the rotary embedding multiplied by a = 0.5 and 2 (positions t -> a t) against the original's: not a symmetry, the frequencies are absolute
#   rotation     the norm weights folded into the readers and the head untied, the stream rotated by a random orthogonal Q (x -> Q x): the logits; the stream after the embedding and after each layer
#                against the original's stream times Q^T (the law of transformation of the stream); the Gram matrix of the stream of each layer against the original's; how far the rotation moves the stream
#                (the embedding); and as a control the rotation without the fold, which is not a symmetry
# For each quantity the largest (or the smallest) over the six sentences is kept: a claim "invariant" is graded on the largest, a claim "moved" on the smallest.
# Usage: systems23.py MODEL_TYPE REPO REV OUT.json [--random-small]       (--random-small: a small random model of the architecture, no download: the smoke run of this code)
import gc, json, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hfx23 as X  # noqa: E402


def experiments(m, tok, cfg, model_type, random_small, out):
    d, nh, nkv, hd, dff = X.dims(m)
    h2 = hd // 2
    seqs = X.encode(tok, model_type, random_small, cfg["vocab_size"])
    T = [s.shape[1] for s in seqs]
    theta = X.rotary_frequencies(m)
    res = {"model_type": model_type, "hidden": d, "heads": nh, "kv_heads": nkv, "head_dim": hd, "layers": m.config.num_hidden_layers, "tied_head": X.is_tied(m), "sequence_lengths": T,
           "torch": torch.__version__, "qk_norm": hasattr(m.model.layers[0].self_attn, "q_norm")}
    res["membership"] = X.membership(m, tok) if tok is not None else None
    base = [X.run(m, ids, cache=True, streams=True, model_type=model_type) for ids in seqs]
    res["float64_islands_patched"] = X.patch_used(model_type)
    again = [X.run(m, ids, model_type=model_type)["logits"] for ids in seqs]
    res["determinism_largest_relative_difference"] = max(X.rho(a, b["logits"]) for a, b in zip(again, base))
    # translation
    res["translation"] = {}
    for c in (1, 16, 128):
        vals = [X.rho(X.run(m, ids, positions=(torch.arange(ids.shape[1]) + c)[None], model_type=model_type)["logits"], b["logits"]) for ids, b in zip(seqs, base)]
        res["translation"][str(c)] = {"largest": max(vals), "smallest": min(vals)}
    # the weight-level phase element equal to a shift of 16 positions
    c = 16
    shifted = [X.run(m, ids, positions=(torch.arange(ids.shape[1]) + c)[None], cache=True, model_type=model_type) for ids in seqs]
    m2 = X.phase_gauge(m, [[c * theta[p] for p in range(h2)] for _ in range(nkv)])
    ph = [X.run(m2, ids, cache=True, model_type=model_type) for ids in seqs]
    del m2
    gc.collect()
    res["phase"] = {"shift": c,
                    "logits": {"largest": max(X.rho(p["logits"], b["logits"]) for p, b in zip(ph, base)), "smallest": min(X.rho(p["logits"], b["logits"]) for p, b in zip(ph, base))},
                    "keys_of_the_gauged_model_against_the_original_at_the_shifted_positions": {"largest": max(X.rho(p["keys"][i], s["keys"][i]) for p, s in zip(ph, shifted) for i in range(len(p["keys"]))),
                                                                                              "smallest": min(X.rho(p["keys"][i], s["keys"][i]) for p, s in zip(ph, shifted) for i in range(len(p["keys"])))},
                    "keys_moved_by_the_shift_control": {"smallest_over_sentences_of_the_mean_over_layers": min(float(np.mean([X.rho(s["keys"][i], b["keys"][i]) for i in range(len(b["keys"]))])) for s, b in zip(shifted, base))}}
    del shifted, ph
    gc.collect()
    # dilation
    res["dilation"] = {}
    for a in (0.5, 2.0):
        vals = [X.rho(X.run(m, ids, pos_scale=a, model_type=model_type)["logits"], b["logits"]) for ids, b in zip(seqs, base)]
        res["dilation"][str(a)] = {"largest": max(vals), "smallest": min(vals)}
    # rotation of the stream
    mf = X.fold_untie(m)
    Q = X.random_orthogonal(d, 77)
    X.rotate(mf, Q, inplace=True)
    mr = mf
    rot = [X.run(mr, ids, streams=True, model_type=model_type) for ids in seqs]
    del mr
    gc.collect()
    Qn = Q.numpy()
    nl = len(base[0]["streams"])
    res["rotation"] = {"layers_compared": nl,
                       "logits": {"largest": max(X.rho(r["logits"], b["logits"]) for r, b in zip(rot, base)), "smallest": min(X.rho(r["logits"], b["logits"]) for r, b in zip(rot, base))},
                       "stream_equals_the_original_times_Q_transpose": {"largest": max(X.rho(r["streams"][l], b["streams"][l] @ Qn.T) for r, b in zip(rot, base) for l in range(nl))},
                       "gram_matrix_of_every_layer": {"largest": max(X.rho(r["streams"][l] @ r["streams"][l].T, b["streams"][l] @ b["streams"][l].T) for r, b in zip(rot, base) for l in range(nl))},
                       "stream_moved_at_the_embedding": {"smallest": min(X.rho(r["streams"][0], b["streams"][0]) for r, b in zip(rot, base))}}
    del rot
    gc.collect()
    mu = X.rotate(m, Q)
    unf = [X.run(mu, ids, model_type=model_type)["logits"] for ids in seqs]
    del mu
    gc.collect()
    res["rotation_without_the_fold_control"] = {"smallest": min(X.rho(u, b["logits"]) for u, b in zip(unf, base)), "largest": max(X.rho(u, b["logits"]) for u, b in zip(unf, base))}
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k in ("translation", "phase", "dilation", "rotation", "rotation_without_the_fold_control", "determinism_largest_relative_difference")}, indent=1), flush=True)


def main(model_type, repo, rev, out, random_small):
    t0 = time.time()
    m, tok, cfg, restore = X.load_model(model_type, repo, rev, random_small=random_small)
    print(f"loaded {repo or 'a random small ' + model_type} in {time.time() - t0:.0f}s; tied head {X.is_tied(m)}", flush=True)
    try:
        experiments(m, tok, cfg, model_type, random_small, out)
    finally:
        restore()


if __name__ == "__main__":
    main(sys.argv[1], None if sys.argv[2] == "-" else sys.argv[2], None if sys.argv[3] == "-" else sys.argv[3], sys.argv[4], "--random-small" in sys.argv)
