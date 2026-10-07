# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Batch 0023, language models whose architecture differs from the Llama of batches 0015 to 0022 in the part that the derivation says fixes the group: a normalization of the queries and keys (Qwen3),
# LayerNorm with learned positions (GPT-2), LayerNorm with a parallel residual and a rotary embedding on a quarter of each head (Pythia, GPT-NeoX). The predictions were derived by hand
# from the rules of the derivation (docs/batches/0023.md); none is in the Lean development or the Datalog program. float64, Transformers on the CPU. Measurements only.
#   qwen3     the Llama-family kinematics of systems23.py (translation, the phase element, dilation, rotation) and four transformations of the queries and keys:
#             T1 the scalar of a key/value group and plane on the keys and its inverse on the queries of the group's heads; T2 a rotation of a plane applied to the keys and the queries (random angles);
#             T3 a positive scalar on the query projection of one head; T4 a real scalar per plane moved from the gain of the query norm to the gain of the key norm
#   gpt2      L1 the all-ones vector added to every output of the attention's and the MLP's output projections (and to their biases); L2 the dual shift of the readers of the two LayerNorms of each block;
#             L3 a constant added to every coordinate of the embedding rows of five tokens (the head is tied to the embedding); L4 the positions shifted by c = 1 and 16 (learned absolute positions)
#   pythia    translation of positions (partial rotary embedding, parallel residual); the writer shift of gpt2 in the layout of GPT-NeoX; dilation; the stream rotated by an orthogonal Q that fixes the
#             all-ones vector after the LayerNorms are folded, and by a general orthogonal Q (which does not fix it)
# Usage: outside23.py MODEL_TYPE REPO REV OUT.json [--random-small]       MODEL_TYPE: qwen3, gpt2, gpt_neox
import gc, json, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hfx23 as X  # noqa: E402
import systems23 as S  # noqa: E402


def spread(vals):
    return {"largest": max(vals), "smallest": min(vals)}


def logits_rho(m2, m, seqs, model_type, base, positions_fn=None):
    vals = []
    for ids, b in zip(seqs, base):
        pos = positions_fn(ids) if positions_fn else None
        vals.append(X.rho(X.run(m2, ids, positions=pos, model_type=model_type)["logits"], b["logits"]))
    return spread(vals)


def qwen3_tests(m, seqs, base, res, seed=5):
    d, nh, nkv, hd, dff = X.dims(m)
    h2 = hd // 2
    rs = np.random.RandomState(seed)
    s1 = np.exp(0.5 * rs.standard_normal((nkv, h2)))
    phi = rs.uniform(0, 2 * np.pi, (nkv, h2))
    c3 = np.exp(0.5 * rs.standard_normal(nh))
    s4 = np.exp(0.5 * rs.standard_normal(h2))
    res["qk_norm_transformations"] = {}
    for name, fn in (("T1_scalar_pair_per_group_and_plane", lambda: X.scalar_pair_gauge(m, s1)), ("T2_rotation_of_a_plane_on_keys_and_queries", lambda: X.phase_gauge(m, phi)),
                     ("T3_positive_scalar_on_one_query_head", lambda: X.head_scalar(m, c3)), ("T4_scalar_moved_from_the_query_gain_to_the_key_gain", lambda: X.qk_gain_move(m, s4))):
        m2 = fn()
        res["qk_norm_transformations"][name] = logits_rho(m2, m, seqs, "qwen3", base)
        print(name, res["qk_norm_transformations"][name], flush=True)
        del m2
        gc.collect()
    # the translation element is a rotation of the planes by c theta: with the norm of the queries and keys it is not a symmetry
    theta = X.rotary_frequencies(m)
    m2 = X.phase_gauge(m, [[16 * theta[p] for p in range(h2)] for _ in range(nkv)])
    res["qk_norm_transformations"]["T2_with_the_angles_of_a_shift_of_16_positions"] = logits_rho(m2, m, seqs, "qwen3", base)
    del m2
    gc.collect()


def gpt2_tests(m, tok, seqs, base, res, seed=6, random_small=False):
    nl, ne = m.config.n_layer, m.config.n_embd
    ni = m.config.n_inner or 4 * ne
    rs = np.random.RandomState(seed)
    T = X.DT
    tt = lambda *shape: torch.tensor(0.3 * rs.standard_normal(shape), dtype=T)
    res["layernorm_transformations"] = {}
    m2 = X.gpt2_writer_shift(m, [tt(ne) for _ in range(nl)], [tt(ni) for _ in range(nl)], [float(tt(1)) for _ in range(nl)], [float(tt(1)) for _ in range(nl)])
    res["layernorm_transformations"]["L1_all_ones_added_to_the_outputs_of_the_two_output_projections"] = logits_rho(m2, m, seqs, "gpt2", base)
    del m2
    gc.collect()
    m2 = X.gpt2_reader_shift(m, [tt(3 * ne) for _ in range(nl)], [tt(ni) for _ in range(nl)])
    res["layernorm_transformations"]["L2_dual_shift_of_the_readers_of_the_two_layernorms"] = logits_rho(m2, m, seqs, "gpt2", base)
    del m2
    gc.collect()
    V = m.config.vocab_size
    toks = [int(t) for t in rs.choice(V, 5, replace=False)]
    cs = [float(x) for x in rs.randn(5)]
    m2 = X.gpt2_embedding_shift(m, toks, cs)
    rows = []
    for ids, b in zip(seqs, base):
        cap = []
        h = m.transformer.ln_f.register_forward_hook(lambda mod, i, o: cap.append(o[0].detach().clone()))
        with torch.no_grad():
            m(input_ids=ids, attention_mask=torch.ones_like(ids))
        h.remove()
        s = cap[0].sum(-1).cpu().numpy()                                           # the sum over the coordinates of the output of the final LayerNorm, per position
        z2 = X.run(m2, ids, model_type="gpt2")["logits"]
        delta = z2 - b["logits"]
        mask = np.zeros(V, dtype=bool)
        mask[toks] = True
        predicted = np.outer(s, np.array(cs))
        rows.append({"unshifted_columns_relative_change": float(np.linalg.norm(delta[:, ~mask]) / np.linalg.norm(b["logits"])), "shifted_columns_against_c_times_the_sum": float(np.linalg.norm(delta[:, mask] - predicted) / np.linalg.norm(predicted))})
    del m2
    gc.collect()
    res["layernorm_transformations"]["L3_embedding_rows_of_five_tokens_shifted"] = {"unshifted_columns_largest": max(r["unshifted_columns_relative_change"] for r in rows),
                                                                                    "shifted_columns_against_prediction_largest": max(r["shifted_columns_against_c_times_the_sum"] for r in rows)}
    res["absolute_positions"] = {}
    for c in (1, 16):
        res["absolute_positions"][str(c)] = logits_rho(m, m, seqs, "gpt2", base, positions_fn=lambda ids, c=c: (torch.arange(ids.shape[1]) + c)[None])


def neox_tests(m, seqs, base, res, seed=7):
    nl, d = m.config.num_hidden_layers, m.config.hidden_size
    rs = np.random.RandomState(seed)
    T = X.DT
    tt = lambda *shape: torch.tensor(0.3 * rs.standard_normal(shape), dtype=T)
    res["translation"] = {}
    for c in (1, 16, 128):
        res["translation"][str(c)] = logits_rho(m, m, seqs, "gpt_neox", base, positions_fn=lambda ids, c=c: (torch.arange(ids.shape[1]) + c)[None])
    res["dilation"] = {}
    for a in (0.5, 2.0):
        vals = [X.rho(X.run(m, ids, pos_scale=a, model_type="gpt_neox")["logits"], b["logits"]) for ids, b in zip(seqs, base)]
        res["dilation"][str(a)] = spread(vals)
    V = m.config.vocab_size
    m2 = X.neox_writer_shift(m, [tt(d) for _ in range(nl)], [tt(4 * d) for _ in range(nl)], [float(tt(1)) for _ in range(nl)], [float(tt(1)) for _ in range(nl)], tt(V))
    res["layernorm_transformations"] = {"writer_shift": logits_rho(m2, m, seqs, "gpt_neox", base)}
    del m2
    gc.collect()
    for name, Q in (("rotation_fixing_the_all_ones_vector", X.orthogonal_fixing_ones(d, 11)), ("general_orthogonal_rotation", X.random_orthogonal(d, 12))):
        m2, offset = X.neox_fold_rotate(m, Q)
        vals = []
        for ids, b in zip(seqs, base):
            z = X.run(m2, ids, model_type="gpt_neox")["logits"] + offset.cpu().numpy()[None, :]
            vals.append(X.rho(z, b["logits"]))
        res["layernorm_transformations"][name] = spread(vals)
        del m2
        gc.collect()


def main(model_type, repo, rev, out, random_small):
    t0 = time.time()
    m, tok, cfg, restore = X.load_model(model_type, repo, rev, random_small=random_small)
    print(f"loaded {repo or 'a random small ' + model_type} in {time.time() - t0:.0f}s", flush=True)
    try:
        res = {"model_type": model_type, "torch": torch.__version__, "layers": m.config.num_hidden_layers, "membership": X.membership(m, tok) if tok is not None else None}
        seqs = X.encode(tok, model_type, random_small, cfg["vocab_size"])
        base = [X.run(m, ids, model_type=model_type) for ids in seqs]
        res["float64_islands_patched"] = X.patch_used(model_type)
        if model_type == "qwen3":
            qwen3_tests(m, seqs, base, res)
            S.experiments(m, tok, cfg, model_type, random_small, out)
            merged = json.load(open(out))
            merged.update(res)
            json.dump(merged, open(out, "w"), indent=1)
        else:
            if model_type == "gpt2":
                gpt2_tests(m, tok, seqs, base, res, random_small=random_small)
            else:
                neox_tests(m, seqs, base, res)
            json.dump(res, open(out, "w"), indent=1)
        print(json.dumps(res, indent=1)[:3000], flush=True)
    finally:
        restore()


if __name__ == "__main__":
    main(sys.argv[1], None if sys.argv[2] == "-" else sys.argv[2], None if sys.argv[3] == "-" else sys.argv[3], sys.argv[4], "--random-small" in sys.argv)
