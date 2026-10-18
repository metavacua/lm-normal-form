# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cell 08: the needle by structure (docs/batches/0033.md, amendment 5). Read-only.

usage: needle.py MODEL_DIR NAME OUT_DIR
For each fact (a country and its capital): the structural edges to the target (units whose down column raises the target
among its top 10 tokens, read from the tensors), those also key-matched to the subject (gate row against the subject's
token embedding, top 20 in the layer), and the exact flow of every unit into the margin from the recorded forward pass.
Compares the structural edges with the units that carry the answer, against random units of equal count."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402
from path import capture_all  # noqa: E402

KEY_TOP = 20
VAL_TOP = 10
CARRY = 50


def main():
    model_dir, name, out = sys.argv[1:4]
    os.makedirs(out, exist_ok=True)
    model, tok = core.load(model_dir)
    for p in model.parameters():
        p.requires_grad_(False)
    atoms_ = [(X, C) for X, C in core.atoms() if X in core.T12]
    L = model.config.num_hidden_layers; ffn = model.config.intermediate_size; eps_n = model.config.rms_norm_eps
    E = model.model.embed_tokens.weight.data; g = model.model.norm.weight.data
    # the value side, once: the top tokens of every unit's down column (as down_meta stores them)
    top_tokens = []
    for k in range(L):
        Wd = model.model.layers[k].mlp.down_proj.weight.data                 # (d, ffn)
        sc = E @ Wd                                                          # (V, ffn)
        top_tokens.append(torch.topk(sc, VAL_TOP, dim=0).indices.clone())    # (VAL_TOP, ffn)
        del sc
    results = []
    rng = np.random.default_rng(8)
    for X, C in atoms_:
        t = core.target_ids(tok, [(X, C)])[0]; ids = core.prompt_ids(tok, X); T = ids.shape[1]
        c, z = capture_all(model, ids)
        o = z.clone(); o[t] = float("-inf"); j = int(o.argmax()); margin = float(z[t] - z[j])
        xL = c["xL"][-1]; r = torch.sqrt((xL ** 2).mean() + eps_n); dvec = (E[t] - E[j]) * g / r
        subj_ids = tok(" " + X, add_special_tokens=False)["input_ids"]; subj_pos = [p for p in range(T) if int(ids[0, p]) in subj_ids]
        subj_vec = E[subj_ids].mean(0)
        contrib = torch.stack([(dvec @ model.model.layers[k].mlp.down_proj.weight.data) * c["act"][k][-1] for k in range(L)])   # (L, ffn) signed, toward t over j
        edge = torch.stack([(top_tokens[k] == t).any(0) for k in range(L)])                                                       # (L, ffn)
        keyrank = torch.zeros(L, ffn, dtype=torch.long)
        for k in range(L):
            key = (model.model.layers[k].mlp.gate_proj.weight.data @ subj_vec).abs()
            keyrank[k, torch.argsort(key, descending=True)] = torch.arange(ffn)
        keyed = edge & (keyrank < KEY_TOP)
        flat = contrib.view(-1); N = flat.numel()
        top_idx = torch.topk(flat, CARRY).indices; carry = torch.zeros(N, dtype=torch.bool); carry[top_idx] = True
        n_edge = int(edge.sum()); n_keyed = int(keyed.sum())
        frac_edge_in_carry = float(edge.view(-1)[carry].float().mean()); base_rate = n_edge / N
        # random baseline: units chosen at random with the same count as the carrying set, fraction that are edges
        rnd = [float(edge.view(-1)[torch.tensor(rng.choice(N, CARRY, replace=False))].float().mean()) for _ in range(200)]
        pos_flow = float(flat[flat > 0].sum()); edge_pos = float(contrib[edge][contrib[edge] > 0].sum()) if n_edge else 0.0
        keyed_units = []
        for k, i in torch.nonzero(keyed).tolist():
            keyed_units.append({"layer": k, "index": i, "key_rank": int(keyrank[k, i]), "contribution": float(contrib[k, i]), "act_last": float(c["act"][k][-1][i]),
                                "gate_pre_at_subject": [float(c["gate"][k][p][i]) for p in subj_pos], "value_tokens": tok.convert_ids_to_tokens(top_tokens[k][:, i].tolist()[:5])})
        keyed_units.sort(key=lambda u: -abs(u["contribution"]))
        carriers = [{"layer": int(q // ffn), "index": int(q % ffn), "contribution": float(flat[q]), "is_edge": bool(edge.view(-1)[q]), "key_rank": int(keyrank.view(-1)[q]),
                     "value_tokens": tok.convert_ids_to_tokens(top_tokens[int(q // ffn)][:, int(q % ffn)].tolist()[:5])} for q in top_idx[:12].tolist()]
        results.append({"country": X, "capital": C, "correct": margin > 0, "margin": margin, "target": tok.convert_ids_to_tokens(t), "competitor": tok.convert_ids_to_tokens(j),
                        "units": N, "edges_to_target": n_edge, "edge_base_rate": base_rate, "carrying_units": CARRY, "fraction_of_carriers_that_are_edges": frac_edge_in_carry,
                        "random_fraction_mean": float(np.mean(rnd)), "random_fraction_p95": float(np.percentile(rnd, 95)),
                        "keyed_edges": n_keyed, "layers_with_keyed_edge": int(keyed.any(1).sum()), "keyed_edges_summed_contribution": float(contrib[keyed].sum()) if n_keyed else 0.0,
                        "edge_units_positive_flow": edge_pos, "all_positive_flow": pos_flow, "edge_share_of_positive_flow": edge_pos / pos_flow if pos_flow else None,
                        "subject_positions": subj_pos, "keyed_top": keyed_units[:8], "carriers_top12": carriers})
        print(f"{name} {X:10s} margin {margin:+.2f} correct {margin > 0!s:5} | edges to {tok.convert_ids_to_tokens(t)!r}: {n_edge:4d} ({base_rate:.3%}) | carriers that are edges {frac_edge_in_carry:.0%} (random {np.mean(rnd):.0%}, p95 {np.percentile(rnd, 95):.0%}) | "
              f"keyed edges {n_keyed} in {int(keyed.any(1).sum())} layers, summed {float(contrib[keyed].sum()) if n_keyed else 0:+.2f} | edge share of positive flow {edge_pos / pos_flow:.0%}", flush=True)
    core.dump(os.path.join(out, f"needle_{name}.json"), {"model": name, "layers": L, "ffn": ffn, "key_top": KEY_TOP, "value_top": VAL_TOP, "carry": CARRY, "facts": results})


if __name__ == "__main__":
    main()
