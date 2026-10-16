# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cell 04: structural bounds of every activation of a pre-norm Llama-style decoder, from the weights alone, valid
for every input of every length (docs/batches/0029.md, addendum 1).

Two facts carry every step. (i) n = RMSNorm(x) = x / sqrt(mean(x^2) + eps) has ||n||_2 <= sqrt(d) for every x,
so for a row w of a linear map reading n * g:  |w . (n * g)| = |(w * g) . n| <= ||w * g||_2 * sqrt(d)
(Cauchy-Schwarz); per head, ||W_h (n * g)||_2 <= sigma_max(W_h diag(g)) * sqrt(d). (ii) softmax weights are
non-negative and sum to one, so each coordinate of an attention output lies within the hull of that coordinate
over the value vectors; and RoPE is an isometry on every (i, i + d_h/2) plane. The residual bound after block k is
the embedding hull plus the sum of the first k blocks' attention and FFN boxes: additive, not compounding.

usage: structural.py MODEL_DIR OUT_DIR T1_TAPS_JSON [--control-scale 256]
Writes OUT_DIR/structural_summary.json and OUT_DIR/structural_vectors.json. Numpy in float64; no model is run.
"""
import argparse
import json
import os
import sys
import time

import numpy as np

BF16_MAX = 3.3895313892515355e38      # the largest finite bfloat16 value (as in common.py; common imports torch, this cell needs none)
SILU_MIN = 0.2784645427610738          # |min silu| at x = -1.2784645


def dump(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def silu(x):
    return x / (1.0 + np.exp(-x))


def act_abs_bound(gate_box):
    """|silu(z)| for |z| <= gate_box: the maximum of silu(gate_box) and of |silu| on the negative side."""
    neg = np.abs(silu(-np.minimum(gate_box, 1.2784645)))
    return np.maximum(silu(gate_box), neg)


def read_weights(model_dir):
    """Every tensor of model.safetensors as float64 numpy, by name."""
    from safetensors import safe_open
    import torch
    W = {}
    with safe_open(os.path.join(model_dir, "model.safetensors"), framework="pt") as f:
        for name in f.keys():
            W[name] = f.get_tensor(name).to(torch.float64).numpy()
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    return W, cfg


def bounds(W, cfg, scale_down0=1.0):
    d = cfg["hidden_size"]; L = cfg["num_hidden_layers"]; nh = cfg["num_attention_heads"]; nkv = cfg["num_key_value_heads"]
    hd = d // nh; rep = nh // nkv; sd = np.sqrt(d)
    E = W["model.embed_tokens.weight"]
    e_lo, e_hi = E.min(axis=0), E.max(axis=0)
    R = np.maximum(np.abs(e_lo), np.abs(e_hi))                       # |x| bound per coordinate, after the embedding
    per_block, resid_bounds, gate_bounds, score_bounds = [], [R.copy()], [], []
    for k in range(L):
        p = f"model.layers.{k}."
        g_in = W[p + "input_layernorm.weight"]; g_post = W[p + "post_attention_layernorm.weight"]
        Wq, Wk, Wv, Wo = (W[p + "self_attn.q_proj.weight"], W[p + "self_attn.k_proj.weight"],
                          W[p + "self_attn.v_proj.weight"], W[p + "self_attn.o_proj.weight"])
        Wg, Wu, Wd = W[p + "mlp.gate_proj.weight"], W[p + "mlp.up_proj.weight"], W[p + "mlp.down_proj.weight"]
        if k == 0 and scale_down0 != 1.0:
            Wd = Wd * scale_down0
        # boxes of q, k, v per output coordinate (fact i, row-wise)
        q_box = np.linalg.norm(Wq * g_in, axis=1) * sd
        k_box = np.linalg.norm(Wk * g_in, axis=1) * sd
        v_box = np.linalg.norm(Wv * g_in, axis=1) * sd
        # per-head norms: the smaller of the spectral bound and the box bound, both sound
        def head_norm(Wm, nheads, box):
            out = []
            for h in range(nheads):
                blk = (Wm[h * hd:(h + 1) * hd] * g_in)
                spec = np.linalg.norm(blk, 2) * sd
                out.append(min(spec, float(np.linalg.norm(box[h * hd:(h + 1) * hd]))))
            return np.array(out)
        qn = head_norm(Wq, nh, q_box); kn = head_norm(Wk, nkv, k_box)
        scores = np.array([qn[h] * kn[h // rep] / np.sqrt(hd) for h in range(nh)])      # RoPE is an isometry
        score_bounds.append(scores)
        # attention output: within the hull of v (fact ii); GQA repeats each kv head's box for its group
        o_box = np.concatenate([v_box[(h // rep) * hd:(h // rep + 1) * hd] for h in range(nh)])
        attn_box = np.abs(Wo) @ o_box
        # FFN from the post-attention norm (fact i again; the residual's magnitude does not enter)
        gate_box = np.linalg.norm(Wg * g_post, axis=1) * sd
        up_box = np.linalg.norm(Wu * g_post, axis=1) * sd
        prod = act_abs_bound(gate_box) * up_box
        ffn_box = np.abs(Wd) @ prod
        R = R + attn_box + ffn_box
        resid_bounds.append(R.copy()); gate_bounds.append(gate_box)
        per_block.append({"attn_box_max": float(attn_box.max()), "ffn_box_max": float(ffn_box.max()),
                          "gate_box_max": float(gate_box.max()), "up_box_max": float(up_box.max()),
                          "v_box_max": float(v_box.max()), "score_bound_max": float(scores.max()),
                          "resid_bound_max": float(R.max())})
    g_f = W["model.norm.weight"]
    final_norm_box = sd * np.abs(g_f)
    head = W.get("lm_head.weight", E)
    logit_box = np.linalg.norm(head * g_f, axis=1) * sd
    return {"resid": resid_bounds, "gate": gate_bounds, "final_norm": final_norm_box, "logits": logit_box,
            "scores": score_bounds, "per_block": per_block, "embedding_hull": (e_lo, e_hi)}


def containment(exact, box):
    """exact: {'lo': [...], 'hi': [...]}; box: non-negative bound per coordinate. Violation = how far outside."""
    lo, hi = np.array(exact["lo"]), np.array(exact["hi"])
    viol = float(max(np.maximum(-lo - box, 0).max(), np.maximum(hi - box, 0).max()))
    amax = np.maximum(np.abs(lo), np.abs(hi))
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = np.where(amax > 0, box / amax, np.inf)
    return {"contained": bool(viol == 0.0), "max_violation": viol, "looseness_min": float(np.nanmin(ratio)),
            "looseness_median": float(np.nanmedian(ratio)), "exact_abs_max": float(amax.max()), "bound_max": float(box.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("out_dir"); ap.add_argument("t1_taps")
    ap.add_argument("--control-scale", type=float, default=256.0)
    a = ap.parse_args()
    t0 = time.time()
    W, cfg = read_weights(a.model_dir)
    B = bounds(W, cfg)
    taps = load(a.t1_taps)
    L = cfg["num_hidden_layers"]
    checks = {}
    for k in range(L + 1):
        key = f"resid.{k}"
        if key in taps:
            checks[key] = containment(taps[key], B["resid"][k])
    for k in range(1, L + 1):
        key = f"gate.{k}"
        if key in taps:
            checks[key] = containment(taps[key], B["gate"][k - 1])
    if "final_norm" in taps:
        checks["final_norm"] = containment(taps["final_norm"], B["final_norm"])
    if "logits" in taps:
        checks["logits"] = containment(taps["logits"], B["logits"])
    all_contained = all(v["contained"] for v in checks.values())
    resid30 = float(B["resid"][L].max()); logit_max = float(B["logits"].max())
    score_max = float(max(s.max() for s in B["scores"]))
    # control d: scale layer 0's down-projection; block 1's FFN box must scale exactly, nothing else may move
    Bc = bounds(W, cfg, scale_down0=a.control_scale)
    r1 = Bc["per_block"][0]["ffn_box_max"] / B["per_block"][0]["ffn_box_max"]
    others_same = all(abs(Bc["per_block"][k]["ffn_box_max"] - B["per_block"][k]["ffn_box_max"]) <= 1e-9 * B["per_block"][k]["ffn_box_max"]
                      and abs(Bc["per_block"][k]["attn_box_max"] - B["per_block"][k]["attn_box_max"]) <= 1e-9 * B["per_block"][k]["attn_box_max"]
                      for k in range(1, L))
    out = {
        "config": {k: cfg.get(k) for k in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "intermediate_size", "vocab_size", "rms_norm_eps")},
        "tensors": len(W), "all_contained": all_contained, "checks": checks,
        "resid_after_last_block_bound_max": resid30, "logit_bound_max": logit_max, "score_bound_max": score_max,
        "final_norm_bound_max": float(B["final_norm"].max()),
        "slack_doublings": {"resid": float(np.log2(BF16_MAX / resid30)), "logits": float(np.log2(BF16_MAX / logit_max)),
                            "scores": float(np.log2(BF16_MAX / score_max))},
        "per_block": B["per_block"],
        "score_bounds_per_head": [s.tolist() for s in B["scores"]],
        "control_d": {"scale": a.control_scale, "block1_ffn_ratio": r1, "other_blocks_unchanged": bool(others_same),
                      "ratio_is_exact": bool(abs(r1 - a.control_scale) <= 1e-9 * a.control_scale)},
        "length_independent": "by construction: no bound reads the previous block's bound except as an additive term, and no bound reads the length",
        "seconds": round(time.time() - t0, 1),
    }
    dump(os.path.join(a.out_dir, "structural_summary.json"), out)
    dump(os.path.join(a.out_dir, "structural_vectors.json"),
         {"resid": [r.tolist() for r in B["resid"]], "gate": [g.tolist() for g in B["gate"]],
          "final_norm": B["final_norm"].tolist(), "logits": B["logits"].tolist()})
    print(f"contained at every tap: {all_contained}; resid bound after block {L}: {resid30:.4g}; logit bound {logit_max:.4g}; "
          f"score bound {score_max:.4g}; control d ratio {r1:.6g} others unchanged {others_same}; {out['seconds']} s")
    for key in ("resid.1", "resid.12", f"resid.{L}", "gate.12", "logits"):
        if key in checks:
            print(f"  {key}: {checks[key]}")


if __name__ == "__main__":
    main()
