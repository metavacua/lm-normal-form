# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cell 02: sound interval bounds (auto_LiRPA, IBP) of the residual stream and the logits for every input of a given
length, from the exact coordinate-wise hull of the embedding table.

usage: ibp.py MODEL_DIR OUT_DIR T1_TAPS_JSON [--lengths 1,2,8] [--depths 1,2,4,8,16,30] [--control-scale 256]

The forward pass is written here in plain tensor operations over the model's own weights (held as buffers, not
copied), because auto_LiRPA traces a module and needs every operator explicit: RMSNorm as x / sqrt(mean(x^2) + eps),
SiLU as x * sigmoid(x), rotary embedding as multiplications by constant cos and sin tables, attention as explicit
matmul, mask, softmax, matmul. Its agreement with `transformers` is measured first (exactness), on real inputs; the
bounds are only reported if that agreement holds.

Writes in OUT_DIR: ibp_summary.json (per length and depth: lower/upper extrema, width, slack, soundness against the
exact length-1 taps, the control ratio, errors if any) and ibp_vectors.json (the bound vectors at length 1).
"""
import argparse
import math
import os
import sys
import traceback

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402

MASK_NEG = -1.0e9


class LlamaPrefix(nn.Module):
    """embeddings -> blocks 1..depth -> (final norm -> tied head if head) with every operator explicit."""

    def __init__(self, model, depth, head, T):
        super().__init__()
        cfg = model.config
        self.depth, self.head, self.T = depth, head, T
        self.d = cfg.hidden_size; self.nh = cfg.num_attention_heads; self.nkv = cfg.num_key_value_heads
        self.hd = self.d // self.nh; self.eps = float(cfg.rms_norm_eps); self.rep = self.nh // self.nkv
        self.scale = 1.0 / math.sqrt(self.hd)
        layers = model.model.layers
        for k in range(depth):
            L = layers[k]
            self.register_buffer(f"g_in{k}", L.input_layernorm.weight.data)
            self.register_buffer(f"g_post{k}", L.post_attention_layernorm.weight.data)
            self.register_buffer(f"wq{k}", L.self_attn.q_proj.weight.data.t().contiguous())
            self.register_buffer(f"wk{k}", L.self_attn.k_proj.weight.data.t().contiguous())
            self.register_buffer(f"wv{k}", L.self_attn.v_proj.weight.data.t().contiguous())
            self.register_buffer(f"wo{k}", L.self_attn.o_proj.weight.data.t().contiguous())
            self.register_buffer(f"wg{k}", L.mlp.gate_proj.weight.data.t().contiguous())
            self.register_buffer(f"wu{k}", L.mlp.up_proj.weight.data.t().contiguous())
            self.register_buffer(f"wd{k}", L.mlp.down_proj.weight.data.t().contiguous())
        self.register_buffer("g_final", model.model.norm.weight.data)
        self.register_buffer("w_head", model.lm_head.weight.data.t().contiguous())
        # rotary tables for positions 0..T-1, the transformers convention (rotate_half: pairs (i, i + hd/2)).
        # The inverse frequencies are taken from the loaded model's own rotary module, not recomputed from a config
        # field: transformers 5 keeps rope_theta under rope_parameters, and the first run read it as None.
        inv = model.model.rotary_emb.inv_freq.detach().to(torch.float32).clone()
        assert inv.shape[-1] == self.hd // 2, (inv.shape, self.hd)
        self.theta_from_inv_freq = float(inv[1] ** (-self.hd / 2.0)) if inv.numel() > 1 else None
        pos = torch.arange(T, dtype=torch.float32)
        fr = torch.outer(pos, inv)                               # (T, hd/2)
        emb = torch.cat([fr, fr], dim=-1)                        # (T, hd)
        self.register_buffer("cos", emb.cos().view(1, 1, T, self.hd))
        self.register_buffer("sin", emb.sin().view(1, 1, T, self.hd))
        mask = torch.full((T, T), MASK_NEG, dtype=torch.float32).triu(1).view(1, 1, T, T)
        self.register_buffer("mask", mask)

    def rms(self, x, g):
        # x * reciprocal(sqrt(.)) and not x / sqrt(.): auto_LiRPA's BoundDiv.forward has an ad-hoc LayerNorm branch that
        # fires whenever the divisor is a Sqrt node, assumes the dividend is a mean-subtracted input with a parent node,
        # and both crashes on a root input and computes the wrong thing for RMSNorm (bivariate.py, BoundDiv, "FIXME").
        var = (x * x).mean(dim=-1, keepdim=True)
        return x * torch.reciprocal(torch.sqrt(var + self.eps)) * g

    def rope(self, x):
        half = self.hd // 2
        x1, x2 = x[..., :half], x[..., half:]
        rot = torch.cat([x2 * -1.0, x1], dim=-1)
        return x * self.cos + rot * self.sin

    def forward(self, x):                                        # x: (1, T, d)
        T = self.T
        for k in range(self.depth):
            h = self.rms(x, getattr(self, f"g_in{k}"))
            q = (h @ getattr(self, f"wq{k}")).view(1, T, self.nh, self.hd).transpose(1, 2)      # (1, nh, T, hd)
            kk = (h @ getattr(self, f"wk{k}")).view(1, T, self.nkv, self.hd).transpose(1, 2)    # (1, nkv, T, hd)
            v = (h @ getattr(self, f"wv{k}")).view(1, T, self.nkv, self.hd).transpose(1, 2)
            q, kk = self.rope(q), self.rope(kk)
            if self.rep > 1:
                kk = kk.repeat_interleave(self.rep, dim=1)
                v = v.repeat_interleave(self.rep, dim=1)
            s = (q @ kk.transpose(-1, -2)) * self.scale + self.mask                               # (1, nh, T, T)
            p = torch.softmax(s, dim=-1)
            o = (p @ v).transpose(1, 2).reshape(1, T, self.d)
            x = x + o @ getattr(self, f"wo{k}")
            h2 = self.rms(x, getattr(self, f"g_post{k}"))
            gate = h2 @ getattr(self, f"wg{k}")
            up = h2 @ getattr(self, f"wu{k}")
            act = gate * torch.sigmoid(gate)
            x = x + (act * up) @ getattr(self, f"wd{k}")
        if self.head:
            return self.rms(x, self.g_final) @ self.w_head
        return x


def exactness(model, T, n_tokens, seed=29):
    """The explicit forward against transformers over n_tokens sequences of length T: the max |diff| of the logits,
    per position, and of the residual after block 1 (to localise a disagreement to attention or to the rest)."""
    g = torch.Generator().manual_seed(seed)
    V = model.config.vocab_size
    L = model.config.num_hidden_layers
    full = LlamaPrefix(model, L, True, T).eval()
    one = LlamaPrefix(model, 1, False, T).eval()
    worst = 0.0; per_pos = [0.0] * T; worst_block1 = 0.0
    with torch.inference_mode():
        for i in range(n_tokens):
            ids = torch.randint(0, V, (1, T), generator=g) if T > 1 else torch.tensor([[i * (V // n_tokens)]])
            out = model(input_ids=ids, output_hidden_states=True)
            ref = out.logits.to(torch.float32)
            emb = model.model.embed_tokens(ids)
            got = full(emb)
            d = (ref - got).abs()
            worst = max(worst, float(d.max()))
            for p in range(T):
                per_pos[p] = max(per_pos[p], float(d[0, p].max()))
            worst_block1 = max(worst_block1, float((out.hidden_states[1].to(torch.float32) - one(emb)).abs().max()))
    return {"max_abs_diff_logits": worst, "per_position": per_pos, "max_abs_diff_resid_after_block1": worst_block1,
            "theta_from_inv_freq": full.theta_from_inv_freq}


def bounds_for(module, x_lo, x_hi):
    from auto_LiRPA import BoundedModule, BoundedTensor
    from auto_LiRPA.perturbations import PerturbationLpNorm
    x0 = (x_lo + x_hi) / 2
    bm = BoundedModule(module, torch.empty_like(x0), bound_opts={"conv_mode": "matrix"}, device="cpu", verbose=False)
    ptb = PerturbationLpNorm(norm=np.inf, x_L=x_lo, x_U=x_hi)
    bx = BoundedTensor(x0, ptb)
    lb, ub = bm.compute_bounds(x=(bx,), method="IBP")
    return lb.detach(), ub.detach()


def stat(lb, ub):
    lo, hi = lb.to(torch.float64).numpy(), ub.to(torch.float64).numpy()
    amax = float(max(np.abs(lo).max(), np.abs(hi).max())) if np.isfinite(lo).all() and np.isfinite(hi).all() else float("inf")
    return {"lower_min": float(lo.min()), "upper_max": float(hi.max()), "abs_max": amax,
            "width_max": float((hi - lo).max()), "width_median": float(np.median(hi - lo)),
            "finite": bool(np.isfinite(lo).all() and np.isfinite(hi).all()),
            "slack_doublings": (float(np.log2(common.BF16_MAX / amax)) if 0 < amax < float("inf") else None)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("out_dir"); ap.add_argument("t1_taps")
    ap.add_argument("--lengths", default="1,2,8"); ap.add_argument("--depths", default="1,2,4,8,16,30")
    ap.add_argument("--control-scale", type=float, default=256.0)
    a = ap.parse_args()
    clock = common.Clock()
    os.makedirs(a.out_dir, exist_ok=True)
    model = common.load_model(a.model_dir)
    L = model.config.num_hidden_layers
    lengths = [int(t) for t in a.lengths.split(",")]
    depths = [min(int(d), L) for d in a.depths.split(",")]
    E = model.model.embed_tokens.weight.data
    e_lo, e_hi = E.min(dim=0).values, E.max(dim=0).values
    taps = common.load(a.t1_taps)
    out = {"model": common.model_facts(model), "embedding_hull": {"zero_width_coords": int((e_hi - e_lo == 0).sum()),
           "min": float(e_lo.min()), "max": float(e_hi.max())}, "lengths": {}, "exactness": {}, "errors": {}}
    vectors = {}
    try:
        import auto_LiRPA
        out["auto_lirpa_version"] = getattr(auto_LiRPA, "__version__", "?")
    except Exception as e:  # the tool is absent: everything below fails and says so
        out["errors"]["import"] = repr(e)
    # exactness of the explicit forward against transformers, at each length
    for T in lengths:
        try:
            out["exactness"][str(T)] = exactness(model, T, 16 if T > 1 else 64)
        except Exception as e:
            out["exactness"][str(T)] = {"error": repr(e), "trace": traceback.format_exc()[-2000:]}
        print(f"{clock.s():7.1f}s exactness T={T}: {out['exactness'][str(T)]}", flush=True)
    # bounds
    for T in lengths:
        x_lo = e_lo.view(1, 1, -1).expand(1, T, -1).contiguous()
        x_hi = e_hi.view(1, 1, -1).expand(1, T, -1).contiguous()
        res = {}
        for depth in depths + ["logits"]:
            key = f"resid.{depth}" if depth != "logits" else "logits"
            try:
                mod = LlamaPrefix(model, L if depth == "logits" else depth, depth == "logits", T).eval()
                lb, ub = bounds_for(mod, x_lo, x_hi)
                r = stat(lb, ub)
                if T == 1 and key in taps:                        # soundness against the exact extrema at length 1
                    lo = np.array(taps[key]["lo"]); hi = np.array(taps[key]["hi"])
                    LB = lb[0, 0].to(torch.float64).numpy(); UB = ub[0, 0].to(torch.float64).numpy()
                    viol_lo = float(np.maximum(LB - lo, 0).max()); viol_hi = float(np.maximum(hi - UB, 0).max())
                    r["sound_strict"] = bool(viol_lo == 0 and viol_hi == 0)
                    r["sound_1e-6"] = bool(viol_lo <= 1e-6 and viol_hi <= 1e-6)
                    r["max_violation"] = max(viol_lo, viol_hi)
                    r["exact_abs_max"] = float(max(np.abs(lo).max(), np.abs(hi).max()))
                    r["exact_width_max"] = float((hi - lo).max())
                    vectors[key] = {"lb": LB.tolist(), "ub": UB.tolist()}
                res[key] = r
                del mod
            except Exception as e:
                res[key] = {"error": repr(e), "trace": traceback.format_exc()[-3000:]}
            print(f"{clock.s():7.1f}s T={T} {key}: {str(res[key])[:200]}  rss {common.peak_rss_mb():.0f} MB", flush=True)
        out["lengths"][str(T)] = res
    # control (c): layer 0's down-projection scaled; bound of resid.1 at length 1 must grow
    try:
        base = out["lengths"]["1"]["resid.1"].get("abs_max")
        if base is None:
            raise RuntimeError("no length-1 bound of resid.1 to compare against: " + str(out["lengths"]["1"]["resid.1"].get("error")))
        w = model.model.layers[0].mlp.down_proj.weight
        w.mul_(a.control_scale)
        mod = LlamaPrefix(model, 1, False, 1).eval()
        lb, ub = bounds_for(mod, e_lo.view(1, 1, -1).contiguous(), e_hi.view(1, 1, -1).contiguous())
        scaled = stat(lb, ub)["abs_max"]
        w.div_(a.control_scale)
        out["control_c"] = {"scale": a.control_scale, "abs_max_before": base, "abs_max_after": scaled,
                            "ratio": (scaled / base if base else None)}
    except Exception as e:
        out["control_c"] = {"error": repr(e)}
    out["seconds"] = clock.s(); out["peak_rss_mb"] = round(common.peak_rss_mb())
    common.dump(os.path.join(a.out_dir, "ibp_summary.json"), out)
    common.dump(os.path.join(a.out_dir, "ibp_vectors.json"), vectors)
    print("done", out["seconds"], "s", out["peak_rss_mb"], "MB; errors:", {k: v for k, v in out["errors"].items()})


if __name__ == "__main__":
    main()
