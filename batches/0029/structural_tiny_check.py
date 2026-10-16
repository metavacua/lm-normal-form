# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The structural bounds against brute force on a tiny random model of the same relations (shapes, GQA, RoPE, SiLU,
RMSNorm): 400 random inputs of lengths 1 to 8, every tap must lie inside its bound. Tests the derivation, not the real
model. Run by the workflow before cell 04."""
import sys, numpy as np
sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from structural import bounds, silu
rng = np.random.default_rng(29)
d, nh, nkv, dff, V, L = 16, 4, 2, 24, 50, 3
hd = d // nh
cfg = dict(hidden_size=d, num_hidden_layers=L, num_attention_heads=nh, num_key_value_heads=nkv, intermediate_size=dff, vocab_size=V, rms_norm_eps=1e-5)
W = {"model.embed_tokens.weight": rng.normal(size=(V, d)), "model.norm.weight": rng.normal(size=d)}
for k in range(L):
    p = f"model.layers.{k}."
    W[p+"input_layernorm.weight"] = rng.normal(size=d); W[p+"post_attention_layernorm.weight"] = rng.normal(size=d)
    W[p+"self_attn.q_proj.weight"] = rng.normal(size=(d, d)); W[p+"self_attn.k_proj.weight"] = rng.normal(size=(nkv*hd, d))
    W[p+"self_attn.v_proj.weight"] = rng.normal(size=(nkv*hd, d)); W[p+"self_attn.o_proj.weight"] = rng.normal(size=(d, d))
    W[p+"mlp.gate_proj.weight"] = rng.normal(size=(dff, d)); W[p+"mlp.up_proj.weight"] = rng.normal(size=(dff, d)); W[p+"mlp.down_proj.weight"] = rng.normal(size=(d, dff))
B = bounds(W, cfg)
def rms(x, g): return x / np.sqrt((x*x).mean(-1, keepdims=True) + 1e-5) * g
def rope(x, T):
    inv = 1.0 / (10000 ** (np.arange(0, hd, 2) / hd)); fr = np.outer(np.arange(T), inv); emb = np.concatenate([fr, fr], -1)
    c, s = np.cos(emb)[None], np.sin(emb)[None]; x1, x2 = x[..., :hd//2], x[..., hd//2:]
    return x * c + np.concatenate([-x2, x1], -1) * s
worst = 0.0
for trial in range(400):
    T = int(rng.integers(1, 9)); ids = rng.integers(0, V, size=T); x = W["model.embed_tokens.weight"][ids]
    assert (np.abs(x) <= B["resid"][0] + 1e-12).all()
    for k in range(L):
        p = f"model.layers.{k}."; h = rms(x, W[p+"input_layernorm.weight"])
        q = (h @ W[p+"self_attn.q_proj.weight"].T).reshape(T, nh, hd).transpose(1, 0, 2)
        kk = (h @ W[p+"self_attn.k_proj.weight"].T).reshape(T, nkv, hd).transpose(1, 0, 2); v = (h @ W[p+"self_attn.v_proj.weight"].T).reshape(T, nkv, hd).transpose(1, 0, 2)
        q, kk = rope(q, T), rope(kk, T); rep = nh // nkv; kk = np.repeat(kk, rep, 0); v = np.repeat(v, rep, 0)
        s = q @ kk.transpose(0, 2, 1) / np.sqrt(hd)
        um = np.tril(np.ones((T, T), bool)); assert (np.abs(s)[..., um] <= np.array(B["scores"][k])[:, None] + 1e-9).all()
        s = s + np.triu(np.full((T, T), -1e9), 1)
        pz = np.exp(s - s.max(-1, keepdims=True)); pz /= pz.sum(-1, keepdims=True); o = (pz @ v).transpose(1, 0, 2).reshape(T, d)
        x = x + o @ W[p+"self_attn.o_proj.weight"].T; h2 = rms(x, W[p+"post_attention_layernorm.weight"])
        gate = h2 @ W[p+"mlp.gate_proj.weight"].T; up = h2 @ W[p+"mlp.up_proj.weight"].T
        assert (np.abs(gate) <= B["gate"][k] + 1e-9).all()
        x = x + (silu(gate) * up) @ W[p+"mlp.down_proj.weight"].T
        assert (np.abs(x) <= B["resid"][k+1] + 1e-9).all(), (k, np.abs(x).max(), B["resid"][k+1].max())
        worst = max(worst, float((np.abs(x) / B["resid"][k+1]).max()))
    fn = rms(x, W["model.norm.weight"]); assert (np.abs(fn) <= B["final_norm"] + 1e-9).all()
    lg = fn @ W["model.embed_tokens.weight"].T; assert (np.abs(lg) <= B["logits"] + 1e-9).all()
Bc = bounds(W, cfg, scale_down0=256.0)
print("tiny-model soundness check passed over 400 random inputs of lengths 1-8; tightest exact/bound ratio:", round(worst, 4),
      "; control d ratio:", Bc["per_block"][0]["ffn_box_max"] / B["per_block"][0]["ffn_box_max"])
