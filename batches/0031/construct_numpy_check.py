# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The constructions of batch 0031 run through an explicit numpy forward (the same arithmetic as the transformers Llama
block: RMSNorm, RoPE in the rotate_half convention, causal softmax attention, SiLU-gated FFN, final norm, tied head),
checked against the theories before any model file exists. Numpy only; runs on the development machine and in CI."""
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from construct import CAPITALS, WORDS, LADDER, char_pieces, word_pieces, m1_tensors, m2_tensors  # noqa: E402


def rms(x, g, eps=1e-5):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * g


def rope(x, theta, hd):                       # x: (T, hd), positions 0..T-1
    T = x.shape[0]
    inv = 1.0 / (theta ** (np.arange(0, hd, 2) / hd)); ang = np.outer(np.arange(T), inv); emb = np.concatenate([ang, ang], -1)
    c, s = np.cos(emb), np.sin(emb); x1, x2 = x[:, :hd // 2], x[:, hd // 2:]
    return x * c + np.concatenate([-x2, x1], -1) * s


def forward(tensors, cfg, ids):
    d = cfg["hidden_size"]; L = cfg["num_hidden_layers"]; theta = cfg["rope_theta"]; eps = cfg["rms_norm_eps"]
    x = tensors["model.embed_tokens.weight"][ids]
    T = len(ids)
    for k in range(L):
        p = f"model.layers.{k}."
        h = rms(x, tensors[p + "input_layernorm.weight"], eps)
        q = rope(h @ tensors[p + "self_attn.q_proj.weight"].T, theta, d); kk = rope(h @ tensors[p + "self_attn.k_proj.weight"].T, theta, d)
        v = h @ tensors[p + "self_attn.v_proj.weight"].T
        s = q @ kk.T / math.sqrt(d) + np.triu(np.full((T, T), -1e9), 1)
        pz = np.exp(s - s.max(-1, keepdims=True)); pz /= pz.sum(-1, keepdims=True)
        x = x + (pz @ v) @ tensors[p + "self_attn.o_proj.weight"].T
        h2 = rms(x, tensors[p + "post_attention_layernorm.weight"], eps)
        gate = h2 @ tensors[p + "mlp.gate_proj.weight"].T; up = h2 @ tensors[p + "mlp.up_proj.weight"].T
        x = x + ((gate / (1 + np.exp(-gate))) * up) @ tensors[p + "mlp.down_proj.weight"].T
    return rms(x, tensors["model.norm.weight"], eps) @ tensors["model.embed_tokens.weight"].T


def main():
    # M1 on T1: greedy from "the capital of france is " (characters), stop at </s>
    tensors, cfg, info = m1_tensors()
    pieces = char_pieces(); idx = {p: i for i, p in enumerate(pieces)}
    prompt = [idx["▁" if ch == " " else ch] for ch in "the capital of france is "]
    out = []
    ids = list(prompt)
    for _ in range(10):
        nxt = int(forward(tensors, cfg, np.array(ids))[-1].argmax()); out.append(pieces[nxt]); ids.append(nxt)
        if nxt == 2:
            break
    text = "".join(out)
    assert text == "paris</s>", text
    print(f"M1: width {info['width']} (rank S {info['rank_S']}), greedy after the prompt: {text!r}")
    # M2 on T2 across the ladder
    wp = word_pieces(); widx = {p: i for i, p in enumerate(wp)}
    for d in LADDER:
        tensors, cfg, info = m2_tensors(d, orthogonal=(d >= 52))
        fails = []; margins = []
        for X, Y in CAPITALS.items():
            ids = np.array([widx["▁" + w] for w in ["the", "capital", "of", X, "is"]])
            lg = forward(tensors, cfg, ids)[-1]
            top = np.argsort(-lg)
            margins.append(float(lg[top[0]] - lg[top[1]]))
            if wp[top[0]] != "▁" + Y:
                fails.append((X, wp[top[0]]))
        print(f"M2 d={d}: failures {fails}, min margin {min(margins):.3f}, coherence {info['max_code_coherence']:.3f}")
        if d == 52:
            assert not fails and min(margins) >= 3, (fails, min(margins))


if __name__ == "__main__":
    main()
