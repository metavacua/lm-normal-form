# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Streaming forward pass of a Llama-architecture model (SmolLM2) from its safetensors file, numpy only, one prompt.
usage: fwd_numpy.py MODEL_DIR IDS [OUT_DIR]      MODEL_DIR holds model.safetensors and config.json; IDS are comma-separated token ids.
Each tensor is read from disk when needed (no mmap), so the resident size stays under 100 MB. Writes to OUT_DIR (default .): yhat.npy (the final
normalized vector at the last position), xL.npy (the residual before the final norm), points.npy (the last position's residual after the embedding and
after every sublayer), logits.npy (every logit of the last position) and prompt.json (the ids)."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_common import bf16_to_f32, peak_rss_mb, st_header, st_tensor  # noqa: E402

mdir, ids = sys.argv[1], [int(t) for t in sys.argv[2].split(",")]
odir = sys.argv[3] if len(sys.argv) > 3 else "."
cfg = json.load(open(os.path.join(mdir, "config.json")))
Hd, KV, L, EPS, THETA = cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["num_hidden_layers"], cfg["rms_norm_eps"], float(cfg["rope_theta"])
D = cfg.get("head_dim") or cfg["hidden_size"] // Hd
assert cfg.get("tie_word_embeddings", True), "the head is the embedding matrix here"
hdr, start = st_header(os.path.join(mdir, "model.safetensors"))
f = open(os.path.join(mdir, "model.safetensors"), "rb")


def tensor(name):
    off, shape = st_tensor(hdr, start, name)
    f.seek(off)
    return bf16_to_f32(f.read(2 * int(np.prod(shape)))).reshape(shape)


def rows(name, rs):
    off, (_, cols) = st_tensor(hdr, start, name)
    out = []
    for r in rs:
        f.seek(off + r * cols * 2)
        out.append(bf16_to_f32(f.read(cols * 2)))
    return np.stack(out)


def rmsnorm(x, g):
    return x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + EPS)) * g


def rot(x):
    return np.concatenate([-x[..., D // 2:], x[..., :D // 2]], -1)


T = len(ids)
inv = 1.0 / (THETA ** (np.arange(0, D, 2, dtype=np.float32) / D))
fr = np.outer(np.arange(T, dtype=np.float32), inv)
emb = np.concatenate([fr, fr], -1)
cos, sin = np.cos(emb)[:, None, :], np.sin(emb)[:, None, :]
mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), 1)
x = rows("model.embed_tokens.weight", ids).astype(np.float32)
points = [x[-1].copy()]
for i in range(L):
    p = f"model.layers.{i}."
    h = rmsnorm(x, tensor(p + "input_layernorm.weight"))
    q = (h @ tensor(p + "self_attn.q_proj.weight").T).reshape(T, Hd, D)
    k = (h @ tensor(p + "self_attn.k_proj.weight").T).reshape(T, KV, D)
    v = (h @ tensor(p + "self_attn.v_proj.weight").T).reshape(T, KV, D)
    q = q * cos + rot(q) * sin
    k = k * cos + rot(k) * sin
    outs = []
    for hh in range(Hd):
        kv = hh // (Hd // KV)
        s = (q[:, hh] @ k[:, kv].T) / np.sqrt(D) + mask
        s = s - s.max(-1, keepdims=True)
        pr = np.exp(s)
        pr /= pr.sum(-1, keepdims=True)
        outs.append(pr @ v[:, kv])
    x = x + np.concatenate(outs, -1) @ tensor(p + "self_attn.o_proj.weight").T
    points.append(x[-1].copy())
    h = rmsnorm(x, tensor(p + "post_attention_layernorm.weight"))
    gte = h @ tensor(p + "mlp.gate_proj.weight").T
    up = h @ tensor(p + "mlp.up_proj.weight").T
    x = x + (gte / (1.0 + np.exp(-gte)) * up) @ tensor(p + "mlp.down_proj.weight").T
    points.append(x[-1].copy())
xL = x[-1]
g = tensor("model.norm.weight")
rms = float(np.sqrt((xL * xL).mean() + EPS))
yhat = (xL / rms * g).astype(np.float32)
off, (V, C) = st_tensor(hdr, start, "model.embed_tokens.weight")
z = np.empty(V, dtype=np.float32)
CH = 2048
for r0 in range(0, V, CH):
    r1 = min(V, r0 + CH)
    f.seek(off + r0 * C * 2)
    z[r0:r1] = bf16_to_f32(f.read((r1 - r0) * C * 2)).reshape(r1 - r0, C) @ yhat
order = np.argsort(-z)[:6]
print("argmax", int(order[0]), "top6", [(int(t), round(float(z[t]), 4)) for t in order])
print(f"rms(xL)={rms:.4f} |xL|={np.linalg.norm(xL):.4f} |yhat|={np.linalg.norm(yhat):.4f} points={len(points)}")
os.makedirs(odir, exist_ok=True)
for name, a in (("yhat", yhat), ("xL", xL), ("points", np.stack(points)), ("logits", z)):
    np.save(os.path.join(odir, name + ".npy"), a)
with open(os.path.join(odir, "prompt.json"), "w") as fh:
    json.dump({"ids": ids}, fh)
print("peak rss MiB", int(peak_rss_mb()))
