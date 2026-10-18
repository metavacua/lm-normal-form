"""Streaming float32 forward pass of SmolLM2-135M from the safetensors file, numpy only, one prompt, 5 positions.
Reads each tensor from disk when needed (no mmap), so the resident size stays under 100 MB. Writes yhat.npy and xL.npy."""
import json, struct, sys, resource
import numpy as np
path = sys.argv[1]; ids = [int(t) for t in sys.argv[2].split(",")]
f = open(path, "rb"); n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); start = 8 + n
def bf16_to_f32(b): return (np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
def tensor(name):
    e = hdr[name]; assert e["dtype"] == "BF16"; a, b = e["data_offsets"]; f.seek(start + a); return bf16_to_f32(f.read(b - a)).reshape(e["shape"])
def rows(name, rs):
    e = hdr[name]; cols = e["shape"][1]; a = e["data_offsets"][0]; out = []
    for r in rs: f.seek(start + a + r * cols * 2); out.append(bf16_to_f32(f.read(cols * 2)))
    return np.stack(out)
Hd, KV, D, EPS, THETA, L = 9, 3, 64, 1e-5, 100000.0, 30
T = len(ids); C = 576
def rmsnorm(x, g): return x * (1.0 / np.sqrt((x * x).mean(-1, keepdims=True) + EPS)) * g
inv = 1.0 / (THETA ** (np.arange(0, D, 2, dtype=np.float32) / D)); fr = np.outer(np.arange(T, dtype=np.float32), inv); emb = np.concatenate([fr, fr], -1)
cos, sin = np.cos(emb)[:, None, :], np.sin(emb)[:, None, :]
def rot(x): return np.concatenate([-x[..., D // 2:], x[..., :D // 2]], -1)
mask = np.triu(np.full((T, T), -np.inf, dtype=np.float32), 1)
x = rows("model.embed_tokens.weight", ids).astype(np.float32)
points = [x[-1].copy()]
for i in range(L):
    p = f"model.layers.{i}."
    h = rmsnorm(x, tensor(p + "input_layernorm.weight"))
    q = (h @ tensor(p + "self_attn.q_proj.weight").T).reshape(T, Hd, D); k = (h @ tensor(p + "self_attn.k_proj.weight").T).reshape(T, KV, D); v = (h @ tensor(p + "self_attn.v_proj.weight").T).reshape(T, KV, D)
    q = q * cos + rot(q) * sin; k = k * cos + rot(k) * sin
    outs = []
    for hh in range(Hd):
        kv = hh // (Hd // KV); s = (q[:, hh] @ k[:, kv].T) / np.sqrt(D) + mask
        s = s - s.max(-1, keepdims=True); pr = np.exp(s); pr /= pr.sum(-1, keepdims=True); outs.append(pr @ v[:, kv])
    a = np.concatenate(outs, -1) @ tensor(p + "self_attn.o_proj.weight").T
    x = x + a; points.append(x[-1].copy())
    h = rmsnorm(x, tensor(p + "post_attention_layernorm.weight"))
    gte = h @ tensor(p + "mlp.gate_proj.weight").T; up = h @ tensor(p + "mlp.up_proj.weight").T
    m = (gte / (1.0 + np.exp(-gte)) * up) @ tensor(p + "mlp.down_proj.weight").T
    x = x + m; points.append(x[-1].copy())
xL = x[-1]; g = tensor("model.norm.weight"); rms = float(np.sqrt((xL * xL).mean() + EPS)); yhat = (xL / rms * g).astype(np.float32)
e = hdr["model.embed_tokens.weight"]; V = e["shape"][0]; z = np.empty(V, dtype=np.float32); CH = 2048
for r0 in range(0, V, CH):
    r1 = min(V, r0 + CH); f.seek(start + e["data_offsets"][0] + r0 * C * 2); z[r0:r1] = bf16_to_f32(f.read((r1 - r0) * C * 2)).reshape(r1 - r0, C) @ yhat
order = np.argsort(-z)[:6]
print("argmax", int(order[0]), "top6", [(int(t), round(float(z[t]), 4)) for t in order])
print(f"z[260]={z[260]:.4f} z[7042]={z[7042]:.4f} the-minus-paris={z[260]-z[7042]:.4f} rank_of_7042={int((z > z[7042]).sum())+1}")
print(f"rms(xL)={rms:.4f} |xL|={np.linalg.norm(xL):.4f} |yhat|={np.linalg.norm(yhat):.4f} points={len(points)}")
np.save("yhat.npy", yhat); np.save("xL.npy", xL); np.save("points.npy", np.stack(points)); np.save("logits.npy", z)
print("peak rss MiB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
