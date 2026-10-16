# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cell 05: the instance decompiled to the end, as ground N-Triples, and checked with Oxigraph.

usage: decompile.py HF_DIR OUT_DIR
Triples (every subject and object an IRI or a literal; no blank node anywhere):
  tensors      name, shape, dtype, byte length, SHA-256 of the bytes (the reference into the safetensors carrier)
  symbols      id -> piece
  units        per block k and unit j: the static key readouts of the gate row and the up row over the 50 symbols
               (score_s = (W[j] * g_post_k) . rms(E[s])), and the value readout of the down column through the final
               norm and the unembedding (score_t = E[t] . (g_f * W_down[:, j]))
  heads        per block and head: the QK score of query symbol s at relative offset d (0..3) against key symbol t
               at offset 0, RoPE included, from the block-1-style static input rms(E[.]) * g_in; and the OV readout
               of key symbol t: E @ (g_f * W_o[:, head] @ W_v_head (g_in * rms(E[t]))).
The static readouts are exact for block 1 at length 1 and projections for the others; the predicate says which.
Checks: Oxigraph loads the file; the triple count; ASK for a blank node (must be false); every tensor SHA-256 in
the graph equals the bytes in model.safetensors; size of the N-Triples against the safetensors file.
"""
import hashlib
import json
import os
import struct
import sys

import numpy as np

BASE = "urn:lmnf:batch0030:"
P = BASE + "p:"


def lit_d(x):
    return f'"{float(x)!r}"^^<http://www.w3.org/2001/XMLSchema#double>'


def lit_s(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def lit_i(i):
    return f'"{int(i)}"^^<http://www.w3.org/2001/XMLSchema#integer>'


def st_read(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; header = json.loads(f.read(n)); base = 8 + n
        out = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            a, b = meta["data_offsets"]; f.seek(base + a); raw = f.read(b - a)
            assert meta["dtype"] == "F32", meta
            out[name] = (np.frombuffer(raw, dtype=np.float32).astype(np.float64).reshape(meta["shape"]), hashlib.sha256(raw).hexdigest(), meta)
        return out


def rms(x, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps)


def rope(v, pos, theta, hd):
    inv = 1.0 / (theta ** (np.arange(0, hd, 2) / hd)); ang = pos * inv
    c, s = np.cos(np.concatenate([ang, ang])), np.sin(np.concatenate([ang, ang]))
    x1, x2 = v[:hd // 2], v[hd // 2:]
    return v * c + np.concatenate([-x2, x1]) * s


def main():
    hf, out_dir = sys.argv[1:3]
    os.makedirs(out_dir, exist_ok=True)
    cfg = json.load(open(os.path.join(hf, "config.json"))); build = json.load(open(os.path.join(hf, "build.json")))
    W = st_read(os.path.join(hf, "model.safetensors"))
    d, L, nh, nkv, V = cfg["hidden_size"], cfg["num_hidden_layers"], cfg["num_attention_heads"], cfg["num_key_value_heads"], cfg["vocab_size"]
    hd = d // nh; rep = nh // nkv; eps = cfg["rms_norm_eps"]; theta = float((cfg.get("rope_parameters") or {}).get("rope_theta", cfg.get("rope_theta", 10000.0)))
    E = W["model.embed_tokens.weight"][0]; g_f = W["model.norm.weight"][0]
    N = rms(E, eps)                                                        # (V, d): the normalised embedding of every symbol
    inst = BASE + "instance/" + W["model.embed_tokens.weight"][1][:16]
    nt = os.path.join(out_dir, "instance.nt"); n_tr = 0
    with open(nt, "w", encoding="utf-8") as f:
        def tr(s, p, o):
            nonlocal n_tr
            f.write(f"<{s}> <{p}> {o} .\n"); n_tr += 1
        tr(inst, P + "openTerm", lit_s("llama-prenorm-rmsnorm-gqa-rope-swiglu-tied"))
        for k in ("hidden_size", "num_hidden_layers", "num_attention_heads", "num_key_value_heads", "intermediate_size", "vocab_size"):
            tr(inst, P + k, lit_i(cfg[k]))
        tr(inst, P + "rms_norm_eps", lit_d(eps)); tr(inst, P + "rope_theta", lit_d(theta))
        for name, (arr, sha, meta) in sorted(W.items()):
            t = inst + "/tensor/" + name
            tr(inst, P + "tensor", f"<{t}>"); tr(t, P + "name", lit_s(name)); tr(t, P + "shape", lit_s(",".join(map(str, meta["shape"]))))
            tr(t, P + "dtype", lit_s(meta["dtype"])); tr(t, P + "bytes", lit_i(meta["data_offsets"][1] - meta["data_offsets"][0])); tr(t, P + "sha256", lit_s(sha))
        for i, piece in enumerate(build["pieces"]):
            tr(inst + f"/symbol/{i}", P + "piece", lit_s(piece))
        for k in range(L):
            p = f"model.layers.{k}."
            g_in = W[p + "input_layernorm.weight"][0]; g_post = W[p + "post_attention_layernorm.weight"][0]
            Wg, Wu, Wd = W[p + "mlp.gate_proj.weight"][0], W[p + "mlp.up_proj.weight"][0], W[p + "mlp.down_proj.weight"][0]
            Wq, Wk, Wv, Wo = (W[p + "self_attn.q_proj.weight"][0], W[p + "self_attn.k_proj.weight"][0], W[p + "self_attn.v_proj.weight"][0], W[p + "self_attn.o_proj.weight"][0])
            key_gate = N @ (Wg * g_post).T; key_up = N @ (Wu * g_post).T          # (V, dff)
            value = E @ (Wd * g_f[:, None])                                        # (V, dff): logit contribution of unit j per symbol
            kind = "exactAtLength1" if k == 0 else "staticProjection"
            for j in range(Wg.shape[0]):
                u = inst + f"/unit/b{k+1}/u{j}"
                tr(u, P + "block", lit_i(k + 1)); tr(u, P + "index", lit_i(j)); tr(u, P + "readoutKind", lit_s(kind))
                for s in range(V):
                    tr(u, P + f"gateKey/sym/{s}", lit_d(key_gate[s, j])); tr(u, P + f"upKey/sym/{s}", lit_d(key_up[s, j]))
                    tr(u, P + f"value/sym/{s}", lit_d(value[s, j]))
            H = N * g_in                                                           # (V, d)
            for h in range(nh):
                hh = inst + f"/head/b{k+1}/h{h}"; kv = h // rep
                tr(hh, P + "block", lit_i(k + 1)); tr(hh, P + "index", lit_i(h)); tr(hh, P + "kvGroup", lit_i(kv)); tr(hh, P + "readoutKind", lit_s(kind))
                Q = H @ Wq[h * hd:(h + 1) * hd].T; K = H @ Wk[kv * hd:(kv + 1) * hd].T; Vv = H @ Wv[kv * hd:(kv + 1) * hd].T   # (V, hd)
                K0 = np.stack([rope(K[t], 0, theta, hd) for t in range(V)])
                for off in range(4):
                    Qd = np.stack([rope(Q[s], off, theta, hd) for s in range(V)])
                    S = Qd @ K0.T / np.sqrt(hd)
                    for s in range(V):
                        for t in range(V):
                            tr(hh, P + f"qk/off/{off}/q/{s}/k/{t}", lit_d(S[s, t]))
                OV = E @ ((Wo[:, h * hd:(h + 1) * hd] @ Vv.T) * g_f[:, None])     # (V_out, V_key)
                for t in range(V):
                    for o in range(V):
                        tr(hh, P + f"ov/key/{t}/out/{o}", lit_d(OV[o, t]))
    # checks with Oxigraph
    import pyoxigraph as ox
    store = ox.Store()
    store.bulk_load(path=nt, format=ox.RdfFormat.N_TRIPLES)
    count = sum(1 for _ in store)
    blank = bool(store.query("ASK { ?s ?p ?o FILTER(isBlank(?s) || isBlank(?o)) }"))
    sha_ok = True
    for name, (arr, sha, meta) in W.items():
        q = f'ASK {{ <{inst}/tensor/{name}> <{P}sha256> {lit_s(sha)} }}'
        sha_ok = sha_ok and bool(store.query(q))
    nt_bytes = os.path.getsize(nt); st_bytes = os.path.getsize(os.path.join(hf, "model.safetensors"))
    # round trip: serialise from the store and reload, counts must agree
    rt = os.path.join(out_dir, "instance.roundtrip.nt")
    with open(rt, "wb") as f:
        store.dump(f, ox.RdfFormat.N_TRIPLES)
    store2 = ox.Store(); store2.bulk_load(path=rt, format=ox.RdfFormat.N_TRIPLES); count2 = sum(1 for _ in store2)
    summary = {"triples_written": n_tr, "triples_loaded": count, "triples_after_round_trip": count2, "blank_nodes": blank,
               "tensor_sha256_in_graph_equal_bytes": sha_ok, "nt_bytes": nt_bytes, "safetensors_bytes": st_bytes, "size_ratio": nt_bytes / st_bytes,
               "units": L * Wg.shape[0], "heads": L * nh, "pyoxigraph": ox.__version__ if hasattr(ox, "__version__") else "?"}
    json.dump(summary, open(os.path.join(out_dir, "decompile.json"), "w"), indent=1)
    os.remove(rt)
    print(summary)


if __name__ == "__main__":
    main()
