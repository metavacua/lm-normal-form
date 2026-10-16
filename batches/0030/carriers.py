# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cell 01, part 2: the identity of the tensors across the three carriers of the instance.

usage: carriers.py HF_DIR ONNX_DIR GGUF_FILE OUT_JSON
For every tensor: its name, shape and dtype in safetensors, in the ONNX export's initializers and in the GGUF, and
whether the GGUF payload bytes equal the safetensors bytes (llama.cpp's converter permutes q and k for Llama's
rotary layout, and may transpose or cast; whatever it did is recorded, not assumed). Nothing here is a claim; it is
the inventory that the registration's Q1 is graded on, plus what the bytes say.
"""
import hashlib
import json
import struct
import sys

import numpy as np


def st_tensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            a, b = meta["data_offsets"]
            f.seek(base + a); raw = f.read(b - a)
            out[name] = {"shape": meta["shape"], "dtype": meta["dtype"], "bytes": b - a, "sha256": hashlib.sha256(raw).hexdigest(),
                         "raw": raw}
        return out


def onnx_inits(path):
    import onnx
    m = onnx.load(path, load_external_data=True)
    out = {}
    for t in m.graph.initializer:
        arr = onnx.numpy_helper.to_array(t)
        out[t.name] = {"shape": list(arr.shape), "dtype": str(arr.dtype), "sha256": hashlib.sha256(np.ascontiguousarray(arr).tobytes()).hexdigest()}
    return out, {"ir_version": m.ir_version, "opsets": {o.domain or "ai.onnx": o.version for o in m.opset_import},
                 "nodes": len(m.graph.node), "inputs": [i.name for i in m.graph.input], "outputs": [o.name for o in m.graph.output]}


def gguf_tensors(path):
    from gguf import GGUFReader
    r = GGUFReader(path)
    out = {}
    for t in r.tensors:
        raw = np.ascontiguousarray(t.data).tobytes()
        out[t.name] = {"shape": [int(x) for x in t.shape], "dtype": str(t.tensor_type.name), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
    kv = {}
    for k, f in r.fields.items():
        if k.startswith("tokenizer.ggml.tokens") or k.startswith("tokenizer.ggml.scores") or k.startswith("tokenizer.ggml.token_type"):
            kv[k] = f"<{len(f.data)} entries>"
        else:
            try:
                v = f.parts[f.data[0]]
                kv[k] = v.tobytes().decode("utf-8", "replace") if v.dtype == np.uint8 else v.tolist()
            except Exception:
                kv[k] = "?"
    return out, kv


def main():
    hf, onnx_dir, gguf, out_json = sys.argv[1:5]
    st = st_tensors(f"{hf}/model.safetensors")
    on, on_meta = onnx_inits(f"{onnx_dir}/model.onnx")
    gg, gg_kv = gguf_tensors(gguf)
    # name maps: HF -> GGUF (llama.cpp's naming), HF -> ONNX (export keeps HF names, sometimes with a prefix)
    def gg_name(h):
        h = h.replace("model.embed_tokens.weight", "token_embd.weight").replace("model.norm.weight", "output_norm.weight").replace("lm_head.weight", "output.weight")
        h = h.replace("model.layers.", "blk.").replace("input_layernorm", "attn_norm").replace("post_attention_layernorm", "ffn_norm")
        h = h.replace("self_attn.q_proj", "attn_q").replace("self_attn.k_proj", "attn_k").replace("self_attn.v_proj", "attn_v").replace("self_attn.o_proj", "attn_output")
        return h.replace("mlp.gate_proj", "ffn_gate").replace("mlp.up_proj", "ffn_up").replace("mlp.down_proj", "ffn_down")
    rows = []
    for name, m in st.items():
        g = gg.get(gg_name(name)); o = on.get(name) or next((on[k] for k in on if k.endswith(name)), None)
        f32 = np.frombuffer(m["raw"], dtype=np.float32) if m["dtype"] == "F32" else None
        same_gguf = bool(g and g["sha256"] == m["sha256"])
        rows.append({"hf": name, "shape": m["shape"], "dtype": m["dtype"], "sha256": m["sha256"],
                     "gguf": gg_name(name) if g else None, "gguf_dtype": g["dtype"] if g else None, "gguf_bytes_identical": same_gguf,
                     "onnx_present": o is not None, "onnx_bytes_identical": bool(o and o["sha256"] == m["sha256"]),
                     "finite": bool(np.isfinite(f32).all()) if f32 is not None else None})
    summary = {"safetensors_tensors": len(st), "onnx_initializers": len(on), "gguf_tensors": len(gg), "onnx": on_meta,
               "gguf_kv": gg_kv, "tensors": rows,
               "gguf_identical": sum(r["gguf_bytes_identical"] for r in rows), "onnx_identical": sum(r["onnx_bytes_identical"] for r in rows),
               "gguf_not_identical": [r["hf"] for r in rows if not r["gguf_bytes_identical"]],
               "onnx_not_identical": [r["hf"] for r in rows if not r["onnx_bytes_identical"]]}
    json.dump(summary, open(out_json, "w"), indent=1)
    print(f"safetensors {len(st)} tensors; onnx initializers {len(on)} ({summary['onnx_identical']} byte-identical); gguf {len(gg)} "
          f"({summary['gguf_identical']} byte-identical; not: {summary['gguf_not_identical']})")


if __name__ == "__main__":
    main()
