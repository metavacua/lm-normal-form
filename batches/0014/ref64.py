# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The float64 reference of batch 0014: the forward pass of a Llama-style decoder (pre-norm RMSNorm; grouped-query attention with rotary embeddings in the rotate-half
# pairing and a causal mask; a gated SiLU feed-forward; a tied or untied head), written out in numpy in float64, reading the weights and the configuration of a
# Hugging Face directory. It is the sixth implementation of the batch, and the one the others are measured against: Transformers' float64 run is not a float64 reference
# (its RMSNorm, rotary tables and eager softmax compute in float32 whatever the dtype of the model), and every float32 runtime differs from the true value of the
# function by rounding error that no other float32 runtime can measure. The weights of a bfloat16 or float16 file are read exactly.
#   ref64.py logits WORK                 the logits of every position of the test set, from WORK/variants/orig: WORK/ref64.npy (float64)
#   ref64.py margins RUNTIME DTYPE WORK  top-1 minus top-2 logit, in float64, at every step of the greedy generations that RUNTIME kept for the original (res/<runtime>-<dtype>-orig-r0.json):
#                                        WORK/res/margins-RUNTIME-DTYPE.json. A variant's generation can first differ from the original's only at such a step.
# As a library: Ref(model_dir).forward(ids, kv=False) -> logits (T, V) float64, and with kv=True also the cache: a list over layers of (K, V), each (n_kv, T, head_dim),
# K after the rotary embedding, exactly what a runtime stores. The cache cell (cache.py) uses it.
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import dump


def read_weights(path):
    """All tensors of a safetensors file as float64 numpy arrays (bfloat16 and float16 are read exactly)."""
    try:
        from safetensors.numpy import load_file
        return {k: v.astype(np.float64) for k, v in load_file(path).items()}
    except Exception:                                     # numpy has no bfloat16: read through torch, convert, keep numpy
        import torch
        from safetensors.torch import load_file
        return {k: v.to(torch.float64).numpy() for k, v in load_file(path).items()}


class Ref:
    def __init__(self, model_dir):
        cfg = json.load(open(os.path.join(model_dir, "config.json")))
        assert not cfg.get("rope_scaling") and not cfg.get("attention_bias") and not cfg.get("mlp_bias"), "not a plain Llama"
        self.W = read_weights(os.path.join(model_dir, "model.safetensors"))
        self.L = cfg["num_hidden_layers"]
        self.nh = cfg["num_attention_heads"]
        self.nkv = cfg.get("num_key_value_heads") or self.nh
        self.hd = cfg.get("head_dim") or cfg["hidden_size"] // self.nh
        self.eps = float(cfg["rms_norm_eps"])
        theta = cfg.get("rope_theta") or (cfg.get("rope_parameters") or {}).get("rope_theta") or 10000.0
        self.inv = 1.0 / (float(theta) ** (np.arange(0, self.hd, 2, dtype=np.float64) / self.hd))

    def rms(self, x, w):
        return w * (x / np.sqrt((x * x).mean(-1, keepdims=True) + self.eps))

    def rope(self, x, cos, sin):                          # x (heads, T, hd): x cos + rotate_half(x) sin
        h = self.hd // 2
        return x * cos + np.concatenate([-x[..., h:], x[..., :h]], -1) * sin

    def forward(self, ids, kv=False):
        W, nh, nkv, hd = self.W, self.nh, self.nkv, self.hd
        T = len(ids)
        x = W["model.embed_tokens.weight"][np.asarray(ids)]
        ang = np.arange(T, dtype=np.float64)[:, None] * self.inv[None, :]
        emb = np.concatenate([ang, ang], -1)
        cos, sin = np.cos(emb), np.sin(emb)
        mask = np.triu(np.full((T, T), -np.inf), 1)
        cache = []
        for l in range(self.L):
            p = f"model.layers.{l}."
            h = self.rms(x, W[p + "input_layernorm.weight"])
            q = (h @ W[p + "self_attn.q_proj.weight"].T).reshape(T, nh, hd).transpose(1, 0, 2)
            k = (h @ W[p + "self_attn.k_proj.weight"].T).reshape(T, nkv, hd).transpose(1, 0, 2)
            v = (h @ W[p + "self_attn.v_proj.weight"].T).reshape(T, nkv, hd).transpose(1, 0, 2)
            q, k = self.rope(q, cos, sin), self.rope(k, cos, sin)
            if kv:
                cache.append((k, v))
            rep = nh // nkv
            kk, vv = np.repeat(k, rep, axis=0), np.repeat(v, rep, axis=0)
            s = q @ kk.transpose(0, 2, 1) / np.sqrt(hd) + mask
            s = np.exp(s - s.max(-1, keepdims=True))
            a = (s / s.sum(-1, keepdims=True)) @ vv      # (nh, T, hd)
            x = x + a.transpose(1, 0, 2).reshape(T, nh * hd) @ W[p + "self_attn.o_proj.weight"].T
            h = self.rms(x, W[p + "post_attention_layernorm.weight"])
            g, u = h @ W[p + "mlp.gate_proj.weight"].T, h @ W[p + "mlp.up_proj.weight"].T
            x = x + (g / (1.0 + np.exp(-g)) * u) @ W[p + "mlp.down_proj.weight"].T
        x = self.rms(x, W["model.norm.weight"])
        logits = x @ W.get("lm_head.weight", W["model.embed_tokens.weight"]).T
        return (logits, cache) if kv else logits


def load_inputs(work):
    j = json.load(open(os.path.join(work, "inputs.json")))
    return j["ids"], j["wins"]


def main(argv):
    if argv[0] == "logits":
        work = argv[1]
        ids, _ = load_inputs(work)
        ref = Ref(os.path.join(work, "variants", "orig"))
        np.save(os.path.join(work, "ref64.npy"), np.concatenate([ref.forward(s) for s in ids["test"]]))
        print("ref64: logits of", sum(len(s) for s in ids["test"]), "positions")
    elif argv[0] == "margins":
        runtime, dtype, work = argv[1:4]
        gen = json.load(open(os.path.join(work, "res", f"{runtime}-{dtype}-orig-r0.json"))).get("gen")
        if not gen:
            return
        ids, _ = load_inputs(work)
        ref = Ref(os.path.join(work, "variants", "orig"))
        out = {}
        for kind, gens in gen.items():
            out[kind] = []
            for p, g in zip(ids[kind], gens):
                if not g:
                    out[kind].append([])
                    continue
                lg = ref.forward(list(p) + list(g[:-1]))[len(p) - 1:len(p) - 1 + len(g)]
                top = np.sort(lg, axis=-1)[:, -2:]
                out[kind].append((top[:, 1] - top[:, 0]).tolist())
        dump(out, os.path.join(work, "res", f"margins-{runtime}-{dtype}.json"))
        print("margins:", {k: sum(len(m) for m in v) for k, v in out.items()}, "steps")


if __name__ == "__main__":
    main(sys.argv[1:])
