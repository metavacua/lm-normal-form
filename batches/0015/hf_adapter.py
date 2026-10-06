# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The bridge between the parameters of model.py and a Hugging Face Llama checkpoint: the architecture from config.json, the tensors from model.safetensors (read one at a
# time as float64 numpy arrays, or float32), and back to a directory that Transformers loads. The names: embed <-> model.embed_tokens.weight, lm_head <-> lm_head.weight
# (absent if tied), norm_f <-> model.norm.weight, and per layer attn_norm <-> input_layernorm.weight, wq wk wv wo <-> self_attn.{q,k,v,o}_proj.weight, mlp_norm <->
# post_attention_layernorm.weight, wg wu wd <-> mlp.{gate,up,down}_proj.weight.
import json, os
import numpy as np
from model import Arch, shapes

HF = {"attn_norm": "input_layernorm.weight", "wq": "self_attn.q_proj.weight", "wk": "self_attn.k_proj.weight", "wv": "self_attn.v_proj.weight", "wo": "self_attn.o_proj.weight",
      "mlp_norm": "post_attention_layernorm.weight", "wg": "mlp.gate_proj.weight", "wu": "mlp.up_proj.weight", "wd": "mlp.down_proj.weight"}


def arch_of(cfg):
    d, nh = cfg["hidden_size"], cfg["num_attention_heads"]
    return Arch(d=d, n_heads=nh, n_kv=cfg.get("num_key_value_heads") or nh, hd=cfg.get("head_dim") or d // nh, d_ff=cfg["intermediate_size"],
                n_layers=cfg["num_hidden_layers"], vocab=cfg["vocab_size"], tied=bool(cfg.get("tie_word_embeddings")), pos="rope", causal=True, norm="rms")


def hf_name(k):
    if k == "embed":
        return "model.embed_tokens.weight"
    if k == "lm_head":
        return "lm_head.weight"
    if k == "norm_f":
        return "model.norm.weight"
    i, n = k[1:].split(".", 1)
    return f"model.layers.{i}.{HF[n]}"


def load(model_dir, dtype=np.float64):
    """(arch, params, config) from a directory with config.json and model.safetensors."""
    import ml_dtypes
    BF16 = ml_dtypes.bfloat16      # importing ml_dtypes registers bfloat16 with numpy, which a bfloat16 file needs
    del BF16
    from safetensors import safe_open
    cfg = json.load(open(os.path.join(model_dir, "config.json")))
    a = arch_of(cfg)
    P = {}
    with safe_open(os.path.join(model_dir, "model.safetensors"), framework="np") as f:
        for k, shp in shapes(a).items():
            t = f.get_tensor(hf_name(k)).astype(dtype)
            assert t.shape == tuple(shp), (k, t.shape, shp)
            P[k] = t
    return a, P, cfg


def save(arch, P, src_dir, out_dir, dtype="float32"):
    """Write P as model.safetensors of `dtype` next to the config of src_dir (its tie_word_embeddings set to the architecture's) and the tokenizer files."""
    import shutil
    import torch
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(src_dir):
        if f != "model.safetensors" and os.path.isfile(os.path.join(src_dir, f)):
            shutil.copy(os.path.join(src_dir, f), os.path.join(out_dir, f))
    tdt = {"float64": torch.float64, "float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[dtype]
    keep = np.float64 if dtype == "float64" else np.float32
    save_file({hf_name(k): torch.from_numpy(np.ascontiguousarray(v.astype(keep))).to(tdt) for k, v in P.items()}, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    cfg = json.load(open(os.path.join(out_dir, "config.json")))
    cfg["tie_word_embeddings"] = bool(arch.tied)
    cfg["torch_dtype"] = cfg["dtype"] = dtype
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2)


def params_of(model, arch):
    """The parameters of a Hugging Face Llama (torch) as float64 numpy arrays under the names of model.py."""
    sd = model.state_dict()
    return {k: sd[hf_name(k)].detach().double().numpy().copy() for k in shapes(arch)}


def float64_patches(cfg):
    """Replace the two float32 islands of Hugging Face's Llama, the RMSNorm (which casts to float32 inside) and the rotary tables (computed from a float32 inverse frequency), by float64 versions,
    so that a float64 run is a float64 run. Returns the function that restores the originals. (The eager attention's softmax is a third island; the default SDPA attention has none.)"""
    import torch
    from transformers.models.llama import modeling_llama as M
    theta = float(cfg.get("rope_theta") or (cfg.get("rope_parameters") or {}).get("rope_theta") or 10000.0)

    def norm64(self, h):
        return self.weight * (h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))

    def rope64(self, x, position_ids):
        dim = self.inv_freq.shape[0] * 2
        inv = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float64) / dim))
        freqs = position_ids[:, :, None].to(torch.float64) * inv[None, None, :]
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos().to(x.dtype), emb.sin().to(x.dtype)

    saved = M.LlamaRMSNorm.forward, M.LlamaRotaryEmbedding.forward
    M.LlamaRMSNorm.forward, M.LlamaRotaryEmbedding.forward = norm64, rope64

    def restore():
        M.LlamaRMSNorm.forward, M.LlamaRotaryEmbedding.forward = saved
    return restore
