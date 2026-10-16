# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What the cells of batch 0029 share: loading the model in float32 with eager attention, the taps on the residual
stream, the gate pre-activations and the logits, the bf16 ceiling, JSON written one way, and the hash of a row."""

import hashlib
import json
import os
import resource
import time

import numpy as np
import torch

BF16_MAX = 3.3895313892515355e38          # the largest finite bfloat16 value
THREADS = int(os.environ.get("LMNF_THREADS", "4"))
torch.set_num_threads(THREADS)


def dump(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=_jsonable)
        f.write("\n")


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o).__name__)


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def peak_rss_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


class Clock:
    def __init__(self):
        self.t0 = time.time()

    def s(self):
        return round(time.time() - self.t0, 1)


def load_model(model_dir):
    """The model in float32 on the CPU, eager attention so that the softmax is an explicit node, no cache."""
    from transformers import AutoConfig, AutoModelForCausalLM
    cfg = AutoConfig.from_pretrained(model_dir)
    cfg._attn_implementation = "eager"
    cfg.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(model_dir, config=cfg, torch_dtype=torch.float32, attn_implementation="eager")
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def model_facts(model):
    cfg = model.config
    return {
        "hidden_size": cfg.hidden_size, "num_hidden_layers": cfg.num_hidden_layers, "num_attention_heads": cfg.num_attention_heads,
        "num_key_value_heads": cfg.num_key_value_heads, "intermediate_size": cfg.intermediate_size, "vocab_size": cfg.vocab_size,
        "rms_norm_eps": cfg.rms_norm_eps,
        "rope_theta": (getattr(cfg, "rope_parameters", None) or {}).get("rope_theta", getattr(cfg, "rope_theta", None)),
        "tie_word_embeddings": cfg.tie_word_embeddings,
        "head_is_embedding": bool(model.lm_head.weight.data_ptr() == model.model.embed_tokens.weight.data_ptr()),
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "tensors": int(len(list(model.state_dict().keys()))),
    }


class Taps:
    """Hooks that keep the coordinate-wise minimum and maximum of each tapped activation over everything that passes.
    Taps: resid.0 (after the embedding), resid.k (after block k, k = 1..L), gate.k (the input of the SiLU of block k),
    final_norm, logits. Each tap is reduced over the batch and position axes and kept as two float64 vectors."""

    def __init__(self, model):
        self.model = model
        self.lo, self.hi = {}, {}
        self.handles = []
        layers = model.model.layers
        self.handles.append(model.model.embed_tokens.register_forward_hook(self._hook("resid.0")))
        for k, layer in enumerate(layers, 1):
            self.handles.append(layer.register_forward_hook(self._hook(f"resid.{k}", first=True)))
            self.handles.append(layer.mlp.act_fn.register_forward_pre_hook(self._prehook(f"gate.{k}")))
        self.handles.append(model.model.norm.register_forward_hook(self._hook("final_norm")))
        self.handles.append(model.lm_head.register_forward_hook(self._hook("logits")))

    def _update(self, name, t):
        t = t.detach()
        d = t.shape[-1]
        flat = t.reshape(-1, d).to(torch.float64)
        lo = flat.min(dim=0).values.numpy()
        hi = flat.max(dim=0).values.numpy()
        if name in self.lo:
            np.minimum(self.lo[name], lo, out=self.lo[name])
            np.maximum(self.hi[name], hi, out=self.hi[name])
        else:
            self.lo[name], self.hi[name] = lo.copy(), hi.copy()

    def _hook(self, name, first=False):
        def f(module, inputs, output):
            out = output[0] if (first and isinstance(output, (tuple, list))) else output
            self._update(name, out)
        return f

    def _prehook(self, name):
        def f(module, inputs):
            self._update(name, inputs[0])
        return f

    def remove(self):
        for h in self.handles:
            h.remove()

    def summary(self):
        """Per tap: the global extremum, the largest magnitude, the slack in doublings to the bf16 ceiling."""
        out = {}
        for name in self.lo:
            lo, hi = self.lo[name], self.hi[name]
            amax = float(max(np.abs(lo).max(), np.abs(hi).max()))
            out[name] = {"min": float(lo.min()), "max": float(hi.max()), "abs_max": amax,
                         "finite": bool(np.isfinite(lo).all() and np.isfinite(hi).all()),
                         "slack_doublings": (float(np.log2(BF16_MAX / amax)) if amax > 0 and np.isfinite(amax) else None),
                         "width": int(lo.shape[0])}
        return out

    def vectors(self):
        return {name: {"lo": self.lo[name].tolist(), "hi": self.hi[name].tolist()} for name in self.lo}


def row_sha256(row_f32):
    return hashlib.sha256(np.ascontiguousarray(row_f32, dtype=np.float32).tobytes()).hexdigest()


def entropy_bits(logits_f32):
    """Entropy in bits of softmax(logits), rows of a 2-D float32 array, computed in float64."""
    z = logits_f32.astype(np.float64)
    z = z - z.max(axis=1, keepdims=True)
    p = np.exp(z)
    p /= p.sum(axis=1, keepdims=True)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = -(p * np.where(p > 0, np.log2(p), 0.0)).sum(axis=1)
    return h, p
