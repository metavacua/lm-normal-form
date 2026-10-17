# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The sub-terms of a Llama-style checkpoint, made by deletion only: prefixes of the block stack, single-block
deletions, single key-value-group deletions (a group is the key-value head and the query heads that read it, so the
grouped-query structure stays uniform), and FFN unit thinning by the product of row and column norms. No weight is
rescaled, written or trained. Prefixes and block deletions are homogeneous and can be saved as standard checkpoints;
group deletions and thinning change one block's shapes and are evaluated in process.
"""
import copy
import os

import torch
import torch.nn as nn

from transformers import AutoConfig, AutoModelForCausalLM


def load(model_dir):
    cfg = AutoConfig.from_pretrained(model_dir)
    cfg._attn_implementation = "eager"; cfg.use_cache = False
    m = AutoModelForCausalLM.from_pretrained(model_dir, config=cfg, dtype=torch.float32, attn_implementation="eager").eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


class View:
    """A sub-term as a view on the loaded model: swaps the block list (and, for in-block deletions, one block) and
    restores on exit. Nothing is copied except the one block that is sliced."""

    def __init__(self, model, layers, note):
        self.model, self.layers, self.note = model, layers, note
        self._orig_layers = model.model.layers; self._orig_n = model.config.num_hidden_layers

    def __enter__(self):
        self.model.model.layers = nn.ModuleList(self.layers); self.model.config.num_hidden_layers = len(self.layers)
        return self.model

    def __exit__(self, *a):
        self.model.model.layers = self._orig_layers; self.model.config.num_hidden_layers = self._orig_n


def prefix(model, k):
    return View(model, list(model.model.layers[:k]), f"P{k}")


def delete_block(model, j):
    L = list(model.model.layers)
    return View(model, L[:j - 1] + L[j:], f"D{j}")


def _slice_linear(lin, rows=None, cols=None):
    W = lin.weight.data
    if rows is not None:
        W = W[rows]
    if cols is not None:
        W = W[:, cols]
    new = nn.Linear(W.shape[1], W.shape[0], bias=lin.bias is not None)
    new.weight.data = W.clone()
    if lin.bias is not None:
        new.bias.data = (lin.bias.data[rows] if rows is not None else lin.bias.data).clone()
    return new


def delete_group(model, j, g):
    """Block j (1-based), key-value group g: remove the kv head g and the query heads that read it."""
    cfg = model.config; nh, nkv = cfg.num_attention_heads, cfg.num_key_value_heads; hd = cfg.hidden_size // nh; rep = nh // nkv
    blk = copy.deepcopy(model.model.layers[j - 1]); att = blk.self_attn
    q_keep = [i for i in range(nh * hd) if not (g * rep * hd <= i < (g + 1) * rep * hd)]
    kv_keep = [i for i in range(nkv * hd) if not (g * hd <= i < (g + 1) * hd)]
    att.q_proj = _slice_linear(att.q_proj, rows=q_keep)
    att.k_proj = _slice_linear(att.k_proj, rows=kv_keep)
    att.v_proj = _slice_linear(att.v_proj, rows=kv_keep)
    att.o_proj = _slice_linear(att.o_proj, cols=q_keep)
    # the grouping ratio is unchanged (rep query heads per kv head); the head count is inferred from the shapes
    L = list(model.model.layers); L[j - 1] = blk
    return View(model, L, f"G{j}.{g}")


def thin_ffn(model, j, frac):
    """Block j: remove the fraction `frac` of units with the smallest ||up row|| * ||down column||."""
    blk = copy.deepcopy(model.model.layers[j - 1]); mlp = blk.mlp
    score = mlp.up_proj.weight.data.norm(dim=1) * mlp.down_proj.weight.data.norm(dim=0)
    n = score.numel(); keep_n = n - int(round(frac * n))
    keep = torch.argsort(score, descending=True)[:keep_n].sort().values.tolist()
    mlp.gate_proj = _slice_linear(mlp.gate_proj, rows=keep)
    mlp.up_proj = _slice_linear(mlp.up_proj, rows=keep)
    mlp.down_proj = _slice_linear(mlp.down_proj, cols=keep)
    L = list(model.model.layers); L[j - 1] = blk
    return View(model, L, f"U{j}.{frac}")


def save_homogeneous(model, view, out_dir):
    """Save a prefix or block deletion as a standard checkpoint (the config stays homogeneous)."""
    with view as m:
        m.save_pretrained(out_dir, safe_serialization=True)


def copy_tokenizer_files(src_dir, dst_dir):
    """Copy the checkpoint's own tokenizer files unchanged (tokenizer.json, tokenizer_config.json, vocab.json, merges.txt,
    special_tokens_map.json, generation_config.json): re-saving SmolLM2's tokenizer with transformers 5.20 writes
    extra_special_tokens as a list, which its own loader then rejects."""
    import shutil
    for f in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "special_tokens_map.json", "generation_config.json"):
        if os.path.exists(os.path.join(src_dir, f)):
            shutil.copy(os.path.join(src_dir, f), os.path.join(dst_dir, f))
