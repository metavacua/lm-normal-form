# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What the cells of batch 0033 share: the model, the 189 facts and their margins, exact gradients over every weight,
selection of a support of coordinates, application of a change in bf16, the evaluation of a state (count, bits per
token), and JSON written one way. The path is computed by PyTorch autograd; nothing here is a model of ours."""

import json
import math
import os
import time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
torch.set_num_threads(int(os.environ.get("LMNF_THREADS", "4")))
EPS = 0.5
T12 = ["France", "Germany", "Italy", "Spain", "Japan", "China", "Russia", "Egypt", "Canada", "Brazil", "India", "Australia"]


def dump(path, obj):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True, default=lambda o: o.item() if hasattr(o, "item") else str(o))
        f.write("\n")


def load(model_dir):
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
    cfg = AutoConfig.from_pretrained(model_dir)
    cfg._attn_implementation = "eager"; cfg.use_cache = False
    model = AutoModelForCausalLM.from_pretrained(model_dir, config=cfg, dtype=torch.float32, attn_implementation="eager").eval()
    for p in model.parameters():
        p.requires_grad_(True)
    return model, AutoTokenizer.from_pretrained(model_dir)


def atoms():
    d = json.load(open(os.path.join(HERE, "..", "0032", "countries.json")))
    return [(x["countryLabel"], x["canonical"][0]["capitalLabel"]) for x in d if x["determined"]]


def target_ids(tok, atoms_):
    return [tok(" " + C, add_special_tokens=False)["input_ids"][0] for _, C in atoms_]


def prompt_ids(tok, X):
    return tok(f"The capital of {X} is", return_tensors="pt")["input_ids"]


def margin_of(z, t):
    o = z.clone(); o[t] = float("-inf")
    v, j = o.max(0)
    return z[t] - v, int(j)


@torch.no_grad()
def all_margins(model, tok, atoms_, targs, targets_override=None):
    ms, top1 = [], []
    for k, ((X, _), t) in enumerate(zip(atoms_, targs)):
        z = model(input_ids=prompt_ids(tok, X), use_cache=False).logits[0, -1]
        tt = targets_override[k] if targets_override is not None else t
        m, _ = margin_of(z, tt); ms.append(float(m)); top1.append(int(z.argmax()))
    return np.array(ms), np.array(top1)


def margin_and_grad(model, tok, X, t):
    """The margin of fact (X, t) and its exact gradient, left in p.grad of every parameter."""
    for p in model.parameters():
        p.grad = None
    z = model(input_ids=prompt_ids(tok, X), use_cache=False).logits[0, -1]
    m, _ = margin_of(z, t)
    m.backward()
    return float(m)


def params(model):
    return list(model.named_parameters())


def select_support(model, tok, facts, N):
    """facts: list of (X, t). Returns {tensor index: sorted flat indices} of the N coordinates with the largest
    max-over-facts |gradient| at the current weights."""
    ps = params(model)
    runmax = [torch.zeros_like(p) for _, p in ps]
    for X, t in facts:
        margin_and_grad(model, tok, X, t)
        for (n, p), r in zip(ps, runmax):
            torch.maximum(r, p.grad.abs(), out=r)
    cv, ct, ci = [], [], []
    for ti, r in enumerate(runmax):
        k = min(N, r.numel())
        v, i = torch.topk(r.view(-1), k)
        cv.append(v); ct.append(torch.full_like(i, ti)); ci.append(i)
    V, T, I = torch.cat(cv), torch.cat(ct), torch.cat(ci)
    top = torch.topk(V, N).indices
    T, I = T[top], I[top]
    for p in model.parameters():
        p.grad = None
    return {int(ti): I[T == ti].sort().values for ti in T.unique().tolist()}


def grad_vec(model, idx):
    ps = params(model)
    return torch.cat([ps[ti][1].grad.view(-1)[idx[ti]] for ti in sorted(idx)]).double()


def get_vals(model, idx):
    ps = params(model)
    return torch.cat([ps[ti][1].data.view(-1)[idx[ti]] for ti in sorted(idx)]).clone()


@torch.no_grad()
def set_vals(model, idx, vals):
    ps = params(model); off = 0
    for ti in sorted(idx):
        n = len(idx[ti]); ps[ti][1].data.view(-1)[idx[ti]] = vals[off:off + n]; off += n


@torch.no_grad()
def apply_delta(model, idx, delta):
    """Adds delta to the support, rounds each changed coordinate to bf16, returns the applied (rounded) change."""
    ps = params(model); off = 0; out = []
    for ti in sorted(idx):
        n = len(idx[ti]); flat = ps[ti][1].data.view(-1); old = flat[idx[ti]].clone()
        new = (old + delta[off:off + n].to(torch.float32)).to(torch.bfloat16).to(torch.float32)
        flat[idx[ti]] = new; out.append(new - old); off += n
    return torch.cat(out).double()


def tensor_sort(name):
    if name == "model.embed_tokens.weight":
        return "embedding_head"
    if name == "model.norm.weight":
        return "final_norm"
    for k in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "input_layernorm", "post_attention_layernorm"):
        if k in name:
            return k
    return "other"


def tensor_layer(name):
    if ".layers." in name:
        return int(name.split(".layers.")[1].split(".")[0])
    return -1


def english_ids(tok, path, n_tokens=4096):
    try:
        text = open(path, encoding="utf-8", errors="replace").read()[5000:30000]
    except Exception:
        text = ""
    if len(text) < 2000:
        text = ("Alice was beginning to get very tired of sitting by her sister on the bank, and of having nothing to do. " * 80)
    return tok(text, add_special_tokens=False)["input_ids"][:n_tokens]


@torch.no_grad()
def bits_per_token(model, ids, chunk=256):
    nll = 0.0; n = 0
    for s in range(0, len(ids) - 1, chunk):
        c = torch.tensor([ids[s:s + chunk + 1]])
        if c.shape[1] < 2:
            break
        lp = torch.log_softmax(model(input_ids=c[:, :-1], use_cache=False).logits[0].double(), -1)
        tgt = c[0, 1:]
        nll += float(-lp[torch.arange(len(tgt)), tgt].sum()); n += len(tgt)
    return nll / n / math.log(2)


def state(model, tok, atoms_, targs, text_ids, the_id):
    ms, top1 = all_margins(model, tok, atoms_, targs)
    return {"count": int((ms > 0).sum()), "the_answers": int((np.array(top1) == the_id).sum()), "bits_per_token": bits_per_token(model, text_ids),
            "T12_correct": [X for (X, _), m in zip(atoms_, ms) if X in T12 and m > 0], "margins": ms}


class Clock:
    def __init__(self):
        self.t0 = time.time()

    def s(self):
        return round(time.time() - self.t0, 1)
