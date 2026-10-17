# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cell 14 (docs/batches/0033.md, amendment 13). Read-only.
usage: live.py MODEL_DIR NAME OUT_DIR

For the prompt "The capital of France is": every weight-times-input product of the execution that structurally feeds the last
position's output (all positions for every layer, except the last layer's query, output, post-norm and feed-forward tensors, which
feed the last position only at the last position). Counted below absolute thresholds and below relative thresholds (the product's
share of its own output row's absolute mass; 2^-24 is the float32 resolution of that row, 2^-8 the bf16 resolution). Address level:
a weight is live at a threshold if at least one of its products reaches it. Attention side: a key or value product at position p for
key/value group g is attention-dead if no head of g gives p more than a threshold of mass at the last position. Also written down:
the attention routing record (every head's distribution over the five positions at the last position) and the feed-forward units
whose activation at the last position is above thresholds."""
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import core  # noqa: E402

ABS = [1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3]
REL = [2 ** -24, 2 ** -16, 2 ** -8, 1e-2]
ATT = [1e-8, 1e-6, 1e-4, 1e-3, 1e-2]
ACT = [1e-6, 1e-4, 1e-3, 1e-2, 1e-1]


def capture(model, ids):
    L = model.config.num_hidden_layers
    c = {k: [None] * L for k in ("xin", "yin", "xmid", "ypost", "act", "oin")}
    hs = []
    for k, Ly in enumerate(model.model.layers):
        hs.append(Ly.register_forward_pre_hook(lambda m, i, k=k: c["xin"].__setitem__(k, i[0][0].detach().clone())))
        hs.append(Ly.input_layernorm.register_forward_hook(lambda m, i, o, k=k: c["yin"].__setitem__(k, o[0].detach().clone())))
        hs.append(Ly.post_attention_layernorm.register_forward_pre_hook(lambda m, i, k=k: c["xmid"].__setitem__(k, i[0][0].detach().clone())))
        hs.append(Ly.post_attention_layernorm.register_forward_hook(lambda m, i, o, k=k: c["ypost"].__setitem__(k, o[0].detach().clone())))
        hs.append(Ly.mlp.down_proj.register_forward_pre_hook(lambda m, i, k=k: c["act"].__setitem__(k, i[0][0].detach().clone())))
        hs.append(Ly.self_attn.o_proj.register_forward_pre_hook(lambda m, i, k=k: c["oin"].__setitem__(k, i[0][0].detach().clone())))
    hs.append(model.model.norm.register_forward_pre_hook(lambda m, i: c.__setitem__("xL", i[0][0].detach().clone())))
    hs.append(model.model.norm.register_forward_hook(lambda m, i, o: c.__setitem__("yL", o[0].detach().clone())))
    with torch.no_grad():
        out = model(input_ids=ids, use_cache=False, output_attentions=True)
    for h in hs:
        h.remove()
    c["probs"] = [a[0].detach().clone() for a in out.attentions]
    return c, out.logits[0, -1].detach().clone()


class Tally:
    """Counts of products and of addresses below thresholds, per tensor sort."""

    def __init__(self):
        self.p = {}; self.a = {}

    def add(self, sort, prod, row_axis=-1):
        """prod: (P, ...) products over the positions used; the output row runs along row_axis (the summed axis of the matmul)."""
        ap = prod.abs(); mass = ap.sum(dim=row_axis, keepdim=True); share = torch.where(mass > 0, ap / mass, torch.zeros_like(ap))
        n = ap.numel(); r = self.p.setdefault(sort, {"n": 0, **{f"abs<{t:g}": 0 for t in ABS}, **{f"rel<{t:g}": 0 for t in REL}})
        r["n"] += n
        for t in ABS:
            r[f"abs<{t:g}"] += int((ap < t).sum())
        for t in REL:
            r[f"rel<{t:g}"] += int((share < t).sum())
        mx = ap.amax(dim=0); ms = share.amax(dim=0); na = mx.numel(); q = self.a.setdefault(sort, {"n": 0, **{f"abs<{t:g}": 0 for t in ABS}, **{f"rel<{t:g}": 0 for t in REL}})
        q["n"] += na
        for t in ABS:
            q[f"abs<{t:g}"] += int((mx < t).sum())
        for t in REL:
            q[f"rel<{t:g}"] += int((ms < t).sum())
        return mx, ms


def main():
    model_dir, name, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
    model, tok = core.load(model_dir)
    for p in model.parameters():
        p.requires_grad_(False)
    cfg = model.config; L = cfg.num_hidden_layers; H = cfg.num_attention_heads; KV = cfg.num_key_value_heads; hd = cfg.hidden_size // H; grp = H // KV; eps_n = cfg.rms_norm_eps
    ids = core.prompt_ids(tok, "France"); T = ids.shape[1]
    c, z = capture(model, ids); o = int(z.argmax())
    E = model.model.embed_tokens.weight.data; g = model.model.norm.weight.data
    t = Tally(); att_tally = {f"att<{a:g}": {"kv_products_dead": 0, "kv_addresses_dead": 0} for a in ATT}; kv_n = {"products": 0, "addresses": 0}
    routing = []; ffn_units = []; head_pairs_below = {f"p<{a:g}": 0 for a in ATT}; head_pairs = 0
    rms = lambda x: x / torch.sqrt((x ** 2).mean(-1, keepdim=True) + eps_n)
    for k in range(L):
        Ly = model.model.layers[k]; last = (k == L - 1); P_all = list(range(T)); P_q = [T - 1] if last else P_all
        xin_hat = rms(c["xin"][k]); xmid_hat = rms(c["xmid"][k]); yin = c["yin"][k]; ypost = c["ypost"][k]; act = c["act"][k]; oin = c["oin"][k]; pr = c["probs"][k]  # pr: (H, T, T)
        t.add("input_layernorm", (Ly.input_layernorm.weight.data[None, :] * xin_hat)[P_all], row_axis=-1)
        t.add("q_proj", Ly.self_attn.q_proj.weight.data[None] * yin[P_q][:, None, :])
        # key and value: all positions; attention gating by the mass the last position's heads of each group give to the position
        m = torch.stack([pr[gi * grp:(gi + 1) * grp, T - 1, :].amax(0) for gi in range(KV)])          # (KV, T): mass from the last position to each position, max over the group's heads
        for sort, W in (("k_proj", Ly.self_attn.k_proj.weight.data), ("v_proj", Ly.self_attn.v_proj.weight.data)):
            prod = W[None] * yin[:, None, :]                                                           # (T, KV*hd, d)
            mx, _ = t.add(sort, prod)
            kv_n["products"] += prod.numel(); kv_n["addresses"] += mx.numel()
            gate_pos = m.repeat_interleave(hd, dim=0).T                                                # (T, KV*hd): the mass for each key/value row at each position
            for a in ATT:
                dead = (gate_pos < a)                                                                  # (T, rows)
                att_tally[f"att<{a:g}"]["kv_products_dead"] += int(dead.sum()) * W.shape[1]
                att_tally[f"att<{a:g}"]["kv_addresses_dead"] += int(dead.all(0).sum()) * W.shape[1]
        t.add("o_proj", Ly.self_attn.o_proj.weight.data[None] * oin[P_q][:, None, :])
        t.add("post_attention_layernorm", (Ly.post_attention_layernorm.weight.data[None, :] * xmid_hat)[P_q], row_axis=-1)
        t.add("gate_proj", Ly.mlp.gate_proj.weight.data[None] * ypost[P_q][:, None, :])
        t.add("up_proj", Ly.mlp.up_proj.weight.data[None] * ypost[P_q][:, None, :])
        t.add("down_proj", Ly.mlp.down_proj.weight.data[None] * act[P_q][:, None, :])
        # the routing record of this layer at the last position
        pl = pr[:, T - 1, :]                                                                           # (H, T)
        routing.append({"layer": k, "heads": [{"head": h, "mass": [float(x) for x in pl[h]], "argmax": int(pl[h].argmax()), "out_norm": float(oin[T - 1, h * hd:(h + 1) * hd].norm())} for h in range(H)]})
        head_pairs += H * T
        for a in ATT:
            head_pairs_below[f"p<{a:g}"] += int((pl < a).sum())
        al = act[T - 1].abs()
        ffn_units.append({"layer": k, **{f"act>={a:g}": int((al >= a).sum()) for a in ACT}, "units_act>=1e-2": torch.nonzero(al >= 1e-2).squeeze(1).tolist()})
    # the final norm and the output row of the tied head
    xL_hat = rms(c["xL"][T - 1:T]); t.add("final_norm", (g[None, :] * xL_hat), row_axis=-1)
    t.add("head_row", (E[o] * c["yL"][T - 1])[None, None, :])
    # the five embedding rows are read, not multiplied: their zero count is the static one
    rows = [int(x) for x in ids[0]]; emb_zero = int((E[rows] == 0).sum()); emb_n = len(rows) * E.shape[1]
    # the static count of the same addresses (the cross-check of cone_zeros.py, which ran on the published file)
    stat = {}
    def stat_add(sort, w):
        aw = w.abs().ravel(); r = stat.setdefault(sort, {"n": 0, "zero": 0, "lt_1e-6": 0, "lt_1e-4": 0, "lt_1e-3": 0, "lt_1e-2": 0})
        r["n"] += aw.numel(); r["zero"] += int((aw == 0).sum()); r["lt_1e-6"] += int((aw < 1e-6).sum()); r["lt_1e-4"] += int((aw < 1e-4).sum()); r["lt_1e-3"] += int((aw < 1e-3).sum()); r["lt_1e-2"] += int((aw < 1e-2).sum())
    for n_, p_ in model.named_parameters():
        if n_.startswith("model.layers.") or n_ == "model.norm.weight":
            stat_add(core.tensor_sort(n_), p_.data)
    stat_add("embed_rows", E[sorted(set(rows) | {o})])
    stat_tot = {key: sum(v[key] for v in stat.values()) for key in ("n", "zero", "lt_1e-6", "lt_1e-4", "lt_1e-3", "lt_1e-2")}
    print(f"  static: cone addresses {stat_tot['n']:,} | exactly zero {stat_tot['zero']:,} | <1e-6 {stat_tot['lt_1e-6']:,} | <1e-4 {stat_tot['lt_1e-4']:,} | <1e-3 {stat_tot['lt_1e-3']:,} | <1e-2 {stat_tot['lt_1e-2']:,}", flush=True)
    tot_p = {"n": sum(v["n"] for v in t.p.values()), **{key: sum(v[key] for v in t.p.values()) for key in list(t.p.values())[0] if key != "n"}}
    tot_a = {"n": sum(v["n"] for v in t.a.values()), **{key: sum(v[key] for v in t.a.values()) for key in list(t.a.values())[0] if key != "n"}}
    tot_a["n_with_embedding_rows"] = tot_a["n"] + emb_n
    res = {"model": name, "output_token": tok.convert_ids_to_tokens(o), "output_id": o, "positions": T, "layers": L, "products": {"total": tot_p, "by_sort": t.p}, "addresses": {"total": tot_a, "by_sort": t.a, "embedding_rows": {"n": emb_n, "zero": emb_zero}},
           "attention_gating_of_kv": {"kv_products": kv_n["products"], "kv_addresses": kv_n["addresses"], **att_tally}, "head_position_pairs": {"n": head_pairs, **head_pairs_below},
           "routing_last_position": routing, "ffn_units_last_position": ffn_units, "static": {"total": stat_tot, "by_sort": stat}, "thresholds": {"abs": ABS, "rel": REL, "att": ATT, "act": ACT}}
    core.dump(os.path.join(out, f"live_{name}.json"), res)
    A = tot_a["n"]; Pn = tot_p["n"]
    print(f"{name}: output {tok.convert_ids_to_tokens(o)!r} | cone addresses with a product: {A:,} (+ {emb_n:,} embedding-row weights read, {emb_zero} of them zero) | products feeding the last position: {Pn:,}", flush=True)
    print("  addresses whose every product is below: " + " | ".join(f"abs {k[4:]}: {tot_a[k]:,} ({tot_a[k] / A:.2%})" for k in tot_a if k.startswith("abs<")), flush=True)
    print("  addresses whose every product's row share is below: " + " | ".join(f"{k[4:]}: {tot_a[k]:,} ({tot_a[k] / A:.2%})" for k in tot_a if k.startswith("rel<")), flush=True)
    print("  products below: " + " | ".join(f"abs {k[4:]}: {tot_p[k] / Pn:.2%}" for k in tot_p if k.startswith("abs<")) + " || row share " + " | ".join(f"{k[4:]}: {tot_p[k] / Pn:.2%}" for k in tot_p if k.startswith("rel<")), flush=True)
    print("  by sort, addresses below abs 1e-6 / row share 2^-24 / row share 2^-8: " + ", ".join(f"{s}: {v['abs<1e-06'] / v['n']:.1%} / {v['rel<5.96046e-08'] / v['n']:.1%} / {v['rel<0.00390625'] / v['n']:.1%}" for s, v in t.a.items()), flush=True)
    print("  key/value addresses attention-dead (no head of the group gives the position more mass than): " + " | ".join(f"{k[4:]}: {v['kv_addresses_dead']:,} of {kv_n['addresses']:,} ({v['kv_addresses_dead'] / kv_n['addresses']:.1%}); products {v['kv_products_dead'] / kv_n['products']:.1%}" for k, v in att_tally.items()), flush=True)
    print(f"  head-position pairs at the last position: {head_pairs} | with mass below " + " | ".join(f"{k[2:]}: {v}" for k, v in head_pairs_below.items()), flush=True)
    print("  feed-forward units at the last position with |activation| >= 1e-2 per layer: " + " ".join(str(u["act>=0.01"]) for u in ffn_units) + f" | total {sum(u['act>=0.01'] for u in ffn_units)} of {L * cfg.intermediate_size}; >= 1e-4: {sum(u['act>=0.0001'] for u in ffn_units)}; >= 1e-1: {sum(u['act>=0.1'] for u in ffn_units)}", flush=True)
    print("  attention argmax source per head, layers 20..: " + " ".join(f"L{r['layer']}:" + "".join(str(h["argmax"]) for h in r["heads"]) for r in routing[20:]), flush=True)


if __name__ == "__main__":
    main()
