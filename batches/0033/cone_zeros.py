# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Static count: of the weights in the cone (all layer tensors, the final norm, six embedding rows), how many are exactly zero
or below a threshold in absolute value, by tensor sort; and whether any unit, head or key/value group is dead by its weights alone.
Reads tensors one at a time from the safetensors file by header offset (bf16 -> float32). usage: cone_zeros.py FILE OUT_JSON"""
import json
import struct
import sys

import numpy as np

ROWS = [504, 3575, 282, 4649, 314, 260]          # the five prompt tokens and the output token
THR = [1e-30, 1e-10, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 2e-2]
D, HD, H, KV, F = 576, 64, 9, 3, 1536


def sort_of(n):
    for s in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "input_layernorm", "post_attention_layernorm"):
        if s in n:
            return s
    return "norm" if n == "model.norm.weight" else "embed_rows"


def main():
    path, out = sys.argv[1:3]
    f = open(path, "rb"); n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); hdr.pop("__metadata__", None); base = 8 + n

    def read(name, row=None):
        t = hdr[name]; assert t["dtype"] == "BF16", t["dtype"]; a, b = t["data_offsets"]; shape = t["shape"]
        if row is None:
            f.seek(base + a); buf = f.read(b - a)
        else:
            f.seek(base + a + row * shape[1] * 2); buf = f.read(shape[1] * 2); shape = [shape[1]]
        u = np.frombuffer(buf, dtype=np.uint16).astype(np.uint32) << 16
        return u.view(np.float32).reshape(shape)

    agg = {}; dead = {"ffn_units_any_vector_max_below_1e-3": 0, "ffn_units_any_vector_max_below_1e-2": 0, "heads_q_or_o_max_below_1e-2": 0, "kv_groups_k_or_v_max_below_1e-2": 0, "ffn_unit_min_of_max_abs": 1.0}
    smallest_nonzero = 1.0; nsub = 0; distinct = set()
    layers = sorted({int(k.split(".")[2]) for k in hdr if k.startswith("model.layers.")})
    def acc(sort, w):
        nonlocal smallest_nonzero, nsub
        aw = np.abs(w).ravel(); r = agg.setdefault(sort, {"n": 0, "zero": 0, **{f"lt_{t:g}": 0 for t in THR}, "sum_abs": 0.0, "max_abs": 0.0})
        r["n"] += aw.size; r["zero"] += int((aw == 0).sum()); r["sum_abs"] += float(aw.sum(dtype=np.float64)); r["max_abs"] = max(r["max_abs"], float(aw.max()))
        for t in THR:
            r[f"lt_{t:g}"] += int((aw < t).sum())
        nz = aw[aw > 0]
        if nz.size:
            smallest_nonzero = min(smallest_nonzero, float(nz.min()))
        nsub += int(((aw > 0) & (aw < np.finfo(np.float32).tiny)).sum())
        distinct.update(np.unique(w.view(np.uint32).ravel()).tolist())
    for k in layers:
        P = lambda s: f"model.layers.{k}.{s}.weight"
        W = {s: read(P(s)) for s in ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "input_layernorm", "post_attention_layernorm")}
        for s, w in W.items():
            acc(sort_of(s), w)
        g = np.abs(W["mlp.gate_proj"]).max(1); u = np.abs(W["mlp.up_proj"]).max(1); d = np.abs(W["mlp.down_proj"]).max(0); m = np.minimum(np.minimum(g, u), d)
        dead["ffn_units_any_vector_max_below_1e-3"] += int((m < 1e-3).sum()); dead["ffn_units_any_vector_max_below_1e-2"] += int((m < 1e-2).sum()); dead["ffn_unit_min_of_max_abs"] = min(dead["ffn_unit_min_of_max_abs"], float(m.min()))
        for h in range(H):
            if min(np.abs(W["self_attn.q_proj"][h * HD:(h + 1) * HD]).max(), np.abs(W["self_attn.o_proj"][:, h * HD:(h + 1) * HD]).max()) < 1e-2:
                dead["heads_q_or_o_max_below_1e-2"] += 1
        for gi in range(KV):
            if min(np.abs(W["self_attn.k_proj"][gi * HD:(gi + 1) * HD]).max(), np.abs(W["self_attn.v_proj"][gi * HD:(gi + 1) * HD]).max()) < 1e-2:
                dead["kv_groups_k_or_v_max_below_1e-2"] += 1
    acc("norm", read("model.norm.weight"))
    for r_ in ROWS:
        acc("embed_rows", read("model.embed_tokens.weight", row=r_))
    tot = {"n": sum(v["n"] for v in agg.values()), "zero": sum(v["zero"] for v in agg.values()), **{f"lt_{t:g}": sum(v[f"lt_{t:g}"] for v in agg.values()) for t in THR}}
    res = {"cone_weights": tot["n"], "totals": tot, "by_sort": agg, "dead_by_weights": dead, "smallest_nonzero_abs": smallest_nonzero, "subnormal_count": nsub, "distinct_bf16_bit_patterns": len(distinct), "thresholds": THR}
    json.dump(res, open(out, "w"), indent=1)
    print(f"cone weights {tot['n']:,} | exactly zero {tot['zero']:,} | smallest nonzero |w| {smallest_nonzero:.3e} | subnormals {nsub} | distinct bf16 patterns {len(distinct)}")
    print("threshold      count below      share"); [print(f"  |w| < {t:<7g} {tot[f'lt_{t:g}']:>14,} {tot[f'lt_{t:g}'] / tot['n']:>10.4%}") for t in THR]
    print("by sort: n, zero, <1e-4, <1e-3, <1e-2, mean|w|, max|w|")
    for s, v in agg.items():
        print(f"  {s:26s} {v['n']:>11,} {v['zero']:>8,} {v['lt_0.0001'] / v['n']:>8.3%} {v['lt_0.001'] / v['n']:>8.3%} {v['lt_0.01'] / v['n']:>8.3%} {v['sum_abs'] / v['n']:>9.4f} {v['max_abs']:>8.3f}")
    print("dead by weights alone:", dead)


if __name__ == "__main__":
    main()
