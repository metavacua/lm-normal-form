# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The dead structure of a Llama-style decoder's weights: the rows, columns and rotary planes that are entirely zero, counted where it
# matters for the gauge of symmetry.py. Each gauge scales one part of a pair by 2^k and the other by 2^-k:
#   units  the up row against the down column
#   vo     a value row against the o columns that read it (one per query head of the group)
#   qk     a key rotary plane (the two rows p and p + hd/2) against the same plane of the query heads of the group
# When the part that a canonical form normalizes by is all zero, normalizing by its own content cannot fix the gauge. Three cases:
#   both zero   the gauge changes nothing but the sign bit of the zeros
#   mixed       the first part is zero and the other is not: the gauge has to be fixed by the other part
#   (a nonzero first part is the ordinary case)
# The census works on any function get(name) -> float array, so a model can be read one tensor at a time.
import numpy as np
import forms as F


def census(get, d, untied):
    out = {"layers": [], "embedding": {}, "totals": {}}
    E = get(F.EMBED)
    out["embedding"] = {"zero_columns": int(np.count_nonzero(~E.any(axis=0))), "zero_rows": int(np.count_nonzero(~E.any(axis=1)))}
    if untied:
        Hd = get("lm_head.weight")
        out["embedding"].update({"head_zero_columns": int(np.count_nonzero(~Hd.any(axis=0))), "head_zero_rows": int(np.count_nonzero(~Hd.any(axis=1))),
                                 "columns_zero_in_both": int(np.count_nonzero(~E.any(axis=0) & ~Hd.any(axis=0)))})
    h2 = d.hd // 2
    for i in range(d.L):
        g_, u_, w_ = get(F.key(i, F.G)), get(F.key(i, F.U)), get(F.key(i, F.D))
        up0, gate0, down0 = ~u_.any(axis=1), ~g_.any(axis=1), ~w_.any(axis=0)
        units = {"up_zero": int(up0.sum()), "gate_zero": int(gate0.sum()), "down_zero": int(down0.sum()), "both_zero": int((up0 & down0).sum()),
                 "mixed_up_zero_down_not": int((up0 & ~down0).sum())}
        q, k, v, o = (get(F.key(i, n)) for n in (F.Q, F.K, F.V, F.O))
        qb, kb, vb, ob = q.reshape(d.nh, d.hd, d.H), k.reshape(d.nkv, d.hd, d.H), v.reshape(d.nkv, d.hd, d.H), o.reshape(d.H, d.nh, d.hd)
        v0 = ~vb.any(axis=2)                                                   # (nkv, hd)
        o0 = ~ob.any(axis=0).reshape(d.nkv, d.rep, d.hd)                       # (nkv, rep, hd): the o columns of each head
        o_all0 = o0.all(axis=1)
        vo = {"value_rows_zero": int(v0.sum()), "o_columns_zero": int(o0.sum()), "both_zero": int((v0 & o_all0).sum()),
              "mixed_value_zero_o_not": int((v0 & ~o_all0).sum())}
        k0 = ~(kb[:, :h2].any(axis=2) | kb[:, h2:].any(axis=2))                # (nkv, hd/2)
        q0 = ~(qb[:, :h2].any(axis=2) | qb[:, h2:].any(axis=2))                # (nh, hd/2)
        q_all0 = q0.reshape(d.nkv, d.rep, h2).all(axis=1)
        qk = {"key_planes_zero": int(k0.sum()), "query_planes_zero": int(q0.sum()), "both_zero": int((k0 & q_all0).sum()),
              "mixed_key_zero_query_not": int((k0 & ~q_all0).sum())}
        out["layers"].append({"layer": i, "units": units, "vo": vo, "qk": qk})
    for part in ("units", "vo", "qk"):
        keys = out["layers"][0][part].keys()
        out["totals"][part] = {n: sum(x[part][n] for x in out["layers"]) for n in keys}
    out["totals"]["mixed_cases"] = (out["totals"]["units"]["mixed_up_zero_down_not"] + out["totals"]["vo"]["mixed_value_zero_o_not"]
                                    + out["totals"]["qk"]["mixed_key_zero_query_not"])
    out["totals"]["layers_with_dead_pairs"] = [x["layer"] for x in out["layers"] if x["units"]["both_zero"] or x["vo"]["both_zero"] or x["qk"]["both_zero"]]
    return out
