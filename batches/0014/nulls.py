# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The null variants of batch 0014 (addendum 2) and batch 0024: changes of the weights that are not symmetries and change the arithmetic by the least that float32 can express, one unit in the last
# place in every norm weight. They are the control for the noise of a quantized run: a variant that an exact symmetry makes of a checkpoint should differ from the original by no more than a null does.
#   nullall   every norm weight times 1 + 2^-23 (batch 0014)
#   nullm     every norm weight times 1 - 2^-24 (batch 0024)
#   nullr<k>  every norm weight times 1 + s 2^-23, the sign s of each element drawn from a generator seeded with 1000 + k (batch 0024), k = 1 .. 6
# In float64 and then rounded to float32. This module has no dependency but numpy so that its test (batches/0024/test_nulls.py) runs without the rest of the harness.
import numpy as np

NULLS = ("nullall", "nullm", "nullr1", "nullr2", "nullr3", "nullr4", "nullr5", "nullr6")


def is_norm_weight(key):
    return key.endswith("layernorm.weight") or key == "model.norm.weight"


def make_null(P, variant):
    if variant not in NULLS:
        raise ValueError(variant)
    g = np.random.default_rng(1000 + int(variant[5:])) if variant.startswith("nullr") else None
    out = {}
    for k, a in P.items():
        if not is_norm_weight(k):
            out[k] = a
        elif variant == "nullall":
            out[k] = (a.astype(np.float64) * (1.0 + 2.0 ** -23)).astype(np.float32)
        elif variant == "nullm":
            out[k] = (a.astype(np.float64) * (1.0 - 2.0 ** -24)).astype(np.float32)
        else:
            sgn = np.where(g.random(a.shape) < 0.5, -1.0, 1.0)
            out[k] = (a.astype(np.float64) * (1.0 + sgn * 2.0 ** -23)).astype(np.float32)
    return out
