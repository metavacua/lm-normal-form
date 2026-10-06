# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# What is in the numbers of a safetensors file that a symmetry of the model would have to agree with: exact zeros and
# negative zeros, NaN and infinities, subnormals, the smallest and largest exponent, rows that repeat, the feed-forward
# units that are the same unit twice (gate row, up row, down column together), the columns of the embedding that repeat,
# and the values that an RMSNorm weight takes. A description: no prediction is made about any of it.
# Usage: scan.py MODEL.safetensors OUT.json
import collections, json, sys
import numpy as np
import ml_dtypes
from safetensors import safe_open


def rows_repeated(a):
    """How many rows of a 2-D array are equal to an earlier row."""
    v = np.ascontiguousarray(a).view(np.dtype((np.void, a.dtype.itemsize * a.shape[1])))
    return int(a.shape[0] - len(np.unique(v)))


def main(path, out):
    res, units = {}, {}
    with safe_open(path, framework="np") as f:
        names = sorted(f.keys())
        for name in names:
            raw = f.get_tensor(name)
            a = raw.astype(np.float32) if raw.dtype == ml_dtypes.bfloat16 else raw
            fin = np.isfinite(a)
            nz = a[fin & (a != 0)]
            e = np.frexp(nz)[1] if nz.size else np.array([0])
            r = {"shape": list(a.shape), "zeros": int((a == 0).sum()), "negative_zeros": int((np.signbit(a) & (a == 0)).sum()),
                 "nan": int(np.isnan(a).sum()), "inf": int(np.isinf(a).sum()),
                 "subnormal": int(((np.abs(a) < np.finfo(np.float32).tiny) & (a != 0)).sum()),
                 "exponent_min": int(e.min()), "exponent_max": int(e.max())}
            if a.ndim == 2:
                r["rows_repeated"] = rows_repeated(a)
                r["columns_repeated"] = rows_repeated(a.T) if a.shape[1] <= 4096 else None
            if a.ndim == 1:
                vals, counts = np.unique(a, return_counts=True)
                r["distinct_values"] = int(len(vals))
                r["most_common_value_count"] = int(counts.max())
            res[name] = r
        layers = sorted({int(n.split(".")[2]) for n in names if n.startswith("model.layers.")})
        for i in layers:
            g, u, d = (f.get_tensor(f"model.layers.{i}.mlp.{k}_proj.weight").astype(np.float32) for k in ("gate", "up", "down"))
            unit = np.concatenate([g, u, d.T], axis=1)   # one row per feed-forward unit
            units[i] = rows_repeated(unit)
    tot = collections.Counter()
    for r in res.values():
        for k in ("zeros", "negative_zeros", "nan", "inf", "subnormal"):
            tot[k] += r[k]
    summary = {"tensors": len(res), "totals": dict(tot), "exponent_min": min(r["exponent_min"] for r in res.values()),
               "exponent_max": max(r["exponent_max"] for r in res.values()),
               "matrices_with_a_repeated_row": sum(1 for r in res.values() if r.get("rows_repeated")),
               "matrices_with_a_repeated_column": sum(1 for r in res.values() if r.get("columns_repeated")),
               "layers": len(units), "units_repeated_in_any_layer": sum(units.values()),
               "norm_vectors": sum(1 for r in res.values() if "distinct_values" in r),
               "norm_vectors_with_a_repeated_value": sum(1 for r in res.values() if r.get("distinct_values", 0) and r["distinct_values"] < r["shape"][0])}
    json.dump({"summary": summary, "tensors": res, "units_repeated_by_layer": units}, open(out, "w"), indent=1, sort_keys=True)
    print(json.dumps(summary, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(*sys.argv[1:3])
