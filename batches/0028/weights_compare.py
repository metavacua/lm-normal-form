# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The weights of the two checkpoints themselves, for batch 0028: P1's header facts, P20's scan of each checkpoint for the
values that are blank in a number (a zero, a negative zero, a NaN, an infinity, a subnormal) and for repeated feed-forward
units, and the observed comparison of the base against the Instruct model entry by entry.

All reads are of BF16 bit patterns (uint16), so that no value is converted. A feed-forward unit of a layer is the triple (a
row of gate_proj, the same row of up_proj, the same column of down_proj); two units are repeated when the three are equal in
every bit.

Usage: weights_compare.py BASE.safetensors INSTRUCT.safetensors OUT.json
       weights_compare.py selftest
"""

import collections
import hashlib
import json
import re
import struct
import sys
import tempfile

import numpy as np

import common


def bits(a):
    """The bit pattern of a BF16 or F32 array as unsigned integers."""
    return a.view(np.uint16 if a.dtype.itemsize == 2 else np.uint32)


def scan(arrays):
    """Zeros, negative zeros, NaNs, infinities and subnormals of BF16 tensors, by bit pattern; other dtypes are counted apart."""
    tot = collections.Counter()
    other = []
    for name, a in arrays.items():
        if a.dtype.itemsize != 2:
            other.append(name)
            continue
        u = bits(a).reshape(-1)
        exp, man = (u >> 7) & 0xFF, u & 0x7F
        tot["zero"] += int(((u & 0x7FFF) == 0).sum())
        tot["negative_zero"] += int((u == 0x8000).sum())
        tot["nan"] += int(((exp == 0xFF) & (man != 0)).sum())
        tot["inf"] += int(((exp == 0xFF) & (man == 0)).sum())
        tot["subnormal"] += int(((exp == 0) & (man != 0)).sum())
        tot["entries"] += int(u.size)
    return {k: tot[k] for k in ("entries", "zero", "negative_zero", "nan", "inf", "subnormal")} | {"tensors_not_bf16": other}


def repeated_units(arrays):
    """Per layer, the number of feed-forward units that repeat another unit of the layer in all three tensors."""
    layers = sorted({int(m.group(1)) for k in arrays for m in [re.match(r"model\.layers\.(\d+)\.", k)] if m})
    per_layer, units = {}, 0
    for i in layers:
        g, u, d = (bits(arrays[f"model.layers.{i}.mlp.{n}.weight"]) for n in ("gate_proj", "up_proj", "down_proj"))
        n = g.shape[0]
        seen = collections.Counter(hashlib.sha1(g[j].tobytes() + u[j].tobytes() + np.ascontiguousarray(d[:, j]).tobytes()).digest()
                                   for j in range(n))
        per_layer[i] = sum(c - 1 for c in seen.values() if c > 1)
        units += n
    return {"units": units, "repeated_units": sum(per_layer.values()), "layers_with_repeats": [i for i, v in per_layer.items() if v]}


def embedding_rows(arrays):
    e = bits(arrays["model.embed_tokens.weight"])
    seen = collections.Counter(hashlib.sha1(e[j].tobytes()).digest() for j in range(e.shape[0]))
    return {"rows": int(e.shape[0]), "repeated_rows": sum(c - 1 for c in seen.values() if c > 1)}


def difference(a, b):
    """Base against Instruct: per tensor the fraction of entries equal in every bit; the embedding rows that are equal."""
    rows, tot_eq, tot = {}, 0, 0
    for name in sorted(a):
        if name not in b or a[name].shape != b[name].shape:
            rows[name] = None
            continue
        x, y = bits(a[name]), bits(b[name])
        eq = int((x == y).sum())
        rows[name] = eq / x.size
        tot_eq += eq
        tot += x.size
    e = bits(a["model.embed_tokens.weight"]) == bits(b["model.embed_tokens.weight"])
    vals = [v for v in rows.values() if v is not None]
    return {"entries": tot, "equal_entries": tot_eq, "fraction_equal": tot_eq / tot if tot else None,
            "tensors": len(rows), "tensors_with_an_equal_entry": sum(1 for v in vals if v > 0),
            "tensors_fully_equal": sum(1 for v in vals if v == 1.0), "max_tensor_fraction": max(vals) if vals else None,
            "embedding_rows_equal": int(e.all(axis=1).sum()), "embedding_rows": int(e.shape[0]),
            "fractions_by_tensor": rows}


def header_facts(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = f.read(n)
    stripped = head.rstrip(b" ")
    dup = []

    def hook(pairs):
        keys = [k for k, _ in pairs]
        dup.extend(k for k in set(keys) if keys.count(k) > 1)
        return dict(pairs)

    hdr = json.loads(stripped, object_pairs_hook=hook)
    meta = hdr.pop("__metadata__", None)
    spans = sorted(tuple(d["data_offsets"]) for d in hdr.values())
    return {"header_bytes": n, "trailing_spaces": len(head) - len(stripped), "json_ends_at_brace": stripped.endswith(b"}"),
            "tensors": len(hdr), "dtypes": dict(collections.Counter(d["dtype"] for d in hdr.values())),
            "elements": int(sum(int(np.prod(d["shape"])) for d in hdr.values())), "has_lm_head": "lm_head.weight" in hdr,
            "metadata": meta, "duplicate_names": sorted(dup), "offsets_contiguous": spans[0][0] == 0 and all(a[1] == b[0] for a, b in zip(spans, spans[1:]))}


def run(base, instruct, out):
    res = {}
    arrays = {}
    for tag, path in (("base", base), ("instruct", instruct)):
        arrays[tag], _ = common.st_arrays(path)
        res[tag] = {"header": header_facts(path), "scan": scan(arrays[tag]), "units": repeated_units(arrays[tag]),
                    "embedding": embedding_rows(arrays[tag])}
    res["difference"] = difference(arrays["base"], arrays["instruct"])
    common.dump(out, res)
    d = res["difference"]
    print(json.dumps({"fraction_equal": d["fraction_equal"], "embedding_rows_equal": d["embedding_rows_equal"],
                      "base_zero": res["base"]["scan"]["zero"], "instruct_zero": res["instruct"]["scan"]["zero"]}))
    return res


def selftest():
    import ml_dtypes
    bad = 0

    def check(what, ok):
        nonlocal bad
        print(("ok   " if ok else "FAIL ") + what)
        bad += 0 if ok else 1

    u = np.array([0x0000, 0x8000, 0x7F80, 0x7FC0, 0x0001, 0x3F80, 0xBF80], dtype=np.uint16)
    s = scan({"x": u.view(ml_dtypes.bfloat16)})
    check("scan: one zero and one negative zero, an inf, a NaN, a subnormal",
          (s["zero"], s["negative_zero"], s["inf"], s["nan"], s["subnormal"]) == (2, 1, 1, 1, 1))
    rng = np.random.default_rng(3)
    g = rng.standard_normal((4, 3)).astype(ml_dtypes.bfloat16)
    up = rng.standard_normal((4, 3)).astype(ml_dtypes.bfloat16)
    dn = rng.standard_normal((3, 4)).astype(ml_dtypes.bfloat16)
    arr = {"model.layers.0.mlp.gate_proj.weight": g, "model.layers.0.mlp.up_proj.weight": up, "model.layers.0.mlp.down_proj.weight": dn}
    check("no repeated unit in random weights", repeated_units(arr)["repeated_units"] == 0)
    g2, up2, dn2 = g.copy(), up.copy(), dn.copy()
    g2[1], up2[1], dn2[:, 1] = g2[0], up2[0], dn2[:, 0]
    arr2 = {"model.layers.0.mlp.gate_proj.weight": g2, "model.layers.0.mlp.up_proj.weight": up2, "model.layers.0.mlp.down_proj.weight": dn2}
    check("a unit copied to another position is one repeated unit", repeated_units(arr2)["repeated_units"] == 1)
    e = rng.standard_normal((5, 4)).astype(ml_dtypes.bfloat16)
    e2 = e.copy()
    e2[2] += 1
    d = difference({"model.embed_tokens.weight": e}, {"model.embed_tokens.weight": e2})
    check("difference: four of five embedding rows equal", d["embedding_rows_equal"] == 4 and 0.7 < d["fraction_equal"] < 0.9)
    e3 = e.copy()
    e3[1] = e3[0]
    check("embedding_rows: one repeated row", embedding_rows({"model.embed_tokens.weight": e3})["repeated_rows"] == 1)
    with tempfile.TemporaryDirectory() as t:
        raw = e.tobytes()
        h = json.dumps({"__metadata__": {"format": "pt"}, "model.embed_tokens.weight": {"dtype": "BF16", "shape": [5, 4],
                                                                                      "data_offsets": [0, len(raw)]}}).encode()
        h += b" " * (-len(h) % 8)
        open(t + "/m.safetensors", "wb").write(struct.pack("<Q", len(h)) + h + raw)
        hf = header_facts(t + "/m.safetensors")
        check("header facts: padded with spaces, metadata read, contiguous", hf["json_ends_at_brace"] and hf["offsets_contiguous"]
              and hf["metadata"] == {"format": "pt"} and hf["dtypes"] == {"BF16": 1})
    return bad


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        sys.exit(1 if selftest() else 0)
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    run(*sys.argv[1:4])
