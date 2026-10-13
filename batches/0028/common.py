# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What the modules of batch 0028 share: the import path of batch 0011's identity readers and batch 0027's
wasm extractor, the arrays of a safetensors file by name, and JSON files written one way."""

import json
import os
import struct
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
for sub in ("0011", "0027"):
    p = os.path.join(HERE, "..", sub)
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np  # noqa: E402

import identity  # noqa: E402  (batches/0011)


def dump(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, sort_keys=True)
        f.write("\n")


def load(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def st_arrays(path):
    """The tensors of a safetensors file by name, as numpy views of the file, and its header as batch 0011 reads it."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        head = f.read(n)
    dup = []

    def hook(pairs):
        keys = [k for k, _ in pairs]
        dup.extend(k for k in set(keys) if keys.count(k) > 1)
        return dict(pairs)

    hdr = json.loads(head, object_pairs_hook=hook)
    meta = hdr.pop("__metadata__", None)
    mm = np.memmap(path, dtype=np.uint8, mode="r")
    base = 8 + n
    arrays = {}
    for name, d in hdr.items():
        s, e = d["data_offsets"]
        dt = identity.ST[d["dtype"]]
        arrays[name] = np.frombuffer(mm[base + s:base + e], dtype=identity.UNNP[dt]).reshape(d["shape"])
    stripped = head.rstrip(b" ")
    info = {"header_bytes": n, "header_trailing_spaces": len(head) - len(stripped), "metadata": meta,
            "duplicate_names": sorted(dup), "tensors": len(hdr), "has_lm_head": "lm_head.weight" in hdr}
    return arrays, info
