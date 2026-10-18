# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Which published model a local GGUF file and a local ONNX file hold, decided by value, not by name: rows of their tied matrix are compared with the
base and the Instruct safetensors files, and the ONNX file is searched for the bytes of each model's row 0, where a float32 copy of the matrix starts.
usage: synth_identify.py BASE.safetensors INSTRUCT.safetensors GGUF ONNX        (needs the gguf package: uv run --with numpy --with gguf python -I ...)"""
import mmap
import os
import sys

import numpy as np
from gguf import GGUFReader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_common import bf16_to_f32, peak_rss_mb, sha256_file, st_header, st_tensor  # noqa: E402

BASE, INST, GG, ON = sys.argv[1:5]


def st(path):
    """A reader of rows of the tied matrix of a safetensors file, and the row width."""
    hdr, start = st_header(path)
    e0, (_, cols) = st_tensor(hdr, start, "model.embed_tokens.weight")
    f = open(path, "rb")

    def row(r):
        f.seek(e0 + r * cols * 2)
        return bf16_to_f32(f.read(cols * 2))
    return row, cols


(rb, C), (ri, _) = st(BASE), st(INST)
print("base vs instruct row 260 equal entries:", int((rb(260) == ri(260)).sum()), "| row 0:", int((rb(0) == ri(0)).sum()))
t = next(t for t in GGUFReader(GG).tensors if t.name == "token_embd.weight")
assert t.tensor_type.name == "F16" and t.data.shape[1:] == (C,), (t.tensor_type.name, t.data.shape)
print("gguf token_embd.weight: F16, data at byte", int(t.data_offset))
for r in (0, 260, 7042):
    gr = t.data[r].astype(np.float32)
    print(f"gguf row {r}: equal to base {int((gr == rb(r)).sum())}/{C}, to instruct {int((gr == ri(r)).sum())}/{C}")
pats = {"base_row0": rb(0).tobytes(), "instruct_row0": ri(0).tobytes()}
L = C * 4
with open(ON, "rb") as o, mmap.mmap(o.fileno(), 0, access=mmap.ACCESS_READ) as mm:
    found = {k: [] for k in pats}
    for k, p in pats.items():
        j = mm.find(p)
        while j >= 0:
            found[k].append(j)
            j = mm.find(p, j + 1)
    print("onnx occurrences:", found)
    for k, poss in found.items():
        src = rb if k.startswith("base") else ri
        for pos in poss:
            res = {r: int((np.frombuffer(mm[pos + r * L:pos + (r + 1) * L], dtype=np.float32) == src(r)).sum()) for r in (260, 7042, 49151)}
            print(f"onnx block for {k} at {pos}: rows equal {res}")
for name, path in (("gguf", GG), ("onnx", ON)):
    print(name, "sha256", sha256_file(path)[:16])
print("peak rss MiB", int(peak_rss_mb()))
