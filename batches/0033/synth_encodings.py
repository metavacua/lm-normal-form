# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Write the patches of one pair into copies of that model's GGUF (f16) and ONNX (f32) files at the located byte addresses, read them back.
usage: synth_encodings.py synthetic_rows.json GGUF ONNX ONNX_BLOCK_BYTE        (needs the gguf package: uv run --with numpy --with gguf python -I ...)
The GGUF address is the data offset of token_embd.weight in the file's own header. ONNX_BLOCK_BYTE is the byte offset of the lookup initializer's data,
found by synth_identify.py's byte search; the graph itself is not loaded here (that takes more memory than this machine allows), so the second,
transposed copy of the tied matrix the export keeps for the head is not written: the CI cell, synth.py, finds both from the graph and is the authority."""
import json
import os
import shutil
import sys

import numpy as np
from gguf import GGUFReader

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_common import bf16_to_f32, peak_rss_mb, sha256_file  # noqa: E402

JS, GG, ON, OPOS = sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4])
assert OPOS > 0
syn = json.load(open(JS))
D = os.path.dirname(os.path.abspath(JS))
C = syn["source"]["row_bytes"] // 2
t = next(t for t in GGUFReader(GG).tensors if t.name == "token_embd.weight")
assert t.tensor_type.name == "F16" and int(t.shape[0]) == C, (t.tensor_type.name, t.shape)
GOFF = int(t.data_offset)
enc = {"gguf": {"file": os.path.basename(GG), "sha256_prefix": sha256_file(GG)[:16], "tensor": "token_embd.weight", "type": "F16", "data_offset": GOFF, "row_bytes": C * 2, "patches": {}},
       "onnx": {"file": os.path.basename(ON), "sha256_prefix": sha256_file(ON)[:16], "initializer": "model.embed_tokens.weight", "type": "FLOAT", "block_byte": OPOS, "row_bytes": C * 4, "patches": {}}}
cg, co = os.path.join(D, "syn.gguf"), os.path.join(D, "syn.onnx")
shutil.copyfile(GG, cg)
shutil.copyfile(ON, co)
saved = []  # (handle, offset, original bytes): undone after each patch, so each copy holds exactly one patch when it is read back


def put(fh, off, data):
    """Write data at off, keeping the bytes it replaces; return what the file holds there afterwards."""
    fh.seek(off)
    saved.append((fh, off, fh.read(len(data))))
    fh.seek(off)
    fh.write(data)
    fh.seek(off)
    return fh.read(len(data))


with open(cg, "r+b") as g, open(co, "r+b") as o:
    for p in syn["patches"]:
        g16, o32 = {}, {}
        for w in p["writes"]:
            v = bf16_to_f32(bytes.fromhex(w["hex"]))
            v16 = v.astype(np.float16)
            w["f16_exact_entries"] = int((v16.astype(np.float32) == v).sum())
            w["f16_hex"] = v16.tobytes().hex()
            g16[w["row"]] = int((np.frombuffer(put(g, GOFF + w["row"] * C * 2, v16.tobytes()), dtype=np.float16).astype(np.float32) == v).sum())
            o32[w["row"]] = int((np.frombuffer(put(o, OPOS + w["row"] * C * 4, v.tobytes()), dtype=np.float32) == v).sum())
        enc["gguf"]["patches"][p["name"]] = {"read_back_equal": g16, "f16_exact": [w["f16_exact_entries"] for w in p["writes"]], "byte_offsets": [GOFF + w["row"] * C * 2 for w in p["writes"]]}
        enc["onnx"]["patches"][p["name"]] = {"read_back_equal": o32, "byte_offsets": [OPOS + w["row"] * C * 4 for w in p["writes"]]}
        print(p["name"], "gguf", g16, "f16-exact", enc["gguf"]["patches"][p["name"]]["f16_exact"], "| onnx", o32)
        for fh, off, b in saved:
            fh.seek(off)
            fh.write(b)
        saved.clear()
os.remove(cg)
os.remove(co)
syn["encodings"] = enc
with open(JS, "w") as fh:
    json.dump(syn, fh, indent=1)
print("peak rss MiB", int(peak_rss_mb()))
