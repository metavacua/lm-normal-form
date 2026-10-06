# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Runtime adapter: llama.cpp at a pinned release. The directory is converted to GGUF by that release's convert_hf_to_gguf.py (f32 or f16, or f32 and then
# llama-quantize to q8_0 or q4_0), and run by lmnf-llama (llamacpp/lmnf_llama.cpp), built by the workflow into the release directory. The dtype is the format:
# float32 -> f32, float16 -> f16, q8_0, q4_0; a suffix -kvTYPE sets the type of the key and value cache (f32, f16, q8_0, q4_0): without it the cache is f32 for float32 weights
# (so that the whole computation is float32; llama.cpp's own default, f16, would add rounding of 2^-11 to every key and value) and llama.cpp's default, f16, for the others.
# LLAMACPP_DIR is the release directory (with the tools and the program), LLAMACPP_SRC the source tree of the same tag.
import json, os, subprocess, sys
import numpy as np
from common import sha256_file
HERE = os.path.dirname(os.path.abspath(__file__))
REL = os.environ.get("LLAMACPP_DIR", "")
SRC = os.environ.get("LLAMACPP_SRC", "")
TYPE = {"float32": "f32", "float16": "f16", "q8_0": "q8_0", "q4_0": "q4_0"}


def version():
    return f"llama.cpp {os.path.basename(REL.rstrip('/')) or '?'}"


def split(dtype):
    """'float32-kvq8_0' -> ('float32', 'q8_0'); 'float32' -> ('float32', 'f32'); 'q8_0' -> ('q8_0', 'f16')."""
    w, _, kv = dtype.partition("-kv")
    return w, kv or ("f32" if w == "float32" else "f16")


def gguf(model_dir, work, dtype):
    name = os.path.basename(model_dir)
    os.makedirs(os.path.join(work, "gguf"), exist_ok=True)
    f32 = os.path.join(work, "gguf", f"{name}-f32.gguf")
    if not os.path.exists(f32):
        subprocess.run([sys.executable, os.path.join(SRC, "convert_hf_to_gguf.py"), model_dir, "--outfile", f32, "--outtype", "f32"], check=True)
    t = TYPE[split(dtype)[0]]
    if t == "f32":
        return f32
    out = os.path.join(work, "gguf", f"{name}-{t}.gguf")
    if not os.path.exists(out):
        if t == "f16":
            subprocess.run([sys.executable, os.path.join(SRC, "convert_hf_to_gguf.py"), model_dir, "--outfile", out, "--outtype", "f16"], check=True)
        else:
            subprocess.run([os.path.join(REL, "llama-quantize"), f32, out, t.upper(), os.environ.get("LMNF_THREADS", "4")], check=True)
    return out


def run(model_dir, ids, wins, dtype, work, tag, full=True):
    path = gguf(model_dir, work, dtype)
    out = os.path.join(work, "res", f"{tag}-llamacpp")
    subprocess.run([os.path.join(REL, "lmnf-llama"), path, os.path.join(work, "inputs.json"), out, "1" if full else "0", os.environ.get("LMNF_THREADS", "4"), split(dtype)[1]], check=True)
    res = json.load(open(out + ".json"))
    res["gguf_bytes"] = os.path.getsize(path)
    res["artifact_sha256"] = sha256_file(path)
    logits = None
    if full:
        logits = np.fromfile(out + ".logits.f32", dtype="<f4").reshape(res["positions"], res["vocab"])
        os.remove(out + ".logits.f32")
    return res, logits
