# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Runtime adapter: candle (Rust), through the program in candle/ (built by the workflow with cargo; not built on the machine where this was written).
import json, os, subprocess
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
EXE = os.path.join(HERE, "candle", "target", "release", "lmnf-candle")


def version():
    return "candle-core, candle-nn, candle-transformers =0.11.0 (lmnf-candle)"


def run(model_dir, ids, wins, dtype, work, tag, full=True):
    out = os.path.join(work, "res", f"{tag}-candle")
    env = dict(os.environ, RAYON_NUM_THREADS=os.environ.get("LMNF_THREADS", "4"))
    subprocess.run([EXE, model_dir, os.path.join(work, "inputs.json"), out, {"float32": "f32", "bfloat16": "bf16"}[dtype], "1" if full else "0"], check=True, env=env)
    res = json.load(open(out + ".json"))
    logits = None
    if full:
        logits = np.fromfile(out + ".logits.f32", dtype="<f4").reshape(res["positions"], res["vocab"])
        os.remove(out + ".logits.f32")
    return res, logits
