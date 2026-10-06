# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Runtime adapter: ONNX Runtime on the CPU, on a graph exported from the directory by optimum-cli (task text-generation: logits of every position, no cache).
# Generation re-runs the whole prefix at each step, so the speed it reports for decode is not comparable with the cached runtimes' and is left out; prefill is.
import os, subprocess
import numpy as np
from common import Timer, peak_rss_mb, ppl_from_logits


def version():
    import onnxruntime
    return f"onnxruntime {onnxruntime.__version__}"


def run(model_dir, ids, wins, dtype, work, tag, full=True):
    import onnxruntime as ort
    assert dtype == "float32"
    out = os.path.join(work, "onnx", os.path.basename(model_dir))
    if not os.path.exists(os.path.join(out, "model.onnx")):
        subprocess.run(["optimum-cli", "export", "onnx", "--model", model_dir, "--task", "text-generation", out], check=True)
    so = ort.SessionOptions()
    so.intra_op_num_threads = int(os.environ.get("LMNF_THREADS", "4"))
    so.inter_op_num_threads = 1
    t = Timer()
    sess = ort.InferenceSession(os.path.join(out, "model.onnx"), so, providers=["CPUExecutionProvider"])
    res = {"load_s": t.lap(), "onnx_bytes": sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out) if os.path.isfile(os.path.join(out, f)))}
    names = {i.name for i in sess.get_inputs()}

    def fwd(seq):
        i = np.array([seq], dtype=np.int64)
        feed = {"input_ids": i, "attention_mask": np.ones_like(i), "position_ids": np.arange(len(seq), dtype=np.int64)[None]}
        return sess.run(None, {k: v for k, v in feed.items() if k in names})[0][0].astype(np.float32)

    logits = None
    if full:
        logits = np.concatenate([fwd(s) for s in ids["test"]])
        eos = set(ids.get("eos", []))
        gen = {}
        for kind in ("plain", "chat"):
            if kind not in ids:
                continue
            outs = []
            for p in ids[kind]:
                seq, o = list(p), []
                for _ in range(32):
                    nxt = int(fwd(seq)[-1].argmax())
                    o.append(nxt)
                    if nxt in eos:
                        break
                    seq.append(nxt)
                outs.append(o)
            gen[kind] = outs
        res["gen"] = gen
        nll, n = 0.0, 0
        for w in wins:
            a, b = ppl_from_logits(fwd(w), w)
            nll, n = nll + a, n + b
        res["ppl_nll"], res["ppl_tokens"] = nll, n
    fwd(wins[0])
    pre = []
    for _ in range(5):
        t.lap()
        fwd(wins[0])
        pre.append(len(wins[0]) / t.lap())
    res["prefill_tps"] = pre
    res["peak_rss_mb"] = peak_rss_mb()
    return res, logits
