# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Runtime adapter: CTranslate2 on the CPU, on the directory converted by ct2-transformers-converter. CTranslate2 works with token strings; the ids of the
# Hugging Face tokenizer are mapped to them and back. The logits of every position come from Generator.forward_batch if this version has it; perplexity from
# score_batch (log-probabilities of the next tokens). dtypes: float32, int8 (the weights of every layer quantized per row, with a dynamic per-row activation scale).
import os, subprocess
import numpy as np
from common import Timer, peak_rss_mb


def version():
    import ctranslate2
    return f"ctranslate2 {ctranslate2.__version__}"


def run(model_dir, ids, wins, dtype, work, tag, full=True):
    import ctranslate2
    from transformers import AutoTokenizer
    threads = int(os.environ.get("LMNF_THREADS", "4"))
    tok = AutoTokenizer.from_pretrained(model_dir)
    quant = {"float32": "float32", "int8": "int8"}[dtype]
    out = os.path.join(work, "ct2", f"{os.path.basename(model_dir)}-{quant}")
    if not os.path.exists(os.path.join(out, "model.bin")):
        subprocess.run(["ct2-transformers-converter", "--model", model_dir, "--output_dir", out, "--quantization", quant, "--force"], check=True)
    t = Timer()
    gen = ctranslate2.Generator(out, device="cpu", compute_type=quant, inter_threads=1, intra_threads=threads)
    res = {"load_s": t.lap(), "ct2_bytes": sum(os.path.getsize(os.path.join(out, f)) for f in os.listdir(out) if os.path.isfile(os.path.join(out, f)))}
    toks = lambda seq: tok.convert_ids_to_tokens(list(seq))
    logits = None
    if full:
        if hasattr(gen, "forward_batch"):
            rows = []
            for s in ids["test"]:
                r = gen.forward_batch([toks(s)])
                rows.append(np.array(r, copy=True).reshape(len(s), -1).astype(np.float32))
            logits = np.concatenate(rows)
        end = [tok.convert_ids_to_tokens(e) for e in ids.get("eos", [])] or None
        g = {}
        for kind in ("plain", "chat"):
            if kind in ids:
                r = gen.generate_batch([toks(p) for p in ids[kind]], max_length=32, sampling_topk=1, include_prompt_in_result=False, end_token=end, return_end_token=True)
                g[kind] = [list(x.sequences_ids[0]) for x in r]
        res["gen"] = g
        sc = gen.score_batch([toks(w) for w in wins])
        res["ppl_nll"] = float(-sum(sum(x.log_probs) for x in sc))
        res["ppl_tokens"] = int(sum(len(x.log_probs) for x in sc))
    w0 = toks(wins[0])
    gen.generate_batch([w0], max_length=1, sampling_topk=1)
    pre = []
    for _ in range(5):
        t.lap()
        gen.generate_batch([w0], max_length=1, sampling_topk=1)
        pre.append(len(w0) / t.lap())
    p16 = toks((list(ids["plain"][0]) + list(wins[0]))[:16])
    dec = []
    for _ in range(3):
        t.lap()
        gen.generate_batch([p16], max_length=64, min_length=64, sampling_topk=1)
        dec.append(64 / t.lap())
    res["prefill_tps"], res["decode_tps"] = pre, dec
    res["peak_rss_mb"] = peak_rss_mb()
    return res, logits
