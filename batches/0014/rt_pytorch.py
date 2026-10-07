# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Runtime adapter: PyTorch with Hugging Face Transformers, on the CPU. run() returns what xrt.py asks of every runtime (see there).
import os
import numpy as np
from common import Timer, peak_rss_mb, ppl_from_logits


def version():
    import torch, transformers
    return f"torch {torch.__version__}, transformers {transformers.__version__}"


def run(model_dir, ids, wins, dtype, work, tag, full=True):
    import torch
    from transformers import AutoModelForCausalLM
    torch.set_num_threads(int(os.environ.get("LMNF_THREADS", "4")))
    dt = {"float32": torch.float32, "bfloat16": torch.bfloat16}[dtype]
    t = Timer()
    model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dt)
    model.eval()
    res = {"load_s": t.lap()}
    logits = None
    with torch.no_grad():
        if full:
            logits = [model(input_ids=torch.tensor([s])).logits[0].float().numpy() for s in ids["test"]]
            res["gen"] = {k: [model.generate(torch.tensor([p]), max_new_tokens=32, do_sample=False, pad_token_id=0)[0][len(p):].tolist() for p in ids[k]]
                          for k in ("plain", "chat") if k in ids}
            nll, n = 0.0, 0
            for w in wins:
                a, b = ppl_from_logits(model(input_ids=torch.tensor([w])).logits[0].float().numpy(), w)
                nll, n = nll + a, n + b
            res["ppl_nll"], res["ppl_tokens"] = nll, n
        x = torch.tensor([wins[0]])
        model(input_ids=x)
        pre = []
        for _ in range(5):
            t.lap()
            model(input_ids=x)
            pre.append(len(wins[0]) / t.lap())
        p16 = torch.tensor([ids["plain"][0][:16]])
        dec = []
        for _ in range(3):
            t.lap()
            model.generate(p16, max_new_tokens=64, min_new_tokens=64, do_sample=False, pad_token_id=0)
            dec.append(64 / t.lap())
    res["prefill_tps"], res["decode_tps"] = pre, dec
    res["peak_rss_mb"] = peak_rss_mb()
    return res, (np.concatenate(logits) if logits is not None else None)
