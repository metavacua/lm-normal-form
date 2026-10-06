# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Does the one definition of the forward pass (model.py), run in float64, compute what Hugging Face's Llama computes? The claims proved over F_p are about that definition; this ties
# it to the model that is run. The checkpoint in MODEL_DIR is run (1) by Transformers in float64 with the two float32 islands of its Llama removed (the RMSNorm casts to float32
# inside, and the rotary tables are computed from a float32 inverse frequency; either would hide differences below 1e-6; the eager softmax is a third, avoided by using the default
# SDPA attention) and (2) by Transformers in float32 as shipped, and both are compared with model.py's float64 logits on the prompts of batch 0010 and 21 sentences.
# Usage: conform.py MODEL_DIR OUT.json
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hf_adapter as H
from model import forward
from ops import FloatOps


def main(model_dir, out):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    a, P, cfg = H.load(model_dir)
    ops = FloatOps(eps=cfg["rms_norm_eps"], base=cfg.get("rope_theta", 10000.0))
    tok = AutoTokenizer.from_pretrained(model_dir)
    texts = open(os.path.join(HERE, "..", "0010", "prompts.txt"), encoding="utf-8").read().splitlines() + open(os.path.join(HERE, "..", "0013", "sentences.txt"), encoding="utf-8").read().splitlines()[:21]
    seqs = [tok(t)["input_ids"] for t in texts]
    mine = [forward(a, P, s, ops) for s in seqs]
    res = {"model_dir": model_dir, "arch": vars(a), "sequences": len(seqs), "positions": int(sum(len(s) for s in seqs)), "comparisons": {}}
    for name, dtype, patch in (("transformers float64, norm and rotary tables in float64", torch.float64, True), ("transformers float32, as shipped", torch.float32, False)):
        restore = H.float64_patches(cfg) if patch else (lambda: None)
        model = AutoModelForCausalLM.from_pretrained(model_dir, dtype=dtype).eval()
        worst, rel, top1 = 0.0, 0.0, 0
        with torch.no_grad():
            for s, m in zip(seqs, mine):
                L = model(input_ids=torch.tensor([s])).logits[0].double().numpy()
                worst = max(worst, float(np.abs(L - m).max()))
                rel = max(rel, float(np.abs(L - m).max() / max(1.0, np.abs(m).max())))
                top1 += int((L.argmax(-1) == m.argmax(-1)).sum())
        res["comparisons"][name] = {"max_abs_difference": worst, "max_relative_to_scale": rel, "top1_agreement": top1 / res["positions"]}
        print(name, res["comparisons"][name], flush=True)
        del model
        restore()
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
