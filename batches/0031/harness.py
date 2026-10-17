# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cell 03: a constructed model in the three harnesses on its theory's prompts.

usage: harness.py MODEL_DIR THEORY ONNX_DIR GGUF LLAMA_BIN_DIR OUT_JSON
transformers greedy (the reference); ONNX Runtime logits against PyTorch; llama.cpp's own tokenization (llama-tokenize)
and greedy continuation (llama-completion) against PyTorch's greedy from the same ids. A missing carrier is recorded,
not an error.
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from construct import CAPITALS  # noqa: E402


def run(cmd, timeout=300):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def main():
    hf, theory, onnx_dir, gguf, lbin, out_json = sys.argv[1:7]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.float32).eval()
    prompts = [f"the capital of {X} is " if theory == "T1" else f"the capital of {X} is" for X in list(CAPITALS)[:3]]
    n_new = 8 if theory == "T1" else 1
    out = {"theory": theory, "errors": {}, "pytorch": {}, "onnxruntime": None, "llamacpp": {}}
    with torch.inference_mode():
        for p in prompts:
            ids = tok(p, return_tensors="pt", add_special_tokens=False)["input_ids"]
            gen = model.generate(ids, max_new_tokens=n_new, do_sample=False, eos_token_id=2, pad_token_id=3)
            out["pytorch"][p] = {"ids": gen[0][ids.shape[1]:].tolist(), "text": tok.decode(gen[0][ids.shape[1]:])}
    if os.path.exists(os.path.join(onnx_dir, "model.onnx")):
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(os.path.join(onnx_dir, "model.onnx"), providers=["CPUExecutionProvider"])
            names = [i.name for i in sess.get_inputs()]; worst = 0.0
            with torch.inference_mode():
                for p in prompts:
                    ids = tok(p, return_tensors="pt", add_special_tokens=False)["input_ids"]
                    feed = {"input_ids": ids.numpy()}
                    if "attention_mask" in names:
                        feed["attention_mask"] = np.ones_like(ids.numpy())
                    if "position_ids" in names:
                        feed["position_ids"] = np.arange(ids.shape[1])[None].astype(np.int64)
                    ref = model(input_ids=ids).logits.numpy(); got = sess.run(None, feed)[0]
                    worst = max(worst, float(np.abs(ref - got).max()))
            out["onnxruntime"] = {"inputs": names, "max_abs_diff_logits": worst}
        except Exception as e:
            out["errors"]["onnxruntime"] = repr(e)
    else:
        out["errors"]["onnxruntime"] = "no ONNX export"
    if os.path.exists(gguf):
        tokz = os.path.join(lbin, "llama-tokenize"); comp = os.path.join(lbin, "llama-completion")
        for p in prompts:
            leg = {}
            try:
                rc, so, se = run([tokz, "-m", gguf, "-p", p, "--ids", "--log-disable"])
                m = re.search(r"\[([0-9,\s]*)\]", so)
                l_ids = [int(x) for x in m.group(1).split(",")] if m and m.group(1).strip() else None
                leg.update({"llama_ids": l_ids, "fast_ids": tok(p, add_special_tokens=False)["input_ids"], "tokenize_rc": rc})
                if l_ids:
                    with torch.inference_mode():
                        g2 = model.generate(torch.tensor([l_ids]), max_new_tokens=n_new, do_sample=False, eos_token_id=2, pad_token_id=3)
                    ref = tok.decode(g2[0][len(l_ids):]).strip()
                    leg["pytorch_from_llama_ids"] = ref
                    rc, so, se = run([comp, "-m", gguf, "-p", p, "-n", str(n_new), "--temp", "0", "--no-display-prompt", "--no-warmup", "--jinja", "-no-cnv"])
                    leg.update({"rc": rc, "continuation": so.strip(), "stderr_tail": se[-200:], "equals_pytorch": bool(rc == 0 and so.strip() == ref)})
            except Exception as e:
                leg["error"] = repr(e)
            out["llamacpp"][p] = leg
    else:
        out["errors"]["llamacpp"] = "no GGUF"
    json.dump(out, open(out_json, "w"), indent=1)
    print(f"{os.path.basename(hf)}: pytorch {[v['text'] for v in out['pytorch'].values()]}; onnx {out['onnxruntime'] or out['errors'].get('onnxruntime')}; "
          f"llama equal {[v.get('equals_pytorch') for v in out['llamacpp'].values()] or out['errors'].get('llamacpp')}")


if __name__ == "__main__":
    main()
