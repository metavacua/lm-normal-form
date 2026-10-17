# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cell 03: a saved sub-term of SmolLM2 in the three harnesses on three prompts of the theory.
usage: harness32.py MODEL_DIR ONNX_DIR GGUF LLAMA_BIN_DIR OUT_JSON"""
import json
import os
import re
import subprocess
import sys

import numpy as np
import torch

PROMPTS = ["The capital of France is", "The capital of Germany is", "The capital of Japan is"]


def run(cmd, timeout=300):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def main():
    hf, onnx_dir, gguf, lbin, out_json = sys.argv[1:6]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(hf)
    model = AutoModelForCausalLM.from_pretrained(hf, dtype=torch.float32).eval()
    out = {"model_dir": hf, "layers": model.config.num_hidden_layers, "errors": {}, "pytorch": {}, "llamacpp": {}}
    with torch.inference_mode():
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt")["input_ids"]
            g = model.generate(ids, max_new_tokens=6, do_sample=False, use_cache=False, pad_token_id=tok.eos_token_id)
            out["pytorch"][p] = tok.decode(g[0][ids.shape[1]:])
    if os.path.exists(os.path.join(onnx_dir, "model.onnx")):
        try:
            import onnxruntime as ort
            sess = ort.InferenceSession(os.path.join(onnx_dir, "model.onnx"), providers=["CPUExecutionProvider"])
            names = [i.name for i in sess.get_inputs()]; worst = 0.0
            with torch.inference_mode():
                for p in PROMPTS:
                    ids = tok(p, return_tensors="pt")["input_ids"]
                    feed = {"input_ids": ids.numpy()}
                    if "attention_mask" in names:
                        feed["attention_mask"] = np.ones_like(ids.numpy())
                    if "position_ids" in names:
                        feed["position_ids"] = np.arange(ids.shape[1])[None].astype(np.int64)
                    ref = model(input_ids=ids, use_cache=False).logits.numpy(); got = sess.run(None, feed)[0]
                    worst = max(worst, float(np.abs(ref - got).max()))
            out["onnxruntime"] = {"inputs": names, "max_abs_diff_logits": worst}
        except Exception as e:
            out["errors"]["onnxruntime"] = repr(e)
    else:
        out["errors"]["onnxruntime"] = "no ONNX export"
    if os.path.exists(gguf):
        tokz = os.path.join(lbin, "llama-tokenize"); comp = os.path.join(lbin, "llama-completion")
        for p in PROMPTS:
            leg = {}
            try:
                rc, so, se = run([tokz, "-m", gguf, "-p", p, "--ids", "--log-disable"])
                m = re.search(r"\[([0-9,\s]*)\]", so)
                l_ids = [int(x) for x in m.group(1).split(",")] if m and m.group(1).strip() else None
                leg.update({"llama_ids": l_ids, "fast_ids": tok(p)["input_ids"], "tokenize_rc": rc, "tokenize_stderr_tail": se[-200:]})
                if l_ids:
                    with torch.inference_mode():
                        g2 = model.generate(torch.tensor([l_ids]), max_new_tokens=6, do_sample=False, use_cache=False, pad_token_id=tok.eos_token_id)
                    ref = tok.decode(g2[0][len(l_ids):]).strip()
                    rc, so, se = run([comp, "-m", gguf, "-p", p, "-n", "6", "--temp", "0", "--no-display-prompt", "--no-warmup", "--jinja", "-no-cnv"])
                    err = [l for l in se.splitlines() if "GGML_ASSERT" in l or "what()" in l]
                    leg.update({"pytorch_from_llama_ids": ref, "rc": rc, "continuation": so.strip(), "stderr_errors": err[:3], "equals_pytorch": bool(rc == 0 and so.strip() == ref)})
            except Exception as e:
                leg["error"] = repr(e)
            out["llamacpp"][p] = leg
    else:
        out["errors"]["llamacpp"] = "no GGUF"
    json.dump(out, open(out_json, "w"), indent=1)
    print(f"{os.path.basename(hf)} ({out['layers']} blocks): pytorch {list(out['pytorch'].values())}; onnx {out.get('onnxruntime') or out['errors'].get('onnxruntime')}; "
          f"llama equal {[v.get('equals_pytorch') for v in out['llamacpp'].values()] or out['errors'].get('llamacpp')}")


if __name__ == "__main__":
    main()
