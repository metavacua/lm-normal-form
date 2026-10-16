# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cells 02 and 06: the instance in the three harnesses, and the working test against two fixed languages.

usage: working.py HF_DIR ONNX_DIR GGUF_FILE LLAMA_BIN_DIR TEXT_FILE OUT_JSON
02: transformers generate through the chat template; ONNX Runtime logits against PyTorch; llama.cpp greedy text
    against PyTorch greedy text on three prompts.
06: L1, English: bits per character of the text under PyTorch (cross-entropy per symbol) and under llama-perplexity,
    against the uniform floor log2(50). L2, the capital relation: mass on the correct first character after
    "the capital of X is " for twelve countries, against 1/50.
"""
import json
import math
import os
import re
import subprocess
import sys

import numpy as np
import torch

ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789.,?!:-'() "
PROMPTS = ["the capital of france is", "hello, how are you?", "1 2 3 4 5 6"]
CAPITALS = {"france": "p", "germany": "b", "italy": "r", "spain": "m", "japan": "t", "china": "b", "russia": "m",
            "egypt": "c", "canada": "o", "brazil": "b", "india": "n", "australia": "c"}


def run(cmd, timeout=600):
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    return p.returncode, p.stdout, p.stderr


def restrict(text):
    return "".join(ch for ch in text.lower().replace("\n", " ") if ch in ALPHABET)


def main():
    hf, onnx_dir, gguf, lbin, text_file, out_json = sys.argv[1:7]
    from transformers import AutoModelForCausalLM, AutoTokenizer
    out = {"errors": {}}
    try:
        tok = AutoTokenizer.from_pretrained(hf)
    except Exception as e:                       # the same directory through the slow SentencePiece reader
        out["errors"]["fast_tokenizer"] = repr(e)
        tok = AutoTokenizer.from_pretrained(hf, use_fast=False)
    out["tokenizer_class"] = type(tok).__name__
    model = AutoModelForCausalLM.from_pretrained(hf, torch_dtype=torch.float32).eval()
    # 02a transformers, chat
    try:
        msgs = [{"role": "user", "content": "the capital of france is"}]
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt")
        ids = ids["input_ids"] if isinstance(ids, dict) or hasattr(ids, "keys") else ids
        with torch.inference_mode():
            gen = model.generate(ids, max_new_tokens=32, do_sample=False)
        out["chat"] = {"prompt_ids": ids[0].tolist(), "prompt_text": tok.decode(ids[0]), "output_text": tok.decode(gen[0][ids.shape[1]:]),
                       "output_ids": gen[0][ids.shape[1]:].tolist()}
    except Exception as e:
        out["errors"]["chat"] = repr(e)
    # PyTorch greedy on the three prompts (16 tokens), the reference for the other legs
    greedy = {}
    with torch.inference_mode():
        for p in PROMPTS:
            ids = tok(p, return_tensors="pt", add_special_tokens=False)["input_ids"]
            gen = model.generate(ids, max_new_tokens=16, do_sample=False)
            greedy[p] = {"ids": gen[0][ids.shape[1]:].tolist(), "text": tok.decode(gen[0][ids.shape[1]:])}
    out["pytorch_greedy"] = greedy
    # 02b ONNX Runtime logits
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(os.path.join(onnx_dir, "model.onnx"), providers=["CPUExecutionProvider"])
        names = [i.name for i in sess.get_inputs()]
        worst = 0.0
        with torch.inference_mode():
            for p in PROMPTS:
                ids = tok(p, return_tensors="pt", add_special_tokens=False)["input_ids"]
                feed = {"input_ids": ids.numpy()}
                if "attention_mask" in names:
                    feed["attention_mask"] = np.ones_like(ids.numpy())
                if "position_ids" in names:
                    feed["position_ids"] = np.arange(ids.shape[1])[None].astype(np.int64)
                ref = model(input_ids=ids).logits.numpy()
                got = sess.run(None, feed)[0]
                worst = max(worst, float(np.abs(ref - got).max()))
        out["onnxruntime"] = {"inputs": names, "max_abs_diff_logits": worst}
    except Exception as e:
        out["errors"]["onnxruntime"] = repr(e)
    # 02c llama.cpp: its own tokenization of each prompt (llama-tokenize --ids), compared with the fast tokenizer's; its
    # greedy continuation (llama-completion, temperature 0), compared with PyTorch's greedy continuation from the SAME ids
    try:
        tokz = os.path.join(lbin, "llama-tokenize"); comp = os.path.join(lbin, "llama-completion")
        legs = {}
        for p in PROMPTS:
            leg = {"tool": "llama-completion" if os.path.exists(comp) else None}
            rc, so, se = run([tokz, "-m", gguf, "-p", p, "--ids", "--log-disable"])
            m = re.search(r"\[([0-9,\s]*)\]", so)
            l_ids = [int(x) for x in m.group(1).split(",")] if m and m.group(1).strip() else None
            f_ids = tok(p, add_special_tokens=False)["input_ids"]
            leg.update({"llama_ids": l_ids, "fast_ids": f_ids, "tokenization_equal": l_ids == f_ids, "tokenize_rc": rc})
            if l_ids:
                with torch.inference_mode():
                    g2 = model.generate(torch.tensor([l_ids]), max_new_tokens=16, do_sample=False)
                leg["pytorch_from_llama_ids"] = {"ids": g2[0][len(l_ids):].tolist(), "text": tok.decode(g2[0][len(l_ids):])}
            if leg["tool"]:
                rc, so, se = run([comp, "-m", gguf, "-p", p, "-n", "16", "--temp", "0", "--no-display-prompt", "--no-warmup", "--jinja", "-no-cnv"])
                leg.update({"rc": rc, "stdout": so[-400:], "stderr_tail": se[-300:], "continuation": so.strip()})
                ref = (leg.get("pytorch_from_llama_ids") or greedy[p])["text"].strip()
                leg["equals_pytorch"] = bool(rc == 0 and so.strip()[:16] == ref[:16])
            else:
                leg.update({"rc": None, "stdout": "", "stderr_tail": "no llama-completion in " + lbin, "continuation": "", "equals_pytorch": False})
            legs[p] = leg
        out["llamacpp"] = legs
    except Exception as e:
        out["errors"]["llamacpp"] = repr(e)
    # 06 L1: English bits per character
    text = restrict(open(text_file, encoding="utf-8", errors="replace").read())[:16384]
    ids = tok(text, add_special_tokens=False)["input_ids"]
    nll = 0.0; n = 0
    with torch.inference_mode():
        for s in range(0, len(ids) - 1, 64):
            chunk = torch.tensor([ids[s:s + 65]])
            if chunk.shape[1] < 2:
                break
            lg = model(input_ids=chunk[:, :-1]).logits[0].double()
            lp = torch.log_softmax(lg, -1)
            tgt = chunk[0, 1:]
            nll += float(-lp[torch.arange(len(tgt)), tgt].sum()); n += len(tgt)
    bits_pt = nll / n / math.log(2)
    l1 = {"characters": len(text), "tokens": len(ids), "bits_per_char_pytorch": bits_pt, "uniform_floor_bits": math.log2(50),
          "text_sha256": __import__("hashlib").sha256(text.encode()).hexdigest()}
    try:
        with open("work/l1_text.txt", "w") as f:
            f.write(text)
        ppl = os.path.join(lbin, "llama-perplexity")
        rc, so, se = run([ppl, "-m", gguf, "-f", "work/l1_text.txt", "-c", "64", "-b", "64", "--chunks", "64"], timeout=900)
        m = re.search(r"Final estimate: PPL = ([0-9.]+)", so + se)
        l1["llama_perplexity"] = {"rc": rc, "ppl": float(m.group(1)) if m else None, "bits_per_char": (math.log2(float(m.group(1))) if m else None),
                                  "tail": (so + se)[-300:]}
    except Exception as e:
        l1["llama_perplexity"] = {"error": repr(e)}
    out["L1_english"] = l1
    # 06 L2: the capital relation, mass on the correct first character
    l2 = {}
    with torch.inference_mode():
        for country, first in CAPITALS.items():
            ids = tok(f"the capital of {country} is ", return_tensors="pt", add_special_tokens=False)["input_ids"]
            p = torch.softmax(model(input_ids=ids).logits[0, -1].double(), -1)
            cid = tok(first, add_special_tokens=False)["input_ids"][-1]
            l2[country] = {"correct_char": first, "p_correct": float(p[cid]), "ratio_to_uniform": float(p[cid] * 50), "argmax": tok.decode([int(p.argmax())])}
    within3 = sum(1 for v in l2.values() if 1 / 3 <= v["ratio_to_uniform"] <= 3)
    out["L2_capitals"] = {"countries": l2, "within_factor_3_of_uniform": within3}
    json.dump(out, open(out_json, "w"), indent=1)
    print(f"chat: {out.get('chat', {}).get('output_text', out['errors'].get('chat'))!r}; onnx max diff {out.get('onnxruntime', {}).get('max_abs_diff_logits', out['errors'].get('onnxruntime'))}; "
          f"llama.cpp equals pytorch: {[v['equals_pytorch'] for v in out.get('llamacpp', {}).values()] if 'llamacpp' in out else out['errors'].get('llamacpp')}; "
          f"L1 bits/char pytorch {bits_pt:.3f} (floor {math.log2(50):.3f}), llama-perplexity {l1.get('llama_perplexity', {}).get('bits_per_char')}; L2 within x3 of uniform: {within3}/12")


if __name__ == "__main__":
    main()
