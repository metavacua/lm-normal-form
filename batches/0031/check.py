# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cell 02: the exhaustive check of a constructed model against its theory under its interpretation.

usage: check.py MODEL_DIR THEORY OUT_JSON [--control swap|nan|null] [--taps TAPS_JSON]
THEORY is T1 (character alphabet, I1: "the capital of X is " + greedy characters until </s> or space) or T2 (word
alphabet, I2: "the capital of X is" + greedy next token). Reports soundness (every assertion on T's prompts is in T),
completeness (every atom of T is asserted), margins, over-generation on every other X, and the information rules:
input dependence I(X; Y) with X uniform over all candidate subjects, and context use (the same quantity, since the
last prompt token is fixed under both interpretations).
"""
import argparse
import hashlib
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "0029"))
import common  # noqa: E402  (batch 0029: Taps)
sys.path.insert(0, HERE)
from construct import CAPITALS, WORDS, CHARS  # noqa: E402

T1 = {"france": "paris"}
T2 = dict(CAPITALS)


def entropy(p):
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("theory"); ap.add_argument("out_json")
    ap.add_argument("--control", default="none"); ap.add_argument("--taps", default=None)
    a = ap.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model_dir)
    model = AutoModelForCausalLM.from_pretrained(a.model_dir, dtype=torch.float32).eval()
    V = model.config.vocab_size
    control = {"control": a.control}
    with torch.no_grad():
        if a.control == "swap":
            Wo = model.model.layers[0].self_attn.o_proj.weight
            i, j = tok.convert_tokens_to_ids("▁paris"), tok.convert_tokens_to_ids("▁berlin")
            row_i, row_j = Wo[i].clone(), Wo[j].clone(); Wo[i] = row_j; Wo[j] = row_i
            control.update({"swapped_rows": [i, j]})
        elif a.control == "nan":
            model.model.embed_tokens.weight[0, 0] = float("nan")
        elif a.control == "null":
            g = model.model.norm.weight
            v = g[0].item(); g[0] = torch.nextafter(torch.tensor(v), torch.tensor(float("inf"))).item()
            control.update({"before": v, "after": g[0].item()})
    theory = T1 if a.theory == "T1" else T2
    subjects = list(CAPITALS) if a.theory == "T1" else WORDS          # candidate X: the 12 countries (T1) or every word (T2)
    taps = common.Taps(model) if a.taps else None
    results = {}; dists = {}; finite = True; h = hashlib.sha256()
    with torch.inference_mode():
        for X in subjects:
            if a.theory == "T1":
                prompt = f"the capital of {X} is "
                ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
                gen = model.generate(ids, max_new_tokens=10, do_sample=False, eos_token_id=2, pad_token_id=3)
                cont_ids = gen[0][ids.shape[1]:].tolist()
                text = tok.decode([t for t in cont_ids if t != 2])
                assertion = text.split(" ")[0] if " " in text else text
                stopped = (2 in cont_ids) or (" " in text)
                lg = model(input_ids=ids).logits[0, -1].double()
            else:
                prompt = f"the capital of {X} is"
                ids = tok(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"]
                lg = model(input_ids=ids).logits[0, -1].double()
                assertion = tok.decode([int(lg.argmax())]).strip()
                stopped = True
            p = torch.softmax(lg, -1).numpy()
            if not np.isfinite(lg.numpy()).all():
                finite = False
            h.update(np.ascontiguousarray(lg.numpy().astype(np.float32)).tobytes())
            top2 = torch.topk(lg, 2).values.tolist()
            results[X] = {"prompt": prompt, "prompt_ids": ids[0].tolist(), "assertion": assertion, "stopped": stopped,
                          "in_theory": X in theory and theory[X] == assertion, "expected": theory.get(X), "margin": top2[0] - top2[1],
                          "p_top": float(p.max())}
            dists[X] = p
    # soundness / completeness over T's prompts; over-generation on the rest
    on_T = {X: r for X, r in results.items() if X in theory}
    sound = all(r["in_theory"] for r in on_T.values())
    complete = all(results[X]["assertion"] == y for X, y in theory.items())
    over = {X: r["assertion"] for X, r in results.items() if X not in theory}
    # information rules: X uniform over the candidates
    P = np.stack([dists[X] for X in subjects])                   # (n, V)
    marg = P.mean(axis=0)
    mi = entropy(marg) - float(np.mean([entropy(row) for row in P]))
    # the same over T's subjects only
    PT = np.stack([dists[X] for X in theory]); mi_T = entropy(PT.mean(axis=0)) - float(np.mean([entropy(r) for r in PT]))
    out = {"model_dir": a.model_dir, "theory": a.theory, "control": control, "finite": finite, "sound": sound, "complete": complete,
           "assertions_on_T": {X: r["assertion"] for X, r in on_T.items()}, "expected": theory,
           "failures_on_T": [X for X, r in on_T.items() if not r["in_theory"]],
           "min_margin_on_T": min(r["margin"] for r in on_T.values()), "over_generation": over,
           "input_dependence_bits_all_candidates": mi, "input_dependence_bits_T_subjects": mi_T, "log2_T": math.log2(len(theory)),
           "candidates": len(subjects), "logits_sha256": h.hexdigest(), "results": results}
    if taps:
        common.dump(a.taps, taps.vectors()); taps.remove(); out["taps_written"] = a.taps
    json.dump(out, open(a.out_json, "w"), indent=1)
    print(f"{os.path.basename(a.model_dir)} / {a.theory} {a.control}: sound {sound} complete {complete} failures {out['failures_on_T']} "
          f"min margin {out['min_margin_on_T']:.3f} MI(all) {mi:.3f} bits MI(T) {mi_T:.3f} bits over-generation sample {dict(list(over.items())[:4])}")


if __name__ == "__main__":
    main()
