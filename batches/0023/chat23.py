# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Batch 0023, the same conversation, and the same request for a tool call, given to a language model and to variants of it whose weights have been transformed by the gauge group. A language model that
# converses and makes tool calls is a function from the text so far to the next token, iterated; the transformations leave that function unchanged (in exact arithmetic), so the variants must say the same
# thing, token for token, if the claim is right. The checkpoints: Qwen3-0.6B, whose chat template takes a list of tools and whose training makes it answer with a <tool_call> block in an agreed JSON syntax,
# and SmolLM2-135M-Instruct, whose template is a plain conversation. float64, greedy decoding (the largest logit, the first on a tie), 64 new tokens, Transformers on the CPU.
# Variants (each applied to a copy of the checkpoint): H1 a signed permutation of the hidden coordinates with the norm weights permuted along; M1 the units of the feed-forward block permuted and the up
# row of each scaled by +-2^k with the down column by the inverse; A1 an invertible matrix on the value dimensions of each key/value group; A6 two heads of the first group exchanged; H4 the norm weights
# folded, the head untied and the stream rotated by a random orthogonal matrix; for Qwen3 also T3 (a positive scalar on the query projection of each head) and T4 (a scalar per plane moved from the query
# gain to the key gain); ALL the composition of H1, M1, A1, A6 (and T3, T4 for Qwen3). The control `broken`: H1 applied to the layers but not to the embedding (and so not to the tied head).
# Usage: chat23.py KEY REPO REV OUT.json [--random-small]       KEY: qwen3 or smollm2 (the architecture for the random small model of the smoke run)
import gc, json, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hfx23 as X  # noqa: E402

TOOLS = [{"type": "function", "function": {"name": "get_weather", "description": "Get the current weather in a given city.",
                                           "parameters": {"type": "object", "properties": {"city": {"type": "string", "description": "The name of the city."}}, "required": ["city"]}}}]
PROMPTS = {"tool_call": ([{"role": "user", "content": "What is the weather in Paris right now? Use the tool."}], TOOLS),
           "conversation": ([{"role": "user", "content": "Say hello in one short sentence."}], None)}
NEW_TOKENS = 64


def prompt_ids(tok, key, which):
    messages, tools = PROMPTS[which]
    kw = {"tools": tools} if tools else {}
    if key == "qwen3":
        kw["enable_thinking"] = False
    text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kw)
    return tok(text, return_tensors="pt", add_special_tokens=False)["input_ids"]


def generate(m, tok, ids):
    with torch.no_grad():
        out = m.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=NEW_TOKENS, do_sample=False, pad_token_id=tok.eos_token_id if tok.eos_token_id is not None else 0)
    return [int(t) for t in out[0, ids.shape[1]:]]


def variants(m, key):
    d, nh, nkv, hd, dff = X.dims(m)
    rs = np.random.RandomState(3)
    perm = [int(i) for i in rs.permutation(d)]
    signs = [int(s) for s in rs.choice([1, -1], d)]
    uperm = [int(i) for i in rs.permutation(dff)]
    uscale = np.sign(rs.standard_normal(dff)) * 2.0 ** rs.randint(-3, 4, dff)
    As = [np.linalg.qr(rs.standard_normal((hd, hd)))[0] @ np.diag(np.exp(rs.uniform(-0.5, 0.5, hd))) @ np.linalg.qr(rs.standard_normal((hd, hd)))[0] for _ in range(nkv)]
    c3, s4 = np.exp(0.5 * rs.standard_normal(nh)), np.exp(0.5 * rs.standard_normal(hd // 2))
    Q = X.random_orthogonal(d, 78)

    def broken():
        m2 = X.signed_permutation(m, perm, signs)
        with torch.no_grad():
            m2.model.embed_tokens.weight.copy_(m.model.embed_tokens.weight)
        return m2

    def allv():
        m2 = X.signed_permutation(m, perm, signs)
        X.unit_gauge(m2, uperm, uscale, inplace=True)
        X.value_output_gauge(m2, As, inplace=True)
        X.head_swap(m2, inplace=True)
        if key == "qwen3":
            X.head_scalar(m2, c3, inplace=True)
            X.qk_gain_move(m2, s4, inplace=True)
        return m2

    def rot():
        mf = X.fold_untie(m)
        X.rotate(mf, Q, inplace=True)
        return mf
    v = {"H1_signed_permutation": lambda: X.signed_permutation(m, perm, signs), "M1_units": lambda: X.unit_gauge(m, uperm, uscale), "A1_value_output": lambda: X.value_output_gauge(m, As),
         "A6_head_swap": lambda: X.head_swap(m), "H4_rotation_of_the_stream": rot, "ALL": allv}
    if key == "qwen3":
        v["T3_head_scalars"] = lambda: X.head_scalar(m, c3)
        v["T4_gain_move"] = lambda: X.qk_gain_move(m, s4)
    v["broken_control"] = broken
    return v


def main(key, repo, rev, out, random_small):
    mt = "qwen3" if key == "qwen3" else "llama"
    t0 = time.time()
    m, tok, cfg, restore = X.load_model(mt, repo, rev, random_small=random_small)
    print(f"loaded in {time.time() - t0:.0f}s", flush=True)
    res = {"key": key, "repo": repo, "revision": rev, "new_tokens": NEW_TOKENS, "prompts": {}}
    try:
        if tok is None:
            raise SystemExit("the chat test needs a tokenizer with a chat template: no random small run")
        which = ["tool_call", "conversation"] if key == "qwen3" else ["conversation"]
        ids = {w: prompt_ids(tok, key, w) for w in which}
        ref = {w: generate(m, tok, ids[w]) for w in which}
        for w in which:
            text = tok.decode(ref[w], skip_special_tokens=False)
            res["prompts"][w] = {"prompt_tokens": int(ids[w].shape[1]), "original_text": text, "original_contains_tool_call_tag": "<tool_call>" in text, "variants": {}}
            print(w, "->", text[:300].replace("\n", " "), flush=True)
        for name, make in variants(m, key).items():
            m2 = make()
            for w in which:
                g = generate(m2, tok, ids[w])
                first = next((i for i, (a, b) in enumerate(zip(g, ref[w])) if a != b), None if len(g) == len(ref[w]) else min(len(g), len(ref[w])))
                res["prompts"][w]["variants"][name] = {"identical_tokens": g == ref[w], "first_difference": first, "text": tok.decode(g, skip_special_tokens=False)}
                print(f"{w:12s} {name:28s} identical {g == ref[w]}  first difference {first}", flush=True)
            del m2
            gc.collect()
    finally:
        restore()
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], None if sys.argv[2] == "-" else sys.argv[2], None if sys.argv[3] == "-" else sys.argv[3], sys.argv[4], "--random-small" in sys.argv)
