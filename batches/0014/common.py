# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# What every runtime of batch 0014 is given and how what comes out is compared. The runtimes never tokenize: token ids come from the Hugging Face
# tokenizer of the checkpoint, in files, so that a difference between runtimes is a difference of the model's arithmetic and not of a tokenizer
# (the tokenizers themselves are compared in their own cell).
#   test set     the 24 sequences of batch 0013's evaluation set that come first (3 prompts and 21 sentences): the logits of every position
#   generation   greedy, 32 new tokens, from 8 plain completion prompts (and, for an instruct model, 8 chat prompts through its chat template)
#   perplexity   16 windows of 512 tokens of the WikiText-2 test set
#   speed        prefill of 512 tokens (median of 5), decode of 64 tokens after a 16-token prompt (median of 3)
# Metrics (logits against a reference, positions pooled): the largest and the mean absolute difference; the normalized mean square difference,
# sum (a - b)^2 / sum b^2 (the figure llama.cpp's conversion checks use); the mean Kullback-Leibler divergence of the next-token distributions in
# nats; top-1 agreement; whether every logit is bit for bit the same.
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))

PLAIN_PROMPTS = ["The capital of France is", "Water boils at one hundred degrees Celsius at sea level, and", "In 1969, the first human to walk on the Moon was",
                 "The three primary colors are red, blue, and", "def fibonacci(n):\n    if n < 2:\n        return n\n    return", "Once upon a time, in a small village by the sea,",
                 "The derivative of x squared with respect to x is", "To make a cup of tea, first"]
CHAT_PROMPTS = ["What is the capital of France?", "Write one sentence about the Moon.", "Name three primary colors.", "What is 12 times 12?",
                "Explain in one sentence what a matrix is.", "Give me a word that rhymes with cat.", "Translate 'good morning' into Spanish.", "What do bees make?"]


def texts_and_ids(tok, chat):
    """Token ids of every input of the batch: dict with 'test' (list of id lists), 'plain', 'chat'."""
    t13 = os.path.join(HERE, "..", "0013", "sentences.txt")
    t10 = os.path.join(HERE, "..", "0010", "prompts.txt")
    seqs = open(t10, encoding="utf-8").read().splitlines() + open(t13, encoding="utf-8").read().splitlines()
    out = {"test": [tok(t)["input_ids"] for t in seqs[:24]], "plain": [tok(t)["input_ids"] for t in PLAIN_PROMPTS]}
    if chat and getattr(tok, "chat_template", None):
        out["chat"] = [tok.apply_chat_template([{"role": "user", "content": m}], add_generation_prompt=True, tokenize=True, return_dict=True)["input_ids"] for m in CHAT_PROMPTS]
    return out


def wikitext_windows(tok, n=16, size=512):
    """n windows of `size` token ids from the WikiText-2 raw test set, and the text they were cut from (for tools that tokenize themselves)."""
    from huggingface_hub import hf_hub_download
    import pandas as pd
    path = hf_hub_download("Salesforce/wikitext", "wikitext-2-raw-v1/test-00000-of-00001.parquet", repo_type="dataset")
    text = "\n\n".join(pd.read_parquet(path)["text"].tolist())
    ids = tok(text)["input_ids"]
    wins = [ids[i * size:(i + 1) * size] for i in range(n) if (i + 1) * size <= len(ids)]
    return wins, tok.decode(sum(wins, []))


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max(axis=-1, keepdims=True)
    return x - m - np.log(np.exp(x - m).sum(axis=-1, keepdims=True))


def logit_metrics(a, b):
    """Metrics of logits a (the run) against logits b (the reference), both (positions, vocab) float32."""
    d = a.astype(np.float64) - b.astype(np.float64)
    la, lb = log_softmax(a), log_softmax(b)
    kl = (np.exp(lb) * (lb - la)).sum(axis=-1)
    return {"positions": int(len(a)), "max_abs": float(np.abs(d).max()), "mean_abs": float(np.abs(d).mean()),
            "nmse": float((d ** 2).sum() / (b.astype(np.float64) ** 2).sum()), "mean_kl": float(kl.mean()), "max_kl": float(kl.max()),
            "top1_agree": float((a.argmax(-1) == b.argmax(-1)).mean()), "bit_identical": bool(np.array_equal(a, b))}


def ppl_from_logits(logits, ids):
    """Negative log-likelihood sum and token count of ids[1:] given logits for positions 0..n-2 (logits has one row per id)."""
    lp = log_softmax(logits[:-1])
    return float(-lp[np.arange(len(ids) - 1), np.asarray(ids[1:])].sum()), len(ids) - 1


def median(xs):
    return float(np.median(xs)) if len(xs) else None


def first_divergence(a, b):
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


class Timer:
    def __init__(self):
        self.t = time.perf_counter()

    def lap(self):
        now = time.perf_counter()
        r, self.t = now - self.t, now
        return r


def peak_rss_mb():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0


def dump(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=1, sort_keys=True)


if __name__ == "__main__":
    sys.exit("common.py is a library")
