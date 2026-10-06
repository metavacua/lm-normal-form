# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Permuting the tokens, in real models (claims T1 to T6 of claims.py on real numbers). For random non-trivial permutations pi of the tokens of a sentence, the equivariance error
#   E(pi) = max over positions i, outputs v of | out(x o pi)[i][v] - out(x)[pi(i)][v] |
# (zero if the model is permutation equivariant); for the decoder the outputs are logits, and the mean KL of the next-token distributions of the two and the share of positions with the
# same argmax are kept as well.
#   SmolLM2-135M-Instruct, float32: a causal decoder with rotary embeddings (T4, expected: not equivariant)
#   BERT-tiny (Google's, 2 layers, hidden 128) in float64, the hidden states of the encoder, with its position embeddings as they are (expected: not equivariant) and with them set to zero
#   (T1: expected equivariant, to float64 rounding)
#   SmolLM2 and the last position: the logits of the last token under permutations of the earlier tokens (T5/T6: a causal model with one layer would be invariant; a 30-layer one is not)
# Usage: lm_tokenperm.py SMOL_DIR OUT.json       (LMNF_BERT and LMNF_BERT_REV name the encoder)
import json, os, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def kl(a, b):
    a, b = a.astype(np.float64), b.astype(np.float64)
    la = a - a.max(-1, keepdims=True)
    la -= np.log(np.exp(la).sum(-1, keepdims=True))
    lb = b - b.max(-1, keepdims=True)
    lb -= np.log(np.exp(lb).sum(-1, keepdims=True))
    return float((np.exp(lb) * (lb - la)).sum(-1).mean())


def stats(model_logits, seqs, rng, n_perm=4, logits=True):
    """model_logits(ids) -> (T, V). Returns the equivariance error over several permutations of each sequence, and for logits the KL and the argmax agreement."""
    errs, kls, same = [], [], []
    for s in seqs:
        base = model_logits(s)
        for _ in range(n_perm):
            while True:
                pi = rng.permutation(len(s))
                if len(set(s[i] for i in pi)) > 1 and [s[i] for i in pi] != s:
                    break
            L = model_logits([s[i] for i in pi])
            want = base[pi]                              # the equivariant answer: row i of the permuted run is row pi(i) of the original
            errs.append(float(np.abs(L - want).max()) / max(1e-300, float(np.abs(want).max())))
            if logits:
                kls.append(kl(L, want))
                same.append(float((L.argmax(-1) == want.argmax(-1)).mean()))
    r = {"max_relative_equivariance_error": max(errs), "median_relative_equivariance_error": float(np.median(errs)), "permutations": len(errs)}
    if logits:
        r.update({"mean_kl": float(np.mean(kls)), "argmax_agreement": float(np.mean(same))})
    return r


def main(smol_dir, out):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
    torch.set_num_threads(int(os.environ.get("LMNF_THREADS", "4")))
    rng = np.random.default_rng(15)
    res = {}
    tok = AutoTokenizer.from_pretrained(smol_dir)
    sents = open(os.path.join(HERE, "..", "0013", "sentences.txt"), encoding="utf-8").read().splitlines()[:8]
    seqs = [tok(t)["input_ids"] for t in sents]
    m = AutoModelForCausalLM.from_pretrained(smol_dir, dtype=torch.float32).eval()

    def causal(ids):
        with torch.no_grad():
            return m(input_ids=torch.tensor([ids])).logits[0].numpy()
    res["smollm2_causal_rope_float32"] = stats(causal, seqs, rng)
    last = []                                            # the last position under permutations of the earlier tokens
    for s in seqs:
        base = causal(s)[-1]
        for _ in range(4):
            pre = list(rng.permutation(len(s) - 1))
            L = causal([s[i] for i in pre] + [s[-1]])[-1]
            last.append((float(np.abs(L - base).max()), kl(L[None], base[None]), float(L.argmax() == base.argmax())))
    res["smollm2_last_position_under_permutation_of_earlier_tokens"] = {"max_logit_difference": max(x[0] for x in last), "median_logit_difference": float(np.median([x[0] for x in last])),
                                                                          "mean_kl": float(np.mean([x[1] for x in last])), "argmax_agreement": float(np.mean([x[2] for x in last]))}
    m = None
    # BERT-tiny: with and without its position embeddings
    name = os.environ.get("LMNF_BERT", "google/bert_uncased_L-2_H-128_A-2")
    rev = os.environ.get("LMNF_BERT_REV", "30b0a37ccaaa32f332884b96992754e246e48c5f")
    bt = AutoTokenizer.from_pretrained(name, revision=rev)
    bseqs = [bt(t)["input_ids"] for t in sents]
    for label, zero in (("bert_tiny_as_shipped_float64", False), ("bert_tiny_position_embeddings_zero_float64", True)):
        b = AutoModel.from_pretrained(name, revision=rev, dtype=torch.float64, use_safetensors=True).eval()
        if zero:
            b.embeddings.position_embeddings.weight.data.zero_()

        def enc(ids, b=b):
            with torch.no_grad():
                return b(input_ids=torch.tensor([ids])).last_hidden_state[0].numpy()
        res[label] = stats(enc, bseqs, rng, logits=False)
        del b
    json.dump(res, open(out, "w"), indent=1)
    for k, v in res.items():
        print(k, {a: (round(x, 6) if isinstance(x, float) else x) for a, x in v.items()}, flush=True)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
