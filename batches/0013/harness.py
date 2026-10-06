# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The model, what it is run on and how a variant of it is compared with it, shared by batches 0013 and 0014. The reference is the model
# in float32 (its bfloat16 weights widened, which is exact). A variant is stored as float32 or rounded to bfloat16, loaded into the same
# architecture, and run, in float32, on the 63 sequences of sentences.txt and batch 0010's three prompts. Against the reference: the
# largest and the mean absolute difference of the logits at every position, the mean Kullback-Leibler divergence of the next-token
# distribution (nats), the share of positions with the same argmax, whether the five most probable tokens after each prompt are the
# same ids in the same order, and whether anything is not finite.
import os
import torch

HERE = os.path.dirname(os.path.abspath(__file__))


def load(model_id):
    """The model, in float32, at the revision in HF_REVISION (the head of main if it is unset)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer
    rev = os.environ.get("HF_REVISION") or None
    tok = AutoTokenizer.from_pretrained(model_id, revision=rev)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.float32, revision=rev)
    model.eval()
    return model, tok


def sequences(tok, size=16):
    """The sentences, then the prompts, in right-padded chunks: (ids, mask, last position of each sequence)."""
    texts = open(os.path.join(HERE, "sentences.txt"), encoding="utf-8").read().splitlines()
    texts += open(os.path.join(HERE, "..", "0010", "prompts.txt"), encoding="utf-8").read().splitlines()
    out = []
    for i in range(0, len(texts), size):
        e = tok(texts[i:i + size], return_tensors="pt", padding=True, padding_side="right")
        out.append((e["input_ids"], e["attention_mask"].bool()))
    return out, len(texts)


@torch.no_grad()
def run(model, chunks):
    return [model(input_ids=ids, attention_mask=m.long(), use_cache=False).logits.float() for ids, m in chunks]


def to_state(R, storage):
    """float32 tensors for the state dict: stored as float32, or rounded to bfloat16 and widened (the rounding of a stored weight)."""
    if storage == "bf16":
        return {k: torch.from_numpy(v).to(torch.bfloat16).float() for k, v in R.items()}
    return {k: torch.from_numpy(v).float() for k, v in R.items()}


@torch.no_grad()
def evaluate(model, R, storage, chunks, ref, n_prompts):
    model.load_state_dict(to_state(R, storage), strict=False)
    out = run(model, chunks)
    mx = sa = n = 0.0
    kl = agree = 0.0
    bad = False
    top5, last = [], []
    for (ids, m), a, b in zip(chunks, out, ref):
        if not torch.isfinite(a[m]).all():
            bad = True
        d = (a[m] - b[m]).abs()
        d = d[torch.isfinite(d)] if bad else d
        mx, sa, n = max(mx, float(d.max()) if d.numel() else 0.0), sa + float(d.sum()), n + int(m.sum())
        lp_b = torch.log_softmax(b[m].double(), -1)
        lp_a = torch.log_softmax(a[m].double().nan_to_num(), -1)
        kl += float((lp_b.exp() * (lp_b - lp_a)).sum())
        agree += float((a[m].nan_to_num().argmax(-1) == b[m].argmax(-1)).sum())
        for r in range(ids.shape[0]):
            last.append((a[r, int(m[r].sum()) - 1], b[r, int(m[r].sum()) - 1]))
    for a, b in last[-n_prompts:]:
        top5.append(bool((a.nan_to_num().topk(5).indices == b.topk(5).indices).all()))
    elems = n * ref[0].shape[-1]
    return {"max_abs": mx, "mean_abs": sa / elems, "mean_kl": kl / n, "top1": agree / n, "top5_prompts": f"{sum(top5)}/{len(top5)}", "nonfinite": bad}


