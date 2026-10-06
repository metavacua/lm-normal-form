# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# "Weight encryption by a secret permutation", on a real checkpoint. The scheme of the pasted text, as the paper it cites implements it (Xu et al., CVPR 2024): the layers and the
# final norm are permuted along the hidden dimension by a secret P, the embedding and the head are not, and the owner applies P to the embedding output and P^-1 to the body's output
# at run time. The checkpoint is SmolLM2-135M-Instruct; the attacker is given the permuted weights and the public SmolLM2-135M base (the instruct model is a fine-tune of it).
#   1 the model with the permuted body, run as shipped: perplexity of 8 windows of 512 tokens of WikiText-2, and the greedy continuation of 4 prompts
#   2 the model with the key applied at the interface (the embedding and the head permuted as well): the same measurements; the logits against the unpermuted model's; and (claims K8, K9 on
#     real weights, float64) the KV cache of both against the unpermuted model's: the keyed model's cache is the original's, in the public gauge, so it carries nothing of the key
#   3 the attack with a public base: the coordinates of the permuted body are matched to those of the base by the cosine similarity of their rows and columns in every layer, with
#     the Hungarian algorithm; the fraction of coordinates recovered; the model with the recovered permutation undone: perplexity and generations
#   4 the attack without a base, by training: a linear map at the input of the body and a linear map at its output, from the identity, on 8 windows, for a few hundred steps of Adam:
#     perplexity along the way (a budget, not a limit)
# Usage: lm_obfuscation.py INSTRUCT_DIR BASE_DIR OUT.json [STEPS]
import json, os, sys, time
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hf_adapter as H
import gauge as G
from ops import FloatOps
from common import wikitext_windows, PLAIN_PROMPTS


def ppl(model, wins, torch):
    nll, n = 0.0, 0
    with torch.no_grad():
        for w in wins:
            ids = torch.tensor([w])
            lg = model(input_ids=ids).logits[0, :-1].float()
            nll += float(torch.nn.functional.cross_entropy(lg, ids[0, 1:], reduction="sum"))
            n += len(w) - 1
    return float(np.exp(nll / n))


def gens(model, prompts, torch):
    with torch.no_grad():
        return [model.generate(torch.tensor([p]), max_new_tokens=24, do_sample=False, pad_token_id=0)[0][len(p):].tolist() for p in prompts]


def features(P, a):
    """For each hidden coordinate j, the vector of everything that reads or writes it: the columns j of q, k, v, gate, up and the rows j of o and down in every layer."""
    cols = []
    for i in range(a.n_layers):
        for n in ("wq", "wk", "wv", "wg", "wu"):
            cols.append(P[f"l{i}.{n}"].T)                # (d, out)
        for n in ("wo", "wd"):
            cols.append(P[f"l{i}.{n}"])                  # (d, in)
    return np.concatenate(cols, axis=1).astype(np.float32)


def main(inst_dir, base_dir, out, steps=300):
    import torch
    from scipy.optimize import linear_sum_assignment
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(int(os.environ.get("LMNF_THREADS", "4")))
    tok = AutoTokenizer.from_pretrained(inst_dir)
    wins, _ = wikitext_windows(tok, n=8, size=512)
    prompts = [tok(t)["input_ids"] for t in PLAIN_PROMPTS[:4]]
    a, P, cfg = H.load(inst_dir)
    ops = FloatOps(eps=cfg["rms_norm_eps"], base=cfg.get("rope_theta", 10000.0))
    rng = np.random.default_rng(15)
    key = [int(i) for i in rng.permutation(a.d)]
    res = {"key_seed": 15, "hidden_size": a.d}
    work = os.environ.get("LMNF_WORK", "work-obf")

    def model_of(arch, params, name):
        d = os.path.join(work, name)
        H.save(arch, params, inst_dir, d, "float32")
        return AutoModelForCausalLM.from_pretrained(d, dtype=torch.float32).eval()

    m = AutoModelForCausalLM.from_pretrained(inst_dir, dtype=torch.float32).eval()
    res["original"] = {"ppl": ppl(m, wins, torch), "generations": gens(m, prompts, torch)}
    ref_logits = [m(input_ids=torch.tensor([p])).logits[0].detach().numpy() for p in prompts]
    del m
    # 1: body only
    _, body = G.body_only_perm(a, P, ops, key, [1] * a.d)
    m = model_of(a, body, "body")
    res["body_only_permuted"] = {"ppl": ppl(m, wins, torch), "generations": gens(m, prompts, torch)}
    del m
    # 2: with the key
    _, keyed = G.residual_signed_perm(a, P, ops, key, [1] * a.d)
    import lm_gauge as LG
    seqs3 = [tok(t)["input_ids"] for t in PLAIN_PROMPTS[:3]]
    c0 = [LG.cache(a, P, q, ops) for q in seqs3]
    res["cache_float64_relative_difference_from_the_original"] = {"keyed": max(LG.cache_error(b, LG.cache(a, keyed, q, ops)) for b, q in zip(c0, seqs3)),
                                                                  "body_only": max(LG.cache_error(b, LG.cache(a, body, q, ops)) for b, q in zip(c0, seqs3))}
    m = model_of(a, keyed, "keyed")
    res["with_the_key"] = {"ppl": ppl(m, wins, torch), "generations": gens(m, prompts, torch),
                           "max_abs_logit_difference": max(float(np.abs(m(input_ids=torch.tensor([p])).logits[0].detach().numpy() - r).max()) for p, r in zip(prompts, ref_logits))}
    del m
    # 3: matching against the public base
    t = time.time()
    ab, Pb, _ = H.load(base_dir)
    Fp, Fb = features(body, a), features(Pb, ab)
    Fp /= np.linalg.norm(Fp, axis=1, keepdims=True) + 1e-12
    Fb /= np.linalg.norm(Fb, axis=1, keepdims=True) + 1e-12
    cost = -(Fp @ Fb.T)                                   # body coordinate (permuted) against base coordinate
    rows, cols = linear_sum_assignment(cost)
    # the secret: coordinate j of the original is at position key[j] in the body; so body position key[j] should match base coordinate j
    truth = {key[j]: j for j in range(a.d)}
    recovered = float(np.mean([truth[int(r)] == int(c) for r, c in zip(rows, cols)]))
    perm_back = [int(c) for _, c in sorted(zip(rows, cols))]       # the permutation that sends the coordinate at body position r to the base coordinate that r was matched to
    _, fixed = G.residual_signed_perm(a, body, ops, perm_back, [1] * a.d)
    fixed["embed"] = P["embed"]                                     # the embedding and the head were never permuted
    if "lm_head" in P:
        fixed["lm_head"] = P["lm_head"]
    m = model_of(a, fixed, "recovered")
    res["attack_with_public_base"] = {"coordinates_recovered": recovered, "ppl": ppl(m, wins, torch), "generations": gens(m, prompts, torch), "seconds": round(time.time() - t, 1)}
    del m
    json.dump(res, open(out, "w"), indent=1)
    # 4: the attack by training an input and an output adapter
    t = time.time()
    m = model_of(a, body, "body")
    for q in m.parameters():
        q.requires_grad_(False)
    d = a.d
    Ain = torch.nn.Parameter(torch.eye(d))
    Aout = torch.nn.Parameter(torch.eye(d))
    emb, head = m.get_input_embeddings(), m.get_output_embeddings()
    opt = torch.optim.Adam([Ain, Aout], lr=1e-2)
    trace = []
    train = wins[:6]
    for step in range(steps):
        ids = torch.tensor([train[step % len(train)][:128]])
        x = emb(ids) @ Ain
        h = m.model(inputs_embeds=x).last_hidden_state                 # the final norm is inside m.model; the adapter acts after it
        lg = (h @ Aout) @ head.weight.T
        loss = torch.nn.functional.cross_entropy(lg[0, :-1], ids[0, 1:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % 25 == 0 or step == steps - 1:
            trace.append((step, float(np.exp(float(loss)))))
            print("adapter attack", trace[-1], flush=True)
    res["attack_by_training_adapters"] = {"steps": steps, "windows_of_128_tokens": len(train), "perplexity_on_the_training_window_along_the_way": trace, "seconds": round(time.time() - t, 1)}
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 300)
