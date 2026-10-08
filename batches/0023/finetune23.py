# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Batch 0023, the equivariance of fine-tuning under the equivalence transformations, on language models. Two weight vectors that the gauge group relates are the same function. Fine-tune a language model from each of them with the standard
# optimizers of PyTorch (torch.optim.SGD, torch.optim.AdamW without weight decay), in float64, on the same ten steps of the same eight short texts, norm weights frozen: if the optimizer respects the
# transformation, the two runs stay the same function; if it does not, they part. Batch 0015 (Q17) did this on a random Llama of 32 hidden units; here the checkpoints are SmolLM2-135M-Instruct and
# delphi-suite/v0-llama2-100k. The reference is the checkpoint with its norm weights folded and its head untied (the base on which rotations of the stream are exact); the variants are a signed permutation
# of the hidden coordinates, a rotation of the stream, a scaling of the feed-forward units by +-2^k, and an invertible matrix on the value dimensions. What is measured: the function distance of the
# variant's run from the reference's run (the Frobenius norm of the difference of the logits on six sentences) as a multiple of how far the reference's own training moved the function.
# Usage: finetune23.py KEY REPO REV OUT.json [--random-small]
import copy, gc, json, os, sys, time
import numpy as np
import torch
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hfx23 as X  # noqa: E402

STEPS = 10
LR = {"sgd": 0.05, "adamw": 2e-5}


def training_batch(tok, vocab, random_small):
    if random_small:
        return torch.randint(0, vocab, (8, 12), generator=torch.Generator().manual_seed(0))
    lines = [l for l in open(os.path.join(HERE, "..", "0013", "sentences.txt"), encoding="utf-8").read().splitlines() if l.strip()][6:80]
    rows = []
    for l in lines:
        ids = tok(l, return_tensors="pt")["input_ids"][0]
        if len(ids) >= 12:
            rows.append(ids[:12])
        if len(rows) == 8:
            break
    if len(rows) < 8:
        raise SystemExit(f"only {len(rows)} sentences of at least 12 tokens")
    return torch.stack(rows)


def logits_on(m, seqs):
    return np.concatenate([X.run(m, ids)["logits"] for ids in seqs], axis=0)


def fro(a, b):
    return float(np.linalg.norm(a - b))


def train(m, data, opt):
    for n, p in m.named_parameters():
        p.requires_grad_("norm" not in n)
    params = [p for p in m.parameters() if p.requires_grad]
    o = torch.optim.SGD(params, lr=LR["sgd"]) if opt == "sgd" else torch.optim.AdamW(params, lr=LR["adamw"], weight_decay=0.0)
    first = last = None
    for _ in range(STEPS):
        o.zero_grad(set_to_none=True)
        lg = m(input_ids=data, attention_mask=torch.ones_like(data)).logits
        loss = torch.nn.functional.cross_entropy(lg[:, :-1].reshape(-1, lg.shape[-1]), data[:, 1:].reshape(-1))
        loss.backward()
        o.step()
        first = float(loss) if first is None else first
        last = float(loss)
    for p in params:
        p.grad = None
    return first, last


def main(key, repo, rev, out, random_small):
    t0 = time.time()
    m, tok, cfg, restore = X.load_model("llama", repo, rev, random_small=random_small)
    res = {"key": key, "repo": repo, "revision": rev, "steps": STEPS, "learning_rates": LR, "parameters": int(sum(p.numel() for p in m.parameters())), "optimizers": {}}
    try:
        d, nh, nkv, hd, dff = X.dims(m)
        data = training_batch(tok, cfg["vocab_size"], random_small)
        seqs = X.encode(tok, "llama", random_small, cfg["vocab_size"])
        F0 = X.fold_untie(m)
        del m
        gc.collect()
        rs = np.random.RandomState(9)
        perm, signs = [int(i) for i in rs.permutation(d)], [int(s) for s in rs.choice([1, -1], d)]
        Q = X.random_orthogonal(d, 79)
        scale = np.sign(rs.standard_normal(dff)) * 2.0 ** rs.randint(-3, 4, dff)
        As = [np.linalg.qr(rs.standard_normal((hd, hd)))[0] @ np.diag(np.exp(rs.uniform(-0.5, 0.5, hd))) @ np.linalg.qr(rs.standard_normal((hd, hd)))[0] for _ in range(nkv)]
        makers = {"signed_permutation_of_the_hidden_coordinates": lambda: X.signed_permutation(F0, perm, signs), "rotation_of_the_stream": lambda: X.rotate(F0, Q),
                  "scaling_of_the_feed_forward_units": lambda: X.unit_gauge(F0, list(range(dff)), scale), "invertible_matrix_on_the_value_dimensions": lambda: X.value_output_gauge(F0, As)}
        Z0 = logits_on(F0, seqs)
        for opt in ("sgd", "adamw"):
            ref = copy.deepcopy(F0)
            loss_ref = train(ref, data, opt)
            Zr = logits_on(ref, seqs)
            moved = fro(Zr, Z0)
            row = {"loss_first_step": loss_ref[0], "loss_last_step": loss_ref[1], "function_change_of_the_reference_run_relative": moved / float(np.linalg.norm(Z0)), "variants": {}}
            del ref
            gc.collect()
            for name, make in makers.items():
                mv = make()
                before = X.rho(logits_on(mv, seqs), Z0)
                train(mv, data, opt)
                Zv = logits_on(mv, seqs)
                row["variants"][name] = {"difference_before_training_relative": before, "difference_after_training_over_the_change_of_the_reference_run": fro(Zv, Zr) / moved if moved > 0 else None}
                print(f"{opt:6s} {name:46s} before {before:.1e}  after / reference change {row['variants'][name]['difference_after_training_over_the_change_of_the_reference_run']}", flush=True)
                del mv
                gc.collect()
            res["optimizers"][opt] = row
        res["seconds"] = round(time.time() - t0)
    finally:
        restore()
        json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], None if sys.argv[2] == "-" else sys.argv[2], None if sys.argv[3] == "-" else sys.argv[3], sys.argv[4], "--random-small" in sys.argv)
