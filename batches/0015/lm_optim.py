# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Training dynamics under the symmetries. A symmetry g of the function (f(g . theta) = f(theta)) is a symmetry of the gradient flow only if it is compatible with the update rule: an orthogonal
# g maps the gradient of a function that it leaves invariant to the gradient at the image (so SGD from g . theta follows g . (SGD from theta)), but Adam divides each coordinate of the gradient by
# the root of its own second moment, and only signed permutations (and the identity) commute with that (this follows from the update rules; the run illustrates it); a scaling g (the unit gauge of a gated feed-forward block) commutes with neither, because the
# gradient of a scaled parameter is scaled the other way. The research notes name this as the reason that the standard basis of a trained model is not random (Elhage et al. 2023; He et al. 2024).
# A tiny Llama (hidden 32, 4 heads, 2 key/value groups, 2 layers, vocabulary 64, untied head, the norm weights frozen at 1) is trained in float64 (Hugging Face's float32 islands patched) for
# 100 full-batch steps on a fixed synthetic next-token task, by SGD (lr 0.2) and by Adam (lr 0.01), from theta and from g(theta), for g a signed permutation of the hidden coordinates, a random
# orthogonal matrix (the norms being 1, no fold is needed), and a scaling of the feed-forward units by +-2^k. For each: the largest difference of the loss curves, and the relative difference
# between the trained parameters of the transformed run and g applied to the trained parameters of the original run.
# Usage: lm_optim.py OUT.json
import json, os, shutil, sys, tempfile
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import hf_adapter as H
import gauge as G
from model import Arch, shapes
from ops import FloatOps

ARCH = Arch(d=32, n_heads=4, n_kv=2, hd=8, d_ff=64, n_layers=2, vocab=64, tied=False)
CFG = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 2, "num_attention_heads": 4, "num_key_value_heads": 2,
       "head_dim": 8, "vocab_size": 64, "max_position_embeddings": 64, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": False, "hidden_act": "silu", "attention_bias": False,
       "mlp_bias": False, "torch_dtype": "float64"}
STEPS = 100
OPTIMIZERS = {"sgd": lambda ps: __import__("torch").optim.SGD(ps, lr=0.2), "adam": lambda ps: __import__("torch").optim.Adam(ps, lr=0.01)}


def initial(rng):
    return {k: (np.ones(s) if len(s) == 1 else rng.standard_normal(s) * 0.4 / np.sqrt(s[-1])) for k, s in shapes(ARCH).items()}


def data(rng):
    x = rng.integers(0, ARCH.vocab, (64, 17))
    for i in range(16):
        follow = rng.random(64) < 0.9
        x[:, i + 1] = np.where(follow, (5 * x[:, i] + 3) % ARCH.vocab, x[:, i + 1])
    return x


def train(work, name, P, opt_name, x):
    import torch
    from transformers import AutoModelForCausalLM
    src = os.path.join(work, "src")
    os.makedirs(src, exist_ok=True)
    json.dump(CFG, open(os.path.join(src, "config.json"), "w"))
    d = os.path.join(work, name)
    H.save(ARCH, P, src, d, "float64")
    restore = H.float64_patches(CFG)
    try:
        model = AutoModelForCausalLM.from_pretrained(d, dtype=torch.float64)
        for n, p in model.named_parameters():
            if "norm" in n:
                p.requires_grad_(False)
        opt = OPTIMIZERS[opt_name]([p for p in model.parameters() if p.requires_grad])
        xt = torch.tensor(x)
        losses = []
        for _ in range(STEPS):
            opt.zero_grad()
            logits = model(input_ids=xt[:, :-1]).logits
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), xt[:, 1:].reshape(-1))
            loss.backward()
            opt.step()
            losses.append(float(loss))
        return losses, H.params_of(model, ARCH)
    finally:
        restore()


def main(out):
    rng = np.random.default_rng(15)
    ops = FloatOps(eps=CFG["rms_norm_eps"])
    P0, x = initial(rng), data(rng)
    perm, signs = [int(i) for i in rng.permutation(ARCH.d)], [int(s) for s in rng.choice([1, -1], ARCH.d)]
    Q = ops.rand_orthogonal(ARCH.d, rng)
    cs = [np.sign(rng.standard_normal(ARCH.d_ff)) * 2.0 ** rng.integers(-3, 4, ARCH.d_ff) for _ in range(ARCH.n_layers)]
    ident = [np.arange(ARCH.d_ff) for _ in range(ARCH.n_layers)]
    transforms = {"signed permutation of the hidden coordinates": lambda P: G.residual_signed_perm(ARCH, P, ops, perm, signs)[1],
                  "orthogonal rotation of the stream": lambda P: G.residual_orth(ARCH, P, ops, Q, fold=True)[1],
                  "scaling of the feed-forward units by +-2^k": lambda P: G.mlp_gauge(ARCH, P, ops, ident, cs)[1]}
    res = {"steps": STEPS, "results": {}}
    work = tempfile.mkdtemp(prefix="lm_optim_")
    try:
        for opt_name in OPTIMIZERS:
            base_losses, base_end = train(work, f"orig-{opt_name}", P0, opt_name, x)
            res["results"][opt_name] = {"original": {"first_loss": base_losses[0], "last_loss": base_losses[-1]}}
            for tname, g in transforms.items():
                losses, end = train(work, f"{tname[:6].replace(' ', '_')}-{opt_name}", g(P0), opt_name, x)
                want = g(base_end)
                num = sum(float(np.sum((end[k] - want[k]) ** 2)) for k in end if not k.endswith("norm") and k != "norm_f")
                den = sum(float(np.sum(want[k] ** 2)) for k in want if not k.endswith("norm") and k != "norm_f")
                res["results"][opt_name][tname] = {"max_loss_difference": float(np.max(np.abs(np.array(losses) - np.array(base_losses)))), "last_loss": losses[-1],
                                                   "relative_parameter_difference": float(np.sqrt(num / den))}
                print(f"{opt_name:5s} {tname:48s} max|dloss| {res['results'][opt_name][tname]['max_loss_difference']:.2e}  parameters {res['results'][opt_name][tname]['relative_parameter_difference']:.2e}"
                      f"  (loss {base_losses[0]:.3f} -> {base_losses[-1]:.3f})", flush=True)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1])
