# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Batch 0023, the finite instance as a checkpoint that standard runtimes load, run on its whole input space. The finite instance (finite23.py) is the Llama equations at hidden size 4, two layers, two query
# heads and one key/value group of head dimension 2, 5 symbols, with parameters drawn from a seeded generator: it is NOT a language model (no training, no specification beyond the equations; the test of
# hfx23.membership below rejects it). It is written here as a Hugging Face checkpoint (config.json, model.safetensors in float64) and run in Transformers (float64 with the float32 islands removed,
# and float32 as shipped) and in candle (float32, candle/src/main.rs) on every sequence of four symbols (625 of them; the logits at the four positions are the logits of all 780 contexts of length 1 to 4),
# and the logits are compared with those of the numpy forward pass (float64), which is checked against model.py in finite23.selftest and which batch 0015 checked against Transformers on a real checkpoint.
# The comparison is exhaustive: every input of the instance is run in every runtime. What it shows is conformance of the runtimes to the equations at this size; nothing about language models.
#   bridge23.py export OUTDIR                    writes OUTDIR/ckpt (the checkpoint), OUTDIR/reference.npy (numpy logits, (625, 4, 5)), OUTDIR/candle_inputs.json
#   bridge23.py torch OUTDIR RES.json            Transformers float64 and float32 against the reference
#   bridge23.py candle OUTDIR OUT.logits.f32 RES.json     the candle output against the reference
import itertools, json, math, os, shutil, sys
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "0015"))
from finite23 import ARCH, make_spec, init_theta, to_dict, unpack, forward_pb, ops_of, selftest  # noqa: E402

SEED = 41
LENGTH = 4
CFG = {"architectures": ["LlamaForCausalLM"], "model_type": "llama", "hidden_size": ARCH.d, "intermediate_size": ARCH.d_ff, "num_hidden_layers": ARCH.n_layers, "num_attention_heads": ARCH.n_heads,
       "num_key_value_heads": ARCH.n_kv, "head_dim": ARCH.hd, "vocab_size": ARCH.vocab, "max_position_embeddings": 512, "rms_norm_eps": 1e-6, "rope_theta": 10000.0, "tie_word_embeddings": False,
       "hidden_act": "silu", "attention_bias": False, "mlp_bias": False, "bos_token_id": 0, "eos_token_id": 1}


def sequences():
    return np.array(list(itertools.product(range(ARCH.vocab), repeat=LENGTH)), dtype=np.int64)


def reference():
    ops, spec = ops_of(), make_spec(ARCH)
    th = init_theta(spec, np.random.default_rng(SEED), ARCH, ops)
    z = forward_pb(ARCH, unpack(th[None, :], spec), sequences(), ops)[0]
    return th, spec, z


def agree(z, ref):
    return {"relative_difference": float(np.linalg.norm(z - ref) / np.linalg.norm(ref)), "argmax_agreement": float((z.argmax(-1) == ref.argmax(-1)).mean()), "positions": int(ref.shape[0] * ref.shape[1]),
            "contexts": int(sum(ARCH.vocab ** t for t in range(1, LENGTH + 1)))}


def export(out):
    import hf_adapter as H
    selftest()
    th, spec, z = reference()
    os.makedirs(out, exist_ok=True)
    src = os.path.join(out, "src")
    os.makedirs(src, exist_ok=True)
    json.dump(CFG, open(os.path.join(src, "config.json"), "w"), indent=1)
    H.save(ARCH, to_dict(th, spec, ARCH), src, os.path.join(out, "ckpt"), "float64")
    shutil.rmtree(src)
    np.save(os.path.join(out, "reference.npy"), z)
    json.dump({"sequences": sequences().tolist()}, open(os.path.join(out, "candle_inputs.json"), "w"))
    print(f"exported: {th.size} parameters, {z.shape[0]} sequences, logits {z.shape}", flush=True)


def torch_run(out, res_path):
    import torch
    import hf_adapter as H
    from transformers import AutoModelForCausalLM
    ref = np.load(os.path.join(out, "reference.npy"))
    ids = torch.tensor(sequences())
    import transformers
    res = {"transformers": transformers.__version__, "torch": torch.__version__}
    for name, dtype, patch in (("float64", torch.float64, True), ("float32", torch.float32, False)):
        restore = H.float64_patches(CFG) if patch else (lambda: None)
        try:
            m = AutoModelForCausalLM.from_pretrained(os.path.join(out, "ckpt"), dtype=dtype, attn_implementation="sdpa").eval()
            with torch.no_grad():
                z = m(input_ids=ids).logits.double().numpy()
            res[name] = agree(z, ref)
            if name == "float64":
                g = m.generate(input_ids=torch.tensor([[0, 1, 2]]), attention_mask=torch.ones(1, 3, dtype=torch.long), max_new_tokens=8, do_sample=False, pad_token_id=0)
                res["generation_from_0_1_2_float64"] = [int(t) for t in g[0, 3:]]
        finally:
            restore()
        print(name, res[name], flush=True)
    # the registered membership test applied to this instance (a negative control): the mean loss of its next-symbol predictions on uniformly random sequences of its alphabet, over ln(5)
    lp = ref - np.log(np.exp(ref - ref.max(-1, keepdims=True)).sum(-1, keepdims=True)) - ref.max(-1, keepdims=True)
    seq = sequences()
    loss = float(-np.mean([lp[i, t, seq[i, t + 1]] for i in range(len(seq)) for t in range(LENGTH - 1)]))
    res["membership_control"] = {"mean_loss_nats": loss, "ln_output_vocabulary": math.log(ARCH.vocab), "fraction_of_ln_vocabulary": loss / math.log(ARCH.vocab)}
    json.dump(res, open(res_path, "w"), indent=1)


def candle_compare(out, path, res_path):
    ref = np.load(os.path.join(out, "reference.npy"))
    z = np.fromfile(path, dtype="<f4").astype(np.float64).reshape(ref.shape)
    res = {"candle_float32": agree(z, ref)}
    print(res, flush=True)
    json.dump(res, open(res_path, "w"), indent=1)


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export(sys.argv[2])
    elif sys.argv[1] == "torch":
        torch_run(sys.argv[2], sys.argv[3])
    elif sys.argv[1] == "candle":
        candle_compare(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise SystemExit(__doc__)
