# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The variants of a checkpoint that batch 0014 runs in several runtimes, written as ordinary Hugging Face directories (config, tokenizer files and
# one model.safetensors) so that every runtime loads a file, not an array in memory. The transformations are those of batch 0013 (symmetry.py,
# canon.py), read from there and not rewritten:
#   orig    the checkpoint as published
#   scale   signs and powers of two only: the feed-forward up rows and down columns, the value rows and o columns, the key and query planes. Each
#           is exact in every IEEE arithmetic with a fixed order of sums; in a stored dtype it stays on the grid unless a scaled value leaves the
#           exponent range
#   perm    permutations only: of the hidden coordinates (no signs), of the feed-forward units, of the heads and key/value groups
#   all     the whole group of batch 0013: signed permutation of the hidden coordinates, units (permutation, signs, powers of two), heads, value/output,
#           query/key
#   canon   the canonical form of batch 0013 (amended): everything sorted by content, signs and powers of two normalized, zeros +0.0
#   broken  the permutation of the hidden coordinates applied to the layers and the final norm but not to the embedding and the head: a control that
#           every runtime must show to be broken
#   nullall (addendum 2, not in ALL: named on the command line) every norm weight multiplied by 1 + 2^-23, one unit in the last place of float32: a variant that is not an
#           exact symmetry but changes the arithmetic by the least that float32 can express, the control for the noise of quantized runs
#   nullm, nullr1 .. nullr6 (batch 0024, not in ALL: named on the command line) seven more variants of the same kind as nullall, to measure how much the noise of one unit in the last place varies:
#           nullm every norm weight times 1 - 2^-24 (one unit below, in float32); nullr<k> every norm weight times 1 + s 2^-23, the sign s of each element drawn from a generator seeded with 1000 + k
# A variant is written in the dtype of the original file if every value survives the round trip through that dtype, and in float32 if not (the
# manifest says which). Usage: variants.py REPO REVISION OUTDIR [variant ...]
import hashlib, json, os, shutil, sys, time
import numpy as np
import ml_dtypes  # noqa: F401  (registers bfloat16 with numpy)
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "0013"))
sys.path.insert(0, os.path.join(HERE, "..", "0011"))
sys.path.insert(0, HERE)
from common import sha256_file
import forms as F
import symmetry as S
import canon as C
import special
import nulls

SEED = 14
ALL = ("orig", "scale", "perm", "all", "canon", "broken", "heads", "units", "units_blk", "resid", "resid_blk")
BLOCK = 32          # the group size of llama.cpp's Q8_0 and Q4_0 along a row


class Unsupported(Exception):
    """The checkpoint has a feature that the transformations of batch 0013 do not model: the run stops (tier V1 of the verification suite)."""


def gate(cfg):
    """Refuse what is not a plain Llama: the symmetries and the canonical form assume it, and silently processing anything else would be a claim about a model that was not checked."""
    bad = []
    if cfg.get("model_type") != "llama" or list(cfg.get("architectures") or []) != ["LlamaForCausalLM"]:
        bad.append(f"architecture {cfg.get('architectures')} / model_type {cfg.get('model_type')}")
    for k in ("attention_bias", "mlp_bias"):
        if cfg.get(k):
            bad.append(f"{k}")
    for k in ("rope_scaling", "sliding_window", "final_logit_softcapping", "attn_logit_softcapping", "num_local_experts", "use_qk_norm", "qk_norm"):
        if cfg.get(k):
            bad.append(f"{k}={cfg.get(k)}")
    if cfg.get("hidden_act", "silu") != "silu":
        bad.append(f"hidden_act={cfg.get('hidden_act')}")
    if (cfg.get("pretraining_tp") or 1) != 1:
        bad.append(f"pretraining_tp={cfg.get('pretraining_tp')}")
    if bad:
        raise Unsupported("; ".join(bad))
    return {"architecture": "LlamaForCausalLM", "checked": ["model_type", "attention_bias", "mlp_bias", "rope_scaling", "sliding_window", "softcapping", "experts", "qk_norm", "hidden_act", "pretraining_tp"]}


def blk_perm(n, rng, B=BLOCK):
    """A permutation of range(n) that maps every block of B consecutive indices to a block: the blocks permuted, and the indices permuted inside each. n % B == 0."""
    nb = n // B
    order = rng.permutation(nb)
    return np.concatenate([order[j] * B + rng.permutation(B) for j in range(nb)])


def units_blk(P, d, rng):
    """The units permuted by a block-preserving permutation, with one sign and one power of two (2^k, k in [-3, 3]) per block of units on the up rows and the inverse on the down columns."""
    R = dict(P)
    for i in range(d.L):
        p = blk_perm(d.F, rng)
        c = np.repeat(np.where(rng.random(d.F // BLOCK) < 0.5, -1.0, 1.0) * 2.0 ** rng.integers(-3, 4, d.F // BLOCK), BLOCK).astype(P[F.key(i, F.U)].dtype)
        g, u, w = P[F.key(i, F.G)], P[F.key(i, F.U)] * c[:, None], P[F.key(i, F.D)] / c[None, :]
        R[F.key(i, F.G)], R[F.key(i, F.U)], R[F.key(i, F.D)] = g[p], u[p], w[:, p]
    return R


def perm_units(P, d, rng):
    R = dict(P)
    for i in range(d.L):
        p = rng.permutation(d.F)
        g, u, w = P[F.key(i, F.G)], P[F.key(i, F.U)], P[F.key(i, F.D)]
        R[F.key(i, F.G)], R[F.key(i, F.U)], R[F.key(i, F.D)] = g[p], u[p], w[:, p]
    return R


def make(P, d, tied, variant):
    rng = np.random.default_rng(SEED)
    if variant == "orig":
        return dict(P)
    if variant == "scale":
        return S.act(P, d, rng, tied, ("units-signed", "vo", "qk"))
    if variant == "perm":
        R = S.apply_residual(P, d, rng.permutation(d.H), np.ones(d.H))
        R = perm_units(R, d, rng)
        return S.heads(R, d, rng)
    if variant == "all":
        return S.act(P, d, rng, tied, special.PARTS)
    if variant == "canon":
        return C.canonical(P, d)
    if variant == "heads":
        return S.heads(P, d, rng)
    if variant == "units":
        return perm_units(P, d, rng)
    if variant == "units_blk":
        return units_blk(P, d, rng)
    if variant == "resid":
        return S.apply_residual(P, d, rng.permutation(d.H), np.ones(d.H))
    if variant == "resid_blk":
        return S.apply_residual(P, d, blk_perm(d.H, rng), np.ones(d.H))
    if variant in nulls.NULLS:
        return nulls.make_null(P, variant)
    if variant == "broken":
        R = S.apply_residual(P, d, rng.permutation(d.H), np.ones(d.H))
        R[F.EMBED] = P[F.EMBED]
        if "lm_head.weight" in P:
            R["lm_head.weight"] = P["lm_head.weight"]
        return R
    raise ValueError(variant)


def to_dtype(a, dtype):
    return {"bfloat16": ml_dtypes.bfloat16, "float16": np.float16, "float32": np.float32}[dtype]


def to_torch(a, use):
    import torch
    if use == "bfloat16":
        return torch.from_numpy(np.ascontiguousarray(a.astype(ml_dtypes.bfloat16).view(np.int16))).view(torch.bfloat16)
    return torch.from_numpy(np.ascontiguousarray(a.astype(np.float16 if use == "float16" else np.float32)))


def write(src_dir, out_dir, R, stored):
    """Write R (float32 arrays) as model.safetensors in `stored` dtype next to the config and tokenizer files of src_dir. Returns the dtype used and the number
    of tensors whose values do not survive the round trip through the dtype of the original."""
    from safetensors.torch import save_file
    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(src_dir):
        if f != "model.safetensors" and os.path.isfile(os.path.join(src_dir, f)) and not f.endswith((".onnx", ".bin", ".h5", ".msgpack", ".pt")):
            shutil.copy(os.path.join(src_dir, f), os.path.join(out_dir, f))
    inexact = [k for k, v in R.items() if not np.array_equal(v.astype(to_dtype(v, stored)).astype(np.float32), v)]
    use = stored if not inexact else "float32"
    tensors = {k: to_torch(v, use) for k, v in R.items()}
    save_file(tensors, os.path.join(out_dir, "model.safetensors"), metadata={"format": "pt"})
    cfg = json.load(open(os.path.join(out_dir, "config.json")))
    cfg["torch_dtype"] = use
    cfg["dtype"] = use
    for k in ("pad_token_id", "bos_token_id", "eos_token_id"):       # a negative id (the random tiny Llama has pad_token_id -1) stops the GGUF converter; the same change in every variant
        if isinstance(cfg.get(k), int) and cfg[k] < 0:
            cfg[k] = None
    json.dump(cfg, open(os.path.join(out_dir, "config.json"), "w"), indent=2)
    return use, len(inexact)


def significand_digest(a):
    """A digest of the multiset of the magnitudes of the significands of a float32 array. A signed permutation and a multiplication by powers of two change neither (unless a value
    leaves the normal range), so every variant but a broken one must have the original's digest tensor by tensor: the file holds the original's numbers, nothing else."""
    m = np.sort(np.abs(np.frexp(a.astype(np.float32))[0]), axis=None)
    return hashlib.sha1(m.tobytes()).hexdigest()


def config_keys_changed(a, b):
    return sorted(k for k in set(a) | set(b) if a.get(k, "<absent>") != b.get(k, "<absent>"))


def main(repo, revision, outdir, which):
    from huggingface_hub import snapshot_download
    os.environ["HF_REVISION"] = revision
    src = snapshot_download(repo, revision=revision, allow_patterns=["*.json", "*.txt", "*.model", "model.safetensors"])
    P, d, cfg = special.weights(repo)
    gated = gate(cfg)
    if d.F % BLOCK or d.H % BLOCK:
        which = tuple(v for v in which if v not in ("units_blk", "resid_blk"))
    tied = "lm_head.weight" not in P
    stored = cfg.get("torch_dtype") or cfg.get("dtype") or "float32"
    manifest = {"repo": repo, "revision": revision, "dims": vars(d), "tied": tied, "stored_dtype_of_original": stored, "seed": SEED, "gate": gated, "variants": {}}
    digests = {k: significand_digest(a) for k, a in P.items()}
    src_cfg = json.load(open(os.path.join(src, "config.json")))
    for v in which:
        t = time.time()
        R = make(P, d, tied, v)
        use, inexact = write(src, os.path.join(outdir, v), R, stored)
        path = os.path.join(outdir, v, "model.safetensors")
        other = [k for k, a in R.items() if significand_digest(a) != digests[k]]
        manifest["variants"][v] = {"stored_dtype": use, "tensors_not_exact_in_original_dtype": inexact, "seconds": round(time.time() - t, 1),
                                   "safetensors_bytes": os.path.getsize(path), "safetensors_sha256": sha256_file(path),
                                   "tensors": len(R), "differs_from_original": sum(1 for k in P if not np.array_equal(P[k], R[k])) if v != "orig" else 0,
                                   "tensors_with_other_significands": other, "same_tensor_names": sorted(R) == sorted(P),
                                   "config_keys_changed": config_keys_changed(src_cfg, json.load(open(os.path.join(outdir, v, "config.json"))))}
        print(v, {k: (x if k != "tensors_with_other_significands" else len(x)) for k, x in manifest["variants"][v].items()}, flush=True)
        del R
    json.dump(manifest, open(os.path.join(outdir, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3], tuple(sys.argv[4:]) or ALL)
