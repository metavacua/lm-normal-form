# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cell 01: the constructions of docs/batches/0031.md as Hugging Face directories. Closed form, no training.

usage: construct.py OUT_ROOT
Writes OUT_ROOT/M1 (zero blocks, T1), OUT_ROOT/M2_d52 (one block, orthogonal codes, T2) and OUT_ROOT/M2_d{D} for the
ladder widths, each with config.json, model.safetensors, tokenizer.json, tokenizer.model, tokenizer_config.json,
construction.json (what was built, the margins the construction predicts, hashes).
"""
import hashlib
import json
import math
import os
import sys

import numpy as np

SEED = 31
CHARS = "abcdefghijklmnopqrstuvwxyz0123456789.,?!:-'() "      # batch 0030's alphabet (46 characters + 4 specials)
SPECIALS = ["<unk>", "<s>", "</s>", "<pad>"]
CAPITALS = {"france": "paris", "germany": "berlin", "italy": "rome", "spain": "madrid", "japan": "tokyo", "china": "beijing",
            "russia": "moscow", "egypt": "cairo", "canada": "ottawa", "brazil": "brasilia", "india": "delhi", "australia": "canberra"}
TEMPLATE = ["the", "capital", "of", "is"]
FILLERS = ["a", "and", "in", "on", "to", "with", "city", "country", "river", "sea", "north", "south", "east", "west", "old", "new", "big", "small"]
WORDS = TEMPLATE + list(CAPITALS) + list(CAPITALS.values()) + FILLERS          # 4 + 12 + 12 + 18 = 46
LADDER = [52, 40, 32, 24, 16, 12, 8]


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


# ----------------------------------------------------------------------------------------------- tokenizers
def char_pieces():
    return SPECIALS + ["▁" if ch == " " else ch for ch in CHARS]            # 50 pieces, the space as the metaspace mark


def word_pieces():
    return SPECIALS + ["▁" + w for w in WORDS]                              # 50 pieces, each word with the metaspace mark


def write_tokenizer(out, kind):
    """Trains the SentencePiece model first and takes ITS piece order as the piece order of the model (the GGUF converter
    reads tokenizer.model, so the model's rows must be indexed in that order); then writes a fast tokenizer.json over
    exactly those pieces. Returns (tokenizer, pieces, info)."""
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast
    import sentencepiece as spm
    corpus = os.path.join(out, "_corpus.txt")
    with open(corpus, "w", encoding="utf-8") as f:
        if kind == "char":
            for _ in range(20):
                f.write(CHARS + "\nthe capital of france is paris. 0123456789 (a-b), c? d! e: f'g\n")
        else:
            for _ in range(20):
                f.write(" ".join(WORDS) + "\n")
    spm.SentencePieceTrainer.train(input=corpus, model_prefix=os.path.join(out, "tokenizer"), model_type="char" if kind == "char" else "word",
                                   vocab_size=50, character_coverage=1.0, unk_id=0, bos_id=1, eos_id=2, pad_id=3, unk_piece="<unk>", bos_piece="<s>",
                                   eos_piece="</s>", pad_piece="<pad>", add_dummy_prefix=(kind == "word"), remove_extra_whitespaces=False,
                                   normalization_rule_name="identity", byte_fallback=False, num_threads=1, minloglevel=2)
    os.remove(corpus)
    if os.path.exists(os.path.join(out, "tokenizer.vocab")):
        os.remove(os.path.join(out, "tokenizer.vocab"))
    sp = spm.SentencePieceProcessor(model_file=os.path.join(out, "tokenizer.model"))
    pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    intended = char_pieces() if kind == "char" else word_pieces()
    assert sorted(pieces) == sorted(intended), (sorted(set(pieces) ^ set(intended)))
    vocab = [(pc, 0.0 if i >= 4 else -100.0) for i, pc in enumerate(pieces)]
    tk = Tokenizer(models.Unigram(vocab, unk_id=0, byte_fallback=False))
    scheme = "never" if kind == "char" else "first"
    tk.pre_tokenizer = pre_tokenizers.Metaspace(replacement="\u2581", prepend_scheme=scheme, split=(kind == "word"))
    tk.decoder = decoders.Metaspace(replacement="\u2581", prepend_scheme=scheme, split=(kind == "word"))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<pad>",
                                  clean_up_tokenization_spaces=False, model_max_length=64)
    tok.save_pretrained(out)
    probe = "the capital of france is paris" if kind == "word" else "the capital of france is "
    sp_ids = sp.encode(probe, out_type=int); fast_ids = tok(probe, add_special_tokens=False)["input_ids"]
    return tok, pieces, {"pieces": pieces, "same_order_as_sentencepiece": True, "probe": probe, "sentencepiece_ids": sp_ids,
                         "fast_ids": fast_ids, "ids_equal": sp_ids == fast_ids}


# ----------------------------------------------------------------------------------------------- models
def save_model(out, cfg_kwargs, tensors):
    """A LlamaForCausalLM with the given config whose state dict is replaced by `tensors` (name -> numpy float32)."""
    import torch
    from transformers import LlamaConfig, LlamaForCausalLM
    cfg = LlamaConfig(**cfg_kwargs)
    torch.manual_seed(0)
    model = LlamaForCausalLM(cfg)
    sd = model.state_dict()
    missing = [k for k in sd if k not in tensors and k != "lm_head.weight"]
    assert not missing, missing
    with torch.no_grad():
        for k, v in tensors.items():
            sd[k].copy_(torch.tensor(v, dtype=torch.float32))
    model.tie_weights(); model.eval(); model.config.use_cache = False
    model.save_pretrained(out, safe_serialization=True)
    return int(sum(p.numel() for p in model.parameters()))


def m1_tensors(pieces=None):
    """Zero blocks: the symmetric chain space -> p -> a -> r -> i -> s -> </s>, factored exactly. Returns tensors, cfg, info."""
    pieces = pieces or char_pieces(); idx = {p: i for i, p in enumerate(pieces)}
    V = 50
    chain = ["▁", "p", "a", "r", "i", "s", "</s>"]
    S = np.zeros((V, V))
    for k in range(len(chain) - 1):
        a, b = idx[chain[k]], idx[chain[k + 1]]
        S[a, b] = S[b, a] = float(k + 1)
    lam, Q = np.linalg.eigh(S)
    keep = np.abs(lam) > 1e-9
    d1 = int(keep.sum())
    d = d1 if d1 % 2 == 0 else d1 + 1                           # the head size must be even for RoPE even with no blocks
    E = np.zeros((V, d)); g = np.zeros(d)
    E[:, :d1] = Q[:, keep] * np.sqrt(np.abs(lam[keep]))
    g[:d1] = np.sign(lam[keep])
    # predicted logits (up to the positive per-row factor of RMSNorm): e_x G e_y = S[x, y]
    tensors = {"model.embed_tokens.weight": E, "model.norm.weight": g}
    cfg = dict(vocab_size=V, hidden_size=d, intermediate_size=2, num_hidden_layers=0, num_attention_heads=1, num_key_value_heads=1,
               max_position_embeddings=64, rms_norm_eps=1e-5, rope_theta=100000.0, tie_word_embeddings=True, bos_token_id=1,
               eos_token_id=2, pad_token_id=3, attention_bias=False, mlp_bias=False, hidden_act="silu")
    info = {"model": "M1", "theory": "T1", "blocks": 0, "width": d, "rank_S": d1, "chain": chain,
            "S_nonzero": int((S != 0).sum()), "eigenvalues_kept": lam[keep].tolist(), "S": S}
    return tensors, cfg, info


def build_m1(out):
    tok, pieces, tinfo = write_tokenizer(out, "char")
    tensors, cfg, info = m1_tensors(pieces); info = {k: v for k, v in info.items() if k != "S"}
    info["parameters"] = save_model(out, cfg, tensors); info["tokenizer"] = tinfo
    return info


def rope_plane_vectors(d, theta=100000.0):
    """In transformers' rotate_half convention the rotary planes are (i, i + d/2); plane 0 has inverse frequency
    theta^0 = 1 radian per position. Returns unit vectors spanning plane 0 and its frequency."""
    u = np.zeros(d); u[0] = 1.0
    w = np.zeros(d); w[d // 2] = 1.0
    return u, w, 1.0


def m2_tensors(d, orthogonal, seed=SEED, alpha=5.0, beta=1.0, attn_scale=30.0, pieces=None):
    """One block, one head, previous-token attention, o-projection writes the capital's code. d must be even.
    Returns tensors, cfg, info."""
    pieces = pieces or word_pieces(); idx = {p: i for i, p in enumerate(pieces)}
    V = 50
    rng = np.random.default_rng(seed)
    const = d - 2                                                # the constant coordinate; d-1 unused
    E = np.zeros((V, d))
    if orthogonal:
        assert d >= V + 2, d
        for t in range(V):
            E[t, t] = 1.0
    else:
        C = rng.normal(size=(V, const)); C /= np.linalg.norm(C, axis=1, keepdims=True)
        E[:, :const] = C
    E[:, const] = beta
    # attention: q and k read the constant coordinate only, both in rotary plane 0, k rotated by phi = 1 radian so
    # that the score peaks at relative offset 1 (query at n, key at n-1)
    u, w, omega = rope_plane_vectors(d)
    phi = omega * 1.0
    Wq = np.zeros((d, d)); Wk = np.zeros((d, d)); Wv = np.zeros((d, d)); Wo = np.zeros((d, d))
    Wq[:, const] = attn_scale * u                                 # q = attn_scale * h_const * u
    Wk[:, const] = attn_scale * (math.cos(phi) * u + math.sin(phi) * w)
    Wv[:const, :const] = np.eye(const)                            # v = the token part of h
    # o-projection: code of X -> alpha * code of cap(X); least squares V K^+ (exact for orthogonal codes)
    K = np.stack([E[idx["▁" + x], :const] for x in CAPITALS])           # (12, const)
    Vm = np.stack([E[idx["▁" + y], :const] for y in CAPITALS.values()])   # (12, const)
    W = alpha * (np.linalg.pinv(K) @ Vm).T                        # (const, const): W @ k_X ~ alpha * v_capX
    Wo[:const, :const] = W
    g_in = np.ones(d); g_post = np.ones(d)
    g_f = np.ones(d); g_f[const:] = 0.0
    dff = 2
    tensors = {"model.embed_tokens.weight": E, "model.norm.weight": g_f,
               "model.layers.0.input_layernorm.weight": g_in, "model.layers.0.post_attention_layernorm.weight": g_post,
               "model.layers.0.self_attn.q_proj.weight": Wq, "model.layers.0.self_attn.k_proj.weight": Wk,
               "model.layers.0.self_attn.v_proj.weight": Wv, "model.layers.0.self_attn.o_proj.weight": Wo,
               "model.layers.0.mlp.gate_proj.weight": np.zeros((dff, d)), "model.layers.0.mlp.up_proj.weight": np.zeros((dff, d)),
               "model.layers.0.mlp.down_proj.weight": np.zeros((d, dff))}
    cfg = dict(vocab_size=V, hidden_size=d, intermediate_size=dff, num_hidden_layers=1, num_attention_heads=1, num_key_value_heads=1,
               max_position_embeddings=64, rms_norm_eps=1e-5, rope_theta=100000.0, tie_word_embeddings=True, bos_token_id=1,
               eos_token_id=2, pad_token_id=3, attention_bias=False, mlp_bias=False, hidden_act="silu")
    # the association residual of the least-squares map, per subject (0 for orthogonal codes)
    resid = float(np.abs(W @ K.T / alpha - Vm.T).max())
    coh = float(np.abs(E[:, :const] @ E[:, :const].T - np.eye(V)).max())
    info = {"model": f"M2_d{d}", "theory": "T2", "blocks": 1, "width": d, "orthogonal_codes": bool(orthogonal),
            "alpha": alpha, "beta": beta, "attn_scale": attn_scale, "phi": phi, "max_code_coherence": coh, "association_residual": resid}
    return tensors, cfg, info


def build_m2(out, d, orthogonal):
    tok, pieces, tinfo = write_tokenizer(out, "word")
    tensors, cfg, info = m2_tensors(d, orthogonal, pieces=pieces)
    info["parameters"] = save_model(out, cfg, tensors); info["tokenizer"] = tinfo
    return info


def main():
    root = sys.argv[1]
    os.makedirs(root, exist_ok=True)
    infos = []
    p = os.path.join(root, "M1"); os.makedirs(p, exist_ok=True); infos.append(build_m1(p))
    for d in LADDER:
        p = os.path.join(root, f"M2_d{d}"); os.makedirs(p, exist_ok=True)
        infos.append(build_m2(p, d, orthogonal=(d >= 52)))
    for info in infos:
        out = os.path.join(root, info["model"])
        info["sha256"] = {f: sha256(os.path.join(out, f)) for f in sorted(os.listdir(out)) if os.path.isfile(os.path.join(out, f))}
        json.dump(info, open(os.path.join(out, "construction.json"), "w"), indent=1)
        print(f"{info['model']}: blocks {info['blocks']}, width {info['width']}, parameters {info['parameters']}, "
              f"{'rank S ' + str(info['rank_S']) if 'rank_S' in info else 'coherence %.3f residual %.2e' % (info['max_code_coherence'], info['association_residual'])}")
    json.dump(infos, open(os.path.join(root, "constructions.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
