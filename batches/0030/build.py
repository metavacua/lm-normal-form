# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cell 01, part 1: the instance of docs/batches/0030.md as a Hugging Face directory.

usage: build.py OUT_DIR
Writes config.json, model.safetensors (float32), tokenizer.model (SentencePiece character model, 50 pieces),
tokenizer_config.json with the chat template, generation_config.json, and build.json (hashes, the alphabet).

Amendment to the registration, made before any run: the newline is not in the alphabet, because SentencePiece
normalises it to whitespace; the alphabet is 4 specials + 26 letters + 10 digits + 9 punctuation marks + space = 50,
and the chat template separates turns with spaces.
"""
import hashlib
import json
import os
import sys

import torch

SEED = 8
STD = 1.0 / 24.0                       # SmolLM2's init_method.std = 0.041666…
LETTERS = "abcdefghijklmnopqrstuvwxyz"
DIGITS = "0123456789"
PUNCT = ".,?!:-'()"
ALPHABET = LETTERS + DIGITS + PUNCT + " "           # 46 characters; plus <unk> <s> </s> <pad> = 50 pieces
CHAT = ("{% for m in messages %}{% if m['role'] == 'user' %}<s>user: {{ m['content'] }} assistant: "
        "{% elif m['role'] == 'assistant' %}{{ m['content'] }}</s>{% endif %}{% endfor %}")


def sha256(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def main():
    out = sys.argv[1]
    os.makedirs(out, exist_ok=True)
    from transformers import LlamaConfig, LlamaForCausalLM, LlamaTokenizer
    import sentencepiece as spm
    cfg = LlamaConfig(vocab_size=50, hidden_size=16, intermediate_size=24, num_hidden_layers=3, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=64, rms_norm_eps=1e-5, rope_theta=100000.0,
                      tie_word_embeddings=True, initializer_range=STD, bos_token_id=1, eos_token_id=2, pad_token_id=3,
                      attention_bias=False, mlp_bias=False, hidden_act="silu", torch_dtype="float32")
    torch.manual_seed(SEED)
    model = LlamaForCausalLM(cfg)
    # the substitution, stated here and not left to the library's initialiser: normal(0, 1/24) for every weight of a
    # linear map or embedding, 1 for every norm gain, drawn in the order of named_parameters under seed 8
    g = torch.Generator().manual_seed(SEED)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if name.endswith("norm.weight"):
                p.fill_(1.0)
            else:
                p.copy_(torch.randn(p.shape, generator=g) * STD)
    model.tie_weights()
    model.eval()
    model.config.use_cache = False
    model.save_pretrained(out, safe_serialization=True)
    # the tokenizer: a SentencePiece character model whose 46 characters plus 4 specials are exactly 50 pieces
    corpus = os.path.join(out, "alphabet_corpus.txt")
    with open(corpus, "w", encoding="utf-8") as f:
        for _ in range(20):
            f.write(ALPHABET + "\n" + "the capital of france is paris. 0123456789 (a-b), c? d! e: f'g\n")
    spm.SentencePieceTrainer.train(input=corpus, model_prefix=os.path.join(out, "tokenizer"), model_type="char", vocab_size=50,
                                   character_coverage=1.0, unk_id=0, bos_id=1, eos_id=2, pad_id=3, unk_piece="<unk>", bos_piece="<s>",
                                   eos_piece="</s>", pad_piece="<pad>", add_dummy_prefix=False, remove_extra_whitespaces=False,
                                   normalization_rule_name="identity", byte_fallback=False, num_threads=1, minloglevel=2)
    sp = spm.SentencePieceProcessor(model_file=os.path.join(out, "tokenizer.model"))
    pieces = [sp.id_to_piece(i) for i in range(sp.get_piece_size())]
    assert len(pieces) == 50, pieces
    # The fast tokenizer (tokenizer.json), the path by which transformers reads SmolLM2: a Unigram model over the same 50
    # pieces with equal scores (every piece is one character, so segmentation is deterministic), Metaspace as
    # SentencePiece does it. transformers' own converter refuses a `char`-type SentencePiece file ("trained with a
    # different algorithm", first run 38980018769), so the fast file is built directly. tokenizer.model stays beside it
    # for llama.cpp's converter, which reads SentencePiece for the Llama architecture.
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast
    vocab = [(pc, 0.0 if i >= 4 else -100.0) for i, pc in enumerate(pieces)]
    tk = Tokenizer(models.Unigram(vocab, unk_id=0, byte_fallback=False))
    tk.pre_tokenizer = pre_tokenizers.Metaspace(replacement="▁", prepend_scheme="never", split=False)
    tk.decoder = decoders.Metaspace(replacement="▁", prepend_scheme="never", split=False)
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, unk_token="<unk>", bos_token="<s>", eos_token="</s>", pad_token="<pad>",
                                  clean_up_tokenization_spaces=False, model_max_length=64)
    tok.chat_template = CHAT
    tok.save_pretrained(out)
    # the slow SentencePiece path, as a second reading of the same pieces: ids must agree with the fast tokenizer
    slow = LlamaTokenizer(vocab_file=os.path.join(out, "tokenizer.model"), unk_token="<unk>", bos_token="<s>", eos_token="</s>",
                          pad_token="<pad>", add_bos_token=False, add_eos_token=False, legacy=False, clean_up_tokenization_spaces=False)
    probe = "the capital of france is paris. 0123456789 (a-b), c? d! e: f'g"
    slow_ids = slow(probe, add_special_tokens=False)["input_ids"]; fast_ids = tok(probe, add_special_tokens=False)["input_ids"]
    tokenizer_agreement = {"probe": probe, "slow_ids": slow_ids, "fast_ids": fast_ids, "equal": slow_ids == fast_ids,
                           "slow_pieces": slow.convert_ids_to_tokens(slow_ids), "fast_pieces": tok.convert_ids_to_tokens(fast_ids)}
    if not tokenizer_agreement["equal"]:
        print("slow and fast tokenizers disagree on the probe (recorded in build.json):", tokenizer_agreement["slow_pieces"][:12], "vs",
              tokenizer_agreement["fast_pieces"][:12])
    os.remove(corpus)
    os.remove(os.path.join(out, "tokenizer.vocab")) if os.path.exists(os.path.join(out, "tokenizer.vocab")) else None
    with open(os.path.join(out, "generation_config.json"), "w") as f:
        json.dump({"bos_token_id": 1, "eos_token_id": 2, "pad_token_id": 3, "do_sample": False, "max_new_tokens": 32}, f, indent=1)
    # round trip of the alphabet through the tokenizer, and the ids of every symbol
    ids = tok(ALPHABET, add_special_tokens=False)["input_ids"]
    back = tok.decode(ids)
    info = {"seed": SEED, "std": STD, "pieces": pieces, "alphabet": ALPHABET, "alphabet_ids": ids, "alphabet_round_trip": back == ALPHABET,
            "tokenizer_agreement": tokenizer_agreement,
            "parameters": int(sum(p.numel() for p in model.parameters())),
            "tensors": {k: list(v.shape) for k, v in model.state_dict().items()},
            "sha256": {f: sha256(os.path.join(out, f)) for f in sorted(os.listdir(out)) if os.path.isfile(os.path.join(out, f))}}
    with open(os.path.join(out, "build.json"), "w") as f:
        json.dump(info, f, indent=1)
    print("built:", info["parameters"], "parameters,", len(info["tensors"]), "tensors; alphabet round trip", info["alphabet_round_trip"],
          "; safetensors", info["sha256"]["model.safetensors"][:16])


if __name__ == "__main__":
    main()
