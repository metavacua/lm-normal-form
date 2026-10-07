# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Cell 07 of batch 0014: the token ids that llama.cpp's tokenizer gives, from the vocabulary stored in the GGUF file of the original checkpoint, for the 24 test texts, the 8
# plain prompts and the first 2,000 characters of the WikiText-2 test text, against the ids of the Hugging Face tokenizer (work/texts.json, made by xrt.py prepare).
# Usage: cell_tokenize.py WORK        needs LLAMACPP_DIR and LLAMACPP_SRC as rt_llamacpp.py does; writes WORK/tokenizers.json
import json, os, subprocess, sys
import rt_llamacpp as L
from common import dump, first_divergence


def main(work):
    t = json.load(open(os.path.join(work, "texts.json")))
    path = L.gguf(os.path.join(work, "variants", "orig"), work, "float32")
    out = os.path.join(work, "res", "llamacpp-tokens.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    subprocess.run([os.path.join(L.REL, "lmnf-llama"), "tokenize", path, os.path.join(work, "texts.json"), out], check=True)
    lc = json.load(open(out))
    same = [a == b for a, b in zip(lc, t["ids"])]
    res = {"texts": len(same), "identical": sum(same), "first_divergence": [first_divergence(a, b) for a, b, s in zip(lc, t["ids"], same) if not s][:8],
           "tokens": [len(x) for x in t["ids"]]}
    dump(res, os.path.join(work, "tokenizers.json"))
    print(json.dumps({k: res[k] for k in ("texts", "identical", "first_divergence")}))


if __name__ == "__main__":
    main(sys.argv[1])
