# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# A binding error that the bag of tensors cannot see. In a safetensors file the names are the only wiring, so
# the contents of layer 0's gate_proj and up_proj (both [d_ff, d_model]) are exchanged in a copy of the file:
# the bag root is the same, the labelled root is not, and the model's logits on the first prompt move.
# Usage: swap.py MODEL HUB WORKDIR PROMPT_FILE OUT.json
import json, sys
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoModelForCausalLM, AutoTokenizer
import identity, identity_core as core


def main(model, hub, work, prompts, out):
    a, b = "model.layers.0.mlp.gate_proj.weight", "model.layers.0.mlp.up_proj.weight"
    t = load_file(hub)
    s = dict(t)
    s[a], s[b] = t[b].clone(), t[a].clone()
    save_file(s, f"{work}/swapped.safetensors", metadata={"format": "pt"})
    r0 = identity.record("safetensors", identity.read_st(hub)[0], "hub")
    r1 = identity.record("safetensors", identity.read_st(f"{work}/swapped.safetensors")[0], "swapped")
    tok = AutoTokenizer.from_pretrained(model)
    ids = tok(open(prompts, encoding="utf-8").readline().rstrip("\n"), return_tensors="pt")
    m = AutoModelForCausalLM.from_pretrained(model)
    m.eval()
    with torch.no_grad():
        base = m(**ids).logits.float()
        m.load_state_dict(load_file(f"{work}/swapped.safetensors"), strict=False)
        moved = m(**ids).logits.float()
    res = {"swapped": [a, b], "same_shape": list(t[a].shape) == list(t[b].shape),
           "bag_root_equal": r0["subject"]["bag_root"] == r1["subject"]["bag_root"],
           "labelled_root_equal": r0["subject"]["labelled_root"] == r1["subject"]["labelled_root"],
           "max_abs_logit_difference": float((base - moved).abs().max()),
           "argmax_equal_at_every_position": bool((base.argmax(-1) == moved.argmax(-1)).all()),
           "verdicts": [core.verify(r0), core.verify(r1)]}
    json.dump(res, open(out, "w"), indent=1, sort_keys=True)
    print(json.dumps(res, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(*sys.argv[1:6])
