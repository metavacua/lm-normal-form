# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The same tensors written by other programs, and one file read when its header is written another way: what the bytes
# of a file say about its tensors, and what they do not. For the safetensors file HUB (the Hub's own):
#   resave        the tensors read and written again by safetensors.torch.save_file with the same metadata
#   transformers  from_pretrained and save_pretrained in bfloat16, and in the default float32, then the float32
#                 file's tensors cast back to bfloat16 against the original's bytes
#   metadata      one and two metadata keys, saved many times in one process: how many distinct byte strings
#   header        the file with its header written four other ways: the library's reading of each
# Usage: writers.py MODEL HUB WORKDIR OUT.json
import hashlib, json, os, struct, sys
import torch
from safetensors.torch import load_file, save, save_file


def sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()


def header_of(raw):
    n = struct.unpack("<Q", raw[:8])[0]
    return n, raw[8:8 + n], raw[8 + n:]


def with_header(head, data):
    pad = (-len(head)) % 8
    head = head + b" " * pad
    return struct.pack("<Q", len(head)) + head + data


def main(model, hub, work, out):
    from transformers import AutoModelForCausalLM
    res = {"hub_sha256": sha(hub)}
    orig = load_file(hub)
    meta = {"format": "pt"}
    save_file(orig, f"{work}/resave.safetensors", metadata=meta)
    res["resave_sha256_equal_to_hub"] = sha(f"{work}/resave.safetensors") == res["hub_sha256"]
    for tag, kw in (("bf16", {"dtype": torch.bfloat16}), ("f32", {})):
        m = AutoModelForCausalLM.from_pretrained(model, **kw)
        m.save_pretrained(f"{work}/save-{tag}")
        names = sorted(p for p in os.listdir(f"{work}/save-{tag}") if p.endswith(".safetensors"))
        res[f"save_{tag}_files"] = names
        if names == ["model.safetensors"]:
            path = f"{work}/save-{tag}/model.safetensors"
            res[f"save_{tag}_sha256_equal_to_hub"] = sha(path) == res["hub_sha256"]
            t = load_file(path)
            res[f"save_{tag}_has_lm_head"] = "lm_head.weight" in t
            res[f"save_{tag}_tensors"] = len(t)
            res[f"save_{tag}_dtypes"] = sorted({str(v.dtype) for v in t.values()})
            res[f"save_{tag}_data_bytes"] = sum(v.numel() * v.element_size() for v in t.values())
            if tag == "f32":
                res["f32_cast_to_bf16_bytes_equal_to_original"] = all(
                    k in t and torch.equal(t[k].to(torch.bfloat16).view(torch.int16), v.view(torch.int16)) for k, v in orig.items())
        del m
    x = {"w": torch.zeros(1)}
    res["metadata_one_key_distinct_byte_strings_in_40"] = len({save(x, metadata={"a": "1"}) for _ in range(40)})
    res["metadata_two_keys_distinct_byte_strings_in_40"] = len({save(x, metadata={"a": "1", "b": "2"}) for _ in range(40)})
    raw = open(hub, "rb").read()
    n, head, data = header_of(raw)
    j = json.loads(head)
    m0 = j.pop("__metadata__", None)
    variants = {
        "a_trailing_spaces": with_header(head + b" " * 8, data),
        "b_leading_spaces": with_header(b" " * 8 + head, data),
        "c_metadata_last_and_whitespace": with_header(json.dumps({**j, "__metadata__": m0}, indent=1).encode(), data),
        "d_table_in_reverse_order": with_header(json.dumps({"__metadata__": m0, **dict(reversed(list(j.items())))}, separators=(",", ":")).encode(), data),
    }
    del raw
    res["header"] = {}
    for name, blob in variants.items():
        path = f"{work}/{name}.safetensors"
        open(path, "wb").write(blob)
        n2, _, data2 = header_of(blob)
        r = {"file_sha256_equal_to_hub": sha(path) == res["hub_sha256"],
             "bytes_after_header_sha256_equal_to_hub": hashlib.sha256(data2).hexdigest() == hashlib.sha256(data).hexdigest()}
        try:
            t = load_file(path)
            r["library_loads"] = True
            r["tensors_equal_to_original"] = all(torch.equal(t[k], v) for k, v in orig.items()) and len(t) == len(orig)
        except Exception as e:
            r["library_loads"] = False
            r["error"] = f"{type(e).__name__}: {str(e)[:160]}"
        res["header"][name] = r
        os.remove(path)
    json.dump(res, open(out, "w"), indent=1, sort_keys=True)
    print(json.dumps(res, indent=1, sort_keys=True))


if __name__ == "__main__":
    main(*sys.argv[1:5])
