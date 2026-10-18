"""Write the patches of one pair into copies of that model's GGUF (f16) and ONNX (f32) files at the located byte addresses, read them back.
usage: xenc2.py synthetic_rows.json GGUF GGUF_DATA_OFFSET ONNX ONNX_BLOCK_BYTE"""
import json, shutil, sys, hashlib, resource
import numpy as np
JS, GG, GOFF, ON, OPOS = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4], int(sys.argv[5]); C = 576
assert OPOS > 0 and GOFF > 0
syn = json.load(open(JS)); D = JS.rsplit("/", 1)[0] if "/" in JS else "."
def bf16_to_f32(b): return (np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
def sha(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for ch in iter(lambda: fh.read(16 << 20), b""): h.update(ch)
    return h.hexdigest()[:16]
enc = {"gguf": {"file": GG, "sha256_prefix": sha(GG), "tensor": "token_embd.weight", "type": "F16", "data_offset": GOFF, "row_bytes": C * 2, "patches": {}},
       "onnx": {"file": ON, "sha256_prefix": sha(ON), "initializer": "model.embed_tokens.weight", "type": "FLOAT", "block_byte": OPOS, "row_bytes": C * 4, "patches": {}}}
shutil.copyfile(GG, f"{D}/syn.gguf"); shutil.copyfile(ON, f"{D}/syn.onnx")
for p in syn["patches"]:
    g16 = {}; o32 = {}
    for w in p["writes"]:
        v = bf16_to_f32(bytes.fromhex(w["hex"])); v16 = v.astype(np.float16); w["f16_exact_entries"] = int((v16.astype(np.float32) == v).sum()); w["f16_hex"] = v16.tobytes().hex()
        with open(f"{D}/syn.gguf", "r+b") as g: g.seek(GOFF + w["row"] * C * 2); g.write(v16.tobytes())
        with open(f"{D}/syn.onnx", "r+b") as o: o.seek(OPOS + w["row"] * C * 4); o.write(v.tobytes())
        with open(f"{D}/syn.gguf", "rb") as g: g.seek(GOFF + w["row"] * C * 2); g16[w["row"]] = int((np.frombuffer(g.read(C * 2), dtype=np.float16).astype(np.float32) == v).sum())
        with open(f"{D}/syn.onnx", "rb") as o: o.seek(OPOS + w["row"] * C * 4); o32[w["row"]] = int((np.frombuffer(o.read(C * 4), dtype=np.float32) == v).sum())
    enc["gguf"]["patches"][p["name"]] = {"read_back_equal": g16, "f16_exact": [w["f16_exact_entries"] for w in p["writes"]], "byte_offsets": [GOFF + w["row"] * C * 2 for w in p["writes"]]}
    enc["onnx"]["patches"][p["name"]] = {"read_back_equal": o32, "byte_offsets": [OPOS + w["row"] * C * 4 for w in p["writes"]]}
    print(p["name"], "gguf", g16, "f16-exact", enc["gguf"]["patches"][p["name"]]["f16_exact"], "| onnx", o32)
    # restore the copies to the original rows before the next patch, so each copy holds exactly one patch when read back
    shutil.copyfile(GG, f"{D}/syn.gguf"); shutil.copyfile(ON, f"{D}/syn.onnx")
syn["encodings"] = enc; json.dump(syn, open(JS, "w"), indent=1)
print("peak rss MiB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
