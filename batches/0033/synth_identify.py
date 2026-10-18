import json, struct, sys, hashlib, resource
import numpy as np
BASE, INST, GG, ON = sys.argv[1:5]; C = 576
def bf16_to_f32(b): return (np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
def st(path):
    f = open(path, "rb"); n = struct.unpack("<Q", f.read(8))[0]; h = json.loads(f.read(n)); e0 = 8 + n + h["model.embed_tokens.weight"]["data_offsets"][0]
    def row(r): f.seek(e0 + r * C * 2); return bf16_to_f32(f.read(C * 2))
    return row
rb, ri = st(BASE), st(INST)
print("base vs instruct row 260 equal entries:", int((rb(260) == ri(260)).sum()), "| row 0:", int((rb(0) == ri(0)).sum()))
GG_OFF = 1785664
with open(GG, "rb") as g:
    def gr(r): g.seek(GG_OFF + r * C * 2); return np.frombuffer(g.read(C * 2), dtype=np.float16).astype(np.float32)
    for r in (0, 260, 7042):
        print(f"gguf row {r}: equal to base {int((gr(r) == rb(r)).sum())}/576, to instruct {int((gr(r) == ri(r)).sum())}/576")
pats = {"base_row0": rb(0).tobytes(), "instruct_row0": ri(0).tobytes()}
found = {k: [] for k in pats}; CH = 32 << 20; L = C * 4
with open(ON, "rb") as o:
    off = 0; prev = b""
    while True:
        buf = o.read(CH)
        if not buf: break
        win = prev + buf
        for k, p in pats.items():
            j = win.find(p)
            while j >= 0:
                found[k].append(off - len(prev) + j); j = win.find(p, j + 1)
        off += len(buf); prev = buf[-L:]
print("onnx occurrences:", found)
with open(ON, "rb") as o:
    for k, poss in found.items():
        src = rb if k.startswith("base") else ri
        for pos in poss:
            res = {}
            for r in (260, 7042, 49151):
                o.seek(pos + r * L); res[r] = int((np.frombuffer(o.read(L), dtype=np.float32) == src(r)).sum())
            print(f"onnx block for {k} at {pos}: rows equal {res}")
for name, path in (("gguf", GG), ("onnx", ON)):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(16 << 20), b""): h.update(chunk)
    print(name, "sha256", h.hexdigest()[:16])
print("peak rss MiB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
