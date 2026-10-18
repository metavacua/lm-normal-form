"""Synthetic rows for the output side of one pair. usage: syn2.py SRC.safetensors DIR OUT_ROW COMP_ROW TAG REVISION
DIR holds yhat.npy and logits.npy from fwd.py. Writes DIR/synthetic_rows_TAG.json; verifies each patch on DIR/syn.safetensors with the numpy evaluator.
ŷ is unchanged by every patch (neither row is a prompt token), so only z[OUT_ROW] and z[COMP_ROW] can change."""
import json, struct, shutil, sys, resource
import numpy as np
SRC, DIR, OUT, COMP, TAG, REV = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
f = open(SRC, "rb"); n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); start = 8 + n
E = hdr["model.embed_tokens.weight"]; C = E["shape"][1]; V = E["shape"][0]; E0 = start + E["data_offsets"][0]
def bf16_to_f32(b): return (np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)
def f32_to_bf16_bytes(a):
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32); lsb = (u >> 16) & 1; return ((u + 0x7FFF + lsb) >> 16).astype(np.uint16).tobytes()
def row_bytes(fh, r): fh.seek(E0 + r * C * 2); return fh.read(C * 2)
yhat = np.load(f"{DIR}/yhat.npy"); z = np.load(f"{DIR}/logits.npy")
assert int(np.argmax(z)) == OUT, "OUT_ROW is not the argmax"
rO = bf16_to_f32(row_bytes(f, OUT)); rC_ = bf16_to_f32(row_bytes(f, COMP))
assert abs(float(rO @ yhat) - z[OUT]) < 1e-3 and abs(float(rC_ @ yhat) - z[COMP]) < 1e-3
m = float(z[OUT] - z[COMP]); target = float(z[COMP]) - m          # mirror: the competitor wins by the margin the output wins by now
patches = [{"name": "A_swap_rows", "writes": [{"row": OUT, "hex": row_bytes(f, COMP).hex()}, {"row": COMP, "hex": row_bytes(f, OUT).hex()}],
            "prediction": {"argmax": COMP, f"z{OUT}": float(z[COMP]), f"z{COMP}": float(z[OUT]), "others_bit_identical": True, "rounding": "none"}}]
alpha = (target - float(z[OUT])) / float(yhat @ yhat); rB = rO + alpha * yhat; rB_bytes = f32_to_bf16_bytes(rB); rB_q = bf16_to_f32(rB_bytes)
bound = float(2.0 ** -9 * np.abs(rB_q * yhat).sum())
patches.append({"name": "B_minnorm_output_row", "writes": [{"row": OUT, "hex": rB_bytes.hex()}],
                "construction": {"target": target, "alpha": alpha, "change_norm_f32": float(np.linalg.norm(alpha * yhat)), "change_norm_after_bf16": float(np.linalg.norm(rB_q - rO)), "entries_changed": int((rB_q != rO).sum())},
                "prediction": {"argmax": COMP, f"z{OUT}_within_bound_of_target": bound, f"z{OUT}_exact_f32_from_rounded_row": float(rB_q @ yhat), f"z{COMP}": float(z[COMP]), "others_bit_identical": True}})
c = int(np.argmax(np.abs(yhat))); d = (target - float(z[OUT])) / float(yhat[c]); w_old = float(rO[c]); w_new_b = f32_to_bf16_bytes(np.array([w_old + d])); w_new = float(bf16_to_f32(w_new_b)[0])
rCw = rO.copy(); rCw[c] = w_new
grid = bf16_to_f32(np.arange(65536, dtype=np.uint16).tobytes()); grid = grid[np.isfinite(grid) & (np.abs(grid) < 1e4)]
with np.errstate(over="ignore"): ach = float(z[OUT]) + (grid - w_old) * float(yhat[c]); flips = ach < float(z[COMP])
patches.append({"name": "C_one_weight_of_output_row", "writes": [{"row": OUT, "hex": f32_to_bf16_bytes(rCw).hex()}],
                "construction": {"coordinate": c, "yhat_c": float(yhat[c]), "weight_old": w_old, "weight_new": w_new, "weight_new_bf16_hex": w_new_b.hex(), "delta_f32": d,
                                 "bf16_values_below_1e4_that_flip": int(flips.sum()), "of": int(grid.size), "smallest_change_that_flips": float(np.min(np.abs(grid[flips] - w_old))) if flips.any() else None},
                "prediction": {"argmax": COMP, f"z{OUT}_exact_f32": float(rCw @ yhat), f"z{COMP}": float(z[COMP]), "others_bit_identical": True}})
results = []; DST = f"{DIR}/syn.safetensors"
for p in patches:
    shutil.copyfile(SRC, DST)
    with open(DST, "r+b") as g:
        for w in p["writes"]: g.seek(E0 + w["row"] * C * 2); g.write(bytes.fromhex(w["hex"]))
    with open(DST, "rb") as g:
        zz = np.empty(V, dtype=np.float32)
        for r0 in range(0, V, 2048):
            r1 = min(V, r0 + 2048); g.seek(E0 + r0 * C * 2); zz[r0:r1] = bf16_to_f32(g.read((r1 - r0) * C * 2)).reshape(r1 - r0, C) @ yhat
    res = {"name": p["name"], "argmax": int(np.argmax(zz)), f"z{OUT}": float(zz[OUT]), f"z{COMP}": float(zz[COMP]), "logits_changed": np.nonzero(zz != z)[0].tolist(), "second": int(np.argsort(-zz)[1])}
    results.append(res); p["byte_offsets_safetensors"] = [E0 + w["row"] * C * 2 for w in p["writes"]]; print(res)
json.dump({"model": TAG, "source": {"file": SRC, "revision": REV, "tensor": "model.embed_tokens.weight", "row_bytes": C * 2, "data_start": E0},
           "pair": {"prompt_ids": [504, 3575, 282, 4649, 314], "output_row": OUT, "competitor_row": COMP},
           "baseline": {"argmax": OUT, f"z{OUT}": float(z[OUT]), f"z{COMP}": float(z[COMP]), "margin": m, "target_for_B_and_C": target, "yhat_norm": float(np.linalg.norm(yhat)), "yhat_argmax_abs_coordinate": c},
           "patches": patches, "numpy_verification": results}, open(f"{DIR}/synthetic_rows_{TAG}.json", "w"), indent=1)
print(TAG, "target", round(target, 4), "| B change norm", round(patches[1]["construction"]["change_norm_f32"], 5), "entries", patches[1]["construction"]["entries_changed"], "bound", round(bound, 4),
      "| C coord", c, "yhat_c", round(float(yhat[c]), 4), "w", round(w_old, 5), "->", round(w_new, 5), "smallest flip", patches[2]["construction"]["smallest_change_that_flips"], "| peak rss MiB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024)
