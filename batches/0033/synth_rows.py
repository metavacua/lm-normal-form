# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Synthetic rows for the output side of one pair. usage: synth_rows.py SRC.safetensors DIR OUT_ROW COMP_ROW TAG REVISION
DIR holds yhat.npy, logits.npy and prompt.json from fwd_numpy.py. Writes DIR/synthetic_rows_TAG.json; verifies each patch on a copy of the
safetensors file with the numpy evaluator. ŷ is unchanged by every patch (neither row is a prompt token), so only z[OUT_ROW] and z[COMP_ROW] can change."""
import json
import os
import shutil
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from synth_common import bf16_to_f32, f32_to_bf16_bytes, peak_rss_mb, st_header, st_tensor  # noqa: E402

SRC, DIR, OUT, COMP, TAG, REV = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5], sys.argv[6]
hdr, start = st_header(SRC)
E0, (V, C) = st_tensor(hdr, start, "model.embed_tokens.weight")
f = open(SRC, "rb")


def row_bytes(fh, r):
    fh.seek(E0 + r * C * 2)
    return fh.read(C * 2)


def all_logits(fh, yhat):
    """z = E · ŷ for every row of the file fh, streamed in 2,048-row chunks."""
    zz = np.empty(V, dtype=np.float32)
    for r0 in range(0, V, 2048):
        r1 = min(V, r0 + 2048)
        fh.seek(E0 + r0 * C * 2)
        zz[r0:r1] = bf16_to_f32(fh.read((r1 - r0) * C * 2)).reshape(r1 - r0, C) @ yhat
    return zz


yhat = np.load(os.path.join(DIR, "yhat.npy"))
z = np.load(os.path.join(DIR, "logits.npy"))
ids = json.load(open(os.path.join(DIR, "prompt.json")))["ids"]
assert int(np.argmax(z)) == OUT, "OUT_ROW is not the argmax"
assert OUT not in ids and COMP not in ids, "a written row would be a prompt row too, and ŷ would change"
rO = bf16_to_f32(row_bytes(f, OUT))
rC_ = bf16_to_f32(row_bytes(f, COMP))
assert abs(float(rO @ yhat) - z[OUT]) < 1e-3 and abs(float(rC_ @ yhat) - z[COMP]) < 1e-3
m = float(z[OUT] - z[COMP])
target = float(z[COMP]) - m  # mirror: the competitor wins by the margin the output wins by now
patches = [{"name": "A_swap_rows", "writes": [{"row": OUT, "hex": row_bytes(f, COMP).hex()}, {"row": COMP, "hex": row_bytes(f, OUT).hex()}],
            "prediction": {"argmax": COMP, f"z{OUT}": float(z[COMP]), f"z{COMP}": float(z[OUT]), "others_bit_identical": True, "rounding": "none"}}]
alpha = (target - float(z[OUT])) / float(yhat @ yhat)
rB = rO + alpha * yhat
rB_bytes = f32_to_bf16_bytes(rB)
rB_q = bf16_to_f32(rB_bytes)
bound = float(2.0 ** -9 * np.abs(rB_q * yhat).sum())
patches.append({"name": "B_minnorm_output_row", "writes": [{"row": OUT, "hex": rB_bytes.hex()}],
                "construction": {"target": target, "alpha": alpha, "change_norm_f32": float(np.linalg.norm(alpha * yhat)), "change_norm_after_bf16": float(np.linalg.norm(rB_q - rO)), "entries_changed": int((rB_q != rO).sum())},
                "prediction": {"argmax": COMP, f"z{OUT}_within_bound_of_target": bound, f"z{OUT}_exact_f32_from_rounded_row": float(rB_q @ yhat), f"z{COMP}": float(z[COMP]), "others_bit_identical": True}})
c = int(np.argmax(np.abs(yhat)))
d = (target - float(z[OUT])) / float(yhat[c])
w_old = float(rO[c])
w_new_b = f32_to_bf16_bytes(np.array([w_old + d]))
w_new = float(bf16_to_f32(w_new_b)[0])
rCw = rO.copy()
rCw[c] = w_new
grid = bf16_to_f32(np.arange(65536, dtype=np.uint16).tobytes())
grid = grid[np.isfinite(grid) & (np.abs(grid) < 1e4)]
with np.errstate(over="ignore"):
    ach = float(z[OUT]) + (grid - w_old) * float(yhat[c])
    flips = ach < float(z[COMP])
patches.append({"name": "C_one_weight_of_output_row", "writes": [{"row": OUT, "hex": f32_to_bf16_bytes(rCw).hex()}],
                "construction": {"coordinate": c, "yhat_c": float(yhat[c]), "weight_old": w_old, "weight_new": w_new, "weight_new_bf16_hex": w_new_b.hex(), "delta_f32": d,
                                 "bf16_values_below_1e4_that_flip": int(flips.sum()), "of": int(grid.size), "smallest_change_that_flips": float(np.min(np.abs(grid[flips] - w_old))) if flips.any() else None},
                "prediction": {"argmax": COMP, f"z{OUT}_exact_f32": float(rCw @ yhat), f"z{COMP}": float(z[COMP]), "others_bit_identical": True}})
# one copy of the file; each patch is written, evaluated and undone (the original bytes of every row any patch writes), so each evaluation sees one patch
DST = os.path.join(DIR, "syn.safetensors")
shutil.copyfile(SRC, DST)
orig = {r: row_bytes(f, r) for r in sorted({w["row"] for p in patches for w in p["writes"]})}
results = []
with open(DST, "r+b") as g:
    for p in patches:
        for w in p["writes"]:
            g.seek(E0 + w["row"] * C * 2)
            g.write(bytes.fromhex(w["hex"]))
        zz = all_logits(g, yhat)
        res = {"name": p["name"], "argmax": int(np.argmax(zz)), f"z{OUT}": float(zz[OUT]), f"z{COMP}": float(zz[COMP]), "logits_changed": np.nonzero(zz != z)[0].tolist(), "second": int(np.argsort(-zz)[1])}
        results.append(res)
        p["byte_offsets_safetensors"] = [E0 + w["row"] * C * 2 for w in p["writes"]]
        print(res)
        for r, b in orig.items():
            g.seek(E0 + r * C * 2)
            g.write(b)
os.remove(DST)
with open(os.path.join(DIR, f"synthetic_rows_{TAG}.json"), "w") as fh:
    json.dump({"model": TAG, "source": {"file": os.path.basename(SRC), "revision": REV, "tensor": "model.embed_tokens.weight", "row_bytes": C * 2, "data_start": E0},
               "pair": {"prompt_ids": ids, "output_row": OUT, "competitor_row": COMP},
               "baseline": {"argmax": OUT, f"z{OUT}": float(z[OUT]), f"z{COMP}": float(z[COMP]), "margin": m, "target_for_B_and_C": target, "yhat_norm": float(np.linalg.norm(yhat)), "yhat_argmax_abs_coordinate": c},
               "patches": patches, "numpy_verification": results}, fh, indent=1)
print(TAG, "target", round(target, 4), "| B change norm", round(patches[1]["construction"]["change_norm_f32"], 5), "entries", patches[1]["construction"]["entries_changed"], "bound", round(bound, 4),
      "| C coord", c, "yhat_c", round(float(yhat[c]), 4), "w", round(w_old, 5), "->", round(w_new, 5), "smallest flip", patches[2]["construction"]["smallest_change_that_flips"], "| peak rss MiB", int(peak_rss_mb()))
