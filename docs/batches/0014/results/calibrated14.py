# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The cells of P6, P15 and P15n read with the null variant of addendum 2 as the floor: for each cell, the NMSE of every exact variant against the cell's own original, in units of the NMSE
# of the null variant `nullall` against its own original in the same cell (the null from the job `nullvariant`, the variants from the quantization job or the job `norot`: other runners,
# so the unit is good to the 10% by which e itself differs between runners). Post hoc: written after the results of the addendum run were read. The classes are those of the registration.
# Usage: calibrated14.py DIR     DIR holds the downloaded artifacts xrt-quant, xrt-norot, xrt-null
import glob, json, os, sys
d = sys.argv[1]
S = {}
for p in glob.glob(os.path.join(d, "xrt-*", "summary-*.json")):
    s = json.load(open(p))
    S[(os.path.basename(os.path.dirname(p)), s["dtype"])] = s
SAME_W, DIFF_W = ("heads", "units_blk", "resid_blk"), ("units", "resid", "scale", "perm", "all", "canon")
SAME_C, DIFF_C = ("heads", "perm", "resid", "resid_blk", "units", "units_blk"), ("scale", "all", "canon")
ALL9 = ("scale", "perm", "all", "canon", "heads", "units", "units_blk", "resid", "resid_blk")
cells = [("Q8_0 weights", "xrt-quant", "q8_0", SAME_W, DIFF_W), ("Q4_0 weights", "xrt-quant", "q4_0", SAME_W, DIFF_W),
         ("f16 cache", "xrt-quant", "float32-kvf16", ALL9, ()), ("Q8_0 cache, rotated", "xrt-quant", "float32-kvq8_0", SAME_C, DIFF_C), ("Q4_0 cache, rotated", "xrt-quant", "float32-kvq4_0", SAME_C, DIFF_C),
         ("Q8_0 cache, not rotated", "xrt-norot", "float32-kvq8_0-norot", SAME_C, DIFF_C), ("Q4_0 cache, not rotated", "xrt-norot", "float32-kvq4_0-norot", SAME_C, DIFF_C)]
print("| cell | null NMSE | variants claimed the same, in units of the null (smallest to largest) | variants claimed different (smallest to largest) | separated |")
print("|---|---|---|---|---|")
for name, art, dt, same, diff in cells:
    s, n = S.get((art, dt)), S.get(("xrt-null", dt))
    if s is None or n is None:
        print(f"| {name} | n/a | | | |")
        continue
    null = n["variants"]["nullall"]["vs_same_runtime_original"]["nmse"]
    f = lambda v: s["variants"][v]["vs_same_runtime_original"]["nmse"] / null
    a = sorted((f(v), v) for v in same)
    b = sorted((f(v), v) for v in diff)
    sep = "yes" if not b or a[-1][0] < b[0][0] else "no"
    fmt = lambda xs: ", ".join(f"{v} {x:.2g}" for x, v in xs)
    print(f"| {name} | {null:.3g} | {fmt(a)} | {fmt(b) if b else '(none)'} | {sep} |")
