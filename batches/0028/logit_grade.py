# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""P14 of docs/batches/0028.md: the logits of a module (DIR_A) against a reference (DIR_B), as batch 0026's P7 and P8 and
with the same thresholds, and a structured result. The files are the float32 files N.f32 of shape [positions, VOCAB] that
batches/0010/logits.py and batches/0026/host.py write.

P7: for each prompt the five most probable next tokens (the last position, softmax in float64) are the same ids in the same
order, and each of the five probabilities differs by at most 1e-4. P8: the largest absolute difference of logits over every
position and the whole vocabulary is at most 1e-3.

Usage: logit_grade.py DIR_A DIR_B VOCAB OUT.json
       logit_grade.py selftest
"""

import glob
import json
import os
import sys
import tempfile

import numpy as np

P7_DP, P8_MAX = 1e-4, 1e-3


def softmax(x):
    e = np.exp(x.astype("float64") - x.max())
    return e / e.sum()


def grade(dir_a, dir_b, vocab):
    files = sorted(glob.glob(os.path.join(dir_a, "*.f32")))
    if not files:
        return {"error": f"no logits in {dir_a}", "P7": False, "P8": False, "prompts": []}
    out, ok7, ok8 = [], True, True
    for f in files:
        g = os.path.join(dir_b, os.path.basename(f))
        if not os.path.exists(g):
            return {"error": f"{os.path.basename(f)} missing in {dir_b}", "P7": False, "P8": False, "prompts": out}
        a, b = (np.fromfile(x, "float32").reshape(-1, vocab) for x in (f, g))
        if a.shape != b.shape:
            return {"error": f"shapes {a.shape} {b.shape}", "P7": False, "P8": False, "prompts": out}
        pa, pb = softmax(a[-1]), softmax(b[-1])
        ja, jb = np.argsort(-pa)[:5], np.argsort(-pb)[:5]
        same = bool((ja == jb).all())
        dp = float(np.abs(pa[ja] - pb[ja]).max())
        mx = float(np.abs(a - b).max())
        p7, p8 = same and dp <= P7_DP, mx <= P8_MAX
        ok7, ok8 = ok7 and p7, ok8 and p8
        out.append({"file": os.path.basename(f), "positions": int(a.shape[0]), "top5_same_order": same, "max_dp": dp,
                    "max_abs": mx, "P7": p7, "P8": p8})
    return {"error": None, "P7": ok7, "P8": ok8, "prompts": out, "max_abs": max(p["max_abs"] for p in out)}


def selftest():
    bad = 0

    def check(what, ok):
        nonlocal bad
        print(("ok   " if ok else "FAIL ") + what)
        bad += 0 if ok else 1

    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as d:
        for sub in ("a", "b", "c"):
            os.makedirs(f"{d}/{sub}")
        x = rng.standard_normal((3, 50)).astype("float32")
        x.tofile(f"{d}/a/0.f32")
        x.tofile(f"{d}/b/0.f32")
        y = x.copy()
        y[0, 0] += 0.5
        y.tofile(f"{d}/c/0.f32")
        same = grade(f"{d}/a", f"{d}/b", 50)
        moved = grade(f"{d}/a", f"{d}/c", 50)
        check("equal logits pass P7 and P8", same["P7"] and same["P8"] and same["max_abs"] == 0.0)
        check("the control: one value moved by 0.5 is reported as 0.5 and fails P8", moved["max_abs"] == 0.5 and not moved["P8"])
        check("a missing file is an error and not a pass", grade(f"{d}/a", f"{d}/none", 50)["error"] is not None)
    return bad


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        sys.exit(1 if selftest() else 0)
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    res = grade(sys.argv[1], sys.argv[2], int(sys.argv[3]))
    with open(sys.argv[4], "w") as fh:
        json.dump(res, fh, indent=1, sort_keys=True)
    print(json.dumps({k: res[k] for k in ("error", "P7", "P8") if k in res}))
