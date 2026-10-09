# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# P7 and P8 of docs/batches/0026.md: logits of the module (DIR_A) against a reference (DIR_B), the
# float32 files N.f32 of shape [positions, VOCAB] that batches/0010/logits.py and the runner write.
# P7: for each prompt the five most probable next tokens (the last position, softmax in float64) are
# the same ids in the same order, and each of the five probabilities differs by at most 1e-4.
# P8: the largest absolute difference of logits over every position and the whole vocabulary is at
# most 1e-3. The numbers are printed for each prompt whatever the verdict. Exit status 0 if both
# hold for every prompt, 1 if one does not, 2 if a file is missing or the shapes differ.
# Usage: grade.py DIR_A DIR_B VOCAB
import glob, os, sys
import numpy as np

A, B, V = sys.argv[1], sys.argv[2], int(sys.argv[3])
P7_DP, P8_MAX = 1e-4, 1e-3
softmax = lambda l: (lambda e: e / e.sum())(np.exp(l.astype("float64") - l.max()))
files = sorted(glob.glob(A + "/*.f32"))
if not files:
    print("no logits in", A)
    sys.exit(2)
ok7 = ok8 = True
for f in files:
    g = os.path.join(B, os.path.basename(f))
    if not os.path.exists(g):
        print(os.path.basename(f), "missing in", B)
        sys.exit(2)
    a, b = [np.fromfile(x, "float32").reshape(-1, V) for x in (f, g)]
    if a.shape != b.shape:
        print(os.path.basename(f), "shapes", a.shape, b.shape)
        sys.exit(2)
    pa, pb = softmax(a[-1]), softmax(b[-1])
    ja, jb = np.argsort(-pa)[:5], np.argsort(-pb)[:5]
    same = bool((ja == jb).all())
    dp = float(np.abs(pa[ja] - pb[ja]).max())
    mx = float(np.abs(a - b).max())
    p7, p8 = same and dp <= P7_DP, mx <= P8_MAX
    ok7, ok8 = ok7 and p7, ok8 and p8
    print(os.path.basename(f), a.shape, "top5 same order", same, "max dp", dp, "| max abs", mx, "| P7", p7, "P8", p8)
print("P7", ok7, "P8", ok8)
sys.exit(0 if ok7 and ok8 else 1)
