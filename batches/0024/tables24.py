#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Markdown tables of batch 0024 from the summaries of the cells (the same files that grade24.py grades; the classes and the cells are those of grade24.py): per cell, the spread of the eight nulls and the
# position of each variant in units of the median and of the largest null. Usage: tables24.py RESULTS_DIR > tables.md
import json, os, statistics, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from grade24 import CELLS, NULLS  # noqa: E402

R = sys.argv[1]
summary, detail = [], []
for stem, label, same, diff in CELLS:
    path = os.path.join(R, stem + ".json")
    if not os.path.exists(path):
        summary.append(f"| {label} | not run | | | | | |")
        continue
    v = json.load(open(path))["variants"]
    nmse = {k: x["vs_same_runtime_original"]["nmse"] for k, x in v.items() if k != "orig"}
    nul = [nmse[k] for k in NULLS]
    med, lo, hi = statistics.median(nul), min(nul), max(nul)
    s_max = max((nmse[k] for k in same), default=None)
    d_min = min((nmse[k] for k in diff), default=None)
    summary.append(f"| {label} | {lo:.3g} to {hi:.3g} | {hi / lo:.2f} | {s_max / med:.2f} | " + ("" if d_min is None else f"{d_min / hi:.2f}") + f" | {nmse['broken'] / hi:.3g} | {len(NULLS)} |")
    rows = []
    for k in sorted((k for k in nmse if k not in NULLS), key=lambda k: nmse[k]):
        cls = "same" if k in same else "different" if k in diff else "control (broken)" if k == "broken" else "-"
        rows.append(f"| {k} | {cls} | {nmse[k]:.3g} | {nmse[k] / med:.2f} | {nmse[k] / hi:.2f} |")
    detail.append(f"#### {label}\n\nThe eight nulls: {lo:.3g} to {hi:.3g}, median {med:.3g}.\n\n| variant | class claimed | NMSE | in units of the median null | in units of the largest null |\n|---|---|---|---|---|\n" + "\n".join(rows) + "\n")
print("| cell | NMSE of the eight nulls | largest / smallest null (N1: at most 3) | largest variant claimed the same / median null (N2: at most 4) | smallest variant claimed different / largest null (N3: at least 1.5) | broken / largest null | nulls |")
print("|---|---|---|---|---|---|---|")
print("\n".join(summary))
print()
print("\n".join(detail))
