# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The quantities of docs/batches/0014.md that were observed and not predicted ("Observed, no prediction"), from the summaries that the jobs kept: e of every (model, runtime, dtype),
# the cache-type errors, the quantization errors, the perplexities of the originals, the throughput of the originals.   Usage: observed14.py DIR
import glob, json, os, sys
d = sys.argv[1]
S = {}
for p in glob.glob(os.path.join(d, "xrt-*", "summary-*.json")):
    s = json.load(open(p))
    S[(s.get("model") or "quant", s["runtime"], s["dtype"])] = s


def row(m, rt, dt):
    s = S[(m, rt, dt)]
    o = s["variants"]["orig"]
    e = (o.get("vs_float64_original") or {}).get("nmse")
    ppl = o.get("ppl")
    ppl = ppl.get("ppl") if isinstance(ppl, dict) else ppl
    return e, ppl, o.get("prefill_tps"), o.get("decode_tps"), o.get("peak_rss_mb")


def g(x, spec):
    return "n/a" if x is None else format(x, spec)


print("| model | runtime | dtype | e (NMSE against the float64 reference) | perplexity | prefill tok/s | decode tok/s | peak RSS MB |")
print("|---|---|---|---|---|---|---|---|")
for (m, rt, dt) in sorted(S, key=lambda k: (k[0], k[1], k[2])):
    e, ppl, pf, dc, rss = row(m, rt, dt)
    print(f"| {m} | {rt} | {dt} | {g(e, '.3g')} | {g(ppl, '.4g')} | {g(pf, '.0f')} | {g(dc, '.1f')} | {g(rss, '.0f')} |")
