# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cell 03: the exact tables of the instance at lengths 1 to MAX: every sequence over the vocabulary, the next-token
distribution after it, the extrema of every tapped activation.

usage: tables.py MODEL_DIR OUT_DIR [--max-len 4] [--variant null|nan] [--batch 8192]
Per length T, in OUT_DIR: tables_T{T}.json (summary, taps extrema and slack), taps_T{T}.json (the vectors, for the
structural check), rows_T{T}.tsv for T <= 2 (sequence ids, argmax, p_max, entropy, sha256 of the row).
"""
import argparse
import hashlib
import itertools
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "0029"))
import common  # noqa: E402  (batch 0029: load_model, Taps, entropy_bits, dump)


def apply_variant(model, variant):
    if variant == "null":
        g = model.model.layers[-1].post_attention_layernorm.weight
        v = g[0].item(); g[0] = torch.nextafter(torch.tensor(v), torch.tensor(float("inf"))).item()
        return {"variant": "null", "before": v, "after": g[0].item()}
    if variant == "nan":
        model.model.layers[0].mlp.down_proj.weight[0, 0] = float("nan")
        return {"variant": "nan"}
    return {"variant": "none"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("out_dir"); ap.add_argument("--max-len", type=int, default=4)
    ap.add_argument("--variant", default="none"); ap.add_argument("--batch", type=int, default=8192)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    model = common.load_model(a.model_dir)
    vinfo = apply_variant(model, a.variant)
    V = model.config.vocab_size
    clock = common.Clock()
    for T in range(1, a.max_len + 1):
        n = V ** T
        taps = common.Taps(model)
        h_all = hashlib.sha256(); digests = np.empty((n, 16), dtype=np.uint8); am_count = np.zeros(V, dtype=np.int64)
        finite = True; ent_sum = 0.0; ent_min = np.inf; ent_max = -np.inf; pmax_sum = 0.0
        rows = open(os.path.join(a.out_dir, f"rows_T{T}.tsv"), "w") if T <= 2 else None
        if rows:
            rows.write("sequence\targmax\tp_max\tentropy_bits\tsha256\n")
        i0 = 0
        with torch.inference_mode():
            # sequences in lexicographic order, generated as base-V numbers so that no list of V^T tuples is held
            for start in range(0, n, a.batch):
                idx = torch.arange(start, min(n, start + a.batch), dtype=torch.long)
                seqs = torch.stack([(idx // (V ** (T - 1 - p))) % V for p in range(T)], dim=1)          # (B, T)
                logits = model(input_ids=seqs).logits[:, -1, :].to(torch.float32).numpy()
                if not np.isfinite(logits).all():
                    finite = False
                h_all.update(np.ascontiguousarray(logits).tobytes())
                hb, p = common.entropy_bits(logits)
                am = logits.argmax(axis=1); np.add.at(am_count, am, 1)
                ent_sum += float(hb.sum()); ent_min = min(ent_min, float(hb.min())); ent_max = max(ent_max, float(hb.max()))
                pm = p[np.arange(len(am)), am]; pmax_sum += float(pm.sum())
                for j in range(logits.shape[0]):
                    d = hashlib.sha256(np.ascontiguousarray(logits[j]).tobytes()).digest()
                    digests[i0 + j] = np.frombuffer(d[:16], dtype=np.uint8)
                    if rows:
                        rows.write(f"{','.join(map(str, seqs[j].tolist()))}\t{am[j]}\t{pm[j]:.6g}\t{hb[j]:.4f}\t{d.hex()}\n")
                i0 += logits.shape[0]
        if rows:
            rows.close()
        distinct = int(len(np.unique(digests, axis=0)))
        ts = taps.summary()
        summary = {"length": T, "sequences": n, "variant": vinfo, "finite": bool(finite), "distinct_rows": distinct,
                   "distinct_argmax": int((am_count > 0).sum()), "argmax_top": [(int(i), int(am_count[i])) for i in np.argsort(-am_count)[:8]],
                   "table_sha256": h_all.hexdigest(), "entropy_bits": {"min": ent_min, "mean": ent_sum / n, "max": ent_max},
                   "p_max_mean": pmax_sum / n, "taps": ts,
                   "resid_abs_max": max(v["abs_max"] for k, v in ts.items() if k.startswith("resid.")), "logit_abs_max": ts["logits"]["abs_max"],
                   "seconds": clock.s(), "peak_rss_mb": round(common.peak_rss_mb())}
        common.dump(os.path.join(a.out_dir, f"tables_T{T}.json"), summary)
        common.dump(os.path.join(a.out_dir, f"taps_T{T}.json"), taps.vectors())
        taps.remove()
        print(f"T={T}: {n} sequences, finite {finite}, distinct rows {distinct}, distinct argmax {summary['distinct_argmax']}, "
              f"resid |max| {summary['resid_abs_max']:.4g}, logits |max| {summary['logit_abs_max']:.4g}, table {summary['table_sha256'][:16]}, "
              f"{clock.s()} s, {summary['peak_rss_mb']} MB", flush=True)


if __name__ == "__main__":
    main()
