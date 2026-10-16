# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cell 01: the exact table at length 1. Every token id of the vocabulary as a one-token prompt, in id order.

usage: enumerate_t1.py MODEL_DIR OUT_DIR [--limit N] [--variant null|nan] [--batch B]

Writes in OUT_DIR:
  t1_rows.tsv      one line per id: id, argmax, p_max, entropy_bits, logit_min, logit_max, top16 ids, row sha256
  t1_summary.json  finiteness, distinct rows, distinct argmax, the matrix sha256, the taps' extrema and slack, time, memory
  t1_taps.json     the full coordinate-wise min and max vectors of every tap (for the soundness check of cell 02)
--variant null  flips one unit in the last place of one gain of the last block's post-attention norm (batch 0014's null variant)
--variant nan   sets one entry of layer 0's down-projection to NaN (control b)
"""
import argparse
import hashlib
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def apply_variant(model, variant):
    if variant == "null":
        g = model.model.layers[-1].post_attention_layernorm.weight
        v = g[0].item()
        g[0] = torch.nextafter(torch.tensor(v, dtype=torch.float32), torch.tensor(float("inf"), dtype=torch.float32)).item()
        return {"variant": "null", "tensor": f"model.layers.{len(model.model.layers)-1}.post_attention_layernorm.weight[0]",
                "before": v, "after": g[0].item()}
    if variant == "nan":
        w = model.model.layers[0].mlp.down_proj.weight
        v = w[0, 0].item()
        w[0, 0] = float("nan")
        return {"variant": "nan", "tensor": "model.layers.0.mlp.down_proj.weight[0,0]", "before": v, "after": "nan"}
    return {"variant": "none"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir"); ap.add_argument("out_dir")
    ap.add_argument("--limit", type=int, default=0); ap.add_argument("--variant", default="none"); ap.add_argument("--batch", type=int, default=512)
    a = ap.parse_args()
    clock = common.Clock()
    model = common.load_model(a.model_dir)
    facts = common.model_facts(model)
    vinfo = apply_variant(model, a.variant)
    V = facts["vocab_size"]
    n = a.limit if a.limit else V
    taps = common.Taps(model)
    os.makedirs(a.out_dir, exist_ok=True)
    matrix_h = hashlib.sha256()
    row_hashes = set()
    argmax_count = np.zeros(V, dtype=np.int64)
    finite = True
    ent_all, pmax_all = [], []
    with open(os.path.join(a.out_dir, "t1_rows.tsv"), "w", encoding="utf-8") as rows:
        rows.write("id\targmax\tp_max\tentropy_bits\tlogit_min\tlogit_max\ttop16\tsha256\n")
        with torch.inference_mode():
            for start in range(0, n, a.batch):
                ids = torch.arange(start, min(n, start + a.batch), dtype=torch.long).unsqueeze(1)   # (B, 1)
                out = model(input_ids=ids)
                logits = out.logits[:, -1, :].to(torch.float32).numpy()                             # (B, V)
                if not np.isfinite(logits).all():
                    finite = False
                matrix_h.update(np.ascontiguousarray(logits).tobytes())
                h, p = common.entropy_bits(logits)
                am = logits.argmax(axis=1)
                top16 = np.argsort(-logits, axis=1)[:, :16]
                for i in range(logits.shape[0]):
                    rh = common.row_sha256(logits[i])
                    row_hashes.add(rh)
                    argmax_count[am[i]] += 1
                    rows.write(f"{start+i}\t{am[i]}\t{p[i, am[i]]:.6g}\t{h[i]:.4f}\t{logits[i].min():.6g}\t{logits[i].max():.6g}\t"
                               f"{','.join(map(str, top16[i].tolist()))}\t{rh}\n")
                ent_all.append(h); pmax_all.append(p[np.arange(len(am)), am])
                if (start // a.batch) % 8 == 0:
                    print(f"{clock.s():7.1f}s rows {start+logits.shape[0]}/{n}  rss {common.peak_rss_mb():.0f} MB", flush=True)
    ent = np.concatenate(ent_all); pmax = np.concatenate(pmax_all)
    tap_summary = taps.summary()
    summary = {
        "model": facts, "variant": vinfo, "rows": int(n), "finite": bool(finite),
        "distinct_rows": int(len(row_hashes)), "distinct_argmax": int((argmax_count > 0).sum()),
        "argmax_top": [(int(i), int(argmax_count[i])) for i in np.argsort(-argmax_count)[:10]],
        "matrix_sha256": matrix_h.hexdigest(),
        "entropy_bits": {"min": float(ent.min()), "median": float(np.median(ent)), "max": float(ent.max())},
        "p_max": {"min": float(pmax.min()), "median": float(np.median(pmax)), "max": float(pmax.max())},
        "taps": tap_summary,
        "resid_abs_max": max(v["abs_max"] for k, v in tap_summary.items() if k.startswith("resid.")),
        "logit_abs_max": tap_summary["logits"]["abs_max"],
        "seconds": clock.s(), "peak_rss_mb": round(common.peak_rss_mb()),
        "threads": common.THREADS, "torch": torch.__version__,
    }
    common.dump(os.path.join(a.out_dir, "t1_summary.json"), summary)
    common.dump(os.path.join(a.out_dir, "t1_taps.json"), taps.vectors())
    taps.remove()
    print(f"done: rows {n} finite {finite} distinct_rows {summary['distinct_rows']} distinct_argmax {summary['distinct_argmax']} "
          f"resid_abs_max {summary['resid_abs_max']:.4g} logit_abs_max {summary['logit_abs_max']:.4g} matrix {summary['matrix_sha256'][:16]} "
          f"{summary['seconds']}s {summary['peak_rss_mb']} MB")


if __name__ == "__main__":
    main()
