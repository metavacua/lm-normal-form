# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The graded table of batch 0029, from the outputs of cells 01 and 02 and the controls, against docs/batches/0029.md.

usage: grade29.py OUT_ROOT   (expects OUT_ROOT/t1/, OUT_ROOT/t1_null/, OUT_ROOT/t1_nan/, OUT_ROOT/ibp/)
Writes OUT_ROOT/graded.md and OUT_ROOT/graded.tsv. Exit status 0 whatever the grades; 2 if an input is missing.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def rd(root, *parts):
    p = os.path.join(root, *parts)
    return common.load(p) if os.path.exists(p) else None


def main():
    root = sys.argv[1]
    t1 = rd(root, "t1", "t1_summary.json"); null = rd(root, "t1_null", "t1_summary.json")
    nan = rd(root, "t1_nan", "t1_summary.json"); ibp = rd(root, "ibp", "ibp_summary.json")
    rows = []

    def line(pid, verdict, measured, criterion):
        rows.append((pid, verdict, measured, criterion))

    if t1 is None or ibp is None:
        print("missing inputs:", "t1" if t1 is None else "", "ibp" if ibp is None else ""); sys.exit(2)
    # P1
    errs = {T: [k for k, v in d.items() if "error" in v] for T, d in ibp["lengths"].items()}
    ex_err = {T: v.get("error") for T, v in ibp["exactness"].items() if "error" in v}
    p1 = not any(errs.values()) and not ex_err and "import" not in ibp["errors"]
    line("P1", "as predicted" if p1 else "refuted", f"errors by length: {errs}; exactness errors: {ex_err}; import: {ibp['errors'].get('import')}",
         "IBP builds and bounds at lengths 1, 2, 8 without error")
    # exactness of the explicit forward (instrument)
    ex = {T: v.get("max_abs_diff_logits") for T, v in ibp["exactness"].items()}
    line("instrument: explicit forward = transformers", "holds" if all(v is not None and v < 1e-3 for v in ex.values()) else "FAILS",
         f"max |diff| by length: {ex}", "below 1e-3 at every length")
    # P2
    l1 = ibp["lengths"].get("1", {})
    sound = {k: v.get("sound_strict") for k, v in l1.items() if "sound_strict" in v}
    viol = {k: v.get("max_violation") for k, v in l1.items() if "max_violation" in v}
    p2 = bool(sound) and all(sound.values())
    line("P2", "as predicted" if p2 else ("refuted" if sound else "not graded"), f"strict containment: {sound}; max violation: {viol}",
         "exact length-1 extrema inside the IBP bounds at every tapped point")
    # P3
    p3 = t1["finite"] and t1["distinct_rows"] == t1["rows"] and t1["distinct_argmax"] >= 100
    line("P3", "as predicted" if p3 else "refuted",
         f"finite {t1['finite']}; distinct rows {t1['distinct_rows']}/{t1['rows']}; duplicates {t1.get('duplicate_groups')}; distinct argmax {t1['distinct_argmax']}",
         "finite, all rows distinct, at least 100 argmax values")
    # P4
    p4 = t1["resid_abs_max"] < 1e3 and t1["logit_abs_max"] < 1e2
    line("P4", "as predicted" if p4 else "refuted", f"resid |max| {t1['resid_abs_max']:.4g}; logit |max| {t1['logit_abs_max']:.4g}",
         "residual below 1e3, logits below 1e2, at length 1")
    # P5
    fin = {T: all(v.get("finite", False) for v in d.values()) for T, d in ibp["lengths"].items() if T in ("2", "8")}
    w8 = ibp["lengths"].get("8", {}).get("logits", {}).get("width_max")
    wx = l1.get("logits", {}).get("exact_width_max")
    factor = (w8 / wx) if (w8 and wx) else None
    p5 = all(fin.values()) and bool(fin) and factor is not None and factor >= 10
    line("P5", "as predicted" if p5 else "refuted", f"finite at 2 and 8: {fin}; logit width at 8 / exact width at 1: {factor}",
         "bounds finite at lengths 2 and 8; width factor at least 10")
    # control a
    if null:
        changed = None
        try:
            A = [l.split("\t")[1] for l in open(os.path.join(root, "t1", "t1_rows.tsv")).read().splitlines()[1:]]
            B = [l.split("\t")[1] for l in open(os.path.join(root, "t1_null", "t1_rows.tsv")).read().splitlines()[1:]]
            changed = sum(1 for x, y in zip(A, B) if x != y) / max(1, len(A))
        except Exception as e:
            changed = f"error {e!r}"
        ok = t1["matrix_sha256"] != null["matrix_sha256"] and isinstance(changed, float) and changed < 1e-3
        line("control a", "as predicted" if ok else "refuted", f"matrix hash changed {t1['matrix_sha256'] != null['matrix_sha256']}; argmax changed fraction {changed}",
             "hash changes; argmax changes in fewer than 0.1% of rows")
    else:
        line("control a", "not run", "", "")
    # control b
    if nan:
        line("control b", "as predicted" if not nan["finite"] else "refuted", f"finite reported: {nan['finite']}", "the NaN copy is reported non-finite")
    else:
        line("control b", "not run", "", "")
    # control c
    c = ibp.get("control_c", {})
    okc = c.get("ratio") is not None and c["ratio"] >= 16
    line("control c", "as predicted" if okc else ("refuted" if c.get("ratio") is not None else "not graded (no bound to compare)"), f"{c}",
         "resid.1 bound grows by at least 16 under 2^8 scaling of layer 0 down")
    # addendum 1: the structural bounds (cell 04)
    st = rd(root, "structural", "structural_summary.json")
    if st:
        bad = {k: v for k, v in st["checks"].items() if not v["contained"]}
        line("P6", "as predicted" if st["all_contained"] else "refuted",
             f"taps checked {len(st['checks'])}; not contained: {list(bad)[:8]}; max violation {max((v['max_violation'] for v in st['checks'].values()), default=None)}",
             "structural bounds contain the exact length-1 extrema at every tapped point")
        p7 = st["resid_after_last_block_bound_max"] < 1e6 and st["logit_bound_max"] < 1e4
        line("P7", "as predicted" if p7 else "refuted", f"resid bound after last block {st['resid_after_last_block_bound_max']:.4g}; logit bound {st['logit_bound_max']:.4g}",
             "residual bound below 1e6, logit bound below 1e4")
        line("P8", "as predicted" if st["score_bound_max"] < 1e3 else "refuted", f"score bound max {st['score_bound_max']:.4g}", "attention-score bound below 1e3 for every head")
        cd = st["control_d"]
        line("control d", "as predicted" if (cd["ratio_is_exact"] and cd["other_blocks_unchanged"]) else "refuted", f"{cd}",
             "block 1's FFN box scales by exactly 2^8; no other block changes")
    else:
        line("P6, P7, P8, control d", "not run", "", "")
    # write
    md = ["# Batch 0029, graded", "", "| line | verdict | measured | criterion |", "|---|---|---|---|"]
    md += [f"| {a} | {b} | {str(c).replace('|', '/')} | {d} |" for a, b, c, d in rows]
    md += ["", "## Observed", "",
           f"- length-1 table: {t1['rows']} rows, matrix sha256 `{t1['matrix_sha256']}`, entropy bits {t1['entropy_bits']}, p_max {t1['p_max']}, "
           f"argmax top {t1['argmax_top'][:5]}, {t1['seconds']} s, {t1['peak_rss_mb']} MB",
           "- taps at length 1 (abs max, slack in doublings): " + "; ".join(f"{k} {v['abs_max']:.4g}/{v['slack_doublings'] and round(v['slack_doublings'],1)}" for k, v in t1["taps"].items()),
           "- IBP by length: " + "; ".join(f"T={T}: " + ", ".join(f"{k} |max| {v.get('abs_max')}" for k, v in d.items()) for T, d in ibp["lengths"].items()),
           f"- embedding hull: {ibp['embedding_hull']}; IBP {ibp['seconds']} s, {ibp['peak_rss_mb']} MB; auto_LiRPA {ibp.get('auto_lirpa_version')}"]
    if st:
        md += ["- structural bounds (every length), per block: resid bound max / attn box max / ffn box max / score bound max: " +
               "; ".join(f"{k+1}: {b['resid_bound_max']:.4g}/{b['attn_box_max']:.4g}/{b['ffn_box_max']:.4g}/{b['score_bound_max']:.3g}" for k, b in enumerate(st["per_block"])),
               "- structural looseness (bound / exact |max|, min over coordinates) at resid.1, resid.12, last resid, gate.12, logits: " +
               ", ".join(f"{k}={st['checks'][k]['looseness_min']:.3g}" for k in ("resid.1", "resid.12", f"resid.{st['config']['num_hidden_layers']}", "gate.12", "logits") if k in st["checks"]),
               f"- structural slack in doublings: {st['slack_doublings']}; final-norm bound {st['final_norm_bound_max']:.4g}; {st['seconds']} s"]
    open(os.path.join(root, "graded.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    open(os.path.join(root, "graded.tsv"), "w", encoding="utf-8").write("\n".join("\t".join(map(str, r)) for r in rows) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
