# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Grades the predictions of docs/batches/0014.md (P1 to P15 and the controls) from the summaries that the jobs kept. Written before any result of the registered cells was
# read; it states each prediction as the registration does and prints what was seen beside it. The rules, with the numbers fixed in the registration:
#   e(run)  = NMSE of the run's logits against the float64 reference of the original (ref64.py): the rounding error of that run.
#   R1      a variant is invisible in a (runtime, dtype) iff its NMSE against the same runtime's original is at most 4 max(e(original), e(variant)).
#   G0      a runtime is comparable on a model iff e(original) <= 1e-7 and its top-1 agreement with the float64 reference is at least 0.99 (float32).
#   ratio   for the quantized forms: the NMSE of a variant against the quantized original divided by e(quantized original); "same" is <= 0.2, "differs" is >= 0.5.
#   margin  a generation that differs from the original's is benign iff the float64 top-1 minus top-2 logit at its first divergence is below 4 x the largest |logit
#           difference| of that variant against the runtime's own original.
# Usage: grade14.py DIR [OUT.tsv]     DIR holds the downloaded artifacts: xrt-<runtime>-<model>/ (summary-*.json, manifest.json, tokenizers.json, cache-<model>.json) and xrt-quant/
import glob, json, os, sys

MODELS = ("smol-instruct", "smol-base", "floatlm-99m", "trilm-99m")
EXACT = ("scale", "perm", "all", "canon", "heads", "units", "units_blk", "resid", "resid_blk")
OTHERS = EXACT[1:]
FLOAT_RUNTIMES = ("onnxruntime", "candle", "llamacpp", "ctranslate2")
rows = []


def say(pid, subject, ok, seen):
    rows.append((pid, subject, {True: "as predicted", False: "REFUTED", None: "not graded"}[ok], seen))


def f(x):
    return "n/a" if x is None else f"{x:.3g}"


def load(d):
    S, manifests, toks, caches = {}, {}, {}, {}
    for p in glob.glob(os.path.join(d, "xrt-*", "summary-*.json")):
        s = json.load(open(p))
        S[(s.get("model") or "?", s["runtime"], s["dtype"])] = s
    for p in glob.glob(os.path.join(d, "xrt-*", "manifest.json")):
        manifests[os.path.basename(os.path.dirname(p))] = json.load(open(p))
    for p in glob.glob(os.path.join(d, "xrt-llamacpp-*", "tokenizers.json")):
        toks[os.path.basename(os.path.dirname(p)).replace("xrt-llamacpp-", "")] = json.load(open(p))
    for p in glob.glob(os.path.join(d, "xrt-pytorch-*", "cache-*.json")):
        caches[os.path.basename(p)[len("cache-"):-len(".json")]] = json.load(open(p))
    return S, manifests, toks, caches


def var(S, m, rt, dt, v):
    return ((S.get((m, rt, dt)) or {}).get("variants") or {}).get(v)


def e64(o):
    return ((o or {}).get("vs_float64_original") or {}).get("nmse")


def same(o):
    return (o or {}).get("vs_same_runtime_original") or {}


def comparable(S, m, rt, dt="float32"):
    r = (var(S, m, rt, dt, "orig") or {}).get("vs_float64_original")
    return None if not r else bool(r["nmse"] <= 1e-7 and r["top1_agree"] >= 0.99)


def r1(S, m, rt, dt, v):
    """(ok, nmse, bound) of rule R1, or None if the numbers are missing."""
    eo, ev, d = e64(var(S, m, rt, dt, "orig")), e64(var(S, m, rt, dt, v)), same(var(S, m, rt, dt, v)).get("nmse")
    if eo is None or ev is None or d is None:
        return None
    return d <= 4 * max(eo, ev), d, 4 * max(eo, ev)


def check_r1(pid, S, m, rt, dt, subject, top1=None):
    bad, seen = [], []
    for v in OTHERS:
        r = r1(S, m, rt, dt, v)
        if r is None:
            continue
        o = same(var(S, m, rt, dt, v))
        ok = r[0] and (top1 is None or o.get("top1_agree", 0) >= top1)
        seen.append(f"{v} {f(r[1])}<={f(r[2])}")
        if not ok:
            bad.append(v)
    say(pid, subject, None if not seen else not bad, "NMSE against own original, bound 4 max e: " + ", ".join(seen) + (f"; outside: {bad}" if bad else ""))


def gens(o):
    g = (o or {}).get("generation") or {}
    return sum(x["identical"] for x in g.values()), sum(x["of"] for x in g.values())


def benign(s, v):
    """(unexplained divergences, number of divergent generations) of variant v by the margin rule."""
    o = s["variants"][v]
    tau = 4 * same(o).get("max_abs", 0.0)
    margins = s.get("margins") or {}
    bad, n = [], 0
    for kind, d in (o.get("generation") or {}).items():
        for i, k in enumerate(d["first_divergences"]):
            if k is None:
                continue
            n += 1
            ms = margins.get(kind) or []
            m = ms[i] if i < len(ms) else None
            if m is None or k >= len(m) or m[k] > tau:
                bad.append(f"{kind}[{i}]@{k}" + ("" if m is None or k >= len(m) else f" margin {m[k]:.3g}>{tau:.3g}"))
    return bad, n


def grade_text(pid, S, m, rt, dt, subject):
    s = S.get((m, rt, dt))
    if not s or "scale" not in s["variants"]:
        return say(pid, subject, None, "no summary")
    a, b = gens(s["variants"]["scale"])
    bad, seen = [], [f"scale {a}/{b}"]
    if a != b:
        bad.append("scale")
    for v in OTHERS:
        if v not in s["variants"] or not s["variants"][v].get("generation"):
            continue
        x, n = benign(s, v)
        seen.append(f"{v} {n} divergent")
        if x:
            bad.append(f"{v}: {x[:3]}")
    say(pid, subject, not bad, "; ".join(seen) + (f"; unexplained: {bad}" if bad else ""))


def grade_ratio(pid, S, m, rt, dt, subject, same_set, differs_set):
    o = var(S, m, rt, dt, "orig")
    e = e64(o)
    if e is None or e <= 0:
        return say(pid, subject, None, "no float64 error of the original")
    bad, seen = [], []
    for v in same_set + differs_set:
        x = var(S, m, rt, dt, v)
        d = same(x).get("nmse")
        if d is None:
            bad.append(v + " (missing)")
            continue
        q = d / e
        ok = q <= 0.2 if v in same_set else q >= 0.5
        seen.append(f"{v} {q:.3g}")
        if not ok:
            bad.append(v)
    say(pid, subject, not bad, f"e = {f(e)}; NMSE against own original / e: " + ", ".join(seen) + (f"; outside: {bad}" if bad else ""))


def grade_cache(C, m):
    classes = (("identity", ("resid", "resid_blk", "units", "units_blk")), ("unit", ("perm", "heads")), ("pow2", ("scale", "all", "canon")))
    for view, V in C["views"].items():
        vs = V["variants"]
        bad, seen = [], []
        for cls, names in classes:
            for v in names:
                if v not in vs:
                    continue
                s = vs[v]["summary"]
                ok = s["all_matched"] and s["all_bijection"] and {"identity": s["all_identity"] and s["all_unit"], "unit": s["all_unit"], "pow2": s["all_pow2"]}[cls]
                seen.append(f"{v}:{cls} resid {s['max_resid']:.1e}")
                if not ok:
                    bad.append(f"{v} ({cls}; layers not matched {s['layers_not_matched'][:4]})")
        b = vs.get("broken")
        if b is not None:
            say("P14c", f"{m}, {view}", not b["summary"]["all_matched"], f"broken: matched in all layers {b['summary']['all_matched']}, min cos {b['summary']['min_cos']:.3f}")
        say("P14", f"{m}, {view}", None if not seen else not bad, f"tolerance {V['tolerance']:g}: " + ", ".join(seen) + (f"; outside: {bad}" if bad else ""))


def main(d, tsv=None):
    S, manifests, toks, caches = load(d)

    # C0: the reference itself
    for m in MODELS:
        r = (S.get((m, "pytorch", "float32")) or {}).get("reference_float32_vs_float64")
        say("C0", m, None if not r else r["nmse"] <= 1e-9, "float64 reference against PyTorch float32: " + ("not available" if not r else f"NMSE {f(r['nmse'])}, top-1 {f(r['top1_agree'])}, max|d| {f(r['max_abs'])}"))

    # P1, P2, P3: PyTorch
    for m in MODELS:
        o = var(S, m, "pytorch", "float32", "scale")
        say("P1", m, None if o is None else bool(same(o).get("bit_identical")), "no PyTorch float32 summary" if o is None else f"scale against original: max|d| {f(same(o).get('max_abs'))}, bit identical {same(o).get('bit_identical')}")
        check_r1("P2", S, m, "pytorch", "float32", m, top1=0.998)
        o = var(S, m, "pytorch", "bfloat16", "scale")
        say("P3", m + " scale", None if o is None else bool(same(o).get("bit_identical")), "no bfloat16 summary" if o is None else f"bit identical {same(o).get('bit_identical')}, max|d| {f(same(o).get('max_abs'))}")
        check_r1("P3", S, m, "pytorch", "bfloat16", m + " others")

    # P4: comparable runtimes; P4b: the default cache of llama.cpp
    for m in MODELS:
        for rt in FLOAT_RUNTIMES:
            c = comparable(S, m, rt)
            r = (var(S, m, rt, "float32", "orig") or {}).get("vs_float64_original") or {}
            say("P4", f"{rt} / {m}", c, "no summary" if c is None else f"e(original) {f(r.get('nmse'))}, top-1 {f(r.get('top1_agree'))}, max|d| {f(r.get('max_abs'))}")
    a, b = e64(var(S, "smol-instruct", "llamacpp", "float32", "orig")), e64(var(S, "smol-instruct", "llamacpp", "float32-kvf16", "orig"))
    say("P4b", "llama.cpp float32, f16 against f32 cache", None if a is None or b is None else bool(b >= 10 * a), f"e with f32 cache {f(a)}, with f16 cache {f(b)}")

    # P5
    for m in MODELS:
        for rt in FLOAT_RUNTIMES:
            c = comparable(S, m, rt)
            if not c:
                if c is False:
                    say("P5", f"{rt} / {m}", None, "not comparable: variants shown in the summaries, no verdict")
                continue
            o = var(S, m, rt, "float32", "scale")
            say("P5a", f"{rt} / {m}", bool(same(o).get("bit_identical")), f"scale: bit identical {same(o).get('bit_identical')}, max|d| {f(same(o).get('max_abs'))}, NMSE {f(same(o).get('nmse'))}")
            check_r1("P5b", S, m, rt, "float32", f"{rt} / {m}")

    # P6, P7: the quantized forms
    for fmt in ("q8_0", "q4_0"):
        if ("smol-instruct", "llamacpp", fmt) in S:
            grade_ratio("P6", S, "smol-instruct", "llamacpp", fmt, f"llama.cpp {fmt}", ("heads", "units_blk", "resid_blk"), ("units", "resid", "scale", "perm", "all", "canon"))
        else:
            say("P6", fmt, None, "no summary")
    if ("smol-instruct", "ctranslate2", "int8") in S:
        grade_ratio("P7", S, "smol-instruct", "ctranslate2", "int8", "CTranslate2 int8", ("heads", "units", "resid", "perm"), ("scale", "all", "canon"))
    else:
        say("P7", "ctranslate2 int8", None, "no summary")

    # P8, P9: generations
    for m in MODELS:
        grade_text("P8", S, m, "pytorch", "float32", f"pytorch / {m}")
        for rt in FLOAT_RUNTIMES:
            if comparable(S, m, rt):
                grade_text("P9", S, m, rt, "float32", f"{rt} / {m}")

    # P10: the control
    for (m, rt, dt), s in sorted(S.items()):
        if m not in MODELS or dt not in ("float32", "bfloat16", "float16"):
            continue
        o, p0 = s["variants"].get("broken"), (s["variants"].get("orig") or {}).get("ppl")
        if o is None or not p0:
            continue
        a, b = gens(o)
        ok = (o.get("ppl") or 0) >= 100 * p0 and same(o).get("top1_agree", 1) <= 0.2 and same(o).get("nmse", 0) >= 0.01 and a <= 1
        say("P10", f"{rt} {dt} / {m}", ok, f"broken: perplexity {f(o.get('ppl'))} against {f(p0)}, top-1 agreement {f(same(o).get('top1_agree'))}, NMSE {f(same(o).get('nmse'))}, identical generations {a}/{b}")

    # P11: perplexity
    for m in MODELS:
        for rt in ("pytorch",) + FLOAT_RUNTIMES:
            if rt != "pytorch" and not comparable(S, m, rt):
                continue
            p0 = (var(S, m, rt, "float32", "orig") or {}).get("ppl")
            if not p0:
                continue
            seen = [(v, abs(var(S, m, rt, "float32", v)["ppl"] / p0 - 1)) for v in EXACT if (var(S, m, rt, "float32", v) or {}).get("ppl")]
            bad = [v for v, x in seen if x > 1e-5]
            say("P11", f"{rt} / {m}", not bad, f"perplexity {f(p0)}; relative change: " + ", ".join(f"{v} {x:.2g}" for v, x in seen) + (f"; outside: {bad}" if bad else ""))

    # P12: speed and size, against the original's own spread
    for (m, rt, dt), s in sorted(S.items()):
        if m not in MODELS or dt not in ("float32", "bfloat16", "float16"):
            continue
        o0 = s["variants"].get("orig") or {}
        stored = lambda v: ((manifests.get(f"xrt-{rt}-{m}") or {}).get("variants") or {}).get(v, {}).get("stored_dtype")
        bad, seen = [], []
        for v in EXACT:
            o = s["variants"].get(v)
            if o is None:
                continue
            for key in ("prefill_tps", "decode_tps"):
                allv, med = o0.get(key + "_all"), o.get(key)
                if allv and med:
                    lo, hi = min(allv) * 0.97, max(allv) * 1.03
                    if not lo <= med <= hi:
                        bad.append(f"{v} {key} {med:.3g} outside [{lo:.3g}, {hi:.3g}]")
            if o.get("peak_rss_mb") and o0.get("peak_rss_mb") and not 0.95 <= o["peak_rss_mb"] / o0["peak_rss_mb"] <= 1.05:
                bad.append(f"{v} peak memory {o['peak_rss_mb'] / o0['peak_rss_mb']:.2f}")
            if o.get("bytes") and o0.get("bytes") and o["bytes"] != o0["bytes"] and stored(v) == stored("orig"):
                bad.append(f"{v} sizes {o['bytes']} against {o0['bytes']}")
            seen.append(v)
        say("P12", f"{rt} {dt} / {m}", None if not seen else not bad, f"{len(seen)} variants inside the original's own spread (3% beyond it) in prefill and decode, memory within 5%, sizes equal" if not bad else "; ".join(bad[:4]))

    # P13: tokenizers
    for m in MODELS:
        t = toks.get(m)
        say("P13", m, None if t is None else t["identical"] == t["texts"], "no tokenizers.json" if t is None else f"{t['identical']} of {t['texts']} texts have the ids of the Hugging Face tokenizer" + ("" if t["identical"] == t["texts"] else f"; first divergences {t['first_divergence']}"))

    # P14: the cache cell
    for m in MODELS:
        if m in caches:
            grade_cache(caches[m], m)
        else:
            say("P14", m, None, "no cache cell")

    # P15: the type of the KV cache in llama.cpp
    for t in ("f16", "q8_0", "q4_0"):
        dt = f"float32-kv{t}"
        if ("smol-instruct", "llamacpp", dt) not in S:
            say("P15", f"cache {t}", None, "no summary")
        elif t == "f16":
            grade_ratio("P15", S, "smol-instruct", "llamacpp", dt, f"cache {t}", EXACT, ())
        else:
            grade_ratio("P15", S, "smol-instruct", "llamacpp", dt, f"cache {t}", ("heads", "perm", "resid", "resid_blk", "units", "units_blk"), ("scale", "all", "canon"))

    # addendum 1: the quantized cache with the rotation disabled (P15n), and what the rotation does (P15r)
    for t in ("q8_0", "q4_0"):
        dt = f"float32-kv{t}-norot"
        if ("smol-instruct", "llamacpp", dt) not in S:
            say("P15n", f"cache {t}, rotation disabled", None, "no summary")
        else:
            grade_ratio("P15n", S, "smol-instruct", "llamacpp", dt, f"cache {t}, rotation disabled", ("heads", "perm", "resid", "resid_blk", "units", "units_blk"), ("scale", "all", "canon"))
        a, b = e64(var(S, "smol-instruct", "llamacpp", f"float32-kv{t}", "orig")), e64(var(S, "smol-instruct", "llamacpp", dt, "orig"))
        lim = 0.8 if t == "q4_0" else 1.0
        say("P15r", f"cache {t}, rotated against not rotated", None if a is None or b is None else bool(a <= lim * b), f"e with the rotation {f(a)}, without {f(b)} (limit {lim} x)")

    # addendum 2: the control for the noise of quantized runs (P16): the null variant, one unit in the last place of float32 in every norm weight, against the original of the same cell
    fine, coarse = ("q8_0", "float32-kvf16", "float32-kvq8_0"), ("q4_0", "float32-kvq4_0")
    for dt in ("float32",) + fine + coarse:
        o, nv = var(S, "smol-instruct-null", "llamacpp", dt, "orig"), var(S, "smol-instruct-null", "llamacpp", dt, "nullall")
        e, d = e64(o), same(nv).get("nmse")
        if e is None or d is None:
            say("P16", f"null variant, {dt}", None, "no summary")
        elif dt == "float32":
            ev = e64(nv)
            say("P16", f"null variant, {dt}", bool(ev is not None and d <= 4 * max(e, ev)), f"NMSE against own original {f(d)}, bound 4 max e = {f(4 * max(e, ev)) if ev is not None else 'n/a'}")
        elif dt in fine:
            say("P16", f"null variant, {dt}", bool(d / e >= 0.1), f"e = {f(e)}; NMSE against own original / e = {f(d / e)} (at least 0.1 predicted)")
        else:
            say("P16", f"null variant, {dt}", bool(d / e < 0.2), f"e = {f(e)}; NMSE against own original / e = {f(d / e)} (below 0.2 predicted)")
    for dt in ("float32-kvq8_0-norot", "float32-kvq4_0-norot"):
        o, nv = var(S, "smol-instruct-null", "llamacpp", dt, "orig"), var(S, "smol-instruct-null", "llamacpp", dt, "nullall")
        e, d = e64(o), same(nv).get("nmse")
        say("observed", f"null variant, {dt}", None, "no summary" if e is None or d is None else f"e = {f(e)}; NMSE against own original / e = {f(d / e)}")

    # addendum 3: the null and the variants in one job on one machine (P17)
    same_w, diff_w = ("heads", "units_blk", "resid_blk"), ("units", "resid", "scale", "perm", "all", "canon")
    same_c, diff_c = ("heads", "perm", "resid", "resid_blk", "units", "units_blk"), ("scale", "all", "canon")
    for dt, label, same_v, diff_v in (("q8_0", "Q8_0 weights", same_w, diff_w), ("q4_0", "Q4_0 weights", same_w, diff_w), ("float32-kvf16", "f16 cache", EXACT, ()),
                                      ("float32-kvq8_0", "Q8_0 cache", same_c, diff_c), ("float32-kvq4_0", "Q4_0 cache", same_c, diff_c),
                                      ("float32-kvq8_0-norot", "Q8_0 cache, not rotated", same_c, diff_c), ("float32-kvq4_0-norot", "Q4_0 cache, not rotated", same_c, diff_c)):
        m = "smol-instruct-nullsame"
        null = same(var(S, m, "llamacpp", dt, "nullall")).get("nmse")
        nv = {v: same(var(S, m, "llamacpp", dt, v)).get("nmse") for v in same_v + diff_v}
        if null is None or any(x is None for x in nv.values()):
            say("P17", label, None, "no summary")
            continue
        a = sorted((nv[v] / null, v) for v in same_v)
        b = sorted((nv[v] / null, v) for v in diff_v)
        ok_a = a[-1][0] <= 4
        ok_b = (not b) or b[0][0] > a[-1][0]
        fmt = lambda xs: ", ".join(f"{v} {x:.3g}" for x, v in xs)
        say("P17", label, bool(ok_a and ok_b), f"null NMSE {f(null)}; claimed the same, in units of the null: {fmt(a)}" + (f"; claimed different: {fmt(b)}" if b else "") + ("" if ok_a else " [a variant claimed the same is above 4 x the null]") + ("" if ok_b else " [the classes overlap]"))

    # controls
    for (m, rt, dt), s in sorted(S.items()):
        dtm = s.get("determinism")
        if dtm is not None and m in MODELS:
            say("control", f"determinism {rt} {dt} / {m}", all(dtm.values()), f"the original run twice: {dtm}")
    for key, mf in sorted(manifests.items()):
        if key in ("xrt-null", "xrt-nullsame"):
            continue          # its variant nullall has other significands by design
        vs = mf["variants"]
        if any("tensors_with_other_significands" in x for x in vs.values()):
            bad = [v for v, x in vs.items() if x.get("tensors_with_other_significands") or not x.get("same_tensor_names", True)]
            say("control", f"{key} significands", not bad, "every tensor of every variant has the original's multiset of significands and the original's name" + (f"; outside: {bad}" if bad else ""))
            allowed = {"torch_dtype", "dtype", "pad_token_id", "bos_token_id", "eos_token_id"}
            bad = [v for v, x in vs.items() if not set(x.get("config_keys_changed", [])) <= allowed]
            say("control", f"{key} config", not bad, f"config keys changed by the writer: {sorted({k for x in vs.values() for k in x.get('config_keys_changed', [])})}")
            hs = [x.get("safetensors_sha256") for x in vs.values()]
            say("control", f"{key} files", len(set(hs)) == len(hs), f"{len(set(hs))} distinct safetensors hashes of {len(hs)} variants")
        for v in ("scale", "perm", "all", "canon"):
            x = vs.get(v)
            if x is not None and mf["stored_dtype_of_original"] == "bfloat16":
                say("control", f"{key} {v} stored dtype", x["stored_dtype"] == "bfloat16" and x["tensors_not_exact_in_original_dtype"] == 0, f"stored as {x['stored_dtype']}, {x['tensors_not_exact_in_original_dtype']} tensors outside the grid")
    for (m, rt, dt), s in sorted(S.items()):
        hs = [o.get("artifact_sha256") for o in s["variants"].values()]
        if m in MODELS and hs and all(hs):
            say("control", f"converted artifacts {rt} {dt} / {m}", len(set(hs)) == len(hs), f"{len(set(hs))} distinct hashes of {len(hs)} converted files (a stale conversion would repeat one)")

    order = lambda r: (r[0] if r[0] != "control" else "Z", r[1])
    print("| prediction | subject | result | seen |\n|---|---|---|---|")
    for pid, subject, res, seen in sorted(rows, key=order):
        print(f"| {pid} | {subject} | {res} | {seen} |")
    print(f"\n{len(rows)} lines; as predicted {sum(1 for r in rows if r[2] == 'as predicted')}, refuted {sum(1 for r in rows if r[2] == 'REFUTED')}, not graded {sum(1 for r in rows if r[2] == 'not graded')}")
    if tsv:
        with open(tsv, "w") as out:
            out.write("prediction\tsubject\tresult\tseen\n")
            for pid, subject, res, seen in sorted(rows, key=order):
                out.write(f"{pid}\t{subject}\t{res}\t{seen.replace(chr(9), ' ').replace(chr(10), ' ')}\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
