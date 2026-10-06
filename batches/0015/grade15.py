# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Grades the predictions Q1 to Q17 of docs/batches/0015.md from the files that the jobs kept. Written before any result of the real-checkpoint cells or of symdim.py was read; it states
# each prediction as the registration does and prints what was seen beside it.
# Usage: grade15.py EXACT_DIR LM_DIR [OUT.tsv]    EXACT_DIR: claims-{poly,oracle,float}.json, degrees-{poly,oracle}.txt, symdim.json; LM_DIR: conform.json, tokenperm.json, gauge.json, obfuscation.json, optim.json
import json, os, re, sys

rows = []
SYM = [("H1", "H1 signed permutation of the hidden coordinates"), ("A6", "A6 heads and key/value groups permuted"), ("M1", "M1 units permuted and scaled"),
       ("A1-1", "A1 value/output gauge, kappa 1"), ("A1-10", "A1 value/output gauge, kappa 10"), ("A1-100", "A1 value/output gauge, kappa 100"), ("A1-1000", "A1 value/output gauge, kappa 1000"),
       ("A3", "A3 query/key complex scalars"), ("N1", "N1 norm gauge"), ("H4", "H4 random orthogonal Q, norms folded, head untied"), ("C1", "C1 composition")]
CONTROL = "H8 control: the embedding centered (QuaRot, SpinQuant)"
EXPECTED_SYMDIM = {"untied-eps0-logits": 142, "untied-eps0-logprobs": 148, "untied-eps-logits": 141, "tied-eps0-logits": 121, "tied-eps-logits": 120, "mha-untied-eps0-logits": 82, "nope-untied-eps0-logits": 190}


def say(qid, subject, ok, seen):
    rows.append((qid, subject, {True: "as predicted", False: "REFUTED", None: "not graded"}[ok], seen))


def load(d, name):
    p = os.path.join(d, name)
    return json.load(open(p)) if os.path.exists(p) else None


def f(x):
    return "n/a" if x is None else f"{x:.3g}"


def main(exact, lm, tsv=None):
    # cell 00: the exact tier
    for mode, n in (("poly", 6), ("oracle", 4), ("float", 6)):
        j = load(exact, f"claims-{mode}.json")
        if j is None:
            say("Q1", mode, None, "no file")
            continue
        bad = [c["id"] for c in j["claims"] if not c["as_expected"]]
        say("Q1", f"{mode}, {len(j['claims'])} claims, {j['trials']} trials", len(j["claims"]) == 46 and not bad and j["trials"] == n, f"unexpected: {bad}")
    for mode, limit in (("oracle", 2.7e-31), ("poly", 7e-8)):
        p = os.path.join(exact, f"degrees-{mode}.txt")
        m = re.search(r"largest failure probability per trial over the claims \(\w+\): ([0-9.e+-]+)", open(p).read()) if os.path.exists(p) else None
        say("Q2", mode, None if m is None else float(m.group(1)) <= limit, "no file" if m is None else f"largest per-trial bound {m.group(1)} (limit {limit:g})")
    j = load(exact, "symdim.json")
    if j is None:
        say("Q3", "symdim", None, "no file")
    else:
        for r in j:
            want = EXPECTED_SYMDIM.get(r["config"])
            say("Q3", r["config"], r["deficiency"] == want, f"rank deficiency {r['deficiency']}, predicted {want} (formula {r['expected_deficiency']})")

    # cell 01: conformance
    j = load(lm, "conform.json")
    if j is None:
        say("Q4", "conform", None, "no file")
    else:
        for name, c in j["comparisons"].items():
            lim = 1e-10 if "float64" in name else 1e-4
            say("Q4", name, c["max_relative_to_scale"] <= lim and c["top1_agreement"] >= (1.0 if "float64" in name else 0.99), f"max relative {f(c['max_relative_to_scale'])} (limit {lim:g}), top-1 {f(c['top1_agreement'])}")

    # cell 02: token permutations
    j = load(lm, "tokenperm.json")
    if j is None:
        for q in ("Q5", "Q6", "Q7a", "Q7b"):
            say(q, "tokenperm", None, "no file")
    else:
        s = j["smollm2_causal_rope_float32"]
        say("Q5", "SmolLM2 causal, RoPE", s["argmax_agreement"] <= 0.5 and s["mean_kl"] >= 0.3, f"argmax agreement {f(s['argmax_agreement'])}, mean KL {f(s['mean_kl'])}")
        s = j["smollm2_last_position_under_permutation_of_earlier_tokens"]
        say("Q6", "SmolLM2 last position", s["median_logit_difference"] >= 0.1 and s["mean_kl"] >= 0.01, f"median logit difference {f(s['median_logit_difference'])}, mean KL {f(s['mean_kl'])}")
        s = j["bert_tiny_position_embeddings_zero_float64"]
        say("Q7a", "BERT-tiny, positions zero", s["max_relative_equivariance_error"] <= 1e-9, f"max relative equivariance error {f(s['max_relative_equivariance_error'])}")
        s = j["bert_tiny_as_shipped_float64"]
        say("Q7b", "BERT-tiny, as shipped", s["median_relative_equivariance_error"] >= 1e-3, f"median relative equivariance error {f(s['median_relative_equivariance_error'])}")

    # cell 03: the gauges
    g = (load(lm, "gauge.json") or {}).get("gauges")
    if not g:
        for q in ("Q8", "Q9", "Q10", "Q11", "Q12"):
            say(q, "gauge", None, "no file")
    else:
        bad = [k for k, name in SYM if g[name]["formal_float64_max_abs_difference"] > 1e-8]
        say("Q8", "formal float64, 11 gauges", not bad, ", ".join(f"{k} {f(g[n]['formal_float64_max_abs_difference'])}" for k, n in SYM) + (f"; outside: {bad}" if bad else ""))
        pred = [k for k, n in SYM if "cache_difference_from_the_predicted" in g[n]]
        bad = [k for k, n in SYM if "cache_difference_from_the_predicted" in g[n] and g[n]["cache_difference_from_the_predicted"] > 1e-9]
        moved = [(k, n) for k, n in SYM if k.startswith(("A6", "A1", "A3"))]
        bad2 = [k for k, n in moved if g[n]["cache_difference_from_the_original"] < 1e-2]
        say("Q9", "cache against the predicted, float64", not bad and not bad2, "predicted: " + ", ".join(f"{k} {f(g[n]['cache_difference_from_the_predicted'])}" for k, n in SYM if k in pred) + "; moved: " + ", ".join(f"{k} {f(g[n]['cache_difference_from_the_original'])}" for k, n in moved) + (f"; outside: {bad + bad2}" if bad + bad2 else ""))
        ok = True
        seen = []
        for k, n in SYM:
            r = g[n]["float32"]
            lim, gens = (1e-2, 0) if k == "C1" else (1e-3, 3)
            if k in ("H1", "A6", "M1", "N1", "H4", "C1"):
                good = r["max_abs_difference"] <= lim and r["top1_agreement"] >= 0.99 and r["generations_identical"] >= gens
                seen.append(f"{k} {f(r['max_abs_difference'])}/{r['generations_identical']}")
                ok = ok and good
        say("Q10", "float32 on the written files", ok, "max|d| / generations identical of 4: " + ", ".join(seen))
        e32 = {k: g[n]["float32"]["max_abs_difference"] for k, n in SYM if k.startswith("A1")}
        e16 = {k: g[n]["bfloat16"]["max_abs_difference"] for k, n in SYM if k.startswith("A1")}
        ok = e32["A1-1000"] >= 5 * e32["A1-1"] and e16["A1-100"] >= 3 * e16["A1-1"] and e16["A1-1000"] >= 10 * e16["A1-1"]
        say("Q11", "the cost of conditioning", ok, "float32 " + ", ".join(f"kappa {k.split('-')[1]} {f(v)}" for k, v in e32.items()) + "; bfloat16 " + ", ".join(f"kappa {k.split('-')[1]} {f(v)}" for k, v in e16.items()))
        c = g[CONTROL]
        say("Q12", "embedding centering, the control", c["formal_float64_max_abs_difference"] >= 1e-3, f"formal float64 max|d| {f(c['formal_float64_max_abs_difference'])}, float32 {f(c['float32']['max_abs_difference'])}, generations identical {c['float32']['generations_identical']}/{c['float32']['of']}")

    # cell 04: obfuscation
    o = load(lm, "obfuscation.json")
    if o is None:
        for q in ("Q13", "Q14", "Q15", "Q16"):
            say(q, "obfuscation", None, "no file")
    else:
        p0 = o["original"]["ppl"]
        b, k = o["body_only_permuted"], o["with_the_key"]
        say("Q13", "body permuted, and with the key", b["ppl"] >= 100 * p0 and abs(k["ppl"] / p0 - 1) <= 1e-4 and k["max_abs_logit_difference"] <= 1e-3,
            f"perplexity original {f(p0)}, body only {f(b['ppl'])}, with the key {f(k['ppl'])}; logit difference with the key {f(k['max_abs_logit_difference'])}")
        a = o.get("attack_with_public_base")
        say("Q14", "matching against the public base", None if a is None else a["coordinates_recovered"] >= 0.99 and abs(a["ppl"] / p0 - 1) <= 0.01,
            "no attack" if a is None else f"coordinates recovered {f(a['coordinates_recovered'])}, repaired perplexity {f(a['ppl'])} against {f(p0)}")
        t = (o.get("attack_by_training_adapters") or {}).get("perplexity_on_the_training_window_along_the_way")
        say("Q15", "the adapter attack", None if not t else t[0][1] >= 3 * min(x[1] for x in t), "no trace" if not t else f"training perplexity {f(t[0][1])} at step 0, minimum {f(min(x[1] for x in t))} in {t[-1][0] + 1} steps")
        c = o.get("cache_float64_relative_difference_from_the_original")
        say("Q16", "the cache of the keyed and of the body-only model", None if c is None else c["keyed"] <= 1e-9 and c["body_only"] >= 1e-2, "no file" if c is None else f"keyed {f(c['keyed'])}, body only {f(c['body_only'])}")

    # cell 05: dynamics
    j = load(lm, "optim.json")
    if j is None:
        say("Q17", "optimizers", None, "no file")
    else:
        R = j["results"]
        sp, ro, sc = "signed permutation of the hidden coordinates", "orthogonal rotation of the stream", "scaling of the feed-forward units by +-2^k"
        checks = [("sgd", sp, lambda r: r["max_loss_difference"] <= 1e-8 and r["relative_parameter_difference"] <= 1e-8), ("adam", sp, lambda r: r["max_loss_difference"] <= 1e-8 and r["relative_parameter_difference"] <= 1e-8),
                  ("sgd", ro, lambda r: r["max_loss_difference"] <= 1e-8 and r["relative_parameter_difference"] <= 1e-8), ("adam", ro, lambda r: r["max_loss_difference"] >= 1e-4 and r["relative_parameter_difference"] >= 1e-2),
                  ("sgd", sc, lambda r: r["max_loss_difference"] >= 1e-4), ("adam", sc, lambda r: r["max_loss_difference"] >= 1e-4)]
        bad, seen = [], []
        for opt, t, test in checks:
            r = R[opt][t]
            seen.append(f"{opt}/{t.split()[0]} loss {f(r['max_loss_difference'])} params {f(r['relative_parameter_difference'])}")
            if not test(r):
                bad.append(f"{opt}/{t.split()[0]}")
        say("Q17", "training dynamics under the symmetries", not bad, "; ".join(seen) + (f"; outside: {bad}" if bad else ""))

    print("| prediction | subject | result | seen |\n|---|---|---|---|")
    for q, s, r, w in rows:
        print(f"| {q} | {s} | {r} | {w} |")
    print(f"\n{len(rows)} lines; as predicted {sum(1 for r in rows if r[2] == 'as predicted')}, refuted {sum(1 for r in rows if r[2] == 'REFUTED')}, not graded {sum(1 for r in rows if r[2] == 'not graded')}")
    if tsv:
        with open(tsv, "w") as out:
            out.write("prediction\tsubject\tresult\tseen\n")
            for q, s, r, w in rows:
                out.write(f"{q}\t{s}\t{r}\t{w.replace(chr(9), ' ').replace(chr(10), ' ')}\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else None)
