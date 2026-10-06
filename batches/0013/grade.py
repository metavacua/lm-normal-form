# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Grades the predictions of docs/batches/0013.md (P1 to P11 and the controls) from the JSON that the cells kept. Written after the
# registration and before any result was read; it states each prediction as the registration does and prints what was seen beside it.
# Usage: grade.py DIR [RUN1DIR|-] [OUT.tsv]   DIR holds one directory per model tag (smol-instruct, smol-base, floatlm-99m, trilm-99m, trilm-390m),
#                          each with check.json, canon.json, naive.json (and function.json for smol-instruct) and outcomes.json;
#                          RUN1DIR, the same for the first run, is for the predictions that compare the two runs (P17).
# P12 to P18 were registered after the first run and before the second (docs/batches/0013.md, "After the first run"); on the data of
# the first run their rows read "not graded: the cell did not record it"
import json, os, sys

FLOAT, TERNARY, ALL = ("smol-instruct", "smol-base", "floatlm-99m"), ("trilm-99m", "trilm-390m"), ("smol-instruct", "smol-base", "floatlm-99m", "trilm-99m", "trilm-390m")
RECORD = {"bag_root": "a946aed21e6ec2017c515d587d5ad72d654eb7c8f92fd9e04c4ca99898d9d3e2",
          "labelled_root": "476ff7a1fd9097fb56d98c60df3c25e52cf08f54bb02a7ec3f523a049754fca9"}
rows = []


def load(d, tag, name):
    p = os.path.join(d, tag, name)
    return json.load(open(p)) if os.path.exists(p) else None


def say(pid, tag, ok, seen):
    """ok is True, False, or None for not graded (the cell did not keep its result)."""
    rows.append((pid, tag, {True: "as predicted", False: "REFUTED", None: "not graded"}[ok], seen))


def main(d, d1=None, tsv=None):
    data = {t: {n: load(d, t, n + ".json") for n in ("check", "canon", "naive", "function", "outcomes")} for t in ALL}
    for t in ALL:
        c = data[t]["check"]
        # P1 and P2: ties of unlike items in one matrix
        if c is None:
            for pid in ("P1" if t in FLOAT else "P2", "P3", "P4", "P5"):
                say(pid, t, None, "check.json missing")
            continue
        ties = {(cls, ax): c["l0"][cls][ax]["matrices_where_signature_ties_unlike_items"] for cls in c["l0"] for ax in ("rows", "columns")}
        lost = {(cls, ax): c["l0"][cls][ax]["unlike_items_lost_to_signature_ties"] for cls in c["l0"] for ax in ("rows", "columns")}
        nz = {f"{cls} {ax}": f"{n} matrices, {lost[(cls, ax)]} items" for (cls, ax), n in ties.items() if n}
        if t in FLOAT:
            say("P1", t, not nz, "no tie of unlike items in any class" if not nz else f"ties: {nz}")
        else:
            say("P2", t, bool(nz), f"ties: {nz}" if nz else "no tie of unlike items in any class")
        s = c["l1_summary"]
        say("P3", t, s["coordinates_anchored"], f"embedding columns distinct up to sign: {s['coordinates_anchored']}")
        say("P4", t, s["units_not_told_apart_by_joint_signature"] == 0,
            f"units not told apart by the joint signature: {s['units_not_told_apart_by_joint_signature']}; layers discrete by separate signatures {s['layers_units_discrete_by_separate_signatures']} / joint {s['layers_units_discrete_by_joint_signature']} / anchored {s['layers_units_discrete_anchored']} of {s['layers']}")
        say("P5", t, s["heads_discrete_anchored_layers"] == s["layers"] and s["heads_discrete_index_free_layers"] == s["layers"],
            f"layers with heads and groups discrete anchored {s['heads_discrete_anchored_layers']}, heads discrete index-free {s['heads_discrete_index_free_layers']}, of {s['layers']}")
    ok6 = {t: all(r[2] == "as predicted" for r in rows if r[1] == t and r[0] in ("P3", "P4", "P5")) for t in ALL}
    for t in ALL:
        got = [r for r in rows if r[1] == t and r[0] in ("P3", "P4", "P5")]
        say("P6", t, None if len(got) < 3 or any(r[2] == "not graded" for r in got) else ok6[t], "P3, P4 and P5 hold" if ok6[t] else "one of P3, P4, P5 does not")

    # P7: SmolLM2-135M-Instruct, the canonical form against the group
    k = data["smol-instruct"]["canon"]
    if k is None:
        say("P7", "smol-instruct", None, "canon.json missing")
    else:
        k = k["canon"]
        n = k["tensors"]
        whole = [x["canonical_tensors_equal"] for x in k["all_parts"]]
        raw = [x["raw_tensors_changed"] for x in k["all_parts"]]
        parts = {p: ([x["canonical_tensors_equal"] for x in v], [x["raw_tensors_changed"] for x in v]) for p, v in k["single_parts"].items()}
        want_raw = {"residual": (272, 272), "units": (90, 90), "units-signed": (60, 60), "vo": (60, 60), "qk": (60, 60), "heads": (90, 120)}
        ok = (n == 272 and len(whole) == 8 and all(e == 272 for e in whole) and all(r == 272 for r in raw) and k["canonical_form_idempotent"]
              and k["canonical_roots"] == k["all_parts"][0]["canonical_roots_of_the_image"] and k["raw_roots"] != k["all_parts"][0]["raw_roots_of_the_image"]
              and all(all(e == 272 for e in parts[p][0]) and len(parts[p][0]) == 2 and all(want_raw[p][0] <= r <= want_raw[p][1] for r in parts[p][1]) for p in want_raw))
        say("P7", "smol-instruct", ok, f"tensors {n}; whole group canonical equal {whole}, raw changed {raw}; idempotent {k['canonical_form_idempotent']}; canonical roots equal "
            f"{k['canonical_roots'] == k['all_parts'][0]['canonical_roots_of_the_image']}; raw roots differ {k['raw_roots'] != k['all_parts'][0]['raw_roots_of_the_image']}; "
            + "; ".join(f"{p}: equal {parts[p][0]}, raw {parts[p][1]}" for p in want_raw))
        # controls: the raw roots against batch 0011's record
        say("control", "smol-instruct", k["raw_roots"] == RECORD, f"raw roots {k['raw_roots']}")
        # P9
        neg = k["negative_controls"]
        g = neg["one gate row negated"]
        e = neg["one unit's gate and up rows exchanged"]
        u = neg["one float32 ulp in one query weight"]
        c2 = neg["two embedding columns exchanged, nothing else"]
        L3 = sorted(f"model.layers.3.mlp.{m}_proj.weight" for m in ("gate", "up", "down"))
        ok9 = (g["differing_count"] == 3 and sorted(g["differing"]) == L3 and e["differing_count"] == 3 and sorted(e["differing"]) == L3
               and 1 <= u["differing_count"] <= 4 and u["layers_touched"] == [5] and "model.layers.5.self_attn.q_proj.weight" in u["differing"]
               and all(".self_attn." in x for x in u["differing"])
               and c2["differing_count"] >= 200 and "model.embed_tokens.weight" not in c2["differing"] + [] and c2["canonical_tensors_equal"] <= 272 - 200)
        say("P9", "smol-instruct", ok9, f"gate row negated: {g['differing_count']} {g['differing']}; gate/up exchanged: {e['differing_count']} {e['differing']}; ulp: {u['differing_count']} {u['differing']} layers {u['layers_touched']}; "
            f"embedding columns exchanged: {c2['differing_count']} differ, {c2['canonical_tensors_equal']} equal, listing {c2['differing']}")
    # P8
    for t in ("smol-base", "floatlm-99m", "trilm-99m", "trilm-390m"):
        k = data[t]["canon"]
        if k is None:
            say("P8", t, None, "canon.json missing")
            continue
        k = k["canon"]
        n = k["tensors"]
        whole = [x["canonical_tensors_equal"] for x in k["all_parts"]]
        parts = {p: [x["canonical_tensors_equal"] for x in v] for p, v in k["single_parts"].items()}
        ok = (all(e == n for e in whole) and all(all(e == n for e in v) for v in parts.values()) and k["canonical_form_idempotent"]
              and k["canonical_roots"] == k["all_parts"][0]["canonical_roots_of_the_image"])
        say("P8", t, ok, f"tensors {n}; whole group {whole}; parts {parts}; idempotent {k['canonical_form_idempotent']}; canonical roots equal {k['canonical_roots'] == k['all_parts'][0]['canonical_roots_of_the_image']}")
    # P10
    k = data["smol-instruct"]["naive"]
    if k is None:
        say("P10", "smol-instruct", None, "naive.json missing")
    else:
        k = k["naive"]
        mats = [m for m in k if m.startswith("model.layers.0.")]
        small = [m for m in k if m.startswith("12x20")]
        ok = (len(mats) == 3 and all(k[m]["distinct_outputs_sort_rows_then_columns"] >= 10 and k[m]["distinct_outputs_signature_then_columns"] == 1 for m in mats)
              and len(small) == 1 and k[small[0]]["distinct_outputs_signature_then_columns"] > 1)
        say("P10", "smol-instruct", ok, "; ".join(f"{m.split('layers.0.')[-1]}: naive {k[m]['distinct_outputs_sort_rows_then_columns']}, signature {k[m]['distinct_outputs_signature_then_columns']} of {k[m]['images']}" for m in mats + small))
    for t in ALL[1:]:
        k = data[t]["naive"]
        if k:
            k = k["naive"]
            say("observed", t, True, "naive/signature distinct outputs: " + "; ".join(f"{m.split('layers.0.')[-1]}: {v['distinct_outputs_sort_rows_then_columns']}/{v['distinct_outputs_signature_then_columns']}" for m, v in k.items() if m.startswith("model.layers.0.")))
    # P11
    f = data["smol-instruct"]["function"]
    if f is None:
        say("P11", "smol-instruct", None, "function.json missing")
    else:
        v = f["variants"]
        exact = ("units-signed", "vo", "qk")
        moved = ("units", "heads", "residual", "all")
        ok = (v["identity"]["max_abs"] == 0 and all(v[p]["max_abs"] == 0 and v[p]["mean_kl"] == 0 and v[p]["top1"] == 1.0 for p in exact)
              and all(1e-7 <= v[p]["max_abs"] <= 1e-3 and v[p]["top1"] >= 0.998 for p in moved))
        say("P11", "smol-instruct", ok, "; ".join(f"{p}: max|d| {v[p]['max_abs']:.3g}, KL {v[p]['mean_kl']:.3g}, top1 {v[p]['top1']:.4f}, top5 {v[p]['top5_prompts']}" for p in ("identity",) + exact + moved))
    # P12 to P17: after the first run; skipped on data in which no cell recorded what they need (the first run's)
    second = any("zero_census" in ((data[t]["canon"] or {}).get("canon") or {}) for t in ALL)
    for t in (ALL if second else ()):
        k = data[t]["canon"]
        c = data[t]["check"]
        if k is None or "raised" in k or "zero_census" not in k.get("canon", {}):
            for pid in ("P12", "P13", "P14", "P18"):
                say(pid, t, None, "raised: " + str(k["raised"]) if k and "raised" in k else "the cell did not record it (first run)" if k else "canon.json missing")
            continue
        k = k["canon"]
        n = k["tensors"]
        whole = [x["canonical_tensors_equal"] for x in k["all_parts"]]
        parts = {p: [x["canonical_tensors_equal"] for x in v] for p, v in k["single_parts"].items()}
        ok12 = (all(e == n for e in whole) and all(all(e == n for e in v) for v in parts.values()) and "zsigns" in parts and k["canonical_form_idempotent"]
                and k["canonical_roots"] == k["all_parts"][0]["canonical_roots_of_the_image"] and not (c and "raised" in c["l1_summary"]))
        say("P12", t, ok12, f"tensors {n}; whole group and the sign of zeros {whole}; parts {parts}; idempotent {k['canonical_form_idempotent']}; roots equal "
            f"{k['canonical_roots'] == k['all_parts'][0]['canonical_roots_of_the_image']}; nothing raised")
        tot = c["dead"]["totals"]
        zero_other = all(v == 0 for part in ("units", "vo") for v in tot[part].values())
        if t == "trilm-390m":
            ok13 = (tot["qk"]["both_zero"] == 173 and tot["qk"]["key_planes_zero"] == 173 and tot["qk"]["query_planes_zero"] == 173 and tot["qk"]["mixed_key_zero_query_not"] == 0
                    and tot["layers_with_dead_pairs"] == [0] and zero_other and tot["mixed_cases"] == 0 and c["dead"]["embedding"]["zero_columns"] == 0)
        else:
            ok13 = zero_other and all(v == 0 for v in tot["qk"].values()) and tot["mixed_cases"] == 0 and c["dead"]["embedding"]["zero_columns"] == 0
        say("P13", t, ok13, f"dead pairs: units {tot['units']['both_zero']}, vo {tot['vo']['both_zero']}, qk {tot['qk']['both_zero']} (layers {tot['layers_with_dead_pairs']}); mixed {tot['mixed_cases']}; "
            f"zero embedding columns {c['dead']['embedding']['zero_columns']}")
        zc = k["zero_census"]
        zs = [x["raw_tensors_changed"] for x in k["single_parts"]["zsigns"]]
        lo, hi = zc["tensors_with_16_or_more_zeros"], zc["tensors_with_16_or_more_zeros"] + zc["tensors_with_1_to_15_zeros"]
        say("P14", t, all(lo <= r <= hi for r in zs) and all(e == n for e in parts["zsigns"]), f"raw tensors changed by the sign of zeros {zs}, tensors with 16 or more zeros {lo}, with 1 to 15 {zc['tensors_with_1_to_15_zeros']}, "
            f"with none {zc['tensors_with_no_zeros']}; canonical equal {parts['zsigns']}")
        e = c["l0"]["embed"]["rows"]["unlike_items_lost_to_signature_ties"]
        anywhere = any(c["l0"][cls][ax]["matrices_where_signature_ties_unlike_items"] for cls in c["l0"] for ax in ("rows", "columns"))
        say("P18", t, e == 0 and anywhere == (t in TERNARY), f"unlike embedding rows lost to signature ties: {e}; a tie of unlike items anywhere: {anywhere} (predicted {t in TERNARY})")
        if d1:
            k1 = load(d1, t, "canon.json")
            if k1 and "canon" in k1 and "canonical_roots" in k1["canon"]:
                same = k1["canon"]["canonical_roots"] == k["canonical_roots"]
                no_zeros = t in ("smol-instruct", "smol-base")
                say("P17", t, same == no_zeros, f"canonical roots equal the first run's: {same} (predicted {no_zeros}); first run {k1['canon']['canonical_roots']['bag_root'][:16]}…, now {k['canonical_roots']['bag_root'][:16]}…")
    f = data["smol-instruct"]["function"]
    if not second:
        pass
    elif f and "canonical" in f["variants"]:
        v = f["variants"]["canonical"]
        say("P16", "smol-instruct", 1e-7 <= v["max_abs"] <= 1e-3 and v["top1"] >= 0.998 and v["mean_kl"] < 1e-9, f"canonical form as a model: max|d| {v['max_abs']:.3g}, KL {v['mean_kl']:.3g}, top1 {v['top1']:.4f}, top5 {v['top5_prompts']}")
    else:
        say("P16", "smol-instruct", None, "the cell did not record it (first run)")
    # controls: the self-test and every kept step
    for t in ALL:
        st = os.path.join(d, t, "selftest.txt")
        txt = open(st).read() if os.path.exists(st) else None
        say("control", t, None if txt is None else "selftest: all ok" in txt, "self-test: " + ("missing" if txt is None else txt.strip().splitlines()[-1]))
        o = data[t]["outcomes"]
        if o:
            say("control", t, all(x.get("outcome", x.get("conclusion")) in ("success", "skipped") for x in o.values() if isinstance(x, dict) and ("outcome" in x or "conclusion" in x)),
                "steps: " + ", ".join(f"{name} {x.get('outcome', x.get('conclusion'))}" for name, x in o.items() if isinstance(x, dict)))
        else:
            say("control", t, None, "outcomes.json missing")
    print("| prediction | model | result | seen |\n|---|---|---|---|")
    for pid, tag, res, seen in sorted(rows, key=lambda r: (r[0] if r[0] != "control" and r[0] != "observed" else "Z" + r[0], ALL.index(r[1]))):
        print(f"| {pid} | {tag} | {res} | {seen} |")
    if tsv:
        with open(tsv, "w") as out:
            out.write("prediction\tmodel\tresult\tseen\n")
            for pid, tag, res, seen in sorted(rows, key=lambda r: (r[0] if r[0] != "control" and r[0] != "observed" else "Z" + r[0], ALL.index(r[1]))):
                out.write(f"{pid}\t{tag}\t{res}\t{seen.replace(chr(9), ' ').replace(chr(10), ' ')}\n")
    refuted = [r for r in rows if r[2] == "REFUTED"]
    print(f"\n{len(rows)} lines; as predicted {sum(1 for r in rows if r[2] == 'as predicted')}, refuted {len(refuted)}, not graded {sum(1 for r in rows if r[2] == 'not graded')}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] != "-" else None, sys.argv[3] if len(sys.argv) > 3 else None)
