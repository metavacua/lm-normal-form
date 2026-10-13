# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The verdicts of docs/batches/0028.md: each prediction P1 to P20 and each control C1 to C5, graded on the observations that
combine.py wrote. A prediction is `supported`, `refuted`, `not checked` (an observation it needs is missing, or a control
it depends on failed) or `not applicable` (it concerns a model that this run did not include). The exit status is 0 only if every
applicable prediction is supported and every control holds; a prediction that is not checked fails the run, as the registration says.

Usage: grade28.py OBSERVATIONS.json OUT_PREFIX      writes OUT_PREFIX.json and OUT_PREFIX.md
       grade28.py selftest
"""

import copy
import json
import sys

SMOL = ("base", "instruct")
INSTRUCT_ONNX_SHA = "7583ca8fdc74af2ca98fb5cc254ea305f840ec3a2a1998c4be1f7c9a35a9df47"
EIGHT = ["forward", "ids_alloc", "memory", "model_load", "out_ptr", "panic_len", "panic_ptr", "weights_alloc"]
SEVERITY = {"P1": "low", "P2": "low", "P3": "moderate", "P4": "high", "P5": "moderate", "P6": "moderate", "P7": "high",
            "P8": "low", "P9": "moderate", "P10": "moderate", "P11": "moderate", "P12": "moderate", "P13": "low", "P14": "high",
            "P15": "low", "P16": "moderate", "P17": "moderate", "P18": "moderate", "P19": "low", "P20": "low"}


class NotChecked(Exception):
    pass


def need(x, what):
    if x is None:
        raise NotChecked(what)
    return x


def get(o, *path):
    for p in path:
        if o is None:
            return None
        o = o.get(p) if isinstance(o, dict) else None
    return o


def both(obs, f):
    """f(model) for the base and the Instruct model; the first failure is the verdict."""
    msgs = []
    for m in SMOL:
        ok, msg = f(m)
        if not ok:
            return False, f"{m}: {msg}"
        msgs.append(f"{m}: {msg}")
    return True, "; ".join(msgs)


def each_tag(obs, f):
    msgs = []
    for t in obs["tags"]:
        ok, msg = f(t)
        if not ok:
            return False, f"{t}: {msg}"
        msgs.append(f"{t}: {msg}")
    return True, "; ".join(msgs)


# ---- the predictions -----------------------------------------------------------------------------------------------

def p1(obs):
    def f(m):
        h = need(get(obs, m, "st_header"), "safetensors header")
        ok = (h["tensors"] == 272 and h["dtypes"] == {"BF16": 272} and h["elements"] == 134515008 and not h["has_lm_head"]
              and h["metadata"] == {"format": "pt"} and h["offsets_contiguous"] and h["json_ends_at_brace"] and not h["duplicate_names"])
        return ok, f"{h['tensors']} tensors {h['dtypes']}, {h['elements']} elements, lm_head {h['has_lm_head']}, metadata {h['metadata']}"
    return both(obs, f)


def p2(obs):
    def f(m):
        c = need(get(obs, m, "onnx_record_census"), "ONNX census")
        ok = c["initializers"] == 273 and c["initializer_elements"] == 162826560 and c["nodes"] == 7360
        sha = get(obs, m, "onnx_sha256")
        if m == "instruct":
            ok = ok and sha == INSTRUCT_ONNX_SHA
        return ok, f"{c['initializers']} initializers, {c['initializer_elements']} elements, {c['nodes']} nodes, sha256 {str(sha)[:8]}"
    return both(obs, f)


def p3(obs):
    def f(m):
        s = need(get(obs, m, "st_onnx"), "match of safetensors and ONNX")
        h = get(obs, "queries", "st_counterparts", f"{m}/onnx")
        ok = (s["ref_distinct_keys"] == 272 and s["target_distinct_keys"] == 272 and s["keys_only_ref"] == 0 and s["keys_only_target"] == 0
              and s["relations"] == {"transposed": 211, "identical": 62}
              and s["ref_tensors_matched_by_more_than_one"] == {"model.embed_tokens.weight": 2})
        if h is not None:
            ok = ok and h == {"1": 271, "2": 1}
        if get(obs, m, "matcher_control") is not None and not get(obs, m, "matcher_control", "holds"):
            raise NotChecked("C1 for identity.match")
        return ok, f"relations {s['relations']}, matched twice {s['ref_tensors_matched_by_more_than_one']}, SPARQL histogram {h}"
    return both(obs, f)


def p4(obs):
    def f(m):
        g = need(get(obs, m, "gguf"), "GGUF maps")
        for c in need(get(obs, m, "gguf_controls"), "GGUF controls").values():
            if not c:
                raise NotChecked("C1 or C2")
        for k in ("f32", "bf16"):
            if need(get(obs, m, "gguf_independent", k), "C5")["mismatch_count"]:
                raise NotChecked("C5")
        h = get(obs, "queries", "st_counterparts", f"{m}/gguf-f32")
        ok = (g["tensors"] == 272 and g["relations"] == {"identical": 212, "permuted": 60} and g["permuted_by_n"] == {"9": 30, "3": 30}
              and g["permuted_roles"] == {"q_proj": 30, "k_proj": 30} and g["name_pairing_agrees"] == 272)
        if h is not None:
            ok = ok and h == {"1": 272}
        return ok, f"{g['tensors']} tensors, relations {g['relations']}, permuted by n {g['permuted_by_n']}, SPARQL {h}"
    return both(obs, f)


def p5(obs):
    def f(m):
        r = need(get(obs, m, "gguf_bf16_vs_f32"), "bf16 against f32")
        c = need(get(obs, m, "gguf_census"), "GGUF census")
        ok = (r["bag_roots_equal"] and r["keys_in_both"] == 272 and c["bf16"]["dtypes"] == {"BF16": 211, "F32": 61}
              and c["f32"]["dtypes"] == {"F32": 272})
        return ok, f"bag roots equal {r['bag_roots_equal']}, bf16 {c['bf16']['dtypes']}, f32 {c['f32']['dtypes']}"
    return both(obs, f)


def p6(obs):
    def f(m):
        r = need(get(obs, m, "onnx_bpk"), "ONNX against burnpack")
        ok = r["large"] == {"identical": 271, "both": 2} and r["small"] == {"identical": 153, "reshaped": 243}
        return ok, f"large {r['large']}, small {r['small']}"
    return both(obs, f)


def p7(obs):
    c = need(obs.get("cross"), "the comparison of the two models")
    w = c["weights_shared_keys"]
    for k in ("st", "onnx", "gguf-f32", "bpk"):
        need(w.get(k), f"cross {k}")
    s = need(c.get("bpk_small"), "the small tensors of the burnpacks")
    q = get(obs, "queries", "cross_shared")
    ok = (all(w[k] == 0 for k in ("st", "onnx", "gguf-f32", "bpk")) and s["digests_equal"] and s["unmatched"] == 0
          and s["tensors_base"] == 396 and s["tensors_instruct"] == 396)
    if q is not None:
        ok = ok and q.get("bpk", 0) == 396 and all(q.get(k, 0) == 0 for k in ("st", "onnx", "gguf-f32"))
    return ok, f"shared weight keys {w}, small tensors {s['tensors_base']}/{s['tensors_instruct']} equal {s['digests_equal']}, SPARQL {q}"


def p8(obs):
    def f(m):
        rows = need(get(obs, "config", m), "key-value table against config.json")
        bad = [r["key"] for r in rows if not r["ok"]]
        return (not bad and len(rows) >= 12), f"{len(rows)} keys checked, wrong: {bad}"
    return both(obs, f)


def p9(obs):
    k = need(get(obs, "cross", "kv"), "the key-value tables of the two models")
    moved = k["differ"] + k["only_base"] + k["only_instruct"]
    allowed = lambda key: key.startswith("general.") or key in (  # noqa: E731
        "tokenizer.chat_template", "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id", "tokenizer.ggml.padding_token_id")
    bad = [x for x in moved if not allowed(x)]
    arrays = [x for x in moved if x in ("tokenizer.ggml.tokens", "tokenizer.ggml.merges", "tokenizer.ggml.token_type")]
    llama = [x for x in moved if x.startswith("llama.")]
    return (not bad and not arrays and not llama), f"differing keys {len(moved)}; outside the allowed set: {bad}"


def p10(obs):
    s = need(get(obs, "cross", "skeleton"), "the skeletons")
    t = need(get(obs, "cross", "structure_triples"), "the structure triples")
    for m in SMOL:
        c = need(get(obs, m, "onnx_skeleton"), "C3")
        if not (c.get("control_changed_operator_differs") and c.get("control_restored_equals")):
            raise NotChecked("C3")
    return (s["equal"] and t["equal"]), f"skeleton {str(s['base'])[:8]} / {str(s['instruct'])[:8]}, structure triples {t['base']} / {t['instruct']}"


def p11(obs):
    r = need(get(obs, "cross", "rust"), "the generated sources")
    b, f = need(r.get("bytes-model.rs"), "Bytes sources"), need(r.get("model.rs"), "File sources")
    return (b["differing_lines"] == 1 and f["differing_lines"] == 2), \
        f"Bytes: {b['differing_lines']} line(s) differ, File: {f['differing_lines']}"


def p12(obs):
    w = need(get(obs, "cross", "wasm"), "the two modules")
    return w["equal"], f"sha256 {str(w['base'])[:8]} / {str(w['instruct'])[:8]}" + (f", {w.get('differing_bodies')} bodies differ" if not w["equal"] else "")


def p13(obs):
    def f(m):
        s = need(get(obs, m, "wasm_search"), "the weight search")
        if not s["planted_window_found"]:
            raise NotChecked("C4")
        ok = s["tensors_searched"] == 273 and s["found_in_module"] == [] and s["found_in_burnpack"] == 273
        return ok, f"{s['tensors_searched']} searched, in the module {len(s['found_in_module'])}, in the burnpack {s['found_in_burnpack']}"
    return both(obs, f)


def p14(obs):
    def f(m):
        lg = need(get(obs, m, "logits"), "logits") or None
        a, n = get(lg, "ort_all"), get(lg, "ort_none")
        need(a, "logits against ONNX Runtime, optimization on")
        need(n, "logits against ONNX Runtime, optimization off")
        c = need(get(lg, "control_other_model"), "the control against the other model")
        if c["error"] or not c["max_abs"] > 0.1:
            raise NotChecked("the control of P14")
        ok = a["error"] is None and n["error"] is None and a["P7"] and a["P8"] and n["P7"] and n["P8"]
        return ok, f"on: P7 {a['P7']} P8 {a['P8']} max {a.get('max_abs')}; off: P7 {n['P7']} P8 {n['P8']}; control {c['max_abs']:.3g}"
    return both(obs, f)


def p15(obs):
    g = need(obs.get("ground"), "the ground check")
    ox = g.get("oxigraph")
    need(ox, "Oxigraph")
    if not (ox.get("control_ground_terms") == "true" and g["control_python_blank_statements"] >= 1 and g["control_differs"]):
        raise NotChecked("the control of P15")
    info = [get(obs, t, k) for t in obs["tags"] for k in ("onnx_graph", "rust_graph", "wasm_graph")]
    info = [i for i in info if i]
    ok = (g["python_blank_statements"] == 0 and ox.get("ground-terms") == "false" and ox.get("ground-graph-names") == "false"
          and all(ok_ for _, ok_ in g["determinism"]) and g["collapsed_triples"] == 0
          and all(i["deterministic"] and i["collapsed"] == 0 and i["blank_terms"] == 0 for i in info))
    for t in obs["tags"]:
        o, c = get(obs, t, "onnx_graph"), get(obs, t, "onnx_census")
        if o and c and o["nodes"] != c["nodes"]:
            ok = False
        w, wc = get(obs, t, "wasm_graph"), get(obs, t, "wasm_census")
        if w and wc and w["functions"] != wc["functions"]:
            ok = False
        r, rc = get(obs, t, "rust_graph"), get(obs, t, "rust_census", "bytes")
        if r and rc and r["statements"] != rc["top_level_statements"]:
            ok = False
    return ok, (f"{g['graphs']} graphs; blank statements (Python) {g['python_blank_statements']}; Oxigraph terms {ox.get('ground-terms')}, "
                f"graph names {ox.get('ground-graph-names')}; control {ox.get('control_ground_terms')}; collapsed {g['collapsed_triples']}; "
                f"determinism {g['determinism']}")


def p16(obs):
    def f(t):
        c = need(get(obs, t, "onnx_census"), "ONNX census")
        return (c["nodes_unnamed"] == 0 and c["node_names_repeated"] == 0), f"{c['nodes_unnamed']} unnamed, {c['node_names_repeated']} repeated of {c['nodes']}"
    return each_tag(obs, f)


def p17(obs):
    def f(t):
        d = need(get(obs, t, "bpk_file_vs_bytes"), "the comparison of the two burnpacks")
        n = need(get(obs, t, "bpk_census", "file"), "the burnpack census")["tensors"]
        ok = d["param_ids_differing"] == n and d["metadata_equal_without_param_id"] and d["data_region_sha256_equal"]
        if t in SMOL:
            ok = ok and n == 669
        return ok, f"{d['param_ids_differing']} of {n} param_ids differ, descriptors equal without them {d['metadata_equal_without_param_id']}, data equal {d['data_region_sha256_equal']}"
    msgs = []
    for t in obs.get("burn_tags", obs["tags"]):
        ok, msg = f(t)
        if not ok:
            return False, f"{t}: {msg}"
        msgs.append(f"{t}: {msg}")
    return True, "; ".join(msgs)


def p18(obs):
    def f(t):
        c = need(get(obs, t, "onnx_census"), "ONNX census")
        ok = c["control_flow_nodes"] == 0 and c["graph_attributes"] == 0 and c["acyclic"] and c["inputs_used_before_made"] == 0
        return ok, f"control flow {c['control_flow_nodes']}, graph attributes {c['graph_attributes']}, acyclic {c['acyclic']}"
    return each_tag(obs, f)


def p19(obs):
    def f(m):
        c = need(get(obs, m, "wasm_census"), "the module's census")
        return (c["imports"] == 0 and sorted(c["exports"]) == EIGHT), f"{c['imports']} imports, exports {c['exports']}"
    return both(obs, f)


def p20(obs):
    w = need(obs.get("weights"), "the scan of the weights")

    def f(m):
        s, u = w[m]["scan"], w[m]["units"]
        if s["tensors_not_bf16"]:
            raise NotChecked("a tensor that is not BF16")
        ok = all(s[k] == 0 for k in ("zero", "negative_zero", "nan", "inf", "subnormal")) and u["repeated_units"] == 0
        return ok, f"zeros {s['zero']}, -0 {s['negative_zero']}, NaN {s['nan']}, inf {s['inf']}, subnormal {s['subnormal']}, repeated units {u['repeated_units']}"
    return both(obs, f)


NEEDS_BURN = {"P6", "P11", "P12", "P13", "P14", "P17", "P19"}

PREDICTIONS = {"P1": (p1, SMOL), "P2": (p2, SMOL), "P3": (p3, SMOL), "P4": (p4, SMOL), "P5": (p5, SMOL), "P6": (p6, SMOL), "P7": (p7, SMOL),
               "P8": (p8, SMOL), "P9": (p9, SMOL), "P10": (p10, SMOL), "P11": (p11, SMOL), "P12": (p12, SMOL), "P13": (p13, SMOL),
               "P14": (p14, SMOL), "P15": (p15, ()), "P16": (p16, ()), "P17": (p17, ()), "P18": (p18, ()), "P19": (p19, SMOL),
               "P20": (p20, SMOL)}


def controls(obs):
    out = {}
    smol = [m for m in SMOL if m in obs["tags"]]
    out["C1"] = all(get(obs, m, "gguf_controls") and get(obs, m, "gguf_controls", "C1_moved_tensor_is_none")
                    and get(obs, m, "gguf_controls", "C1_unmoved_tensor_is_found") and get(obs, m, "matcher_control", "holds") for m in smol) if smol else None
    out["C2"] = all(get(obs, m, "gguf_controls", "C2_wrong_head_count_is_none") and get(obs, m, "gguf_controls", "C2_right_head_count_is_found")
                    for m in smol) if smol else None
    sk = [get(obs, t, "onnx_skeleton") for t in obs["tags"]]
    out["C3"] = all(s and s.get("control_changed_operator_differs") and s.get("control_restored_equals") for s in sk) if sk else None
    out["C4"] = all(get(obs, t, "wasm_search", "planted_window_found") for t in obs["tags"] if get(obs, t, "wasm_search") is not None)
    out["C5"] = all(get(obs, m, "gguf_independent", k, "mismatch_count") == 0 for m in smol for k in ("f32", "bf16")) if smol else None
    return out


def grade(obs):
    tags = obs["tags"]
    burn = obs.get("burn_tags", tags)
    rows = []
    for pid, (fn, needs) in PREDICTIONS.items():
        have = burn if pid in NEEDS_BURN else tags
        if any(t not in tags for t in needs) or any(t not in have for t in needs) or (pid == "P17" and not burn):
            rows.append({"id": pid, "severity": SEVERITY[pid], "status": "not applicable",
                         "detail": f"needs {list(needs)} (burn jobs: {burn}); run has {tags}"})
            continue
        try:
            ok, detail = fn(obs)
            rows.append({"id": pid, "severity": SEVERITY[pid], "status": "supported" if ok else "refuted", "detail": detail})
        except NotChecked as e:
            rows.append({"id": pid, "severity": SEVERITY[pid], "status": "not checked", "detail": f"needs: {e}"})
        except (KeyError, TypeError, IndexError, ZeroDivisionError) as e:
            rows.append({"id": pid, "severity": SEVERITY[pid], "status": "not checked", "detail": f"observation malformed: {e!r}"})
    return rows, controls(obs)


def report(rows, ctl):
    lines = ["| prediction | severity | verdict | what was seen |", "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['severity']} | **{r['status']}** | {r['detail']} |")
    lines += ["", "| control | holds |", "|---|---|"] + [f"| {k} | {v} |" for k, v in ctl.items()]
    return "\n".join(lines) + "\n"


def exit_status(rows, ctl):
    bad = [r for r in rows if r["status"] in ("refuted", "not checked")]
    return 1 if bad or any(v is False for v in ctl.values()) else 0


# ---- the selftest: every prediction can be supported and can be refuted ---------------------------------------------

def expected():
    """The observations under which every prediction is supported: the contract between combine.py and this file."""
    onnx_census = {"initializers": 273, "initializer_elements": 162826560, "nodes": 7360, "nodes_unnamed": 0, "node_names_repeated": 0,
                   "control_flow_nodes": 0, "graph_attributes": 0, "acyclic": True, "inputs_used_before_made": 0}
    gguf_ctl = {"C1_moved_tensor_is_none": True, "C1_unmoved_tensor_is_found": True, "C2_wrong_head_count_is_none": True,
                "C2_right_head_count_is_found": True}
    cfg = [{"key": f"k{i}", "ok": True} for i in range(12)]

    def model(m):
        return {
            "st_header": {"tensors": 272, "dtypes": {"BF16": 272}, "elements": 134515008, "has_lm_head": False, "metadata": {"format": "pt"},
                          "offsets_contiguous": True, "json_ends_at_brace": True, "duplicate_names": []},
            "onnx_record_census": dict(onnx_census), "onnx_census": dict(onnx_census),
            "onnx_sha256": INSTRUCT_ONNX_SHA if m == "instruct" else "00" * 32,
            "onnx_skeleton": {"control_changed_operator_differs": True, "control_restored_equals": True, "skeleton_sha256": "aa"},
            "st_onnx": {"ref_distinct_keys": 272, "target_distinct_keys": 272, "keys_only_ref": 0, "keys_only_target": 0,
                        "relations": {"transposed": 211, "identical": 62}, "ref_tensors_matched_by_more_than_one": {"model.embed_tokens.weight": 2}},
            "matcher_control": {"holds": True},
            "gguf": {"tensors": 272, "relations": {"identical": 212, "permuted": 60}, "permuted_by_n": {"9": 30, "3": 30},
                     "permuted_roles": {"q_proj": 30, "k_proj": 30}, "name_pairing_agrees": 272},
            "gguf_controls": dict(gguf_ctl), "gguf_independent": {"f32": {"mismatch_count": 0}, "bf16": {"mismatch_count": 0}},
            "gguf_bf16_vs_f32": {"bag_roots_equal": True, "keys_in_both": 272},
            "gguf_census": {"bf16": {"dtypes": {"BF16": 211, "F32": 61}}, "f32": {"dtypes": {"F32": 272}}},
            "onnx_bpk": {"large": {"identical": 271, "both": 2}, "small": {"identical": 153, "reshaped": 243}},
            "wasm_search": {"tensors_searched": 273, "found_in_module": [], "found_in_burnpack": 273, "planted_window_found": True},
            "logits": {"ort_all": {"error": None, "P7": True, "P8": True, "max_abs": 7e-5}, "ort_none": {"error": None, "P7": True, "P8": True},
                       "control_other_model": {"error": None, "max_abs": 3.2}},
            "onnx_graph": {"nodes": 7360, "deterministic": True, "collapsed": 0, "blank_terms": 0},
            "rust_graph": {"statements": 10, "deterministic": True, "collapsed": 0, "blank_terms": 0},
            "rust_census": {"bytes": {"top_level_statements": 10}},
            "wasm_graph": {"functions": 5, "deterministic": True, "collapsed": 0, "blank_terms": 0},
            "wasm_census": {"functions": 5, "imports": 0, "exports": list(EIGHT)},
            "bpk_census": {"file": {"tensors": 669}},
            "bpk_file_vs_bytes": {"param_ids_differing": 669, "metadata_equal_without_param_id": True, "data_region_sha256_equal": True},
        }

    return {
        "tags": ["base", "instruct"], "base": model("base"), "instruct": model("instruct"),
        "config": {"base": cfg, "instruct": cfg},
        "cross": {"weights_shared_keys": {"st": 0, "onnx": 0, "gguf-f32": 0, "bpk": 0},
                  "bpk_small": {"tensors_base": 396, "tensors_instruct": 396, "digests_equal": True, "unmatched": 0},
                  "kv": {"differ": ["general.name", "tokenizer.chat_template", "tokenizer.ggml.eos_token_id"], "only_base": [],
                         "only_instruct": ["tokenizer.chat_template"]},
                  "skeleton": {"base": "aa", "instruct": "aa", "equal": True}, "structure_triples": {"base": 5, "instruct": 5, "equal": True},
                  "rust": {"bytes-model.rs": {"differing_lines": 1}, "model.rs": {"differing_lines": 2}},
                  "wasm": {"base": "bb", "instruct": "bb", "equal": True}},
        "queries": {"st_counterparts": {"base/onnx": {"1": 271, "2": 1}, "instruct/onnx": {"1": 271, "2": 1},
                                        "base/gguf-f32": {"1": 272}, "instruct/gguf-f32": {"1": 272}},
                    "cross_shared": {"bpk": 396}},
        "ground": {"graphs": 40, "python_blank_statements": 0, "oxigraph": {"ground-terms": "false", "ground-graph-names": "false",
                                                                          "control_ground_terms": "true"},
                   "control_python_blank_statements": 1, "control_differs": True, "determinism": [["x", True]], "collapsed_triples": 0},
        "weights": {m: {"scan": {"zero": 0, "negative_zero": 0, "nan": 0, "inf": 0, "subnormal": 0, "tensors_not_bf16": []},
                        "units": {"repeated_units": 0}} for m in SMOL},
    }


def selftest():
    bad = 0

    def check(what, ok):
        nonlocal bad
        print(("ok   " if ok else "FAIL ") + what)
        bad += 0 if ok else 1

    obs = expected()
    rows, ctl = grade(obs)
    check("all twenty predictions are supported by the expected observations", [r["status"] for r in rows].count("supported") == 20
          and exit_status(rows, ctl) == 0)
    check("every control holds", all(ctl.values()))
    mutations = {
        "P1": lambda o: o["base"]["st_header"].update(tensors=271), "P2": lambda o: o["base"]["onnx_record_census"].update(nodes=7361),
        "P3": lambda o: o["instruct"]["st_onnx"].update(relations={"transposed": 210, "identical": 63}),
        "P4": lambda o: o["base"]["gguf"].update(relations={"identical": 272}),
        "P5": lambda o: o["base"]["gguf_census"]["bf16"].update(dtypes={"BF16": 212, "F32": 60}),
        "P6": lambda o: o["base"]["onnx_bpk"].update(small={"identical": 154, "reshaped": 242}),
        "P7": lambda o: o["cross"]["weights_shared_keys"].update(st=1), "P8": lambda o: o["config"]["base"][0].update(ok=False),
        "P9": lambda o: o["cross"]["kv"]["differ"].append("llama.block_count"),
        "P10": lambda o: o["cross"]["skeleton"].update(equal=False),
        "P11": lambda o: o["cross"]["rust"]["bytes-model.rs"].update(differing_lines=2),
        "P12": lambda o: o["cross"]["wasm"].update(equal=False), "P13": lambda o: o["base"]["wasm_search"].update(found_in_module=["x"]),
        "P14": lambda o: o["base"]["logits"]["ort_none"].update(P8=False),
        "P15": lambda o: o["ground"].update(python_blank_statements=1), "P16": lambda o: o["base"]["onnx_census"].update(nodes_unnamed=1),
        "P17": lambda o: o["base"]["bpk_file_vs_bytes"].update(param_ids_differing=668),
        "P18": lambda o: o["instruct"]["onnx_census"].update(control_flow_nodes=1),
        "P19": lambda o: o["base"]["wasm_census"].update(imports=1),
        "P20": lambda o: o["weights"]["instruct"]["scan"].update(zero=1),
    }
    for pid, mutate in mutations.items():
        o = copy.deepcopy(obs)
        mutate(o)
        r, _ = grade(o)
        st = {x["id"]: x["status"] for x in r}
        check(f"{pid} is refuted by one change to its observation", st[pid] == "refuted")
    o = copy.deepcopy(obs)
    o["base"]["gguf_controls"]["C2_wrong_head_count_is_none"] = False
    r, c = grade(o)
    check("a failed control leaves P4 not checked and fails the run", {x["id"]: x["status"] for x in r}["P4"] == "not checked"
          and c["C2"] is False and exit_status(r, c) == 1)
    o = copy.deepcopy(obs)
    o["tags"] = ["tiny"]
    o["tiny"] = o["base"]
    r, c = grade(o)
    st = {x["id"]: x["status"] for x in r}
    check("a tiny-only run grades what applies to it and not the rest", st["P1"] == "not applicable" and st["P16"] == "supported"
          and st["P18"] == "supported")
    o = copy.deepcopy(obs)
    del o["cross"]["wasm"]
    check("a missing observation is `not checked`", {x["id"]: x["status"] for x in grade(o)[0]}["P12"] == "not checked")
    return bad


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        sys.exit(1 if selftest() else 0)
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    observations = json.load(open(sys.argv[1]))
    rows_, ctl_ = grade(observations)
    with open(sys.argv[2] + ".json", "w") as fh:
        json.dump({"predictions": rows_, "controls": ctl_}, fh, indent=1, sort_keys=True)
    with open(sys.argv[2] + ".md", "w") as fh:
        fh.write(report(rows_, ctl_))
    print(report(rows_, ctl_))
    sys.exit(exit_status(rows_, ctl_))
