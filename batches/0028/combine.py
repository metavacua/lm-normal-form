# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Cell 06 of batch 0028: the graphs and maps of all cells as one ground dataset, checked in Oxigraph, and the observations
that grade28.py grades.

Reads the artifacts of the jobs (ART/<job>-<tag>/<tag>/...), builds from the identity records the tensor graphs and the maps
between them (one named graph per artifact and per map; one predicate per relation, so that nothing is reified), adds the
structure graphs that the cells wrote, loads the dataset into Oxigraph and runs the queries of queries/0028, compares the two
models, and writes OUT/observations.json, the per-model tensor tables, the census table and the dataset.

Usage: combine.py ART OUT TAGS_JSON BURN_TAGS_JSON [OXIGRAPH]
       combine.py selftest
"""

import collections
import difflib
import glob
import gzip
import hashlib
import json
import os
import re
import subprocess
import sys

import common
import ground as G
from common import identity

HERE = os.path.dirname(os.path.abspath(__file__))
QUERIES = os.path.join(HERE, "..", "..", "queries", "0028")
REL_PRED = {"identical": ["identicalTo"], "transposed": ["transposeOf"], "both": ["identicalTo", "transposeOf"],
            "reshaped": ["reshapeOf"], "permuted": ["llamaCppPermutationOf"]}
ALLOWED_KV = ("general.",)
ALLOWED_KV_EXACT = {"tokenizer.chat_template", "tokenizer.ggml.bos_token_id", "tokenizer.ggml.eos_token_id",
                    "tokenizer.ggml.padding_token_id"}
EIGHT_EXPORTS = ["forward", "ids_alloc", "memory", "model_load", "out_ptr", "panic_len", "panic_ptr", "weights_alloc"]


class Art:
    """The downloaded artifacts: ROOT/<job>-<tag>/<tag>/<name>."""

    def __init__(self, root):
        self.root = root

    def p(self, job, tag, name):
        return os.path.join(self.root, f"{job}-{tag}", tag, name)

    def j(self, job, tag, name):
        path = self.p(job, tag, name)
        return common.load(path) if os.path.exists(path) else None

    def text(self, job, tag, name):
        path = self.p(job, tag, name)
        if not os.path.exists(path):
            return None
        opener = gzip.open if path.endswith(".gz") else open
        with opener(path, "rt", encoding="utf-8") as f:
            return f.read()


def label(e):
    return next(iter(e["labels"].values()))


def role_of(name):
    for pat, role in (("embed_tokens", "embedding"), ("norm", "norm"), ("q_proj", "query"), ("k_proj", "key"), ("v_proj", "value"),
                      ("o_proj", "output"), ("gate_proj", "gate"), ("up_proj", "up"), ("down_proj", "down")):
        if pat in name:
            return role
    return "other"


# ---- graphs ------------------------------------------------------------------------------------------------------

def tensor_graph(tag, artifact, rec):
    gr = G.Graph(tag, artifact + "-tensors")
    art = G.iri("a", tag, artifact)
    gr.typed(art, "Artifact")
    gr.add(art, G.pred("format"), G.lit(rec["subject"]["container"]))
    lists = [(rec["subject"]["tensors"], False), (rec["subject"].get("constants", []), True)]
    for ents, const in lists:
        for e in ents:
            t = G.tensor_iri(tag, artifact, label(e))
            gr.typed(t, "Tensor")
            gr.add(t, G.pred("inArtifact"), art)
            gr.add(t, G.pred("label"), G.lit(label(e)))
            gr.add(t, G.pred("dtype"), G.lit(e["dtype"]))
            gr.add(t, G.pred("storedDtype"), G.lit(e["stored_dtype"]))
            gr.add(t, G.pred("shape"), G.lit("x".join(map(str, e["shape"])) or "scalar"))
            gr.add(t, G.pred("digest"), G.lit(e["digest"]))
            if e.get("digest_t"):
                gr.add(t, G.pred("digestTransposed"), G.lit(e["digest_t"]))
            if "x-bytes" in e:
                gr.add(t, G.pred("bytes"), G.num(e["x-bytes"]))
            if "x-role" in e:
                gr.add(t, G.pred("role"), G.lit(e["x-role"]))
            if const:
                gr.add(t, G.pred("constant"), G.lit("true"))
    return gr


def relation_graph(tag_tgt, tag_ref, art_ref, art_tgt, rows, kind="maps"):
    """Edges from the tensors of the target to the tensors of the reference they are related to, one predicate per relation."""
    gr = G.Graph(kind, f"{tag_tgt}:{art_ref}:{art_tgt}" if tag_tgt == tag_ref else f"{tag_tgt}:{tag_ref}:{art_tgt}")
    for r in rows:
        t = G.tensor_iri(tag_tgt, art_tgt, r["label"])
        if r["relation"] in ("none", "unread"):
            gr.add(t, G.pred("unmatched"), G.lit("true"))
            continue
        for pr in REL_PRED[r["relation"]]:
            for ref in r["ref"]:
                gr.add(t, G.pred(pr), G.tensor_iri(tag_ref, art_ref, ref))
    return gr


def gguf_rows(maps):
    return [{"label": r["gguf"], "relation": r["relation"], "ref": r["refs"]} for r in maps["rows"]]


# ---- the two models -----------------------------------------------------------------------------------------------

def partition_by_bytes(rows, tgt_entries):
    large = [r for r, e in zip(rows, tgt_entries) if e.get("x-bytes", 0) >= 256]
    small = [r for r, e in zip(rows, tgt_entries) if e.get("x-bytes", 0) < 256]
    return large, small


def rel_counts(rows):
    return dict(collections.Counter(r["relation"] for r in rows))


def tensor_table(m, st, onnx_rows, gguf_maps_rows, bpk_rows):
    """One row per safetensors tensor: its counterparts in ONNX, GGUF and the burnpack, with the relation of each."""
    onnx_of, gguf_of, bpk_of = collections.defaultdict(list), collections.defaultdict(list), collections.defaultdict(list)
    for r in onnx_rows or []:
        for ref in r["ref"]:
            onnx_of[ref].append((r["label"], r["relation"]))
    for r in gguf_maps_rows or []:
        for ref in r["ref"]:
            gguf_of[ref].append((r["label"], r["relation"]))
    for r in bpk_rows or []:
        for ref in r["ref"]:
            bpk_of[ref].append((r["label"], r["relation"]))
    lines = ["safetensors\trole\tshape\tonnx\tgguf\tburnpack\n"]
    for e in st["subject"]["tensors"]:
        n = label(e)
        o = onnx_of.get(n, [])
        b = list(dict.fromkeys(x for on, _ in o for x in bpk_of.get(on, [])))
        fmt = lambda xs: "|".join(f"{a}:{rel}" for a, rel in xs) or "-"  # noqa: E731
        lines.append(f"{n}\t{role_of(n)}\t{'x'.join(map(str, e['shape']))}\t{fmt(o)}\t{fmt(gguf_of.get(n, []))}\t{fmt(b)}\n")
    return "".join(lines)


def matcher_control(st, onnx):
    """C1 for identity.match: one tensor of the target given another digest is `none` and no other row changes."""
    import copy
    base_rows, _ = identity.match(st, onnx)
    pert = copy.deepcopy(onnx)
    e = pert["subject"]["tensors"][0]
    flip = lambda h: ("1" if h[0] != "1" else "2") + h[1:]  # noqa: E731
    e["digest"] = flip(e["digest"])
    if e.get("digest_t"):
        e["digest_t"] = flip(e["digest_t"])
    e.pop("x-flat", None)
    rows, _ = identity.match(st, pert)
    changed = [i for i, (a, b) in enumerate(zip(base_rows, rows)) if a["relation"] != b["relation"] or a["ref"] != b["ref"]]
    return {"holds": changed == [0] and rows[0]["relation"] == "none", "changed_rows": changed, "first_relation": rows[0]["relation"]}


def models_maps(A, m, graphs):
    out = {}
    st, onnx = A.j("inputs", m, "st-record.json"), A.j("inputs", m, "onnx-record.json")
    g32, g16 = A.j("gguf", m, "gguf-f32-record.json"), A.j("gguf", m, "gguf-bf16-record.json")
    bpk, onnxpub = A.j("burn", m, "bpk-file-record.json"), A.j("inputs", m, "onnxpub-record.json")
    recs = {"st": st, "onnx": onnx, "gguf-f32": g32, "gguf-bf16": g16, "bpk": bpk, "onnxpub": onnxpub}
    for art, rec in recs.items():
        if rec:
            graphs.append(tensor_graph(m, art, rec))
    onnx_rows = gguf_r = bpk_rows = None
    if st and onnx:
        rows, summ = identity.match(st, onnx)
        onnx_rows, out["st_onnx"] = rows, summ
        out["matcher_control"] = matcher_control(st, onnx)
        graphs.append(relation_graph(m, m, "st", "onnx", rows))
    if st and onnxpub:
        rows, summ = identity.match(st, onnxpub)
        out["st_onnxpub"] = summ
        graphs.append(relation_graph(m, m, "st", "onnxpub", rows))
    if onnx and bpk:
        rows, summ = identity.match(onnx, bpk, constants=True)
        bpk_rows = rows
        large, small = partition_by_bytes(rows, bpk["subject"]["tensors"])
        out["onnx_bpk"] = {"summary": summ, "large": rel_counts(large), "small": rel_counts(small),
                           "large_tensors": len(large), "small_tensors": len(small)}
        graphs.append(relation_graph(m, m, "onnx", "bpk", rows))
    gm = A.j("gguf", m, "gguf-maps.json")
    if gm:
        gguf_r = gguf_rows(gm)
        out["gguf"] = gm["summary"]
        graphs.append(relation_graph(m, m, "st", "gguf-f32", gguf_r))
    if g32 and g16:
        rows, summ = identity.match(g32, g16)
        out["gguf_bf16_vs_f32"] = {"bag_roots_equal": summ["bag_roots_equal"], "keys_in_both": summ["keys_in_both"],
                                   "relations": summ["relations"]}
        graphs.append(relation_graph(m, m, "gguf-f32", "gguf-bf16", rows))
    out["gguf_census"] = {"f32": A.j("gguf", m, "gguf-f32-census.json"), "bf16": A.j("gguf", m, "gguf-bf16-census.json")}
    out["gguf_controls"] = A.j("gguf", m, "gguf-controls.json")
    out["gguf_independent"] = {"f32": A.j("gguf", m, "gguf-f32-independent.json"), "bf16": A.j("gguf", m, "gguf-bf16-independent.json")}
    out["gguf_config"] = A.j("gguf", m, "gguf-config.json")
    out["tensor_table"] = tensor_table(m, st, onnx_rows, gguf_r, bpk_rows) if st else None
    return out


def kv_diff(a, b):
    ka, kb = set(a), set(b)
    differ = [k for k in sorted(ka & kb) if any(a[k].get(f) != b[k].get(f) for f in ("type", "value", "len", "sha256"))]
    return {"only_base": sorted(ka - kb), "only_instruct": sorted(kb - ka), "differ": differ}


def line_diff(a, b):
    x, y = a.splitlines(), b.splitlines()
    if len(x) == len(y):
        idx = [i for i, (p, q) in enumerate(zip(x, y)) if p != q]
        return {"lines": [len(x), len(y)], "differing_lines": len(idx), "first": [x[i][:160] for i in idx[:3]]}
    d = [line for line in difflib.unified_diff(x, y, lineterm="", n=0) if line[:1] in "+-" and line[:3] not in ("+++", "---")]
    return {"lines": [len(x), len(y)], "differing_lines": len(d), "first": d[:3]}


def structure_lines(text, tag):
    """The statements of an artifact graph without the declared labels, the model's name replaced: what a program is."""
    out = []
    for line in text.splitlines():
        if ":p:label>" in line:
            continue
        out.append(line.replace(f":{tag}:", ":M:"))
    return sorted(out)


def cross(A, graphs):
    out = {"weights_shared_keys": {}, "bpk_small": None}
    for kind, job, name in (("st", "inputs", "st-record.json"), ("onnx", "inputs", "onnx-record.json"),
                            ("gguf-f32", "gguf", "gguf-f32-record.json"), ("bpk", "burn", "bpk-file-record.json")):
        rb, ri = A.j(job, "base", name), A.j(job, "instruct", name)
        if not (rb and ri):
            out["weights_shared_keys"][kind] = None
            continue
        rows, summ = identity.match(rb, ri)
        graphs.append(relation_graph("instruct", "base", kind, kind, rows, kind="cross"))
        tgt = ri["subject"]["tensors"]
        if kind == "bpk":
            large, small = partition_by_bytes(rows, tgt)
            out["weights_shared_keys"][kind] = sum(1 for r in large if r["relation"] != "none")
            sb = {label(e): e["digest"] for e in rb["subject"]["tensors"] if e.get("x-bytes", 0) < 256}
            si = {label(e): e["digest"] for e in tgt if e.get("x-bytes", 0) < 256}
            out["bpk_small"] = {"tensors_base": len(sb), "tensors_instruct": len(si), "digests_equal": sorted(sb.values()) == sorted(si.values()),
                                "same_name_same_digest": sum(1 for k, v in si.items() if sb.get(k) == v),
                                "relations": rel_counts(small), "unmatched": sum(1 for r in small if r["relation"] == "none")}
        else:
            out["weights_shared_keys"][kind] = sum(1 for r in rows if r["relation"] != "none")
    kb, ki = A.j("gguf", "base", "gguf-kv.json"), A.j("gguf", "instruct", "gguf-kv.json")
    out["kv"] = kv_diff(kb, ki) if kb and ki else None
    sb, si = A.j("inputs", "base", "onnx-skeleton.json"), A.j("inputs", "instruct", "onnx-skeleton.json")
    out["skeleton"] = {"base": sb and sb["skeleton_sha256"], "instruct": si and si["skeleton_sha256"],
                       "equal": bool(sb and si and sb["skeleton_sha256"] == si["skeleton_sha256"])} if sb and si else None
    gb, gi = A.text("inputs", "base", "onnx-graph.nq.gz"), A.text("inputs", "instruct", "onnx-graph.nq.gz")
    if gb and gi:
        xb, xi = structure_lines(gb, "base"), structure_lines(gi, "instruct")
        out["structure_triples"] = {"base": len(xb), "instruct": len(xi), "equal": xb == xi}
    out["rust"] = {}
    for fname in ("bytes-model.rs", "model.rs"):
        a, b = A.text("burn", "base", fname), A.text("burn", "instruct", fname)
        out["rust"][fname] = line_diff(a, b) if a and b else None
    wb, wi = A.j("burn", "base", "wasm-census.json"), A.j("burn", "instruct", "wasm-census.json")
    if wb and wi:
        w = {"base": wb["sha256"], "instruct": wi["sha256"], "equal": wb["sha256"] == wi["sha256"]}
        if not w["equal"]:
            hb, hi = A.j("burn", "base", "wasm-bodies.json"), A.j("burn", "instruct", "wasm-bodies.json")
            if hb and hi:
                w["functions"] = [len(hb), len(hi)]
                w["differing_bodies"] = sum(1 for p, q in zip(hb, hi) if p != q) + abs(len(hb) - len(hi))
        out["wasm"] = w
    return out


# ---- Oxigraph ----------------------------------------------------------------------------------------------------

def tsv_rows(text):
    rows = [line.split("\t") for line in text.strip("\n").split("\n") if line]
    clean = lambda s: re.sub(r'\^\^<[^>]*>$', "", s).strip('"<>')  # noqa: E731
    return [[clean(c) for c in r] for r in rows[1:]]


def oxigraph_run(ox, graphs, out, controls_text):
    """Load the dataset, run every query of queries/0028, and run the control (a copy of one graph with a blank node)."""
    import shutil
    os.makedirs(os.path.join(out, "sparql"), exist_ok=True)
    nq = os.path.join(out, "dataset.nq")
    with open(nq, "w", encoding="utf-8", newline="\n") as f:
        for g in graphs:
            f.write(g.text())
    store = os.path.join(out, "store")
    shutil.rmtree(store, ignore_errors=True)
    subprocess.run([ox, "load", "--location", store, "--file", nq], check=True, capture_output=True, text=True)
    res = {}
    for q in sorted(glob.glob(os.path.join(QUERIES, "*.rq"))):
        name = os.path.basename(q)[:-3]
        r = subprocess.run([ox, "query", "--location", store, "--query-file", q, "--union-default-graph", "--results-format", "tsv"],
                           capture_output=True, text=True)
        if r.returncode:
            res[name] = {"error": r.stderr[-300:]}
            continue
        with open(os.path.join(out, "sparql", name + ".tsv"), "w") as f:
            f.write(r.stdout)
        res[name] = r.stdout.strip() if name.startswith("ground-") else tsv_rows(r.stdout)
    # the control, through the same engine: one graph of the dataset with one blank node
    cstore, cnq = os.path.join(out, "store-control"), os.path.join(out, "control.nq")
    shutil.rmtree(cstore, ignore_errors=True)
    with open(cnq, "w", encoding="utf-8", newline="\n") as f:
        f.write(controls_text)
    subprocess.run([ox, "load", "--location", cstore, "--file", cnq], check=True, capture_output=True, text=True)
    r = subprocess.run([ox, "query", "--location", cstore, "--query-file", os.path.join(QUERIES, "ground-terms.rq"),
                        "--union-default-graph", "--results-format", "tsv"], capture_output=True, text=True)
    res["control_ground_terms"] = r.stdout.strip()
    shutil.rmtree(store, ignore_errors=True)
    shutil.rmtree(cstore, ignore_errors=True)
    os.remove(cnq)
    return res, nq


def histogram(rows, key_cols, count_col):
    out = collections.defaultdict(dict)
    for r in rows:
        out["/".join(r[:key_cols])][r[key_cols]] = int(r[count_col])
    return {k: dict(v) for k, v in out.items()}


# ---- census table ------------------------------------------------------------------------------------------------

def census_table(obs):
    rows = ["| artifact | A anonymous | B absent | C neutral | D unused capacity | E non-portable label | F unbounded control |",
            "|---|---|---|---|---|---|---|"]
    for tag in ("base", "instruct", "tiny"):
        t = obs.get(tag) or {}
        st = (t.get("st_header") or {}) if t else {}
        oc = t.get("onnx_census")
        if st:
            rows.append(f"| {tag} safetensors | tensors without a name: 0; duplicate names: {len(st.get('duplicate_names', []))} | "
                        f"`lm_head.weight` not stored (alias): {int(not st.get('has_lm_head', True))} | | "
                        f"header padding bytes: {st.get('trailing_spaces')} | | |")
        if oc:
            rows.append(f"| {tag} ONNX | nodes unnamed: {oc['nodes_unnamed']}; names repeated: {oc['node_names_repeated']} | "
                        f"empty input slots: {oc['empty_input_slots']}; empty output slots: {oc['empty_output_slots']} | "
                        f"Identity: {oc['identity_nodes']} | initializers unread: {oc['initializers_unread']}; dead nodes: "
                        f"{oc['nodes_dead']}; graph inputs unread: {len(oc['graph_inputs_unread'])} | counter-like value names: "
                        f"{oc['values_counter_prefixed'] + oc['values_digits_only']} of {oc['values']} | control-flow nodes: "
                        f"{oc['control_flow_nodes']}; graph attributes: {oc['graph_attributes']} |")
        gc = ((t.get("gguf_census") or {}).get("f32")) if t else None
        if gc:
            rows.append(f"| {tag} GGUF (F32) | | `output.weight` not stored (alias): {int(not gc['has_output_weight'])}; empty arrays: "
                        f"{gc['kv_empty_arrays']}; empty strings: {gc['kv_empty_strings']} | | padding before data: "
                        f"{gc['padding_before_data_bytes']}; between tensors: {gc['padding_between_tensors_bytes']}; trailing: "
                        f"{gc['trailing_bytes']} bytes | | |")
        bc = ((t.get("bpk_census") or {}).get("file")) if t else None
        if bc:
            rows.append(f"| {tag} burnpack | | | | padding after CBOR: {bc.get('padding_after_cbor_bytes')}; between tensors: "
                        f"{bc.get('padding_between_tensors_bytes')} bytes | `param_id`s: {bc['param_ids']} (distinct "
                        f"{bc.get('param_ids_distinct')}) | |")
        rc = ((t.get("rust_census") or {}).get("bytes")) if t else None
        if rc:
            rows.append(f"| {tag} generated Rust | | | | `Param::uninitialized` slots: {rc['param_uninitialized']}; statements never "
                        f"used: {rc['statements_never_used']} | `ParamId::new()`: {rc['param_id_new']}; path in first line: "
                        f"{rc['first_line_names_a_path']} | `for`/`while`/`loop`: {rc['loops']} |")
        wc = t.get("wasm_census") if t else None
        if wc:
            ic = wc["instruction_census"]
            rows.append(f"| {tag} wasm module | functions unnamed: {wc['functions_unnamed']}; names repeated: {wc['names_repeated']} | | "
                        f"`nop`: {ic['nop']}; `drop`: {ic['drop']} | data segment zero bytes: {wc['data_segment_zero_bytes']} of "
                        f"{wc['data_segment_bytes']}; memory pages declared: {wc['memory_pages_declared']} | names with a build "
                        f"hash: {wc['names_with_build_hash']} of {wc['functions_named']} | `loop`: {ic['loop_empty'] + ic['loop_typed']}; "
                        f"`br_table`: {ic['br_table']}; `call_indirect` sites: {wc['indirect_call_sites']}; functions in call "
                        f"cycles: {wc['functions_in_call_cycles']} |")
    w27 = obs.get("closed_module")
    if w27:
        ic = w27["instruction_census"]
        rows.append(f"| closed module (batch 0027) | functions unnamed: {w27['functions_unnamed']}; names repeated: {w27['names_repeated']} "
                    f"| | `nop`: {ic['nop']}; `drop`: {ic['drop']} | data segment zero bytes: {w27['data_segment_zero_bytes']} of "
                    f"{w27['data_segment_bytes']}; memory pages declared: {w27['memory_pages_declared']} | names with a build hash: "
                    f"{w27['names_with_build_hash']} of {w27['functions_named']} | `loop`: {ic['loop_empty'] + ic['loop_typed']}; "
                    f"`br_table`: {ic['br_table']}; `call_indirect` sites: {w27['indirect_call_sites']}; functions in call cycles: "
                    f"{w27['functions_in_call_cycles']} |")
    return "\n".join(rows) + "\n"


# ---- the whole ---------------------------------------------------------------------------------------------------

def collect(A, tags):
    obs = {"tags": tags}
    for t in tags:
        o = obs.setdefault(t, {})
        o["revision"] = (A.text("inputs", t, "revision.txt") or "").strip() or None
        o["onnx_census"], o["onnx_skeleton"] = A.j("inputs", t, "onnx-census.json"), A.j("inputs", t, "onnx-skeleton.json")
        o["onnx_graph"], o["onnx_record_census"] = A.j("inputs", t, "onnx-graph.json"), (A.j("inputs", t, "onnx-record.json") or {}).get("census")
        o["onnx_sha256"] = ((A.text("inputs", t, "model.sha256") or "").split() or [None])[0]
        o["st_header"] = None      # from weights.json, which weights_compare.py writes from the file itself
        o["st_census"] = A.j("inputs", t, "st-census.json")
        o["st_record_census"] = (A.j("inputs", t, "st-record.json") or {}).get("census")
        o["onnxpub"] = {"census": A.j("inputs", t, "onnxpub-census.json"), "skeleton": A.j("inputs", t, "onnxpub-skeleton.json")}
        o["gguf_summary"] = A.j("gguf", t, "gguf-summary.json")
        o["bpk_census"] = {"file": A.j("burn", t, "bpk-file-census.json"), "bytes": A.j("burn", t, "bpk-bytes-census.json")}
        o["bpk_file_vs_bytes"] = A.j("burn", t, "bpk-file-vs-bytes.json")
        o["rust_census"] = {"file": A.j("burn", t, "rust-file-census.json"), "bytes": A.j("burn", t, "rust-bytes-census.json")}
        o["rust_graph"], o["onnx_to_rust"] = A.j("burn", t, "rust-graph.json"), A.j("burn", t, "onnx-to-rust.json")
        o["wasm_census"], o["wasm_graph"] = A.j("burn", t, "wasm-census.json"), A.j("burn", t, "wasm-graph.json")
        o["wasm_search"] = A.j("burn", t, "weights-search.json")
        o["wasm_validate"] = (A.text("burn", t, "validate.log") or "")
        o["wasm_imports_exports"] = (A.text("burn", t, "imports-exports.txt") or "")
        o["host_log"] = (A.text("burn", t, "host.log") or "")[-1500:]
        o["burn_outcomes"] = {k: v.get("outcome") for k, v in (A.j("burn", t, "outcomes.json") or {}).items()}
        o["inputs_outcomes"] = {k: v.get("outcome") for k, v in (A.j("inputs", t, "outcomes.json") or {}).items()}
        vocab = (A.text("inputs", t, "vocab.txt") or "").strip()
        o["vocab"] = int(vocab) if vocab.isdigit() else None
        o["logits"] = {}
    return obs


def grade_logits(A, tags, obs):
    import logit_grade
    for t in tags:
        v = obs[t].get("vocab")
        mod = A.p("burn", t, "wasm-logits")
        if not (v and os.path.isdir(mod)):
            continue
        res = {}
        for level in ("all", "none"):
            ref = A.p("inputs", t, f"ort-{level}")
            if os.path.isdir(ref):
                res[f"ort_{level}"] = logit_grade.grade(mod, ref, v)
        other = {"base": "instruct", "instruct": "base"}.get(t)
        if other and os.path.isdir(A.p("inputs", other, "ort-all")):
            res["control_other_model"] = logit_grade.grade(mod, A.p("inputs", other, "ort-all"), v)
        obs[t]["logits"] = res


def run(art_root, out, tags, burn_tags, ox):
    os.makedirs(out, exist_ok=True)
    A = Art(art_root)
    obs = collect(A, tags)
    obs["burn_tags"] = burn_tags
    wpath = os.path.join(out, "weights.json")
    weights = common.load(wpath) if os.path.exists(wpath) else None
    for t in ("base", "instruct"):
        if t in tags and weights:
            obs[t]["st_header"] = weights[t]["header"]
    grade_logits(A, tags, obs)
    graphs = []
    models = [t for t in tags if t in ("base", "instruct")]
    twice = []
    for m in models:
        first, second = [], []
        mm = models_maps(A, m, first)
        models_maps(A, m, second)
        twice.append((f"tensor and map graphs of {m}", sorted(g.sha256() for g in first) == sorted(g.sha256() for g in second)))
        graphs.extend(first)
        table = mm.pop("tensor_table")
        if table:
            with open(os.path.join(out, f"tensor-map-{m}.tsv"), "w") as f:
                f.write(table)
        obs[m].update(mm)
    if len(models) == 2:
        first, second = [], []
        obs["cross"] = cross(A, first)
        cross(A, second)
        twice.append(("cross graphs", sorted(g.sha256() for g in first) == sorted(g.sha256() for g in second)))
        graphs.extend(first)
    obs["closed_module"] = A.j("wasm27", "closed", "closed-census.json") if os.path.exists(A.p("wasm27", "closed", "closed-census.json")) \
        else None
    obs["config"] = {m: obs[m].get("gguf_config") for m in models}
    # the structure graphs the cells wrote, read as the lines they are
    struct_texts, struct_info = [], {}
    for t in tags:
        for job, name in (("inputs", "onnx-graph.nq.gz"), ("inputs", "onnxpub-graph.nq.gz"), ("gguf", "gguf-f32-kv.nq.gz"),
                          ("burn", "rust-graph.nq.gz"), ("burn", "wasm-graph.nq.gz")):
            txt = A.text(job, t, name)
            if txt:
                struct_texts.append(txt)
                struct_info[f"{t}/{name}"] = {"lines": txt.count("\n"), "blank_terms": G.blank_terms(txt)}
    w27 = A.text("wasm27", "closed", "closed-graph.nq.gz") if os.path.exists(A.p("wasm27", "closed", "closed-graph.nq.gz")) else None
    if w27:
        struct_texts.append(w27)
        struct_info["closed/closed-graph.nq.gz"] = {"lines": w27.count("\n"), "blank_terms": G.blank_terms(w27)}

    class Raw:      # a graph that is already text
        def __init__(self, text):
            self._t = text

        def text(self):
            return self._t

    all_graphs = graphs + [Raw(t) for t in struct_texts]
    texts_py = sum(G.blank_terms(g.text()) for g in all_graphs)
    ctrl_src = (graphs[0].text() if graphs else (struct_texts[0] if struct_texts else ""))
    ctrl_text = G.add_blank_node(ctrl_src) if ctrl_src else ""
    ground = {"graphs": len(all_graphs), "python_blank_statements": texts_py, "structure": struct_info,
              "determinism": [[n, bool(ok)] for n, ok in twice],
              "collapsed_triples": sum(g.collapsed() for g in graphs), "control_python_blank_statements": G.blank_terms(ctrl_text),
              "control_differs": ctrl_text != ctrl_src}
    if ox and all_graphs:
        res, nq = oxigraph_run(ox, all_graphs, out, ctrl_text)
        ground["oxigraph"] = {k: v for k, v in res.items() if k.startswith(("ground-", "control_"))}
        obs["queries"] = {"st_counterparts": histogram(res.get("st-counterparts", []), 2, 3) if isinstance(res.get("st-counterparts"), list)
                          else None,
                          "cross_shared": {r[0]: int(r[1]) for r in res.get("cross-shared", [])} if isinstance(res.get("cross-shared"), list)
                          else None,
                          "relations_by_graph": res.get("relations-by-graph"), "classes": res.get("classes-by-artifact"),
                          "triples_by_graph": res.get("counts-by-graph"),
                          "errors": {k: v["error"] for k, v in res.items() if isinstance(v, dict) and "error" in v}}
        h = hashlib.sha256()
        with open(nq, "rb") as f, gzip.open(os.path.join(out, "dataset.nq.gz"), "wb", compresslevel=6) as z:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
                z.write(chunk)
        ground["dataset_sha256"], ground["dataset_bytes"] = h.hexdigest(), os.path.getsize(nq)
        os.remove(nq)
    obs["ground"] = ground
    obs["weights"] = weights
    common.dump(os.path.join(out, "observations.json"), obs)
    with open(os.path.join(out, "census-table.md"), "w") as f:
        f.write(census_table(obs))
    print(json.dumps({"tags": tags, "graphs": len(all_graphs), "ground": {k: ground[k] for k in ("python_blank_statements",
                                                                                                "collapsed_triples")}}))
    return obs


def selftest():
    bad = 0

    def check(what, ok):
        nonlocal bad
        print(("ok   " if ok else "FAIL ") + what)
        bad += 0 if ok else 1

    a = {"k1": {"type": "UINT32", "value": 1}, "k2": {"type": "STRING", "value": "x"}, "arr": {"type": "ARRAY<STRING", "len": 3, "sha256": "aa"}}
    b = {"k1": {"type": "UINT32", "value": 2}, "k2": {"type": "STRING", "value": "x"}, "arr": {"type": "ARRAY<STRING", "len": 3, "sha256": "aa"},
         "k3": {"type": "UINT32", "value": 0}}
    d = kv_diff(a, b)
    check("kv_diff: one differing value, one key only in the second", d["differ"] == ["k1"] and d["only_instruct"] == ["k3"])
    check("line_diff: one line of three", line_diff("a\nb\nc\n", "a\nX\nc\n")["differing_lines"] == 1)
    check("line_diff of unequal lengths counts added and removed lines", line_diff("a\nb\n", "a\nb\nc\n")["differing_lines"] == 1)
    check("structure_lines drops labels and renames the model", structure_lines(
        '<urn:lmnf:b28:n:base:onnx:1> <urn:lmnf:b28:p:label> "x" <g> .\n<urn:lmnf:b28:n:base:onnx:1> <urn:lmnf:b28:p:index> "1" <g> .\n',
        "base") == ['<urn:lmnf:b28:n:M:onnx:1> <urn:lmnf:b28:p:index> "1" <g> .'])
    check("tsv_rows strips datatypes and brackets", tsv_rows('?a\t?b\n<urn:x>\t"3"^^<http://www.w3.org/2001/XMLSchema#integer>\n') == [["urn:x", "3"]])
    check("role_of", [role_of(n) for n in ("model.embed_tokens.weight", "model.layers.0.self_attn.k_proj.weight",
                                           "model.layers.0.input_layernorm.weight", "x")] == ["embedding", "key", "norm", "other"])
    return bad


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        sys.exit(1 if selftest() else 0)
    if len(sys.argv) in (5, 6):
        run(sys.argv[1], sys.argv[2], json.loads(sys.argv[3]), json.loads(sys.argv[4]), sys.argv[5] if len(sys.argv) == 6 else None)
    else:
        sys.exit(__doc__)
