# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Burn for batch 0028: the burnpack of each load strategy as an identity record and a census, the comparison of the
`param_id`s of the two, the generated Rust as a graph of fields and statements, and the map from ONNX nodes to
what burn-onnx generated for them.

The generated source is read as text, with the structure that burn-onnx writes: `pub struct SubmoduleN` with
one field per parameter or submodule, `impl SubmoduleN` with a `new` and a `forward`, and in each `forward` one
top-level `let` per node at eight spaces of indentation (a tuple pattern may span lines). A statement uses
an earlier statement of its scope when the earlier name occurs in its text; it uses a field when `self.FIELD` does.

The map from ONNX nodes follows the naming rule of onnx-ir that batches/0010/align.py states: nodes in file
order, named by operator type in lower case and the ordinal among nodes of that type; a MatMul whose second
input is neither a node's output nor a graph input becomes a Linear and takes the name `linear<ordinal among
Linears>`; the first output of a node is `<name>_out1`.

Usage: burn_tools.py run TAG ONNX FILE.bpk FILE.rs BYTES.bpk BYTES.rs OUTDIR
       burn_tools.py selftest [SOURCE.rs]
"""

import collections
import json
import os
import re
import sys

import common
import ground as G
from common import identity

IDENT = re.compile(r"\b[a-z_][a-z0-9_]*\b")
FIELD_USE = re.compile(r"self\.([a-z_][a-z0-9_]*)")
STRUCT = re.compile(r"^pub struct (\w+) \{$", re.M)
IMPL = re.compile(r"^impl (\w+) \{$", re.M)
FIELD = re.compile(r"^ {4}(\w+): (.+),$", re.M)
LET = re.compile(r"^ {8}let (?:mut )?(\w+)\b", re.M)
LET_TUPLE_OPEN = re.compile(r"^ {8}let \($", re.M)


def parse_rust(text):
    """Fields by struct, and the top-level statements of each `forward`, with the names they use."""
    structs = {}
    for m in STRUCT.finditer(text):
        end = text.index("\n}\n", m.end())
        fields = []
        for f in FIELD.finditer(text[m.end():end]):
            fields.append((f.group(1), f.group(2)))
        structs[m.group(1)] = fields
    impls = [(m.start(), m.group(1)) for m in IMPL.finditer(text)]

    def scope_at(pos):
        s = None
        for p, n in impls:
            if p <= pos:
                s = n
        return s

    # top-level statements: (position, [names defined])
    marks = []
    for m in LET.finditer(text):
        marks.append((m.start(), [m.group(1)]))
    for m in LET_TUPLE_OPEN.finditer(text):
        close = text.index("\n        ) =", m.end())
        names = [x.strip().rstrip(",") for x in text[m.end():close].split("\n") if x.strip()]
        marks.append((m.start(), names))
    marks.sort()
    fn_forward = [m.start() for m in re.finditer(r"^ {4}pub fn forward\(", text, re.M)]
    stmts = []
    for i, (pos, names) in enumerate(marks):
        # a statement belongs to a forward when the nearest preceding `pub fn forward(` is after the nearest `pub fn new(`
        prev_fwd = max((p for p in fn_forward if p <= pos), default=None)
        prev_new = max((m.start() for m in re.finditer(r"^ {4}pub fn new\(", text, re.M) if m.start() <= pos), default=-1)
        if prev_fwd is None or prev_new > prev_fwd:
            continue
        nxt = marks[i + 1][0] if i + 1 < len(marks) else len(text)
        end_fn = text.find("\n    }\n", pos)
        nxt = min(nxt, end_fn if end_fn != -1 else nxt)
        stmts.append({"scope": scope_at(pos), "names": names, "pos": pos, "body": text[pos:nxt]})
    return structs, stmts


def rust_census(text, structs, stmts):
    n_fields = sum(len(v) for v in structs.values())
    defined = collections.defaultdict(set)
    used = collections.defaultdict(set)
    for s in stmts:
        for n in s["names"]:
            defined[s["scope"]].add(n)
    for s in stmts:
        toks = set(IDENT.findall(s["body"])) - set(s["names"])
        used[s["scope"]] |= toks
    unused = sum(1 for sc, names in defined.items() for n in names if n not in used[sc])
    first = text.split("\n", 1)[0]
    return {
        "bytes": len(text.encode()), "lines": text.count("\n"), "structs": len(structs), "fields": n_fields,
        "fields_linear": sum(1 for v in structs.values() for _, t in v if t == "Linear"),
        "fields_param": sum(1 for v in structs.values() for _, t in v if t.startswith("burn::module::Param<")),
        "fields_submodule": sum(1 for v in structs.values() for _, t in v if re.fullmatch(r"Submodule\d+", t)),
        "param_uninitialized": text.count("Param::uninitialized("), "param_id_new": text.count("ParamId::new()"),
        "top_level_statements": sum(len(s["names"]) for s in stmts),
        "statements_never_used": unused, "from_file": text.count("fn from_file"), "from_bytes": text.count("fn from_bytes"),
        "default_impl": text.count("impl Default for Model"), "std_mentions": len(re.findall(r"std::|extern crate std", text)),
        "first_line": first, "first_line_names_a_path": bool(re.search(r'"[^"]*/[^"]*"', first)),
        "loops": len(re.findall(r"\bfor \w+ in\b|\bwhile\b|\bloop \{", text)),
    }


def rust_graph(text, structs, stmts, tag, artifact):
    gr = G.Graph(tag, artifact)
    art = G.iri("a", tag, artifact)
    gr.typed(art, "Artifact")
    gr.add(art, G.pred("format"), G.lit("rust"))
    field_iri = {}
    for sc, fields in structs.items():
        for name, ty in fields:
            f = G.iri("f", tag, artifact, sc, name)
            field_iri[(sc, name)] = f
            gr.typed(f, "Field")
            gr.add(f, G.pred("inArtifact"), art)
            gr.add(f, G.pred("scope"), G.lit(sc))
            gr.add(f, G.pred("label"), G.lit(name))
            gr.add(f, G.pred("fieldType"), G.lit(ty))
    stmt_iri = {}
    for i, s in enumerate(stmts):
        for name in s["names"]:
            n = G.iri("s", tag, artifact, s["scope"], name)
            stmt_iri.setdefault((s["scope"], name), n)
            gr.typed(n, "Statement")
            gr.add(n, G.pred("inArtifact"), art)
            gr.add(n, G.pred("scope"), G.lit(s["scope"]))
            gr.add(n, G.pred("label"), G.lit(name))
            gr.add(n, G.pred("index"), G.num(i))
    earlier = collections.defaultdict(dict)
    for s in stmts:
        sc = s["scope"]
        toks = set(IDENT.findall(s["body"])) - set(s["names"])
        for name in s["names"]:
            n = stmt_iri[(sc, name)]
            for t in sorted(toks):
                if t in earlier[sc]:
                    gr.add(n, G.pred("usesStatement"), earlier[sc][t])
            for fld in sorted(set(FIELD_USE.findall(s["body"]))):
                if (sc, fld) in field_iri:
                    gr.add(n, G.pred("usesField"), field_iri[(sc, fld)])
        for name in s["names"]:
            earlier[sc][name] = stmt_iri[(sc, name)]
    return gr, stmt_iri, field_iri


def onnx_nodes(path):
    import onnx
    m = onnx.load(path, load_external_data=False)
    g = m.graph
    return [(i, n.op_type, list(n.input), list(n.output)) for i, n in enumerate(g.node)], [v.name for v in g.input]


def map_nodes(nodes, graph_inputs, structs, stmts, tag, artifact_onnx, artifact_rust, gr, stmt_iri, field_iri):
    """Each ONNX node to the statement or field that burn-onnx generated for it, or `folded`."""
    produced = {o for _, _, _, outs in nodes for o in outs}
    fields = {(sc, name): ty for sc, fs in structs.items() for name, ty in fs}
    by_field_name = collections.defaultdict(list)
    for (sc, name) in fields:
        by_field_name[name].append(sc)
    by_stmt_name = collections.defaultdict(list)
    for s in stmts:
        for n in s["names"]:
            by_stmt_name[n].append(s["scope"])
    ordinal, linears = collections.Counter(), 0
    cats = collections.Counter()
    by_op = collections.defaultdict(collections.Counter)
    rows = []
    for i, op, ins, outs in nodes:
        ordinal[op] += 1
        base = f"{op.lower()}{ordinal[op]}"
        is_linear = op == "MatMul" and len(ins) > 1 and ins[1] not in produced and ins[1] not in graph_inputs
        if is_linear:
            linears += 1
            base = f"linear{linears}"
        target = None
        if is_linear or op == "Constant":
            if base in by_field_name:
                sc = by_field_name[base][0]
                target, cat = field_iri[(sc, base)], "linear" if is_linear else "param"
            else:
                cat = "folded"
        else:
            nm = f"{base}_out1"
            if nm in by_stmt_name:
                sc = by_stmt_name[nm][0]
                target, cat = stmt_iri[(sc, nm)], "statement"
            else:
                cat = "folded"
        cats[cat] += 1
        by_op[op][cat] += 1
        rows.append([i, op, base, cat])
        if target:
            gr.add(G.iri("n", tag, artifact_onnx, i), G.pred("generatedAs"), target)
    return {"categories": dict(cats), "nodes": len(nodes),
            "by_operator": {op: dict(c) for op, c in sorted(by_op.items())}}, rows


def bpk_extra(path, meta, census):
    """The empty space of a burnpack: zero bytes between the CBOR and the data, and between the tensors."""
    spans = sorted(tuple(d["data_offsets"]) for d in meta["tensors"].values())
    gaps = sum(max(0, b[0] - a[1]) for a, b in zip(spans, spans[1:]))
    return {"padding_after_cbor_bytes": census["data_start"] - (10 + census["metadata_size"]),
            "padding_between_tensors_bytes": gaps, "param_ids_distinct": len({d.get("param_id") for d in meta["tensors"].values()})}


def run(tag, onnx_path, file_bpk, file_rs, bytes_bpk, bytes_rs, outdir):
    os.makedirs(outdir, exist_ok=True)
    out = {"tag": tag}
    for label, bpk in (("file", file_bpk), ("bytes", bytes_bpk)):
        ents, c, meta = identity.read_bpk(bpk)
        c.update(bpk_extra(bpk, meta, c))
        common.dump(os.path.join(outdir, f"bpk-{label}-census.json"), c)
        if label == "file":
            common.dump(os.path.join(outdir, "bpk-file-record.json"), identity.record("burnpack", ents, os.path.basename(bpk)))
            out["file_tensors"] = len(ents)
    common.dump(os.path.join(outdir, "bpk-file-vs-bytes.json"), identity.bpkdiff(file_bpk, bytes_bpk))
    nodes, gin = onnx_nodes(onnx_path)
    for label, rs in (("file", file_rs), ("bytes", bytes_rs)):
        text = open(rs, encoding="utf-8").read()
        structs, stmts = parse_rust(text)
        common.dump(os.path.join(outdir, f"rust-{label}-census.json"), rust_census(text, structs, stmts))
        if label == "bytes":
            gr, stmt_iri, field_iri = rust_graph(text, structs, stmts, tag, "rust")
            deterministic = gr.sha256() == rust_graph(text, structs, stmts, tag, "rust")[0].sha256()
            summary, rows = map_nodes(nodes, gin, structs, stmts, tag, "onnx", "rust", gr, stmt_iri, field_iri)
            common.dump(os.path.join(outdir, "onnx-to-rust.json"), summary)
            with open(os.path.join(outdir, "onnx-to-rust.tsv"), "w") as f:
                f.write("node\toperator\tgenerated_name\tcategory\n" + "".join("\t".join(map(str, r)) + "\n" for r in rows))
            G.write(os.path.join(outdir, "rust-graph.nq.gz"), gr.text())
            common.dump(os.path.join(outdir, "rust-graph.json"), {
                "triples": len(gr.triples), "collapsed": gr.collapsed(), "graph_sha256": gr.sha256(),
                "deterministic": deterministic, "blank_terms": G.blank_terms(gr.text()),
                "statements": sum(len(s["names"]) for s in stmts), "fields": sum(len(v) for v in structs.values())})
            out["map"] = summary["categories"]
    common.dump(os.path.join(outdir, "burn-summary.json"), out)
    print(json.dumps(out, sort_keys=True))


def selftest(source=None):
    bad = 0

    def check(what, ok):
        nonlocal bad
        print(("ok   " if ok else "FAIL ") + what)
        bad += 0 if ok else 1

    mini = '''// Generated from ONNX "x/model.onnx" by burn-onnx
#[derive(Module, Debug)]
pub struct Submodule1 {
    constant1: burn::module::Param<Tensor<1>>,
    linear1: Linear,
    #[module(skip)]
    device: Device,
}
impl Submodule1 {
    pub fn new(device: &Device) -> Self {
        let constant1: burn::module::Param<Tensor<1>> = burn::module::Param::uninitialized(
            burn::module::ParamId::new(),
        );
        Self { constant1 }
    }
    pub fn forward(&self, x: Tensor<2>) -> (Tensor<2>, Tensor<2>) {
        let linear1_out1 = self.linear1.forward(x);
        let relu1_out1 = burn::tensor::activation::relu(linear1_out1.clone());
        let unused1_out1 = relu1_out1.clone();
        (relu1_out1, linear1_out1)
    }
}
pub struct Model {
    submodule1: Submodule1,
}
impl Model {
    pub fn forward(&self, x: Tensor<2>) -> Tensor<2> {
        let (
            relu1_out1,
            linear1_out1,
        ) = self.submodule1.forward(x);
        relu1_out1
    }
}
'''
    structs, stmts = parse_rust(mini)
    check("two structs, three fields in the first", set(structs) == {"Submodule1", "Model"} and len(structs["Submodule1"]) == 3)
    names = [(s["scope"], tuple(s["names"])) for s in stmts]
    check("statements of both forwards, the tuple pattern read, none of `new`",
          names == [("Submodule1", ("linear1_out1",)), ("Submodule1", ("relu1_out1",)), ("Submodule1", ("unused1_out1",)),
                    ("Model", ("relu1_out1", "linear1_out1"))])
    c = rust_census(mini, structs, stmts)
    check("census: one uninitialized param, one ParamId::new, one never used", c["param_uninitialized"] == 1 and
          c["param_id_new"] == 1 and c["statements_never_used"] >= 1 and c["first_line_names_a_path"])
    gr, si, fi = rust_graph(mini, structs, stmts, "t", "rust")
    check("graph: relu uses linear; linear uses its field", any("usesStatement" in line and "relu1_out1" in line for line in gr.lines())
          and any("usesField" in line for line in gr.lines()))
    check("graph has no blank node", G.blank_terms(gr.text()) == 0)
    nodes = [(0, "Constant", [], ["c0"]), (1, "MatMul", ["x", "w"], ["m0"]), (2, "Relu", ["m0"], ["r0"]), (3, "Foo", ["r0"], ["f0"])]
    summ, rows = map_nodes(nodes, ["x"], structs, stmts, "t", "onnx", "rust", gr, si, fi)
    check("map: constant to its param, matmul to its linear, relu to its statement, Foo folded",
          [r[3] for r in rows] == ["param", "linear", "statement", "folded"])
    if source:
        text = open(source, encoding="utf-8").read()
        st, sm = parse_rust(text)
        c = rust_census(text, st, sm)
        print(json.dumps(c, sort_keys=True))
        check("the real source: submodules, statements and params found", c["structs"] >= 2 and c["top_level_statements"] > 100
              and c["param_uninitialized"] > 0)
    return bad


if __name__ == "__main__":
    if sys.argv[1:2] == ["selftest"]:
        sys.exit(1 if selftest(*sys.argv[2:3]) else 0)
    if len(sys.argv) == 9 and sys.argv[1] == "run":
        run(*sys.argv[2:])
    else:
        sys.exit(__doc__)
