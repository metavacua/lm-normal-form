# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The graph an ONNX file states, for batch 0028: its skeleton digest, its census of blank and empty
constructs, and the graph as ground RDF.

The skeleton is the file with its names erased: nodes in order with domain, operator type and attributes
(tensor-valued attributes by their lmnf-tensor digest), the wiring with each value name replaced by the
index of its first occurrence, and the type and shape of the inputs, outputs and initializers, but not the
contents of the initializers. Two models with the same program have the same skeleton.

Node IRIs are positional (the index in the node list), value IRIs are the index of the value's first
occurrence; the declared names are literals (`label`), never part of an IRI. A tensor's IRI is its declared
name (ground.tensor_iri), as in the records of docs/interfaces/identity.md.

Usage: onnxgraph.py MODEL.onnx TAG ARTIFACT OUTDIR     writes ARTIFACT-skeleton.json, -census.json, -graph.nq.gz
       onnxgraph.py selftest
"""

import collections
import hashlib
import json
import os
import re
import struct
import sys

import ground as G

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "0011"))

CONTROL_FLOW = ("If", "Loop", "Scan")
COUNTER_PREFIXED = re.compile(r"^onnx::[A-Za-z0-9_]+_\d+$")
DIGITS_ONLY = re.compile(r"^\d+$")


def tensor_canon(t):
    """A tensor-valued attribute by its content: dtype, shape and the lmnf-tensor digest."""
    import identity
    from onnx import numpy_helper
    try:
        a = numpy_helper.to_array(t)
        dt, d = identity.digest(a)
        return ["tensor", dt, list(a.shape), d]
    except Exception:   # a dtype numpy cannot hold: the serialized message stands for it
        return ["tensor-raw", hashlib.sha256(t.SerializeToString()).hexdigest()]


def attr_canon(a):
    from onnx import AttributeProto as A
    t = a.type
    if t == A.INT:
        return ["i", a.i]
    if t == A.FLOAT:
        return ["f", struct.pack("<f", a.f).hex()]
    if t == A.STRING:
        return ["s", a.s.hex()]
    if t == A.INTS:
        return ["is", list(a.ints)]
    if t == A.FLOATS:
        return ["fs", [struct.pack("<f", x).hex() for x in a.floats]]
    if t == A.STRINGS:
        return ["ss", [x.hex() for x in a.strings]]
    if t == A.TENSOR:
        return tensor_canon(a.t)
    if t == A.TENSORS:
        return ["ts", [tensor_canon(x) for x in a.tensors]]
    if t in (A.GRAPH, A.GRAPHS):
        return ["graph"]
    return ["other", int(t), a.SerializeToString().hex()]


def value_canon(v, alpha):
    tt = v.type.tensor_type
    dims = [(d.dim_value if d.HasField("dim_value") else d.dim_param) for d in tt.shape.dim]
    return [alpha(v.name), int(tt.elem_type), dims]


class Alpha:
    """Value names replaced by the index of their first occurrence; the empty name (an absent optional) stays empty."""

    def __init__(self):
        self.ix = {}

    def __call__(self, name):
        if name == "":
            return ""
        if name not in self.ix:
            self.ix[name] = len(self.ix)
        return self.ix[name]


def seed_alpha(g):
    """The one order in which names are numbered: graph inputs, initializers, then node inputs and outputs."""
    al = Alpha()
    for v in g.input:
        al(v.name)
    for t in g.initializer:
        al(t.name)
    for n in g.node:
        for x in n.input:
            al(x)
        for x in n.output:
            al(x)
    return al


def skeleton(model):
    g = model.graph
    al = seed_alpha(g)
    nodes = []
    for n in g.node:
        attrs = sorted(([a.name, attr_canon(a)] for a in n.attribute), key=lambda x: x[0])
        nodes.append([n.domain, n.op_type, attrs, [al(x) for x in n.input], [al(x) for x in n.output]])
    obj = {
        "opsets": sorted([o.domain, o.version] for o in model.opset_import),
        "inputs": [value_canon(v, al) for v in g.input],
        "outputs": [value_canon(v, al) for v in g.output],
        "initializers": [[al(t.name), int(t.data_type), list(t.dims)] for t in g.initializer],
        "nodes": nodes,
    }
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(blob).hexdigest(), len(blob), obj


def acyclic(g):
    """Kahn's algorithm on the node dependency graph: (acyclic, nodes used before the node that makes them)."""
    produced_by = {}
    for i, n in enumerate(g.node):
        for o in n.output:
            if o:
                produced_by.setdefault(o, i)
    deps = [set() for _ in g.node]
    out_edges = collections.defaultdict(list)
    early = 0
    for i, n in enumerate(g.node):
        for x in n.input:
            j = produced_by.get(x)
            if j is not None and j != i:
                deps[i].add(j)
                out_edges[j].append(i)
                early += j > i
            elif j == i:
                early += 1
    indeg = [len(d) for d in deps]
    ready = [i for i, d in enumerate(indeg) if d == 0]
    done = 0
    while ready:
        i = ready.pop()
        done += 1
        for k in out_edges[i]:
            indeg[k] -= 1
            if indeg[k] == 0:
                ready.append(k)
    return done == len(g.node), early


def census(model, path=None):
    g = model.graph
    nodes = list(g.node)
    names = [n.name for n in nodes]
    nonempty = [x for x in names if x]
    consumed = collections.Counter(x for n in nodes for x in n.input if x)
    outs = {v.name for v in g.output}
    init_names = [t.name for t in g.initializer]
    producers = collections.Counter(o for n in nodes for o in n.output if o)
    live = set(outs)
    prod = {}
    for i, n in enumerate(nodes):
        for o in n.output:
            if o:
                prod[o] = i
    stack = list(outs)
    live_nodes = set()
    while stack:
        v = stack.pop()
        i = prod.get(v)
        if i is not None and i not in live_nodes:
            live_nodes.add(i)
            stack.extend(x for x in nodes[i].input if x)
    ok, early = acyclic(g)
    all_values = set(x for n in nodes for x in list(n.input) + list(n.output) if x) | set(init_names)
    all_values |= {v.name for v in g.input} | outs
    from onnx import AttributeProto as A
    c = {
        "file_bytes": os.path.getsize(path) if path else None,
        "ir_version": model.ir_version,
        "opsets": sorted([o.domain, o.version] for o in model.opset_import),
        "producer": [model.producer_name, model.producer_version],
        "nodes": len(nodes),
        "nodes_unnamed": len(names) - len(nonempty),
        "node_names_repeated": len(nonempty) - len(set(nonempty)),
        "initializers": len(init_names),
        "initializer_names_repeated": len(init_names) - len(set(init_names)),
        "initializers_unread": sum(1 for x in init_names if x not in consumed and x not in outs),
        "initializers_zero_size": sum(1 for t in g.initializer if 0 in list(t.dims)),
        "initializers_external": sum(1 for t in g.initializer if t.data_location == 1),
        "graph_inputs": [v.name for v in g.input],
        "graph_inputs_unread": [v.name for v in g.input if v.name not in consumed],
        "graph_outputs": [v.name for v in g.output],
        "nodes_dead": len(nodes) - len(live_nodes),
        "values": len(all_values),
        "values_counter_prefixed": sum(1 for x in all_values if COUNTER_PREFIXED.match(x)),
        "values_digits_only": sum(1 for x in all_values if DIGITS_ONLY.match(x)),
        "values_path_like": sum(1 for x in all_values if x.startswith("/")),
        "value_names_produced_twice": sum(1 for k, v in producers.items() if v > 1),
        "empty_input_slots": sum(1 for n in nodes for x in n.input if x == ""),
        "empty_output_slots": sum(1 for n in nodes for x in n.output if x == ""),
        "nodes_with_empty_input": sum(1 for n in nodes if "" in list(n.input)),
        "identity_nodes": sum(1 for n in nodes if n.op_type == "Identity"),
        "constant_nodes": sum(1 for n in nodes if n.op_type == "Constant"),
        "control_flow_nodes": sum(1 for n in nodes if n.op_type in CONTROL_FLOW),
        "graph_attributes": sum(1 for n in nodes for a in n.attribute if a.type in (A.GRAPH, A.GRAPHS)),
        "functions": len(model.functions),
        "metadata_props": len(model.metadata_props),
        "acyclic": ok,
        "inputs_used_before_made": early,
        "operators": {f"{d}/{o}": k for (d, o), k in collections.Counter(
            (n.domain, n.op_type) for n in nodes).most_common()},
    }
    return c


def build_graph(model, tag, artifact):
    """The graph as ground RDF: positional nodes and values, one predicate per input and output position."""
    g = model.graph
    gr = G.Graph(tag, artifact)
    art = G.iri("a", tag, artifact)
    gr.typed(art, "Artifact")
    gr.add(art, G.pred("format"), G.lit("onnx"))
    al = seed_alpha(g)
    seen = set()

    def value(name):
        k = al(name)
        v = G.iri("v", tag, artifact, k)
        if k not in seen:
            seen.add(k)
            gr.typed(v, "Value")
            gr.add(v, G.pred("inArtifact"), art)
            gr.add(v, G.pred("label"), G.lit(name))
        return v

    for k, v in enumerate(g.input):
        gr.add(value(v.name), G.pred("graphInput"), G.num(k))
    for k, v in enumerate(g.output):
        gr.add(value(v.name), G.pred("graphOutput"), G.num(k))
    for t in g.initializer:
        gr.add(value(t.name), G.pred("holds"), G.tensor_iri(tag, artifact, t.name))
    for i, n in enumerate(g.node):
        node = G.iri("n", tag, artifact, i)
        gr.typed(node, "Node")
        gr.add(node, G.pred("inArtifact"), art)
        gr.add(node, G.pred("index"), G.num(i))
        gr.add(node, G.pred("opType"), G.lit(n.op_type))
        if n.domain:
            gr.add(node, G.pred("domain"), G.lit(n.domain))
        if n.name:
            gr.add(node, G.pred("label"), G.lit(n.name))
        for k, x in enumerate(n.input):
            if x:
                gr.add(node, G.pred(f"in{k}"), value(x))
            else:
                gr.add(node, G.pred(f"absentIn{k}"), G.lit("true"))
        for k, x in enumerate(n.output):
            if x:
                gr.add(node, G.pred(f"out{k}"), value(x))
            else:
                gr.add(node, G.pred(f"absentOut{k}"), G.lit("true"))
    return gr


def run(path, tag, artifact, outdir):
    import onnx
    model = onnx.load(path, load_external_data=False)
    os.makedirs(outdir, exist_ok=True)
    sha, size, _ = skeleton(model)
    first = model.graph.node[0]
    was = first.op_type
    first.op_type = was + "_changed"          # C3 on the real graph: one operator type of one node changed
    changed = skeleton(model)[0]
    first.op_type = was
    sk = {"skeleton_sha256": sha, "skeleton_bytes": size, "nodes": len(model.graph.node),
          "control_changed_operator_differs": changed != sha, "control_restored_equals": skeleton(model)[0] == sha}
    with open(os.path.join(outdir, f"{artifact}-skeleton.json"), "w") as f:
        json.dump(sk, f, indent=1, sort_keys=True)
    c = census(model, path)
    with open(os.path.join(outdir, f"{artifact}-census.json"), "w") as f:
        json.dump(c, f, indent=1, sort_keys=True)
    gr = build_graph(model, tag, artifact)
    G.write(os.path.join(outdir, f"{artifact}-graph.nq.gz"), gr.text())
    info = {"triples": len(gr.triples), "collapsed": gr.collapsed(), "graph_sha256": gr.sha256(),
            "deterministic": gr.sha256() == build_graph(model, tag, artifact).sha256(), "nodes": len(model.graph.node),
            "blank_terms": G.blank_terms(gr.text())}
    with open(os.path.join(outdir, f"{artifact}-graph.json"), "w") as f:
        json.dump(info, f, indent=1, sort_keys=True)
    print(json.dumps({**sk, **info}, sort_keys=True))
    return sk, c, info


def selftest():
    import numpy as np
    import onnx
    from onnx import TensorProto, helper, numpy_helper
    bad = 0

    def check(what, ok):
        nonlocal bad
        print(("ok   " if ok else "FAIL ") + what)
        bad += 0 if ok else 1

    def model(names=("a", "b", "c"), op2="Relu", w=1.0, eps=1e-5, with_empty=True):
        w1 = numpy_helper.from_array(np.full((2, 2), w, dtype=np.float32), names[0] + "_w")
        c1 = helper.make_node("Constant", [], [names[1] + "_eps"], value=numpy_helper.from_array(
            np.array(eps, dtype=np.float32)))
        n1 = helper.make_node("MatMul", ["x", names[0] + "_w"], [names[1]], name="")
        n2 = helper.make_node(op2, [names[1]], [names[2]], name="relu")
        ins = [names[2], ""] if with_empty else [names[2]]
        n3 = helper.make_node("Clip", ins + [names[1] + "_eps"], ["y"], name="relu")  # a repeated name on purpose
        gg = helper.make_graph([c1, n1, n2, n3], "g", [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
                               [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])], [w1])
        return helper.make_model(gg, opset_imports=[helper.make_opsetid("", 13)])

    m1, m2, m3 = model(), model(w=2.0), model(names=("p", "q", "r"), w=2.0)
    s1 = skeleton(m1)[0]
    check("the skeleton ignores the contents of initializers", skeleton(m2)[0] == s1)
    check("the skeleton ignores every name", skeleton(m3)[0] == s1)
    check("C3: the skeleton changes when one operator type changes", skeleton(model(op2="Sigmoid"))[0] != s1)
    check("the skeleton sees the value of a Constant", skeleton(model(eps=2e-5))[0] != s1)
    c = census(m1)
    check("census: two unnamed nodes, one repeated name", c["nodes_unnamed"] == 2 and c["node_names_repeated"] == 1)
    check("census: one empty input slot", c["empty_input_slots"] == 1 and c["nodes_with_empty_input"] == 1)
    check("census: acyclic, no control flow, one constant", c["acyclic"] and c["control_flow_nodes"] == 0
          and c["constant_nodes"] == 1)
    gr, gr2 = build_graph(m1, "t", "onnx"), build_graph(m2, "t", "onnx")
    t = gr.text()
    check("the graph has no blank node", G.blank_terms(t) == 0)
    check("the graph is deterministic", gr.sha256() == build_graph(m1, "t", "onnx").sha256())

    def structure(g_):
        return sorted(line for line in g_.lines() if ":p:label>" not in line)

    check("two models with one program give one graph apart from the labels", structure(gr) == structure(gr2))
    check("an absent optional is a triple, not a node", ":p:absentIn1>" in t)
    return bad


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        sys.exit(1 if selftest() else 0)
    if len(sys.argv) != 5:
        sys.exit(__doc__)
    run(*sys.argv[1:5])
