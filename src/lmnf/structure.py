# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A report read directly from an ONNX model, independent of the RDF rendering.

It exists so that the RDF graph has something to be checked against.
"""

import collections

from onnx import AttributeProto, TensorProto

from . import tensors

GRAPH_ATTRIBUTES = (AttributeProto.GRAPH, AttributeProto.GRAPHS)


def subgraphs(node):
    for attribute in node.attribute:
        if attribute.type == AttributeProto.GRAPH:
            yield attribute.name, attribute.g
        elif attribute.type == AttributeProto.GRAPHS:
            for position, graph in enumerate(attribute.graphs):
                yield f"{attribute.name}/{position}", graph


def value_info(info):
    tensor_type = info.type.tensor_type
    dims = []
    for dim in tensor_type.shape.dim:
        if dim.HasField("dim_value"):
            dims.append(dim.dim_value)
        elif dim.HasField("dim_param"):
            dims.append(dim.dim_param)
        else:
            dims.append(None)
    return {"name": info.name, "type": TensorProto.DataType.Name(tensor_type.elem_type), "dims": dims}


def function_calls_are_acyclic(model):
    """Model-local functions may call one another; a cycle would be recursion."""
    names = {(f.domain, f.name) for f in model.functions}
    calls = {
        (f.domain, f.name): {(n.domain, n.op_type) for n in f.node if (n.domain, n.op_type) in names}
        for f in model.functions
    }
    state = {}

    def visit(key):
        if state.get(key) == "done":
            return True
        if state.get(key) == "open":
            return False
        state[key] = "open"
        ok = all(visit(callee) for callee in sorted(calls[key]))
        state[key] = "done"
        return ok

    return all(visit(key) for key in sorted(calls))


def report(model):
    histogram = collections.Counter()
    types = collections.Counter()
    totals = collections.Counter()
    control_flow = []
    undefined = set()

    def walk(graph, graph_id, outer):
        local = {i.name for i in graph.input} | {t.name for t in graph.initializer}
        produced_at = {out: index for index, node in enumerate(graph.node) for out in node.output if out}
        for tensor in graph.initializer:
            totals["initializers"] += 1
            totals["elements"] += tensors.element_count(tensor)
            types[TensorProto.DataType.Name(tensor.data_type)] += 1
            if tensors.has_bytes(tensor):
                totals["bytes"] += len(tensors.content(tensor))
        for index, node in enumerate(graph.node):
            totals["nodes"] += 1
            histogram[(node.domain, node.op_type)] += 1
            for name in node.input:
                if not name:
                    continue
                if name in produced_at:
                    totals["order_violations"] += produced_at[name] >= index
                elif name not in local and name not in outer:
                    undefined.add(name)
            nested = list(subgraphs(node))
            if nested:
                control_flow.append({"graph": graph_id, "index": index, "op": node.op_type})
            for attribute, graph_ in nested:
                walk(graph_, f"{graph_id}/node/{index}/attr/{attribute}/graph", outer | local | set(produced_at))

    walk(model.graph, "graph", set())
    return {
        "ir_version": model.ir_version,
        "producer": f"{model.producer_name} {model.producer_version}".strip(),
        "opsets": {o.domain: o.version for o in model.opset_import},
        "functions": len(model.functions),
        "functions_acyclic": function_calls_are_acyclic(model),
        "nodes": totals["nodes"],
        "op_histogram": dict(histogram),
        "control_flow": control_flow,
        "order_violations": totals["order_violations"],
        "undefined_inputs": len(undefined),
        "initializers": {
            "count": totals["initializers"],
            "elements": totals["elements"],
            "bytes": totals["bytes"],
        },
        "initializer_types": dict(types),
        "inputs": [value_info(i) for i in model.graph.input],
        "outputs": [value_info(o) for o in model.graph.output],
    }
