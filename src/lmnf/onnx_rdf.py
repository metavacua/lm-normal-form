# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Render an ONNX model's structure as RDF.

Everything about the graph is written: nodes, the values that flow between them,
attributes, graph inputs and outputs, opset imports, and for every stored tensor
its type, shape, size and SHA-256 digest. Tensor contents are not written, apart
from small tensors (shapes, axes, scalar constants) whose values are structure.

Every resource has an IRI derived from the model IRI, so the output is the same
bytes on every run and contains no blank nodes.
"""

import hashlib
import math
from urllib.parse import quote

import pyoxigraph as ox
from onnx import AttributeProto, TensorProto

from . import tensors
from .structure import subgraphs
from .vocab import NS, RDF_TYPE, XSD

SMALL_INTEGERS = 64
SMALL_FLOATS = 16

_TYPE = ox.NamedNode(RDF_TYPE)
_INTEGER = ox.NamedNode(XSD + "integer")
_DOUBLE = ox.NamedNode(XSD + "double")
_BOOLEAN = ox.NamedNode(XSD + "boolean")


def _int(value):
    return ox.Literal(str(int(value)), datatype=_INTEGER)


def _double(value):
    value = float(value)
    if math.isnan(value):
        text = "NaN"
    elif math.isinf(value):
        text = "INF" if value > 0 else "-INF"
    else:
        text = repr(value)
    return ox.Literal(text, datatype=_DOUBLE)


def _true():
    return ox.Literal("true", datatype=_BOOLEAN)


class _Scope:
    """Names visible in a graph: its own, then those of the graphs that enclose it."""

    def __init__(self, graph_id, parent=None):
        self.graph_id = graph_id
        self.parent = parent
        self.names = {}

    def define(self, name):
        self.names.setdefault(name, f"{self.graph_id}/value/{quote(name, safe='')}")
        return self.names[name]

    def find(self, name):
        scope = self
        while scope is not None:
            if name in scope.names:
                return scope.names[name], True
            outermost, scope = scope, scope.parent
        return outermost.define(name), False


class _Emitter:
    def __init__(self, model_iri):
        self.base = model_iri
        self.graph = ox.NamedNode(model_iri)
        self.declared = set()
        self.typed = set()
        self.counts = {"nodes": 0, "values": 0, "initializers": 0, "quads": 0}

    def iri(self, suffix):
        return ox.NamedNode(f"{self.base}#{suffix}")

    def quad(self, subject, predicate, obj):
        self.counts["quads"] += 1
        predicate = _TYPE if predicate == "a" else ox.NamedNode(NS + predicate)
        if isinstance(obj, str):
            obj = ox.Literal(obj)
        return ox.Quad(subject, predicate, obj, self.graph)

    def kind(self, subject, name):
        return self.quad(subject, "a", ox.NamedNode(NS + name))

    # ── values ──

    def value(self, suffix, name):
        subject = self.iri(suffix)
        if suffix not in self.declared:
            self.declared.add(suffix)
            self.counts["values"] += 1
            yield self.kind(subject, "Value")
            yield self.quad(subject, "name", name)

    def dims(self, suffix, sizes):
        subject = self.iri(suffix)
        yield self.quad(subject, "rank", _int(len(sizes)))
        for axis, size in enumerate(sizes):
            dim = self.iri(f"{suffix}/dim/{axis}")
            yield self.quad(subject, "dim", dim)
            yield self.quad(dim, "axis", _int(axis))
            if isinstance(size, int):
                yield self.quad(dim, "size", _int(size))
            elif size is not None:
                yield self.quad(dim, "dimParam", size)

    def tensor(self, suffix, tensor):
        subject = self.iri(suffix)
        count = tensors.element_count(tensor)
        yield self.quad(subject, "elemType", TensorProto.DataType.Name(tensor.data_type))
        yield from self.dims(suffix, list(tensor.dims))
        yield self.quad(subject, "elementCount", _int(count))
        if tensors.is_external(tensor):
            yield self.quad(subject, "external", _true())
            return
        data = tensors.content(tensor)
        yield self.quad(subject, "byteLength", _int(len(data)))
        yield self.quad(subject, "sha256", hashlib.sha256(data).hexdigest())
        if tensor.data_type in tensors.INTEGER_TYPES and count <= SMALL_INTEGERS:
            yield from self.elements(suffix, tensors.elements(tensor), "int", _int)
        elif tensor.data_type in tensors.FLOAT_TYPES and count <= SMALL_FLOATS:
            yield from self.elements(suffix, tensors.elements(tensor), "float", _double)

    def elements(self, suffix, values, predicate, literal):
        subject = self.iri(suffix)
        for position, value in enumerate(values):
            element = self.iri(f"{suffix}/el/{position}")
            yield self.quad(subject, "element", element)
            yield self.quad(element, "position", _int(position))
            yield self.quad(element, predicate, literal(value))

    def typed_value(self, suffix, info):
        if suffix in self.typed:
            return
        self.typed.add(suffix)
        tensor_type = info.type.tensor_type
        if not info.type.HasField("tensor_type"):
            return
        yield self.quad(self.iri(suffix), "elemType", TensorProto.DataType.Name(tensor_type.elem_type))
        if tensor_type.HasField("shape"):
            sizes = []
            for dim in tensor_type.shape.dim:
                if dim.HasField("dim_value"):
                    sizes.append(dim.dim_value)
                elif dim.HasField("dim_param"):
                    sizes.append(dim.dim_param)
                else:
                    sizes.append(None)
            yield from self.dims(suffix, sizes)

    # ── attributes ──

    def attribute(self, node_id, attribute, scope):
        suffix = f"{node_id}/attr/{quote(attribute.name, safe='')}"
        subject = self.iri(suffix)
        yield self.quad(self.iri(node_id), "attribute", subject)
        yield self.quad(subject, "name", attribute.name)
        yield self.quad(subject, "attrType", AttributeProto.AttributeType.Name(attribute.type))
        kind = attribute.type
        if kind == AttributeProto.INT:
            yield self.quad(subject, "int", _int(attribute.i))
        elif kind == AttributeProto.FLOAT:
            yield self.quad(subject, "float", _double(attribute.f))
        elif kind == AttributeProto.STRING:
            yield self.quad(subject, "string", attribute.s.decode("utf-8", "replace"))
        elif kind == AttributeProto.INTS:
            yield self.quad(subject, "elementCount", _int(len(attribute.ints)))
            if len(attribute.ints) <= SMALL_INTEGERS:
                yield from self.elements(suffix, attribute.ints, "int", _int)
        elif kind == AttributeProto.FLOATS:
            yield self.quad(subject, "elementCount", _int(len(attribute.floats)))
            if len(attribute.floats) <= SMALL_FLOATS:
                yield from self.elements(suffix, attribute.floats, "float", _double)
        elif kind == AttributeProto.STRINGS:
            yield self.quad(subject, "elementCount", _int(len(attribute.strings)))
            texts = [s.decode("utf-8", "replace") for s in attribute.strings]
            yield from self.elements(suffix, texts, "string", ox.Literal)
        elif kind == AttributeProto.TENSOR:
            yield self.quad(subject, "tensor", self.iri(f"{suffix}/tensor"))
            yield self.kind(self.iri(f"{suffix}/tensor"), "Tensor")
            yield from self.tensor(f"{suffix}/tensor", attribute.t)

    # ── graphs ──

    def graph_(self, graph, graph_id, parent_scope):
        subject = self.iri(graph_id)
        scope = _Scope(graph_id, parent_scope)
        yield self.kind(subject, "Graph")
        yield self.quad(subject, "name", graph.name)

        stored = {t.name for t in graph.initializer}
        for info in graph.input:
            scope.define(info.name)
        for tensor in graph.initializer:
            scope.define(tensor.name)
        for node in graph.node:
            for name in node.output:
                if name:
                    scope.define(name)

        for tensor in graph.initializer:
            suffix = scope.names[tensor.name]
            self.counts["initializers"] += 1
            yield from self.value(suffix, tensor.name)
            yield self.kind(self.iri(suffix), "Initializer")
            yield from self.tensor(suffix, tensor)
            self.typed.add(suffix)
        for predicate, infos in (("graphInput", graph.input), ("graphOutput", graph.output), (None, graph.value_info)):
            for info in infos:
                suffix, _ = scope.find(info.name)
                yield from self.value(suffix, info.name)
                if predicate:
                    yield self.quad(subject, predicate, self.iri(suffix))
                if info.name not in stored:
                    yield from self.typed_value(suffix, info)

        for index, node in enumerate(graph.node):
            node_id = f"{graph_id}/node/{index}"
            resource = self.iri(node_id)
            self.counts["nodes"] += 1
            yield self.kind(resource, "Node")
            yield self.quad(resource, "inGraph", subject)
            yield self.quad(resource, "index", _int(index))
            yield self.quad(resource, "opType", node.op_type)
            yield self.quad(resource, "domain", node.domain)
            if node.name:
                yield self.quad(resource, "name", node.name)
            ports = (("input", "in", "consumes", node.input), ("output", "out", "produces", node.output))
            for predicate, side, link, names in ports:
                linked = set()
                for position, name in enumerate(names):
                    port = self.iri(f"{node_id}/{side}/{position}")
                    yield self.quad(resource, predicate, port)
                    yield self.quad(port, "position", _int(position))
                    if not name:
                        yield self.quad(port, "absent", _true())
                        continue
                    suffix, known = scope.find(name)
                    yield from self.value(suffix, name)
                    if not known:
                        yield self.quad(self.iri(suffix), "undefined", _true())
                    yield self.quad(port, "value", self.iri(suffix))
                    if suffix not in linked:
                        linked.add(suffix)
                        yield self.quad(resource, link, self.iri(suffix))
            for attribute in node.attribute:
                yield from self.attribute(node_id, attribute, scope)
            for name, nested in subgraphs(node):
                nested_id = f"{node_id}/attr/{quote(name, safe='/')}/graph"
                attribute_iri = self.iri(f"{node_id}/attr/{quote(name.split('/')[0], safe='')}")
                yield self.quad(attribute_iri, "subgraph", self.iri(nested_id))
                yield from self.graph_(nested, nested_id, scope)

    def model(self, model):
        subject = self.graph
        yield self.kind(subject, "Model")
        yield self.quad(subject, "irVersion", _int(model.ir_version))
        for predicate, text in (("producerName", model.producer_name), ("producerVersion", model.producer_version)):
            if text:
                yield self.quad(subject, predicate, text)
        for opset in model.opset_import:
            resource = self.iri(f"opset/{quote(opset.domain, safe='') or 'default'}")
            yield self.quad(subject, "opsetImport", resource)
            yield self.quad(resource, "domain", opset.domain)
            yield self.quad(resource, "version", _int(opset.version))
        for function in model.functions:
            resource = self.iri(f"function/{quote(function.domain, safe='')}/{quote(function.name, safe='')}")
            yield self.quad(subject, "function", resource)
            yield self.quad(resource, "domain", function.domain)
            yield self.quad(resource, "name", function.name)
            yield self.quad(resource, "nodeCount", _int(len(function.node)))
        yield self.quad(subject, "mainGraph", self.iri("graph"))
        yield from self.graph_(model.graph, "graph", None)


def write_nquads(model, model_iri, path):
    """Write the model's structure to `path` as N-Quads and return what was written."""
    emitter = _Emitter(model_iri)
    ox.serialize(emitter.model(model), str(path), ox.RdfFormat.N_QUADS)
    return emitter.counts
