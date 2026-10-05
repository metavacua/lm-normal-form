# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Load a structure graph into Oxigraph and ask it questions in SPARQL."""

import os
from pathlib import Path

import pyoxigraph as ox

from .vocab import XSD

# The query files live at the repository root; LMNF_QUERIES points somewhere else.
QUERIES = Path(os.environ.get("LMNF_QUERIES") or Path(__file__).resolve().parents[2] / "queries")

_INTEGERS = {XSD + name for name in ("integer", "int", "long", "nonNegativeInteger")}
_REALS = {XSD + name for name in ("double", "decimal", "float")}


def load(path):
    store = ox.Store()
    store.bulk_load(path=str(path), format=ox.RdfFormat.N_QUADS)
    return store


def text(name):
    return (QUERIES / f"{name}.rq").read_text(encoding="utf-8")


def _python(term):
    if term is None:
        return None
    if isinstance(term, ox.Literal):
        datatype = term.datatype.value
        if datatype in _INTEGERS:
            return int(term.value)
        if datatype in _REALS:
            return float(term.value)
        if datatype == XSD + "boolean":
            return term.value == "true"
    return term.value


def select(store, query):
    """Run a SELECT query; every row is a dict of plain Python values, unbound as None."""
    result = store.query(query, use_default_graph_as_union=True)
    names = [variable.value for variable in result.variables]
    return [{name: _python(solution[name]) for name in names} for solution in result]


def rows(store, name):
    return select(store, text(name))


def ask(store, name):
    return bool(store.query(text(name), use_default_graph_as_union=True))


def stored_float_shapes(store):
    """The shape of every floating-point tensor the graph stores, read back from the graph."""
    shapes = {}
    for row in rows(store, "stored_float_tensors"):
        shape = shapes.setdefault(row["tensor"], [])
        if row["axis"] is not None:
            shape.append(row["size"])
    return [tuple(shape) for shape in shapes.values()]
