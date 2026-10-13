# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Ground RDF for batch 0028: IRIs, N-Triples and N-Quads, and the checks of "The rule for the graphs"
in docs/batches/0028.md. Standard library only.

A node is an IRI `urn:lmnf:b28:KIND:PART:PART...` whose parts are declared in the artifact (a tensor's name)
or positional (an index). There is no blank-node constructor here: the writer has no way to emit `_:`.
`blank_terms` is a second, independent reading of the written text, used to cross-check Oxigraph's `isBlank`
and as the instrument that the control of P15 turns on.

Usage: ground.py selftest
"""

import gzip
import hashlib
import re
import sys
from urllib.parse import quote

NS = "urn:lmnf:b28:"
RDF_TYPE = "<http://www.w3.org/1999/02/22-rdf-syntax-ns#type>"
XSD_INTEGER = "http://www.w3.org/2001/XMLSchema#integer"

_TERM = re.compile(
    r'<[^<>"{}|^`\\\s]*>'                                 # an IRI reference
    r'|"(?:[^"\\]|\\.)*"(?:\^\^<[^<>\s]*>|@[A-Za-z0-9-]+)?'  # a literal, with its datatype or language tag
    r"|_:\S+"                                              # a blank node label
    r"|\."                                                 # the end of the statement
)


def enc(part):
    """A part of an IRI: everything but the unreserved characters and the slash is percent-encoded, the colon too,
    so that two different tuples of parts cannot give one IRI."""
    return quote(str(part), safe="/")


def iri(kind, *parts):
    return "<" + NS + kind + ":" + ":".join(enc(p) for p in parts) + ">"


def tensor_iri(tag, artifact, name):
    """A tensor of an artifact, by the name that the artifact declares for it."""
    return iri("t", tag, artifact, name)


def pred(name):
    return "<" + NS + "p:" + name + ">"


def cls(name):
    return "<" + NS + "c:" + name + ">"


def lit(text):
    out = []
    for ch in str(text):
        o = ord(ch)
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\r":
            out.append("\\r")
        elif ch == "\t":
            out.append("\\t")
        elif o < 0x20 or o == 0x7F:
            out.append("\\u%04X" % o)
        else:
            out.append(ch)
    return '"' + "".join(out) + '"'


def num(n):
    return '"%d"^^<%s>' % (int(n), XSD_INTEGER)


class Graph:
    """A set of ground triples in one named graph. `added` counts every call of `add`; a triple added twice
    is one triple (RDF graphs are sets), and `added - len(triples)` is how many collapsed."""

    def __init__(self, kind, *parts):
        self.name = iri("g", kind, *parts)
        self.triples = set()
        self.added = 0

    def add(self, s, p, o):
        for term in (s, p):
            if not term.startswith("<"):
                raise ValueError("a subject and a predicate are IRIs here, got " + term[:40])
        if not (o.startswith("<") or o.startswith('"')):
            raise ValueError("an object is an IRI or a literal here, got " + o[:40])
        self.added += 1
        self.triples.add((s, p, o))

    def typed(self, s, class_name):
        self.add(s, RDF_TYPE, cls(class_name))

    def lines(self, quads=True):
        for s, p, o in sorted(self.triples):
            yield (f"{s} {p} {o} {self.name} ." if quads else f"{s} {p} {o} .")

    def text(self, quads=True):
        return "".join(line + "\n" for line in self.lines(quads))

    def sha256(self, quads=True):
        return hashlib.sha256(self.text(quads).encode("utf-8")).hexdigest()

    def collapsed(self):
        return self.added - len(self.triples)


def write(path, text):
    opener = gzip.open if path.endswith(".gz") else open
    with opener(path, "wt", encoding="utf-8", newline="\n") as f:
        f.write(text)


def blank_terms(text):
    """The number of statements of an N-Triples or N-Quads text that have a blank node as a term."""
    n = 0
    for line in text.splitlines():
        if line.strip() and any(t.startswith("_:") for t in _TERM.findall(line)):
            n += 1
    return n


def add_blank_node(text):
    """The control: the first statement of the text with its subject replaced by a blank node."""
    lines = text.splitlines(keepends=True)
    s = _TERM.match(lines[0]).group(0)
    lines[0] = "_:b0" + lines[0][len(s):]
    return "".join(lines)


def selftest():
    bad = 0

    def check(what, ok):
        nonlocal bad
        print(("ok   " if ok else "FAIL ") + what)
        bad += 0 if ok else 1

    g = Graph("test", "m")
    a, b = iri("t", "m", "st", "model.layers.0.q_proj.weight"), iri("t", "m", "onnx", "onnx::MatMul_9693")
    check("the colon of a part is encoded", "%3A%3A" in b and b.count(":") == iri("t", "m", "onnx", "x").count(":"))
    g.typed(a, "Tensor")
    g.add(a, pred("label"), lit('a "quoted"\nname with \\ and \x01'))
    g.add(a, pred("rank"), num(2))
    g.add(a, pred("rank"), num(2))
    check("a triple added twice is one triple", len(g.triples) == 3 and g.collapsed() == 1)
    t = g.text()
    check("the text has no blank node", blank_terms(t) == 0)
    check("a literal that holds _: is not a blank node", blank_terms(
        f'{a} {pred("label")} "x _:b0 y" {g.name} .\n') == 0)
    check("the control adds a blank node", blank_terms(add_blank_node(t)) == 1 and add_blank_node(t) != t)
    check("two writings are byte-identical", g.sha256() == g.sha256())
    try:
        g.add("_:b0", pred("p"), lit("x"))
        check("the writer refuses a blank subject", False)
    except ValueError:
        check("the writer refuses a blank subject", True)
    return bad


if __name__ == "__main__":
    if sys.argv[1:] == ["selftest"]:
        sys.exit(1 if selftest() else 0)
    sys.exit(__doc__)
