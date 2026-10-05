#!/usr/bin/env python3
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Experiment 0002: one block of a network evaluated by a single SPARQL query.

The block is embedding lookup, a gated feed-forward layer with ReLU, a residual
connection, unembedding, and choice of the largest logit. Every weight is one
RDF resource with four statements (matrix, row, column, value). The query uses
only SPARQL 1.1: joins, SUM, GROUP BY, IF and ORDER BY.

This is a toy. It shows that the arithmetic is expressible and exact, and how
slow this encoding is. It says nothing about a model of useful size.

Exit status: 0 when the comparison was made (whatever the verdict), 2 when it
could not be made. What is expected is in docs/experiments/0002-sparql-forward-toy.md.
"""

import argparse
import importlib.metadata
import json
import platform
import random
import sys
import time
from pathlib import Path

import pyoxigraph as ox

NS = "https://metavacua.github.io/lm-normal-form/experiments/0002#"
XSD = "http://www.w3.org/2001/XMLSchema#"

# (vocabulary, hidden, intermediate)
SIZES = ((8, 4, 6), (128, 32, 64), (512, 64, 128))
E1 = "E1 the query engine and plain arithmetic agree"
LOGIT_TOLERANCE = 1e-9
INCONCLUSIVE = 2

QUERY = """PREFIX : <%s>
SELECT ?token (SUM(?u * ?h2) AS ?logit) WHERE {
  ?unembed :matrix :Unembed ; :row ?token ; :col ?j ; :value ?u .
  { SELECT ?j ((?h + ?delta) AS ?h2) WHERE {
      ?embed :matrix :Embed ; :row %d ; :col ?j ; :value ?h .
      { SELECT ?j (SUM(?d * ?a) AS ?delta) WHERE {
          ?down :matrix :Down ; :row ?j ; :col ?k ; :value ?d .
          { SELECT ?k (IF(SUM(?g * ?x) > 0, SUM(?g * ?x), 0.0e0) AS ?a) WHERE {
              ?gate :matrix :Gate ; :row ?k ; :col ?i ; :value ?g .
              ?input :matrix :Embed ; :row %d ; :col ?i ; :value ?x .
          } GROUP BY ?k }
      } GROUP BY ?j }
  } }
} GROUP BY ?token ORDER BY DESC(?logit) ?token"""


def build(vocabulary, hidden, intermediate, seed=7):
    rng = random.Random(seed)

    def matrix(rows, columns):
        return [[rng.uniform(-1, 1) for _ in range(columns)] for _ in range(rows)]

    return {
        "Embed": matrix(vocabulary, hidden),
        "Gate": matrix(intermediate, hidden),
        "Down": matrix(hidden, intermediate),
        "Unembed": matrix(vocabulary, hidden),
    }


def count(weights):
    return sum(len(row) for matrix in weights.values() for row in matrix)


def reference(weights, token):
    """The same block in ordinary arithmetic."""
    h = weights["Embed"][token]
    a = [max(0.0, sum(g * x for g, x in zip(row, h))) for row in weights["Gate"]]
    h2 = [x + sum(d * y for d, y in zip(row, a)) for x, row in zip(h, weights["Down"])]
    return [sum(u * x for u, x in zip(row, h2)) for row in weights["Unembed"]]


def load(weights):
    node = lambda name: ox.NamedNode(NS + name)  # noqa: E731
    integer, double = ox.NamedNode(XSD + "integer"), ox.NamedNode(XSD + "double")
    quads = []
    for name, matrix in weights.items():
        for r, row in enumerate(matrix):
            for c, value in enumerate(row):
                weight = node(f"{name}/{r}/{c}")
                quads += [
                    ox.Quad(weight, node("matrix"), node(name)),
                    ox.Quad(weight, node("row"), ox.Literal(str(r), datatype=integer)),
                    ox.Quad(weight, node("col"), ox.Literal(str(c), datatype=integer)),
                    ox.Quad(weight, node("value"), ox.Literal(repr(value), datatype=double)),
                ]
    store = ox.Store()
    store.extend(quads)
    return store


def logits(store, token):
    """(token, logit) pairs, largest logit first, computed by the query engine."""
    return [(int(s["token"].value), float(s["logit"].value)) for s in store.query(QUERY % (NS, token, token))]


def measure(vocabulary, hidden, intermediate):
    weights = build(vocabulary, hidden, intermediate)
    store = load(weights)
    seconds, error, same, complete = [], 0.0, True, True
    for token in sorted({0, vocabulary // 2, vocabulary - 1}):
        started = time.perf_counter()
        ranked = logits(store, token)
        seconds.append(time.perf_counter() - started)
        if len(ranked) != vocabulary:
            complete = False
            continue
        expected = reference(weights, token)
        same = same and ranked[0][0] == max(range(vocabulary), key=expected.__getitem__)
        error = max(error, max(abs(value - expected[index]) for index, value in ranked))
    per_pass = sum(seconds) / len(seconds)
    return {
        "vocabulary": vocabulary,
        "hidden": hidden,
        "intermediate": intermediate,
        "weights": count(weights),
        "quads": len(store),
        "passes": len(seconds),
        "seconds_per_pass": per_pass,
        "weights_per_second": count(weights) / per_pass,
        "complete": complete,
        "same_token": complete and same,
        "largest_logit_error": error if complete else None,
    }


def summary(report):
    lines = ["# Experiment 0002: forward pass in SPARQL (toy)", "", "## Verdicts", ""]
    lines += [f"- {name}: {verdict}" for name, verdict in report["verdicts"].items()]
    if report.get("inconclusive"):
        lines += ["", f"**Inconclusive:** {report['inconclusive']}"]
    lines += ["", "## Measurements", "", "One line per model size. Time is the mean over the forward passes run.", ""]
    for run in report["runs"]:
        error = "not measured" if run["largest_logit_error"] is None else f"{run['largest_logit_error']:.1e}"
        lines.append(
            f"- vocabulary {run['vocabulary']}, hidden {run['hidden']}, intermediate {run['intermediate']}: "
            f"{run['weights']:,} weights, {run['quads']:,} quads, {run['seconds_per_pass'] * 1000:.0f} ms per forward pass "
            f"over {run['passes']} passes ({run['weights_per_second']:,.0f} weights per second), "
            f"same token chosen {run['same_token']}, largest logit error {error}"
        )
    lines += ["", "Versions: " + ", ".join(f"{name} {version}" for name, version in report["versions"].items()) + "."]
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    out = parser.parse_args(argv).out
    out.mkdir(parents=True, exist_ok=True)

    runs = [measure(*size) for size in SIZES]
    report = {
        "runs": runs,
        "versions": {"python": platform.python_version(), "pyoxigraph": importlib.metadata.version("pyoxigraph")},
    }
    incomplete = [run for run in runs if not run["complete"]]
    if incomplete or not runs:
        report["verdicts"] = {E1: "NOT EVALUATED"}
        report["inconclusive"] = (
            f"the query returned the wrong number of rows for {len(incomplete)} of {len(runs)} model sizes"
        )
    else:
        agree = all(run["same_token"] and run["largest_logit_error"] <= LOGIT_TOLERANCE for run in runs)
        report["verdicts"] = {E1: "HOLDS" if agree else "FAILS"}
    (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    text = summary(report)
    (out / "SUMMARY.md").write_text(text, encoding="utf-8")
    print(text)
    return INCONCLUSIVE if "inconclusive" in report else 0


if __name__ == "__main__":
    sys.exit(main())
