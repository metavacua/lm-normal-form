# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""`python -m lmnf check MODEL.onnx --iri IRI --out DIR`

Exit status says how the run ended, not what it found:
  0  every premise held; hypothesis verdicts are in the report
  2  inconclusive: a premise failed or a check could not be evaluated
Anything else is a crash.
"""

import argparse
import json
import sys
from pathlib import Path

import onnx

from . import graphstore, onnx_rdf, structure

INCONCLUSIVE = 2


def inconclusive(reason, report, out):
    report["inconclusive"] = reason
    write(report, out)
    print(f"INCONCLUSIVE: {reason}")
    return INCONCLUSIVE


def write(report, out):
    (out / "report.json").write_text(json.dumps(report, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def jsonable(direct):
    plain = dict(direct)
    plain["op_histogram"] = {f"{domain}::{op}": count for (domain, op), count in sorted(direct["op_histogram"].items())}
    return plain


def check(path, iri, out):
    out.mkdir(parents=True, exist_ok=True)
    report = {"model": str(path), "iri": iri}

    # Premise: the file is ONNX that the reference checker accepts.
    try:
        onnx.checker.check_model(str(path))
    except Exception as error:  # the checker raises several unrelated types
        return inconclusive(f"the ONNX checker rejected the model: {error}", report, out)
    model = onnx.load(str(path))

    direct = structure.report(model)
    report["direct"] = jsonable(direct)
    counts = onnx_rdf.write_nquads(model, iri, out / "model.nq")
    store = graphstore.load(out / "model.nq")
    report["rdf"] = dict(counts, stored_quads=len(store))

    # Premise: the run is not vacuous.
    if direct["nodes"] == 0 or len(store) == 0:
        return inconclusive("the graph has no nodes, so nothing was checked", report, out)

    # Premise: the RDF graph says what the ONNX graph says.
    from_sparql = {
        "nodes": graphstore.rows(store, "node_count")[0]["nodes"],
        "op_histogram": {(r["domain"], r["op"]): r["count"] for r in graphstore.rows(store, "op_histogram")},
        "control_flow": len(graphstore.rows(store, "control_flow")),
        "order_violation": graphstore.ask(store, "order_violation"),
        "undefined_inputs": graphstore.rows(store, "undefined_values")[0]["count"],
        "initializers": graphstore.rows(store, "initializers")[0],
    }
    expected = {
        "nodes": direct["nodes"],
        "op_histogram": direct["op_histogram"],
        "control_flow": len(direct["control_flow"]),
        "order_violation": direct["order_violations"] > 0,
        "undefined_inputs": direct["undefined_inputs"],
        "initializers": direct["initializers"],
    }
    differing = sorted(key for key in expected if expected[key] != from_sparql[key])
    report["sparql"] = dict(from_sparql, op_histogram=jsonable(from_sparql)["op_histogram"])
    report["graph_io"] = graphstore.rows(store, "graph_io")
    if counts["quads"] != len(store):
        differing.append("quads")
    if differing:
        return inconclusive(f"SPARQL and the direct report disagree on: {', '.join(differing)}", report, out)

    # Hypothesis H1: the computation has no control flow and no cycle.
    h1 = (
        not direct["control_flow"]
        and direct["order_violations"] == 0
        and direct["undefined_inputs"] == 0
        and direct["functions_acyclic"]
    )
    report["verdicts"] = {"H1": "HOLDS" if h1 else "FAILS"}
    write(report, out)

    print(f"nodes: {direct['nodes']}  quads: {len(store)}  stored tensors: {direct['initializers']['count']}")
    for (domain, op), count in sorted(direct["op_histogram"].items(), key=lambda item: (-item[1], item[0])):
        print(f"  {count:6d}  {domain + '::' if domain else ''}{op}")
    for node in direct["control_flow"]:
        print(f"  control flow: {node['op']} at {node['graph']}/node/{node['index']}")
    print(f"H1 no control flow: {report['verdicts']['H1']}")
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(prog="lmnf")
    commands = parser.add_subparsers(dest="command", required=True)
    one = commands.add_parser("check", help="render a model's structure as RDF and check it against the model")
    one.add_argument("model", type=Path)
    one.add_argument("--iri", required=True, help="the IRI that identifies this model file")
    one.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    return check(args.model, args.iri, args.out)


if __name__ == "__main__":
    sys.exit(main())
