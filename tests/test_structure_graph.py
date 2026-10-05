# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The RDF graph must say what the ONNX graph says, and the queries must be able to fail."""

import hashlib
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
from onnx import TensorProto, helper, numpy_helper

from conftest import MODEL_IRI, SORTED, branching, empty, gated_block
from lmnf import graphstore, onnx_rdf, structure

ROOT = Path(__file__).resolve().parent.parent


def load(model, tmp_path):
    path = tmp_path / "model.nq"
    counts = onnx_rdf.write_nquads(model, MODEL_IRI, path)
    return counts, graphstore.load(path)


def test_every_node_becomes_a_node_resource(tmp_path):
    counts, store = load(gated_block(), tmp_path)
    assert graphstore.rows(store, "node_count") == [{"nodes": 6}]
    assert counts["nodes"] == 6
    assert counts["quads"] == len(store) > 0


def test_op_histogram_from_sparql_equals_the_direct_report(tmp_path):
    model = gated_block()
    _, store = load(model, tmp_path)
    from_sparql = {(r["domain"], r["op"]): r["count"] for r in graphstore.rows(store, "op_histogram")}
    assert from_sparql == structure.report(model)["op_histogram"]
    assert from_sparql[("", "MatMul")] == 2


def test_a_graph_without_branches_reports_no_control_flow(tmp_path):
    model = gated_block()
    _, store = load(model, tmp_path)
    assert graphstore.rows(store, "control_flow") == []
    assert structure.report(model)["control_flow"] == []


def test_an_if_node_is_control_flow_and_its_branches_are_in_the_graph(tmp_path):
    model = branching()
    _, store = load(model, tmp_path)
    found = graphstore.rows(store, "control_flow")
    assert [r["op"] for r in found] == ["If"]
    assert graphstore.rows(store, "node_count") == [{"nodes": 3}]
    assert [c["op"] for c in structure.report(model)["control_flow"]] == ["If"]
    # Both branches read `x` from the enclosing graph: one value resource, three mentions.
    readers = graphstore.select(
        store,
        "PREFIX lmnf: <https://metavacua.github.io/lm-normal-form/ns#> "
        'SELECT ?op WHERE { ?v lmnf:name "x" . ?n lmnf:consumes ?v ; lmnf:opType ?op } ORDER BY ?op',
    )
    assert [r["op"] for r in readers] == ["Identity", "Neg"]
    assert graphstore.select(
        store,
        "PREFIX lmnf: <https://metavacua.github.io/lm-normal-form/ns#> "
        'SELECT (COUNT(DISTINCT ?v) AS ?n) WHERE { ?v lmnf:name "x" }',
    ) == [{"n": 1}]


def test_the_order_certificate_holds_for_a_sorted_graph(tmp_path):
    model = gated_block()
    _, store = load(model, tmp_path)
    assert graphstore.ask(store, "order_violation") is False
    assert structure.report(model)["order_violations"] == 0


def test_the_order_certificate_fails_for_a_shuffled_graph(tmp_path):
    shuffled = ("sigmoid", "mm_gate", "mul", "mm_down", "add", "reshape")
    model = gated_block(order=shuffled)
    _, store = load(model, tmp_path)
    assert graphstore.ask(store, "order_violation") is True
    assert structure.report(model)["order_violations"] == 1


def test_a_node_that_consumes_its_own_output_breaks_the_certificate(tmp_path):
    model = gated_block()
    model.graph.node[2].input[1] = "a"  # Mul(g, a) -> a
    _, store = load(model, tmp_path)
    assert graphstore.ask(store, "order_violation") is True
    assert structure.report(model)["order_violations"] == 1


def test_initializer_facts_match_the_bytes(tmp_path):
    model = gated_block()
    _, store = load(model, tmp_path)
    gate = next(t for t in model.graph.initializer if t.name == "w_gate")
    raw = numpy_helper.to_array(gate).tobytes()
    facts = graphstore.select(
        store,
        "PREFIX lmnf: <https://metavacua.github.io/lm-normal-form/ns#> "
        "SELECT ?type ?rank ?elements ?bytes ?sha WHERE { "
        '?v a lmnf:Initializer ; lmnf:name "w_gate" ; lmnf:elemType ?type ; lmnf:rank ?rank ; '
        "lmnf:elementCount ?elements ; lmnf:byteLength ?bytes ; lmnf:sha256 ?sha }",
    )
    assert facts == [
        {"type": "FLOAT", "rank": 2, "elements": 24, "bytes": 96, "sha": hashlib.sha256(raw).hexdigest()}
    ]
    assert graphstore.rows(store, "initializers") == [{"count": 3, "elements": 24 + 24 + 4, "bytes": 96 + 96 + 32}]
    assert structure.report(model)["initializers"] == {"count": 3, "elements": 52, "bytes": 224}


def test_small_integer_tensors_can_be_read_back_in_order(tmp_path):
    _, store = load(gated_block(), tmp_path)
    values = graphstore.select(
        store,
        "PREFIX lmnf: <https://metavacua.github.io/lm-normal-form/ns#> "
        'SELECT ?i WHERE { ?v lmnf:name "target_shape" ; lmnf:element ?e . ?e lmnf:position ?p ; lmnf:int ?i } ORDER BY ?p',
    )
    assert [r["i"] for r in values] == [1, -1, 2, 2]


def test_the_gate_matrix_is_found_by_wiring_whatever_it_is_called(tmp_path):
    digests = []
    for gate_name in ("w_gate", "onnx::MatMul_17"):
        model = gated_block(gate_name=gate_name)
        path = tmp_path / f"{len(digests)}.nq"
        onnx_rdf.write_nquads(model, MODEL_IRI, path)
        found = graphstore.rows(graphstore.load(path), "silu_gate")
        assert len(found) == 1
        assert found[0]["name"] == gate_name
        digests.append(found[0]["sha"])
    assert digests[0] == digests[1]


def test_graph_inputs_keep_their_symbolic_dimensions(tmp_path):
    _, store = load(gated_block(), tmp_path)
    dims = graphstore.select(
        store,
        "PREFIX lmnf: <https://metavacua.github.io/lm-normal-form/ns#> "
        'SELECT ?axis ?size ?param WHERE { ?g lmnf:graphInput ?v . ?v lmnf:name "x" ; lmnf:dim ?d . '
        "?d lmnf:axis ?axis . OPTIONAL { ?d lmnf:size ?size } OPTIONAL { ?d lmnf:dimParam ?param } } ORDER BY ?axis",
    )
    assert dims == [{"axis": 0, "size": None, "param": "batch"}, {"axis": 1, "size": 4, "param": None}]


def test_emission_is_deterministic(tmp_path):
    a, b = tmp_path / "a.nq", tmp_path / "b.nq"
    onnx_rdf.write_nquads(gated_block(), MODEL_IRI, a)
    onnx_rdf.write_nquads(gated_block(), MODEL_IRI, b)
    assert a.read_bytes() == b.read_bytes()


def run_check(model_path, out_dir, **extra):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"), **extra)
    return subprocess.run(
        [sys.executable, "-m", "lmnf", "check", str(model_path), "--iri", MODEL_IRI, "--out", str(out_dir)],
        capture_output=True,
        text=True,
        env=env,
        cwd=ROOT,
    )


def test_check_exits_zero_when_every_premise_holds(save, tmp_path):
    done = run_check(save(gated_block()), tmp_path / "out")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "H1 no control flow: HOLDS" in done.stdout
    assert (tmp_path / "out" / "report.json").exists()


def test_check_reports_a_failed_hypothesis_without_failing_the_run(save, tmp_path):
    done = run_check(save(branching()), tmp_path / "out")
    assert done.returncode == 0, done.stdout + done.stderr
    assert "H1 no control flow: FAILS" in done.stdout


def test_check_is_inconclusive_on_an_empty_graph(save, tmp_path):
    done = run_check(save(empty()), tmp_path / "out")
    assert done.returncode == 2, done.stdout + done.stderr
    assert "INCONCLUSIVE" in done.stdout


def test_check_is_inconclusive_when_the_onnx_checker_rejects_the_model(save, tmp_path):
    shuffled = tuple(reversed(SORTED))
    done = run_check(save(gated_block(order=shuffled)), tmp_path / "out")
    assert done.returncode == 2, done.stdout + done.stderr
    assert "INCONCLUSIVE" in done.stdout


def test_check_is_inconclusive_when_sparql_and_the_direct_report_disagree(save, tmp_path):
    queries = tmp_path / "queries"
    queries.mkdir()
    for source in (ROOT / "queries").glob("*.rq"):
        queries.joinpath(source.name).write_text(source.read_text(encoding="utf-8"), encoding="utf-8")
    wrong = queries / "node_count.rq"
    wrong.write_text(wrong.read_text(encoding="utf-8").replace("a lmnf:Node", "a lmnf:Value"), encoding="utf-8")
    done = run_check(save(gated_block()), tmp_path / "out", LMNF_QUERIES=str(queries))
    assert done.returncode == 2, done.stdout + done.stderr
    assert "disagree on: nodes" in done.stdout


def test_an_undefined_name_is_counted_once_however_often_it_is_used(tmp_path):
    model = gated_block()
    model.graph.node[1].input[0] = "ghost"  # Sigmoid(ghost)
    model.graph.node[2].input[0] = "ghost"  # Mul(ghost, s)
    _, store = load(model, tmp_path)
    assert graphstore.rows(store, "undefined_values") == [{"count": 1}]
    assert structure.report(model)["undefined_inputs"] == 1


def test_stored_float_shapes_cover_initializers_and_constant_nodes(tmp_path):
    model = gated_block()
    model.graph.initializer.add().CopyFrom(numpy_helper.from_array(np.float32(0.5), "half"))
    table = numpy_helper.from_array(np.zeros((2, 3, 5), dtype=np.float32), "table")
    model.graph.node.append(helper.make_node("Constant", [], ["c"], value=table))
    _, store = load(model, tmp_path)
    assert sorted(graphstore.stored_float_shapes(store)) == [(), (2, 3, 5), (4, 6), (6, 4)]


def test_every_stored_tensor_comes_back_with_its_element_type(tmp_path):
    _, store = load(gated_block(), tmp_path)
    assert sorted(graphstore.stored_tensors(store)) == [("FLOAT", (4, 6)), ("FLOAT", (6, 4)), ("INT64", (4,))]


def test_a_string_tensor_is_described_without_a_digest(tmp_path):
    model = gated_block()
    words = helper.make_tensor("words", TensorProto.STRING, [2], [b"a", b"b"])
    model.graph.node.append(helper.make_node("Constant", [], ["c"], value=words))
    model.graph.initializer.add().CopyFrom(words)
    counts, store = load(model, tmp_path)
    assert counts["quads"] == len(store)
    facts = graphstore.select(
        store,
        "PREFIX lmnf: <https://metavacua.github.io/lm-normal-form/ns#> "
        'SELECT ?n ?sha WHERE { ?t a lmnf:Tensor ; lmnf:elemType "STRING" ; lmnf:elementCount ?n . '
        "OPTIONAL { ?t lmnf:sha256 ?sha } }",
    )
    assert facts == [{"n": 2, "sha": None}]
    # The string initializer is counted, and its bytes are not: the two reports must still agree.
    assert graphstore.rows(store, "initializers") == [{"count": 4, "elements": 54, "bytes": 224}]
    assert structure.report(model)["initializers"] == {"count": 4, "elements": 54, "bytes": 224}
