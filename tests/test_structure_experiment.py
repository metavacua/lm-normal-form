# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The structure experiment after its downloads, on synthetic models. No network, no runtime."""

import json

import pytest

from conftest import MODEL_IRI, empty, gated_block
from lmnf import feeds
from lmnf.experiments import structure

H1 = "H1 no control flow, no cycle"
H2 = "H2 every parameter is stored in the graph"
H5 = "H5 the graph computes what the PyTorch weights compute"

CELL = {
    "id": "01-synthetic",
    "experiment": "structure",
    "onnx_repository": "owner/name",
    "onnx_file": "onnx/model.onnx",
    "reference_repository": "owner/reference",
    "prompt": "A prompt",
}
SUBJECT = {
    "repository": "owner/name",
    "revision": "abc",
    "file": "onnx/model.onnx",
    "companions": [],
    "bytes": 1234,
    "sha256": "00",
    "reference_repository": "owner/reference",
    "reference_revision": "def",
}
# The synthetic block stores a 4x6 and a 6x4 float matrix and a 4-element integer shape.
STORED = [(6, 4), (4, 6)]


def forward(same=True, difference=1e-6):
    entry = {"id": 1, "token": " Paris", "probability": 0.25}
    encoding = {
        "token_ids": [1, 2],
        "tokens": ["a", "b"],
        "onnx_top5": [entry],
        "pytorch_top5": [entry],
        "same_top5_ids": same,
        "max_abs_logit_difference": 1e-5,
        "max_abs_probability_difference": difference,
    }
    return lambda: {"prompt": "A prompt", "bos_token_id": None, "encodings": {"tokenizer default": encoding}}


def never():
    raise AssertionError("the forward pass must not run")


def evaluate(save, tmp_path, model=None, reference=STORED, forward_pass=None):
    out = tmp_path / "out"
    path = save(model if model is not None else gated_block())
    status = structure.evaluate(CELL, SUBJECT, path, MODEL_IRI, reference, forward_pass or forward(), out)
    report = json.loads((out / "report.json").read_text(encoding="utf-8"))
    return status, report, (out / "SUMMARY.md").read_text(encoding="utf-8")


def test_every_hypothesis_holds_for_a_graph_that_stores_its_parameters(save, tmp_path):
    status, report, text = evaluate(save, tmp_path)
    assert status == 0
    assert report["verdicts"] == {H1: "HOLDS", H2: "HOLDS", H5: "HOLDS"}
    assert report["cell"] == "01-synthetic"
    assert report["experiment"] == "structure"
    assert "inconclusive" not in report
    assert report["headline"]
    assert report["parameters"]["matched"] == 2
    assert f"- {H1}: HOLDS" in text
    assert "`owner/reference` at revision `def`" in text


def test_a_parameter_the_graph_does_not_store_fails_h2_and_not_the_run(save, tmp_path):
    status, report, text = evaluate(save, tmp_path, reference=STORED + [(8,)])
    assert status == 0
    assert report["verdicts"][H2] == "FAILS"
    assert report["parameters"]["missing_from_onnx"] == [[8]]
    assert "3 reference tensors, 2 matched in the graph, 1 missing" in text


def test_integer_tensors_do_not_count_as_parameters_but_are_reported(save, tmp_path):
    _, report, _ = evaluate(save, tmp_path, reference=STORED + [(4,)])
    assert report["verdicts"][H2] == "FAILS"
    assert report["exploratory"]["parameters_matched_counting_every_element_type"] == 3


def test_a_forward_pass_that_disagrees_fails_h5_and_not_the_run(save, tmp_path):
    status, report, _ = evaluate(save, tmp_path, forward_pass=forward(same=False))
    assert status == 0
    assert report["verdicts"][H5] == "FAILS"
    status, report, _ = evaluate(save, tmp_path, forward_pass=forward(difference=0.5))
    assert status == 0
    assert report["verdicts"][H5] == "FAILS"


def test_an_input_no_rule_covers_leaves_h5_unevaluated_and_the_run_inconclusive(save, tmp_path):
    def refused():
        raise feeds.UnknownInput("no rule for graph input 'mystery'")

    status, report, text = evaluate(save, tmp_path, forward_pass=refused)
    assert status == 2
    assert report["verdicts"] == {H1: "HOLDS", H2: "HOLDS", H5: "NOT EVALUATED"}
    assert report["inconclusive"] == "H5: no rule for graph input 'mystery'"
    assert "**Inconclusive:** H5: no rule for graph input 'mystery'" in text


def test_an_empty_reference_inventory_leaves_h2_unevaluated(save, tmp_path):
    status, report, _ = evaluate(save, tmp_path, reference=[])
    assert status == 2
    assert report["verdicts"][H2] == "NOT EVALUATED"
    assert report["inconclusive"].startswith("H2:")


def test_nothing_is_evaluated_when_the_structure_check_is_inconclusive(save, tmp_path):
    status, report, text = evaluate(save, tmp_path, model=empty(), forward_pass=never)
    assert status == 2
    assert report["verdicts"] == {H1: "NOT EVALUATED", H2: "NOT EVALUATED", H5: "NOT EVALUATED"}
    assert "no nodes" in report["inconclusive"]
    assert "**Inconclusive:**" in text


def test_the_summary_states_what_was_found(save, tmp_path):
    _, report, text = evaluate(save, tmp_path)
    assert "# 01-synthetic" in text
    assert "- `MatMul`: 2" in text
    assert "- Nodes: 6" in text
    assert "With tokenizer default (2 tokens" in text
    assert "|" not in text  # lists, not tables: the summary is also read in a terminal
    assert f"onnx {report['versions']['onnx']}" in text


@pytest.mark.parametrize("missing", ["onnx_repository", "onnx_file", "reference_repository", "prompt"])
def test_a_cell_that_lacks_what_the_experiment_needs_is_refused_before_any_download(missing, tmp_path):
    cell = {key: value for key, value in CELL.items() if key != missing}
    with pytest.raises(KeyError, match=missing):
        structure.run(cell, tmp_path / "out")
