# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The experiment's report writer, exercised on a synthetic model. No download, no runtime."""

import importlib.util
import json
from pathlib import Path

from conftest import MODEL_IRI, gated_block
from lmnf.__main__ import check

RUNNER = Path(__file__).resolve().parent.parent / "experiments" / "0001-onnx-structure-graph" / "run.py"


def runner():
    spec = importlib.util.spec_from_file_location("experiment_0001", RUNNER)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_the_summary_states_every_verdict_and_every_operator(save, tmp_path):
    assert check(save(gated_block()), MODEL_IRI, tmp_path) == 0
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    entry = {"id": 1, "token": " Paris", "probability": 0.25}
    report.update(
        subject={"repository": "owner/name", "revision": "abc", "file": "onnx/model.onnx", "bytes": 1234, "sha256": "00"},
        verdicts={"H1 no control flow, no cycle": "HOLDS", "H2 every parameter is stored in the graph": "FAILS"},
        parameters={"reference_tensors": 3, "matched": 2, "missing_from_onnx": [[8]], "only_in_onnx": [],
                    "elements_reference": 56, "elements_onnx": 52},
        forward={"encodings": {"tokenizer default": {
            "token_ids": [1, 2], "same_top5_ids": True, "max_abs_probability_difference": 1e-6,
            "max_abs_logit_difference": 1e-5, "onnx_top5": [entry], "pytorch_top5": [entry]}}},
        exploratory={"silu_gate_matches": 1},
        versions={"onnx": "1.2.3", "pyoxigraph": "0.5.11"},
        inconclusive="H5: no rule for graph input 'mystery'",
    )
    text = runner().summary(report)
    assert "- H1 no control flow, no cycle: HOLDS" in text
    assert "- H2 every parameter is stored in the graph: FAILS" in text
    assert "- `MatMul`: 2" in text
    assert "|" not in text  # lists, not tables: the summary is also read in a terminal
    assert "onnx 1.2.3, pyoxigraph 0.5.11" in text
    assert "3 reference tensors, 2 matched in the graph, 1 missing" in text
    assert "tokenizer default (2 tokens" in text
    assert "**Inconclusive:** H5: no rule for graph input 'mystery'" in text
