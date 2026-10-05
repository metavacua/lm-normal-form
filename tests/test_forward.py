# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""One block of a network, evaluated by query engines: each must choose what plain arithmetic chooses."""

import importlib.util
import json
import os

import pytest

from lmnf.experiments import forward
from lmnf.experiments.forward import block, engines

E1 = "E1 the engine and plain arithmetic agree"


def engine_cases():
    cases = []
    for name in sorted(engines.ENGINES):
        marks = []
        # Outside CI a missing optional engine is skipped. In CI every engine must be there.
        if name == "duckdb" and importlib.util.find_spec("duckdb") is None and not os.environ.get("CI"):
            marks.append(pytest.mark.skip(reason="duckdb is not installed here"))
        cases.append(pytest.param(name, marks=marks))
    return cases


def test_the_four_engines_of_the_first_batch_are_registered():
    assert sorted(engines.ENGINES) == ["duckdb", "oxigraph-scalar", "oxigraph-vector", "sqlite"]


@pytest.mark.parametrize("name", engine_cases())
@pytest.mark.parametrize("size", [(8, 4, 6), (40, 12, 20)])
def test_every_engine_agrees_with_plain_arithmetic(name, size, tmp_path):
    vocabulary = size[0]
    weights = block.build(*size)
    engine = engines.ENGINES[name](weights, tmp_path)
    try:
        for token in range(0, vocabulary, 3):
            expected = block.reference(weights, token)
            ranked = engine.logits(token)
            assert sorted(index for index, _ in ranked) == list(range(vocabulary))
            assert ranked[0][0] == max(range(vocabulary), key=expected.__getitem__)
            values = [value for _, value in ranked]
            assert values == sorted(values, reverse=True)
            assert max(abs(value - expected[index]) for index, value in ranked) < 1e-9
    finally:
        engine.close()


def test_the_block_has_the_weights_it_says_it_has():
    weights = block.build(8, 4, 6)
    assert block.count(weights) == 8 * 4 + 6 * 4 + 4 * 6 + 8 * 4 == 112
    assert block.build(8, 4, 6) == weights  # the same seed gives the same weights


def test_what_each_engine_stores(tmp_path):
    weights = block.build(8, 4, 6)
    stored = {}
    for name in ("oxigraph-scalar", "oxigraph-vector", "sqlite"):
        engine = engines.ENGINES[name](weights, tmp_path)
        stored[name] = engine.stored()
        engine.close()
    assert stored["oxigraph-scalar"] == {"quads": 4 * 112}
    # One resource per row of Embed, Gate and Unembed and per column of Down, three statements each.
    assert stored["oxigraph-vector"] == {"quads": 3 * (8 + 6 + 6 + 8), "vectors": 8 + 6 + 6 + 8}
    assert stored["sqlite"] == {"rows": 112}


# ── the run: verdict, budget, exit status ──


class Shifted:
    """An engine that is wrong by a constant."""

    name = "shifted"

    def __init__(self, weights, work):
        self.weights = weights

    def stored(self):
        return {"nothing": 0}

    def logits(self, token):
        values = block.reference(self.weights, token)
        return sorted(((i, v + 0.5 * (i % 2)) for i, v in enumerate(values)), key=lambda pair: -pair[1])

    def close(self):
        pass


class Silent(Shifted):
    """An engine that answers with nothing."""

    name = "silent"

    def logits(self, token):
        return []


def run(tmp_path, monkeypatch, engine="sqlite", sizes=((8, 4, 6),), budget=5, fake=None):
    if fake is not None:
        monkeypatch.setitem(engines.ENGINES, fake.name, fake)
        engine = fake.name
    cell = {
        "id": "09-probe",
        "experiment": "forward",
        "engine": engine,
        "sizes": [list(s) for s in sizes],
        "budget_seconds": budget,
    }
    status = forward.run(cell, tmp_path / "out")
    report = json.loads((tmp_path / "out" / "report.json").read_text(encoding="utf-8"))
    return status, report, (tmp_path / "out" / "SUMMARY.md").read_text(encoding="utf-8")


def test_a_run_reports_agreement_and_what_it_measured(tmp_path, monkeypatch):
    status, report, text = run(tmp_path, monkeypatch)
    assert status == 0
    assert report["cell"] == "09-probe"
    assert report["experiment"] == "forward"
    assert report["verdicts"] == {E1: "HOLDS"}
    assert report["runs"][0]["weights"] == 112
    assert report["runs"][0]["stored"] == {"rows": 112}
    assert report["runs"][0]["passes"] == 3
    assert report["runs"][0]["seconds_per_pass"] > 0
    assert report["headline"]
    assert f"- {E1}: HOLDS" in text
    assert "112 weights" in text
    assert "|" not in text


def test_a_wrong_answer_fails_the_expectation_and_not_the_run(tmp_path, monkeypatch):
    status, report, text = run(tmp_path, monkeypatch, fake=Shifted)
    assert status == 0
    assert report["verdicts"] == {E1: "FAILS"}
    assert f"- {E1}: FAILS" in text


def test_an_engine_that_returns_too_few_rows_makes_the_run_inconclusive(tmp_path, monkeypatch):
    status, report, text = run(tmp_path, monkeypatch, fake=Silent)
    assert status == 2
    assert report["verdicts"] == {E1: "NOT EVALUATED"}
    assert "**Inconclusive:**" in text


def test_sizes_are_tried_in_order_until_one_is_over_budget(tmp_path, monkeypatch):
    sizes = ((8, 4, 6), (16, 4, 6), (24, 4, 6))
    status, report, text = run(tmp_path, monkeypatch, sizes=sizes, budget=0)
    assert status == 0
    assert [r["vocabulary"] for r in report["runs"]] == [8]
    assert report["not_attempted"] == [[16, 4, 6], [24, 4, 6]]
    assert "not attempted" in text


def test_within_budget_every_size_is_tried(tmp_path, monkeypatch):
    sizes = ((8, 4, 6), (16, 4, 6))
    _, report, _ = run(tmp_path, monkeypatch, sizes=sizes, budget=3600)
    assert [r["vocabulary"] for r in report["runs"]] == [8, 16]
    assert report["not_attempted"] == []
