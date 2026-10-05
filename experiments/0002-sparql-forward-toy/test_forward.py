# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A query engine and ordinary arithmetic must choose the same next token."""

import json

import pytest

import forward


@pytest.mark.parametrize("vocabulary, hidden, intermediate, tokens", [(8, 4, 6, range(8)), (64, 16, 32, (0, 17, 63))])
def test_sparql_and_plain_arithmetic_agree(vocabulary, hidden, intermediate, tokens):
    weights = forward.build(vocabulary, hidden, intermediate)
    store = forward.load(weights)
    for token in tokens:
        expected = forward.reference(weights, token)
        ranked = forward.logits(store, token)
        assert len(ranked) == vocabulary
        assert ranked[0][0] == max(range(vocabulary), key=expected.__getitem__)
        assert max(abs(value - expected[index]) for index, value in ranked) < 1e-9


def test_four_quads_are_stored_for_each_weight():
    weights = forward.build(8, 4, 6)
    assert len(forward.load(weights)) == 4 * forward.count(weights) == 4 * 112


E1 = "E1 the query engine and plain arithmetic agree"


def run(tmp_path, monkeypatch, query=None):
    monkeypatch.setattr(forward, "SIZES", ((8, 4, 6),))
    if query is not None:
        monkeypatch.setattr(forward, "QUERY", query)
    status = forward.main(["--out", str(tmp_path)])
    report = json.loads((tmp_path / "report.json").read_text(encoding="utf-8"))
    return status, report, (tmp_path / "SUMMARY.md").read_text(encoding="utf-8")


def test_a_run_reports_agreement_and_what_it_measured(tmp_path, monkeypatch):
    status, report, text = run(tmp_path, monkeypatch)
    assert status == 0
    assert report["verdicts"] == {E1: "HOLDS"}
    assert report["runs"][0]["weights"] == 112
    assert report["runs"][0]["quads"] == 448
    assert report["runs"][0]["passes"] == 3
    assert f"- {E1}: HOLDS" in text
    assert "112 weights, 448 quads" in text


def test_a_wrong_answer_fails_the_expectation_and_not_the_run(tmp_path, monkeypatch):
    status, report, text = run(tmp_path, monkeypatch, forward.QUERY.replace("(?h + ?delta)", "(?h - ?delta)"))
    assert status == 0
    assert report["verdicts"] == {E1: "FAILS"}
    assert f"- {E1}: FAILS" in text


def test_a_query_that_returns_too_few_rows_is_inconclusive(tmp_path, monkeypatch):
    status, report, text = run(tmp_path, monkeypatch, forward.QUERY.replace(":Unembed", ":Missing"))
    assert status == 2
    assert report["verdicts"] == {E1: "NOT EVALUATED"}
    assert "**Inconclusive:**" in text
