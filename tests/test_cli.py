# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The commands the workflow calls, run the way the workflow runs them."""

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from lmnf import experiments

ROOT = Path(__file__).resolve().parent.parent

CELL = {
    "id": "01-a",
    "experiment": "structure",
    "onnx_repository": "owner/name",
    "onnx_file": "onnx/model.onnx",
    "reference_repository": "owner/name",
    "prompt": "A prompt",
}
OTHER = dict(CELL, id="02-b", onnx_file="onnx/model_q4.onnx")


def lmnf(*arguments, cwd=ROOT):
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    return subprocess.run(
        [sys.executable, "-m", "lmnf", *map(str, arguments)], capture_output=True, text=True, env=env, cwd=cwd
    )


def batch_file(root, cells, number="0007"):
    path = root / "batches" / f"{number}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"batch": number, "title": "A title", "cells": cells}), encoding="utf-8")
    return path


def report_file(reports, name):
    folder = reports / name
    folder.mkdir(parents=True, exist_ok=True)
    data = {"cell": name, "verdicts": {"H1 no control flow, no cycle": "HOLDS"}, "headline": "A headline."}
    (folder / "report.json").write_text(json.dumps(data), encoding="utf-8")


def outputs(done):
    assert done.returncode == 0, done.stdout + done.stderr
    return dict(line.split("=", 1) for line in done.stdout.splitlines())


def test_plan_prints_what_the_workflow_reads(tmp_path):
    batch_file(tmp_path, [CELL, OTHER])
    planned = outputs(lmnf("plan", "--root", tmp_path, "--ref", "batch/0007-anything"))
    assert planned["count"] == "2"
    assert planned["file"] == "batches/0007.json"
    assert json.loads(planned["matrix"]) == {
        "include": [{"id": "01-a", "experiment": "structure"}, {"id": "02-b", "experiment": "structure"}]
    }


def test_plan_runs_no_cells_once_results_are_recorded_and_only_documents_changed(tmp_path):
    batch_file(tmp_path, [CELL])
    recorded = tmp_path / "docs" / "batches" / "0007.results.json"
    recorded.parent.mkdir(parents=True)
    recorded.write_text(json.dumps({"batch": "0007", "commit": "abc1234def", "run": 1}), encoding="utf-8")
    changed = tmp_path / "changed.txt"
    changed.write_text("docs/batches/0007.md\nREADME.md\n", encoding="utf-8")
    planned = outputs(lmnf("plan", "--root", tmp_path, "--ref", "batch/0007-x", "--changed", changed))
    assert planned["count"] == "0"
    assert json.loads(planned["matrix"]) == {"include": []}


def test_plan_fails_loudly_on_a_batch_file_it_cannot_accept(tmp_path):
    batch_file(tmp_path, [CELL, CELL])
    done = lmnf("plan", "--root", tmp_path, "--ref", "batch/0007-x")
    assert done.returncode == 1
    assert "01-a" in done.stderr


def test_collect_sums_the_batch_up_and_fails_while_a_cell_is_missing(tmp_path):
    path = batch_file(tmp_path, [CELL, OTHER])
    reports = tmp_path / "reports"
    report_file(reports, "01-a")
    collected = lmnf("collect", "--batch", path, "--reports", reports, "--out", tmp_path / "out")
    assert collected.returncode == 2, collected.stdout + collected.stderr
    text = (tmp_path / "out" / "BATCH.md").read_text(encoding="utf-8")
    assert "2 cells registered, 1 reported, 1 inconclusive" in text
    assert json.loads((tmp_path / "out" / "batch.json").read_text(encoding="utf-8"))["inconclusive"] == ["02-b"]

    report_file(reports, "02-b")
    collected = lmnf(
        "collect", "--batch", path, "--reports", reports, "--out", tmp_path / "out", "--commit", "abc1234def", "--run", "42"
    )
    assert collected.returncode == 0, collected.stdout + collected.stderr
    assert "2 cells registered, 2 reported, 0 inconclusive" in (tmp_path / "out" / "BATCH.md").read_text(encoding="utf-8")
    # What the run was, so that the results can be recorded against it.
    data = json.loads((tmp_path / "out" / "batch.json").read_text(encoding="utf-8"))
    assert (data["commit"], data["run"]) == ("abc1234def", "42")


def test_cell_refuses_a_name_that_is_not_in_the_batch(tmp_path):
    path = batch_file(tmp_path, [CELL])
    done = lmnf("cell", "--batch", path, "--id", "99-z", "--out", tmp_path / "out")
    assert done.returncode == 1
    assert "99-z" in done.stderr


def test_a_cell_is_handed_to_the_experiment_it_names(tmp_path, monkeypatch):
    from lmnf.experiments import structure

    seen = []
    monkeypatch.setattr(structure, "run", lambda cell, out: seen.append((cell["id"], out)) or 0)
    assert experiments.run(CELL, tmp_path / "out") == 0
    assert seen == [("01-a", tmp_path / "out")]
    with pytest.raises(ValueError, match="guess"):
        experiments.run(dict(CELL, experiment="guess"), tmp_path / "out")


def test_measure_wraps_a_command_and_keeps_its_exit_status(tmp_path):
    out = tmp_path / "resources.json"
    done = lmnf("measure", "--label", "probe", "--out", out, "--", sys.executable, "-c", "raise SystemExit(5)")
    assert done.returncode == 5
    assert json.loads(out.read_text(encoding="utf-8"))["exit_status"] == 5


def test_an_unknown_command_is_an_error_and_says_what_exists():
    done = lmnf("guess")
    assert done.returncode == 1
    assert "collect" in done.stderr


def test_the_first_batch_is_a_valid_batch_of_twelve():
    from lmnf import batch

    cells = batch.load(ROOT / "batches" / "0001.json")["cells"]
    assert len(cells) == 12
    assert len({(cell["onnx_repository"], cell["onnx_file"]) for cell in cells}) == 12
    assert {cell["prompt"] for cell in cells} == {"The capital of France is"}
