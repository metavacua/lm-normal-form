# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The commands the workflow calls, run the way the workflow runs them."""

import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

FORWARD = {"id": "02-b", "experiment": "forward", "engine": "sqlite", "sizes": [[8, 4, 6]], "budget_seconds": 5}
OTHER = dict(FORWARD, id="03-c")


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


def outputs(done):
    assert done.returncode == 0, done.stdout + done.stderr
    return dict(line.split("=", 1) for line in done.stdout.splitlines())


def test_plan_prints_what_the_workflow_reads(tmp_path):
    batch_file(tmp_path, [FORWARD, OTHER])
    planned = outputs(lmnf("plan", "--root", tmp_path, "--ref", "batch/0007-anything"))
    assert planned["count"] == "2"
    assert planned["file"] == "batches/0007.json"
    assert json.loads(planned["matrix"]) == {
        "include": [{"id": "02-b", "experiment": "forward"}, {"id": "03-c", "experiment": "forward"}]
    }


def test_plan_runs_no_cells_for_a_push_that_changed_only_documents(tmp_path):
    batch_file(tmp_path, [FORWARD])
    changed = tmp_path / "changed.txt"
    changed.write_text("docs/batches/0007.md\nREADME.md\n", encoding="utf-8")
    planned = outputs(lmnf("plan", "--root", tmp_path, "--ref", "batch/0007-x", "--changed", changed))
    assert planned["count"] == "0"
    assert json.loads(planned["matrix"]) == {"include": []}


def test_plan_fails_loudly_on_a_batch_file_it_cannot_accept(tmp_path):
    batch_file(tmp_path, [FORWARD, FORWARD])
    done = lmnf("plan", "--root", tmp_path, "--ref", "batch/0007-x")
    assert done.returncode == 1
    assert "02-b" in done.stderr


def test_cell_runs_the_cell_it_is_given_and_collect_sums_the_batch_up(tmp_path):
    path = batch_file(tmp_path, [FORWARD, OTHER])
    reports = tmp_path / "reports"
    done = lmnf("cell", "--batch", path, "--id", "02-b", "--out", reports / "02-b")
    assert done.returncode == 0, done.stdout + done.stderr
    report = json.loads((reports / "02-b" / "report.json").read_text(encoding="utf-8"))
    assert report["cell"] == "02-b"
    assert report["verdicts"] == {"E1 the engine and plain arithmetic agree": "HOLDS"}

    # One of the two cells has not reported: the batch is inconclusive, and says which.
    collected = lmnf("collect", "--batch", path, "--reports", reports, "--out", tmp_path / "out")
    assert collected.returncode == 2, collected.stdout + collected.stderr
    text = (tmp_path / "out" / "BATCH.md").read_text(encoding="utf-8")
    assert "2 cells registered, 1 reported, 1 inconclusive" in text
    assert json.loads((tmp_path / "out" / "batch.json").read_text(encoding="utf-8"))["inconclusive"] == ["03-c"]

    done = lmnf("cell", "--batch", path, "--id", "03-c", "--out", reports / "03-c")
    assert done.returncode == 0, done.stdout + done.stderr
    collected = lmnf("collect", "--batch", path, "--reports", reports, "--out", tmp_path / "out")
    assert collected.returncode == 0, collected.stdout + collected.stderr


def test_cell_refuses_a_name_that_is_not_in_the_batch(tmp_path):
    path = batch_file(tmp_path, [FORWARD])
    done = lmnf("cell", "--batch", path, "--id", "99-z", "--out", tmp_path / "out")
    assert done.returncode == 1
    assert "99-z" in done.stderr


def test_measure_wraps_a_command_and_keeps_its_exit_status(tmp_path):
    out = tmp_path / "resources.json"
    done = lmnf("measure", "--label", "probe", "--out", out, "--", sys.executable, "-c", "raise SystemExit(5)")
    assert done.returncode == 5
    assert json.loads(out.read_text(encoding="utf-8"))["exit_status"] == 5


def test_the_first_batch_is_a_valid_batch_of_twelve():
    planned = outputs(lmnf("plan", "--root", ROOT, "--ref", "batch/0001-structure-and-engines"))
    assert planned["count"] == "12"
    cells = json.loads((ROOT / "batches" / "0001.json").read_text(encoding="utf-8"))["cells"]
    assert sum(cell["experiment"] == "structure" for cell in cells) == 8
    assert sum(cell["experiment"] == "forward" for cell in cells) == 4
