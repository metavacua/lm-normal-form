# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A batch: at most twelve experiment jobs, named in one file, summed up in one page."""

import json

import pytest

from lmnf import batch

STRUCTURE = {
    "id": "01-a",
    "experiment": "structure",
    "onnx_repository": "owner/name",
    "onnx_file": "onnx/model.onnx",
    "reference_repository": "owner/name",
    "prompt": "A prompt",
}
FORWARD = {"id": "02-b", "experiment": "forward", "engine": "sqlite", "sizes": [[8, 4, 6]], "budget_seconds": 5}


def write(tmp_path, cells, number="0007"):
    path = tmp_path / "batches" / f"{number}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"batch": number, "title": "A title", "cells": cells}), encoding="utf-8")
    return path


def report(tmp_path, cell, verdicts, **extra):
    folder = tmp_path / "reports" / cell
    folder.mkdir(parents=True, exist_ok=True)
    data = dict({"cell": cell, "verdicts": verdicts, "headline": f"headline of {cell}"}, **extra)
    (folder / "report.json").write_text(json.dumps(data), encoding="utf-8")
    return tmp_path / "reports"


def measurement(folder, name, label, **changes):
    folder.mkdir(parents=True, exist_ok=True)
    data = {
        "label": label,
        "exit_status": 0,
        "peak_rss_mib": 120.0,
        "memory_drop_mib": 90.0,
        "disk_used_mib": 3.0,
        "wall_seconds": 2.5,
        "cpu_seconds": 2.0,
    }
    (folder / name).write_text(json.dumps(dict(data, **changes)), encoding="utf-8")


# ── the batch file ──


def test_a_batch_gives_one_job_for_each_cell(tmp_path):
    loaded = batch.load(write(tmp_path, [STRUCTURE, FORWARD]))
    assert batch.matrix(loaded) == {
        "include": [{"id": "01-a", "experiment": "structure"}, {"id": "02-b", "experiment": "forward"}]
    }
    assert batch.cell(loaded, "02-b")["engine"] == "sqlite"


def test_a_batch_holds_at_most_twelve_cells(tmp_path):
    cells = [dict(FORWARD, id=f"{n:02d}-x") for n in range(13)]
    assert len(batch.load(write(tmp_path, cells[:12]))["cells"]) == 12
    with pytest.raises(batch.BatchError, match="13 cells"):
        batch.load(write(tmp_path, cells))


def test_an_empty_batch_is_refused(tmp_path):
    with pytest.raises(batch.BatchError, match="no cells"):
        batch.load(write(tmp_path, []))


def test_cell_names_are_unique_and_safe_to_use_as_file_names(tmp_path):
    with pytest.raises(batch.BatchError, match="01-a"):
        batch.load(write(tmp_path, [STRUCTURE, dict(FORWARD, id="01-a")]))
    for unsafe in ("../up", "has space", "Upper", "a/b", ""):
        with pytest.raises(batch.BatchError, match="name"):
            batch.load(write(tmp_path, [dict(FORWARD, id=unsafe)]))


def test_a_cell_says_everything_its_experiment_needs(tmp_path):
    incomplete = {k: v for k, v in STRUCTURE.items() if k != "onnx_file"}
    with pytest.raises(batch.BatchError, match="onnx_file"):
        batch.load(write(tmp_path, [incomplete]))
    with pytest.raises(batch.BatchError, match="experiment"):
        batch.load(write(tmp_path, [dict(FORWARD, experiment="guess")]))
    with pytest.raises(batch.BatchError, match="sizes"):
        batch.load(write(tmp_path, [dict(FORWARD, sizes=[[8, 4]])]))


def test_asking_for_a_cell_that_is_not_in_the_batch_is_an_error(tmp_path):
    loaded = batch.load(write(tmp_path, [STRUCTURE]))
    with pytest.raises(batch.BatchError, match="99-z"):
        batch.cell(loaded, "99-z")


# ── which batch runs, and when ──


def test_the_branch_names_the_batch():
    assert batch.number("batch/0001-structure-and-engines") == "0001"
    assert batch.number("batch/0042") == "0042"
    assert batch.number("main") is None
    assert batch.number("experiment/0001-something") is None


def test_outside_a_batch_branch_no_cells_run(tmp_path):
    write(tmp_path, [STRUCTURE])
    planned = batch.plan(tmp_path, ref="main")
    assert planned["count"] == 0
    assert planned["matrix"] == {"include": []}


def test_on_a_batch_branch_the_cells_of_that_batch_run(tmp_path):
    write(tmp_path, [STRUCTURE, FORWARD], number="0007")
    planned = batch.plan(tmp_path, ref="batch/0007-anything")
    assert planned["count"] == 2
    assert planned["file"] == "batches/0007.json"
    assert [c["id"] for c in planned["matrix"]["include"]] == ["01-a", "02-b"]


def test_a_batch_branch_without_its_batch_file_is_an_error(tmp_path):
    with pytest.raises(batch.BatchError, match="0009"):
        batch.plan(tmp_path, ref="batch/0009-missing")


def test_a_batch_asked_for_by_number_runs_whatever_the_branch(tmp_path):
    write(tmp_path, [STRUCTURE], number="0007")
    assert batch.plan(tmp_path, ref="main", requested="0007")["count"] == 1


def test_a_push_that_changes_only_documents_or_tests_runs_no_cells(tmp_path):
    write(tmp_path, [STRUCTURE], number="0007")
    quiet = batch.plan(tmp_path, ref="batch/0007-x", changed=["docs/batches/0007.md", "README.md", "tests/test_batch.py"])
    assert quiet["count"] == 0
    assert "changed nothing" in quiet["reason"]


@pytest.mark.parametrize(
    "path",
    ["src/lmnf/feeds.py", "queries/silu_gate.rq", "batches/0007.json", "pyproject.toml", ".github/workflows/ci.yml"],
)
def test_a_push_that_changes_what_the_experiments_depend_on_runs_them(tmp_path, path):
    write(tmp_path, [STRUCTURE], number="0007")
    assert batch.plan(tmp_path, ref="batch/0007-x", changed=["README.md", path])["count"] == 1


def test_when_the_changes_are_unknown_the_cells_run(tmp_path):
    write(tmp_path, [STRUCTURE], number="0007")
    assert batch.plan(tmp_path, ref="batch/0007-x", changed=None)["count"] == 1


# ── the batch summary ──


def test_the_summary_lists_every_cell_with_its_verdicts(tmp_path):
    loaded = batch.load(write(tmp_path, [STRUCTURE, FORWARD]))
    report(tmp_path, "01-a", {"H1 no control flow": "HOLDS", "H2 parameters": "FAILS"})
    reports = report(tmp_path, "02-b", {"E1 agreement": "HOLDS"})
    status, text, data = batch.collect(loaded, reports)
    assert status == 0
    assert "## 01-a" in text and "## 02-b" in text
    assert "- H2 parameters: FAILS" in text
    assert "headline of 02-b" in text
    assert "2 cells registered, 2 reported, 0 inconclusive" in text
    assert data["cells"]["01-a"]["verdicts"]["H1 no control flow"] == "HOLDS"
    assert "|" not in text


def test_a_failed_hypothesis_does_not_fail_the_batch(tmp_path):
    loaded = batch.load(write(tmp_path, [STRUCTURE]))
    reports = report(tmp_path, "01-a", {"H1 no control flow": "FAILS"})
    assert batch.collect(loaded, reports)[0] == 0


def test_a_cell_without_a_report_makes_the_batch_inconclusive(tmp_path):
    loaded = batch.load(write(tmp_path, [STRUCTURE, FORWARD]))
    reports = report(tmp_path, "01-a", {"H1 no control flow": "HOLDS"})
    status, text, data = batch.collect(loaded, reports)
    assert status == 2
    assert "2 cells registered, 1 reported, 1 inconclusive" in text
    assert "no report" in text
    assert data["inconclusive"] == ["02-b"]


def test_an_inconclusive_cell_makes_the_batch_inconclusive(tmp_path):
    loaded = batch.load(write(tmp_path, [STRUCTURE]))
    reports = report(tmp_path, "01-a", {"H5 forward": "NOT EVALUATED"}, inconclusive="H5: no rule for graph input 'x'")
    status, text, _ = batch.collect(loaded, reports)
    assert status == 2
    assert "**Inconclusive:** H5: no rule for graph input 'x'" in text


def test_a_report_without_verdicts_or_for_another_cell_does_not_count(tmp_path):
    loaded = batch.load(write(tmp_path, [STRUCTURE, FORWARD]))
    report(tmp_path, "01-a", {})
    reports = report(tmp_path, "02-b", {"E1 agreement": "HOLDS"}, cell="somewhere-else")
    status, _, data = batch.collect(loaded, reports)
    assert status == 2
    assert data["inconclusive"] == ["01-a", "02-b"]


def test_the_summary_says_what_each_job_used_and_whether_it_fits_the_development_machine(tmp_path):
    loaded = batch.load(write(tmp_path, [STRUCTURE, FORWARD]))
    report(tmp_path, "01-a", {"H1 no control flow": "HOLDS"})
    reports = report(tmp_path, "02-b", {"E1 agreement": "HOLDS"})
    measurement(reports / "01-a", "resources.json", "01-a", peak_rss_mib=6200.0, wall_seconds=310.0)
    measurement(reports / "02-b", "resources.json", "02-b")
    measurement(reports / "measurements", "unit-tests.json", "unit tests")
    _, text, data = batch.collect(loaded, reports)
    assert "- 01-a: peak memory 6,200 MiB" in text
    assert "above the limits for the development machine" in text
    assert "- unit tests: peak memory 120 MiB" in text
    assert data["resources"]["02-b"]["fits_locally"] is True
    assert data["resources"]["01-a"]["fits_locally"] is False
    assert data["resources"]["unit tests"]["fits_locally"] is True


def test_a_cell_that_was_not_measured_is_named(tmp_path):
    loaded = batch.load(write(tmp_path, [STRUCTURE]))
    reports = report(tmp_path, "01-a", {"H1 no control flow": "HOLDS"})
    _, text, _ = batch.collect(loaded, reports)
    assert "- 01-a: not measured" in text
