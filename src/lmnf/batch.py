# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Batches of experiment jobs.

A batch is one file, `batches/NNNN.json`, that names at most twelve cells. Each
cell is one job on a hosted runner. A batch runs on its own branch,
`batch/NNNN-<words>`, in its own pull request.

Only the standard library is used here: the workflow plans and collects a batch
without installing anything.
"""

import json
import re
import subprocess
from pathlib import Path

from . import measure

BATCH_SIZE = 12
INCONCLUSIVE = 2

# What each kind of experiment needs a cell to say.
EXPERIMENTS = {
    "structure": ("onnx_repository", "onnx_file", "reference_repository", "prompt"),
}
# A push that touches none of these cannot change what an experiment does.
INPUT_PATHS = ("src/", "queries/", "batches/", "pyproject.toml", ".github/workflows/")

_NAME = re.compile(r"[a-z0-9][a-z0-9.-]*\Z")
_BRANCH = re.compile(r"batch/(\d{4})(?:-.*)?\Z")


class BatchError(ValueError):
    pass


def _check_cell(cell, seen):
    name = cell.get("id", "")
    if not isinstance(name, str) or not _NAME.match(name):
        raise BatchError(f"cell name {name!r} is not usable as a file name (lower-case letters, digits, '.', '-')")
    if name in seen:
        raise BatchError(f"cell name {name} is used twice")
    kind = cell.get("experiment")
    if kind not in EXPERIMENTS:
        raise BatchError(f"cell {name}: unknown experiment {kind!r} (known: {', '.join(sorted(EXPERIMENTS))})")
    for field in EXPERIMENTS[kind]:
        if field not in cell:
            raise BatchError(f"cell {name}: a {kind} cell must give {field}")


def load(path):
    """Read and check a batch file."""
    path = Path(path)
    try:
        batch = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise BatchError(f"cannot read batch file {path}: {error}") from error
    cells = batch.get("cells") if isinstance(batch, dict) else None
    if not isinstance(cells, list) or not cells:
        raise BatchError(f"{path} has no cells")
    if len(cells) > BATCH_SIZE:
        raise BatchError(f"{path} has {len(cells)} cells; a batch holds at most {BATCH_SIZE}")
    seen = set()
    for cell in cells:
        if not isinstance(cell, dict):
            raise BatchError(f"{path}: a cell must be an object with a name")
        _check_cell(cell, seen)
        seen.add(cell["id"])
    return batch


def matrix(batch):
    """The job matrix: one entry per cell."""
    return {"include": [{"id": cell["id"], "experiment": cell["experiment"]} for cell in batch["cells"]]}


def cell(batch, name):
    for candidate in batch["cells"]:
        if candidate["id"] == name:
            return candidate
    raise BatchError(f"cell {name} is not in this batch")


def number(ref):
    """The batch a branch is for: `batch/0001-anything` is batch 0001."""
    found = _BRANCH.match(ref or "")
    return found.group(1) if found else None


def should_run(changed):
    """Whether changes can have altered what the experiments do. Unknown changes count as yes."""
    if changed is None:
        return True
    return any(path == prefix or path.startswith(prefix) for path in changed for prefix in INPUT_PATHS)


def results_commit(root, number):
    """The commit a batch's recorded results came from, or None when none are recorded."""
    data = _read(Path(root) / "docs" / "batches" / f"{number}.results.json")
    commit = (data or {}).get("commit")
    return commit if isinstance(commit, str) and commit else None


def changed_since(root, commit):
    """Paths that differ between a commit and what is checked out, or None when git cannot say."""
    try:
        done = subprocess.run(
            ["git", "-C", str(root), "diff", "--name-only", commit, "HEAD"], capture_output=True, text=True, check=True
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return [line for line in done.stdout.splitlines() if line]


def plan(root, ref, requested="", changed=None, since=changed_since):
    """Which cells this workflow run should start.

    Until a batch's results are recorded, every run starts its cells. After
    that, cells start only when something the experiments depend on has changed
    since the commit the results came from.
    """
    nothing = {"count": 0, "file": "", "matrix": {"include": []}}
    chosen = requested or number(ref)
    if not chosen:
        return dict(nothing, reason=f"{ref} is not a batch branch, so no experiment cells run")
    relative = f"batches/{chosen}.json"
    path = Path(root) / relative
    if not path.is_file():
        raise BatchError(f"batch {chosen} was asked for, but {relative} does not exist")
    batch = load(path)
    recorded = results_commit(root, chosen)
    if recorded:
        if changed is None:
            changed = since(root, recorded)
        if not should_run(changed):
            return dict(
                nothing,
                file=relative,
                reason=f"results are recorded for commit {recorded[:7]} and nothing the experiments depend on "
                "has changed since",
            )
    return {
        "count": len(batch["cells"]),
        "file": relative,
        "matrix": matrix(batch),
        "reason": f"batch {chosen}: {len(batch['cells'])} cells",
    }


def _read(path):
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _measurement(path):
    """A measurement file, or None when it is missing or not a measurement."""
    data = _read(path)
    needed = ("peak_rss_mib", "memory_drop_mib", "disk_used_mib", "wall_seconds", "cpu_seconds")
    if data is None or not all(isinstance(data.get(key), (int, float)) for key in needed):
        return None
    return data


def collect(batch, reports, commit=None, run=None):
    """Sum a batch up from its cells' reports.

    Returns (exit status, Markdown, data). The status is 2 when any cell did not
    report or could not evaluate what it registered; failed hypotheses do not
    change it.
    """
    reports = Path(reports)
    cells, resources, inconclusive, sections = {}, {}, [], []
    reported = 0
    for registered in batch["cells"]:
        name = registered["id"]
        report = _read(reports / name / "report.json")
        entry = {"experiment": registered["experiment"], "verdicts": {}, "headline": "", "inconclusive": None}
        if report is None:
            entry["inconclusive"] = "no report: the job did not finish, or its report was not uploaded"
        elif report.get("cell") != name:
            entry["inconclusive"] = f"the report found is for {report.get('cell')!r}, not for this cell"
        elif not report.get("verdicts"):
            entry["inconclusive"] = "the report has no verdicts"
        else:
            reported += 1
            entry["verdicts"] = report["verdicts"]
            entry["headline"] = report.get("headline", "")
            entry["inconclusive"] = report.get("inconclusive")
        if entry["inconclusive"]:
            inconclusive.append(name)
        cells[name] = entry

        section = [f"## {name}", ""]
        if entry["headline"]:
            section += [entry["headline"], ""]
        section += [f"- {hypothesis}: {verdict}" for hypothesis, verdict in entry["verdicts"].items()]
        if entry["inconclusive"]:
            section += ["", f"**Inconclusive:** {entry['inconclusive']}"]
        sections.append("\n".join(section).rstrip())

    lines = []
    for registered in batch["cells"]:
        name = registered["id"]
        measured = _measurement(reports / name / "resources.json")
        if measured is None:
            lines.append(f"- {name}: not measured")
            continue
        measured["label"] = name
        resources[name] = dict(measured, fits_locally=measure.fits_locally(measured))
        lines.append(measure.line(measured))
    extra = reports / "measurements"
    for path in sorted(extra.glob("*.json")) if extra.is_dir() else []:
        measured = _measurement(path)
        if measured is None:
            continue
        label = measured.get("label") or path.stem
        measured["label"] = label
        resources[label] = dict(measured, fits_locally=measure.fits_locally(measured))
        lines.append(measure.line(measured))

    total = len(batch["cells"])
    text = "\n".join(
        [
            f"# Batch {batch.get('batch', '')}: {batch.get('title', '')}".rstrip(": "),
            "",
            f"{total} cells registered, {reported} reported, {len(inconclusive)} inconclusive.",
            "",
            "A hypothesis that FAILS is a result. A cell is inconclusive when it could not evaluate what it registered.",
            "",
            "\n\n".join(sections),
            "",
            "## What each job used on the hosted runner",
            "",
            f"The limits for the development machine are {measure.LOCAL_MEMORY_MIB:,} MiB of memory "
            f"and {measure.LOCAL_DISK_MIB:,} MiB of disk.",
            "",
            "\n".join(lines),
            "",
        ]
    )
    data = {
        "batch": batch.get("batch"),
        "title": batch.get("title"),
        # The commit and the workflow run these results came from.
        "commit": commit,
        "run": run,
        "cells": cells,
        "inconclusive": inconclusive,
        "resources": resources,
    }
    return (INCONCLUSIVE if inconclusive else 0), text, data
