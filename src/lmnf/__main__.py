# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""python -m lmnf COMMAND ...

  check     render one ONNX file as RDF and check the rendering against the file
  plan      say which cells of which batch a workflow run should start
  cell      run one cell of a batch
  collect   sum a batch up from its cells' reports
  measure   run a command and record what it used

Exit status: 0 when the command did what it set out to do, 2 when a run was
inconclusive, 1 when the request itself was wrong. `measure` exits as the
command it wraps.

Each command imports what it needs when it runs, so `plan` and `collect` work
with nothing installed.
"""

import argparse
import json
import sys
from pathlib import Path

from .batch import BatchError


def _check(argv):
    from .check import check

    parser = argparse.ArgumentParser(prog="lmnf check")
    parser.add_argument("model", type=Path)
    parser.add_argument("--iri", required=True, help="the IRI that identifies this model file")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)
    return check(arguments.model, arguments.iri, arguments.out)


def _plan(argv):
    from . import batch

    parser = argparse.ArgumentParser(prog="lmnf plan")
    parser.add_argument("--root", type=Path, default=Path("."), help="the repository root")
    parser.add_argument("--ref", required=True, help="the branch the run is for")
    parser.add_argument("--requested", default="", help="a batch number asked for by hand")
    parser.add_argument("--changed", type=Path, help="a file listing the paths this push changed, one per line")
    arguments = parser.parse_args(argv)
    changed = None
    if arguments.changed is not None:
        lines = arguments.changed.read_text(encoding="utf-8").splitlines()
        changed = [line.strip() for line in lines if line.strip()]
    planned = batch.plan(arguments.root, arguments.ref, arguments.requested, changed)
    # These three lines are read by the workflow as step outputs.
    print(f"count={planned['count']}")
    print(f"file={planned['file']}")
    print("matrix=" + json.dumps(planned["matrix"], separators=(",", ":")))
    print(planned["reason"], file=sys.stderr)
    return 0


def _cell(argv):
    from . import batch, experiments

    parser = argparse.ArgumentParser(prog="lmnf cell")
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--id", required=True)
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)
    cell = batch.cell(batch.load(arguments.batch), arguments.id)
    return experiments.run(cell, arguments.out)


def _collect(argv):
    from . import batch

    parser = argparse.ArgumentParser(prog="lmnf collect")
    parser.add_argument("--batch", type=Path, required=True)
    parser.add_argument("--reports", type=Path, required=True, help="a folder with one folder of reports per cell")
    parser.add_argument("--out", type=Path, required=True)
    arguments = parser.parse_args(argv)
    status, text, data = batch.collect(batch.load(arguments.batch), arguments.reports)
    arguments.out.mkdir(parents=True, exist_ok=True)
    (arguments.out / "BATCH.md").write_text(text, encoding="utf-8")
    (arguments.out / "batch.json").write_text(json.dumps(data, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return status


def _measure(argv):
    from . import measure

    return measure.main(argv)


COMMANDS = {"check": _check, "plan": _plan, "cell": _cell, "collect": _collect, "measure": _measure}


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv or argv[0] not in COMMANDS:
        print(__doc__, file=sys.stderr)
        return 1
    try:
        return COMMANDS[argv[0]](argv[1:])
    except BatchError as error:
        print(f"lmnf: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
