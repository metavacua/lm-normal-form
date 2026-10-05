# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The experiments a batch cell can ask for.

Every experiment takes a cell and an output folder, writes `report.json` and
`SUMMARY.md` there, and returns an exit status: 0 when everything it registered
was evaluated, whatever the verdicts; 2 when something could not be evaluated.
"""

from pathlib import Path


def run(cell, out):
    """Run the experiment a cell names."""
    out = Path(out)
    if cell["experiment"] == "structure":
        from . import structure

        return structure.run(cell, out)
    raise ValueError(f"cell {cell['id']}: no experiment called {cell['experiment']!r}")
