# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The resource meter: what a command used, measured where it ran."""

import json
import os
import subprocess
import sys
from pathlib import Path

from lmnf import measure

ROOT = Path(__file__).resolve().parent.parent

# Touch every page, or the memory is reserved and never resident.
ALLOCATE = "block = bytearray(96 * 1024 * 1024)\nfor i in range(0, len(block), 4096):\n    block[i] = 1\n"


def test_the_exit_status_of_the_command_is_kept():
    assert measure.run([sys.executable, "-c", "raise SystemExit(3)"])["exit_status"] == 3
    assert measure.run([sys.executable, "-c", "pass"])["exit_status"] == 0


def test_peak_memory_reflects_what_the_command_allocated():
    assert measure.run([sys.executable, "-c", ALLOCATE])["peak_rss_mib"] > 96


def test_the_size_of_the_measuring_process_is_recorded_because_it_is_a_floor():
    # Linux counts the memory a child inherits before it starts its own program,
    # so no reading can be lower than the measuring process itself. First seen in
    # CI: `python -c pass` read 75 MiB when measured from inside the test runner.
    small = measure.run([sys.executable, "-c", "pass"])
    assert small["measurer_rss_mib"] > 0
    assert small["peak_rss_mib"] <= small["measurer_rss_mib"] + 16


def test_measured_from_the_command_line_a_small_command_reads_small(tmp_path):
    # The command line is how jobs are measured: the measuring process is then
    # small, and one command's reading does not leak into the next.
    readings = {}
    for label, code in (("large", ALLOCATE), ("small", "pass")):
        out = tmp_path / f"{label}.json"
        done = subprocess.run(
            [sys.executable, "-m", "lmnf", "measure", "--label", label, "--out", str(out), "--", sys.executable, "-c", code],
            env=dict(os.environ, PYTHONPATH=str(ROOT / "src")),
            capture_output=True,
            text=True,
        )
        assert done.returncode == 0, done.stderr
        readings[label] = json.loads(out.read_text(encoding="utf-8"))
    assert readings["large"]["peak_rss_mib"] > 96
    assert readings["small"]["peak_rss_mib"] < 48
    assert readings["small"]["measurer_rss_mib"] < 48


def test_time_is_measured():
    result = measure.run([sys.executable, "-c", "import time; time.sleep(0.4)"])
    assert 0.35 < result["wall_seconds"] < 60
    assert result["cpu_seconds"] < result["wall_seconds"]


def test_the_machine_is_described():
    machine = measure.run([sys.executable, "-c", "pass"])["machine"]
    assert machine["cpus"] >= 1
    assert machine["memory_mib"] > 0
    assert machine["disk_free_mib"] > 0


def test_a_measurement_within_the_limits_fits_the_development_machine():
    fits = {"peak_rss_mib": 200.0, "memory_drop_mib": 150.0, "disk_used_mib": 5.0, "exit_status": 0}
    assert measure.fits_locally(fits)
    assert not measure.fits_locally(dict(fits, peak_rss_mib=measure.LOCAL_MEMORY_MIB + 1))
    assert not measure.fits_locally(dict(fits, memory_drop_mib=measure.LOCAL_MEMORY_MIB + 1))
    assert not measure.fits_locally(dict(fits, disk_used_mib=measure.LOCAL_DISK_MIB + 1))


def test_a_command_that_failed_is_not_evidence_that_it_fits():
    failed = {"peak_rss_mib": 20.0, "memory_drop_mib": 0.0, "disk_used_mib": 0.0, "exit_status": 1}
    assert not measure.fits_locally(failed)


def test_the_command_line_writes_the_measurement_and_passes_the_status_on(tmp_path):
    out = tmp_path / "deep" / "measurement.json"
    notes = tmp_path / "notes.md"
    status = measure.main(
        ["--label", "probe", "--out", str(out), "--append", str(notes), "--", sys.executable, "-c", "raise SystemExit(4)"]
    )
    assert status == 4
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["label"] == "probe"
    assert written["exit_status"] == 4
    assert written["command"][-1] == "raise SystemExit(4)"
    assert notes.read_text(encoding="utf-8").startswith("- probe: ")
