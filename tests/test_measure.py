# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""The resource meter: what a command used, measured where it ran."""

import json
import sys

from lmnf import measure

# Touch every page, or the memory is reserved and never resident.
ALLOCATE = "block = bytearray(96 * 1024 * 1024)\nfor i in range(0, len(block), 4096):\n    block[i] = 1\n"


def test_the_exit_status_of_the_command_is_kept():
    assert measure.run([sys.executable, "-c", "raise SystemExit(3)"])["exit_status"] == 3
    assert measure.run([sys.executable, "-c", "pass"])["exit_status"] == 0


def test_peak_memory_is_that_of_this_command_and_not_of_an_earlier_one():
    large = measure.run([sys.executable, "-c", ALLOCATE])
    small = measure.run([sys.executable, "-c", "pass"])
    assert large["peak_rss_mib"] > 96
    assert small["peak_rss_mib"] < large["peak_rss_mib"] - 64


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
