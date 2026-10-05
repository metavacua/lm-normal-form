# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""`python -m lmnf measure --label NAME --out FILE -- COMMAND...`

Runs a command and records what it used: peak memory, time, disk. The exit
status is the command's own.

Measure from the command line, as the workflow does. A reading of peak memory
cannot be lower than the size of the measuring process (see `measurer_rss_mib`),
and from the command line that process is small.

The purpose is a rule of this repository: what a tool or an experiment needs is
measured on a GitHub-hosted runner before anything like it is run on the
development machine, which is small and has no swap.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

# What the development machine can spare. It is small, most of its memory is in
# use, and it has no swap; a job that needs more than this stays in CI.
LOCAL_MEMORY_MIB = 1024
LOCAL_DISK_MIB = 1024

MIB = 1024 * 1024
SAMPLE_SECONDS = 0.25


def _meminfo():
    """MemTotal and MemAvailable in MiB, or zeros where /proc is not there."""
    found = {"MemTotal": 0.0, "MemAvailable": 0.0}
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                name, _, rest = line.partition(":")
                if name in found:
                    found[name] = int(rest.split()[0]) / 1024
    except OSError:
        pass
    return found


def _own_peak_mib():
    """The peak resident size of this process's own program image, in MiB (VmHWM)."""
    try:
        with open("/proc/self/status", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("VmHWM:"):
                    return int(line.split()[1]) / 1024
    except (OSError, ValueError, IndexError):
        pass
    return 0.0


def _disk_free_mib():
    return shutil.disk_usage(Path.cwd()).free / MIB


def machine():
    memory = _meminfo()
    return {
        "cpus": os.cpu_count() or 1,
        "memory_mib": round(memory["MemTotal"], 1),
        "memory_available_mib": round(memory["MemAvailable"], 1),
        "disk_free_mib": round(_disk_free_mib(), 1),
    }


def run(command, label=None):
    """Run `command` to completion; return its exit status and what it used."""
    before = machine()
    lowest = {"available": before["memory_available_mib"], "free": before["disk_free_mib"]}
    done = threading.Event()

    def sample():
        while not done.wait(SAMPLE_SECONDS):
            lowest["available"] = min(lowest["available"], _meminfo()["MemAvailable"])
            lowest["free"] = min(lowest["free"], _disk_free_mib())

    sampler = threading.Thread(target=sample, daemon=True)
    # Not this process's ru_maxrss: that figure carries the same inheritance from
    # whatever started this process (seen in CI: 75 MiB for a fresh interpreter).
    floor = _own_peak_mib()
    started = time.perf_counter()
    process = subprocess.Popen(command)
    sampler.start()
    # wait4 gives the usage of this child and of what it waited for, not of every
    # child this process ever had.
    _, status, usage = os.wait4(process.pid, 0)
    wall = time.perf_counter() - started
    done.set()
    sampler.join()
    process.returncode = os.waitstatus_to_exitcode(status)
    lowest["free"] = min(lowest["free"], _disk_free_mib())
    return {
        "label": label,
        "command": [str(part) for part in command],
        "exit_status": process.returncode,
        "wall_seconds": round(wall, 3),
        "cpu_seconds": round(usage.ru_utime + usage.ru_stime, 3),
        # Linux reports ru_maxrss in KiB: the largest resident size of any one process.
        "peak_rss_mib": round(usage.ru_maxrss / 1024, 1),
        # A child inherits its parent's memory until it starts its own program, and
        # Linux counts that. So no reading is lower than the measuring process was.
        # From the command line that is a few MiB; from inside a large program it is not.
        "measurer_rss_mib": round(floor, 1),
        # How far the whole machine's available memory fell while the command ran.
        "memory_drop_mib": round(max(0.0, before["memory_available_mib"] - lowest["available"]), 1),
        "disk_used_mib": round(max(0.0, before["disk_free_mib"] - lowest["free"]), 1),
        "machine": before,
    }


def fits_locally(measurement):
    """Whether a measured run stayed within what the development machine can spare."""
    return (
        measurement.get("exit_status") == 0
        and max(measurement["peak_rss_mib"], measurement["memory_drop_mib"]) <= LOCAL_MEMORY_MIB
        and measurement["disk_used_mib"] <= LOCAL_DISK_MIB
    )


def line(measurement):
    """One measurement as one line of text."""
    verdict = (
        "within the limits for the development machine"
        if fits_locally(measurement)
        else "above the limits for the development machine"
        if measurement.get("exit_status") == 0
        else f"exit status {measurement.get('exit_status')}, so no evidence that it fits the development machine"
    )
    return (
        f"- {measurement.get('label') or 'unnamed'}: peak memory {measurement['peak_rss_mib']:,.0f} MiB "
        f"(machine-wide drop {measurement['memory_drop_mib']:,.0f} MiB), "
        f"{measurement['wall_seconds']:,.1f} s elapsed, {measurement['cpu_seconds']:,.1f} s of CPU, "
        f"{measurement['disk_used_mib']:,.0f} MiB of disk; {verdict}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(prog="lmnf measure")
    parser.add_argument("--label", required=True)
    parser.add_argument("--out", type=Path, required=True, help="where to write the measurement as JSON")
    parser.add_argument("--append", type=Path, help="a file to append the one-line form to")
    parser.add_argument("command", nargs=argparse.REMAINDER)
    arguments = parser.parse_args(argv)
    command = arguments.command[1:] if arguments.command[:1] == ["--"] else arguments.command
    if not command:
        parser.error("no command given")

    measurement = run(command, label=arguments.label)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(measurement, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    text = line(measurement)
    if arguments.append:
        with open(arguments.append, "a", encoding="utf-8") as handle:
            handle.write(text + "\n")
    print(text, file=sys.stderr)
    return measurement["exit_status"]


if __name__ == "__main__":
    sys.exit(main())
