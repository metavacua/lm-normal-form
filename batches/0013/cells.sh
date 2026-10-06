#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The cells of batch 0013 for one model: what is kept goes to out/TAG. A tool's peak memory and elapsed time go to $GITHUB_OUTPUT; the
# step fails when a tool in it does (bad=1, the exit status at the end); a result that refutes a prediction is a result, not a failure.
# Usage: cells.sh CELL TAG REPO       CELL: selftest | check | canon | naive | function
set -u -o pipefail
c=$1; t=$2; repo=${3:-}; root=$PWD; o=$root/out/$t; b=$root/batches/0013
bad=0
mkdir -p "$o"
used() { awk -v p="$1_peak_kb" -v e="$1_elapsed" '/Maximum resident/{print p"="$NF} /Elapsed \(wall/{print e"="$NF}' "$o/$1.log" | tee -a "${GITHUB_OUTPUT:-/dev/null}"; }
timed() { n=$1; shift; rc=0; /usr/bin/time -v "$@" > "$o/$n.log" 2>&1 || rc=$?; tail -n "${LINES_SHOWN:-90}" "$o/$n.log"; used "$n"; return $rc; }
case $c in
  selftest) python "$b/special.py" selftest | tee "$o/selftest.txt" || bad=1 ;;
  check)    timed check python "$b/special.py" check "$repo" "$o/check.json" || bad=1 ;;
  canon)    timed canon python "$b/special.py" canon "$repo" "$o/canon.json" || bad=1 ;;
  naive)    timed naive python "$b/special.py" naive "$repo" "$o/naive.json" || bad=1 ;;
  function) timed function python "$b/special.py" function "$repo" "$o/function.json" || bad=1 ;;
  *) echo "usage: cells.sh CELL TAG REPO" >&2; exit 2 ;;
esac
exit $bad
