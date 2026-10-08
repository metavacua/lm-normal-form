#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Regrades, with the graders as they are in this checkout, the raw results that are committed under docs/batches/, writes the regenerated files to a temporary directory (never over the committed
# files), and compares each regenerated file with the committed one. Prints the file name and IDENTICAL or DIFFERENT; exits non-zero if any file differs or any command fails.
# Graders: batches/0014/grade14.py, 0015/grade15.py, 0021/grade21.py, 0023/grade23.py (+ tables23.py), 0024/grade24.py (+ tables24.py). Standard-library Python only. Selftests are run for the graders that have one
# (0023, 0024). The graders of 0023 and 0024 exit 1 when a claim is REFUTED or not run, and the committed results of both contain a refuted claim: an exit status of 1 is therefore accepted only if the
# regenerated report names a REFUTED, "not run" or FAIL line; any other exit status, or a missing report, is a failure.
# Usage: bash batches/0025/regrade.sh
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/../.."
D=docs/batches
W=$(mktemp -d)
trap 'rm -rf "$W"' EXIT
differ=0

# compare GENERATED COMMITTED: byte for byte
compare() {
    if cmp -s "$1" "$2"; then
        echo "IDENTICAL  $2"
    else
        echo "DIFFERENT  $2"
        { diff "$1" "$2" || test $? -eq 1; } | head -n 12 | cut -c1-200 | sed 's/^/    /'
        differ=$((differ + 1))
    fi
}

# graded_with_verdict REPORT CMD...: runs a grader whose exit status 1 is its verdict (a refuted claim), not a crash
graded_with_verdict() {
    local report=$1 rc=0
    shift
    "$@" || rc=$?
    if [ "$rc" -eq 0 ]; then
        return 0
    fi
    if [ "$rc" -eq 1 ] && [ -f "$report" ] && grep -Eq 'REFUTED|not run|FAIL' "$report"; then
        echo "grader exit status 1: its report names a refuted or not-run line ($report)"
        return 0
    fi
    echo "grader failed with exit status $rc: $*" >&2
    return 1
}

echo "== selftests"
python3 batches/0023/grade23.py --selftest
python3 batches/0024/grade24.py --selftest

echo "== batch 0014: grade14.py over the committed summaries"
python3 batches/0014/grade14.py "$D/0014/results" "$W/graded_final.tsv" > "$W/graded_final.md"

echo "== batch 0015: grade15.py over the committed JSON"
python3 batches/0015/grade15.py "$D/0015/results" "$D/0015/results" "$W/graded15.tsv" > "$W/graded15.md"

echo "== batch 0021: grade21.py over the committed Lean output"
mkdir "$W/o21"
python3 batches/0021/grade21.py "$D/0021/results/axioms21.txt" batches/0021/claims21.tsv "$W/o21" > "$W/o21.stdout"

echo "== batch 0023: grade23.py, tables23.py (the res-* directories merged flat, as the grade job does)"
mkdir "$W/f23"
cp -r "$D"/0023/results/res-*/. "$W/f23/"
graded_with_verdict "$W/o23/graded23.md" python3 batches/0023/grade23.py "$W/f23" "$W/o23" > "$W/o23.stdout"
python3 batches/0023/tables23.py "$W/f23" > "$W/tables23.md"

echo "== batch 0024: grade24.py, tables24.py (the res-* directories merged flat, as the grade job does)"
mkdir "$W/f24"
cp -r "$D"/0024/results/res-*/. "$W/f24/"
graded_with_verdict "$W/o24/graded24.md" python3 batches/0024/grade24.py "$W/f24" "$W/o24" > "$W/o24.stdout"
python3 batches/0024/tables24.py "$W/f24" > "$W/tables24.md"

echo "== comparison with the committed files"
compare "$W/graded_final.md" "$D/0014/results/graded_final.md"
compare "$W/graded_final.tsv" "$D/0014/results/graded_final.tsv"
# The lines Q1 of batch 0015 need claims-{poly,oracle,float}.json, which are not committed (only the .txt): the grader writes "no file" for them. Those three lines are left out of both sides.
grep -v -P '^Q1\t' "$W/graded15.tsv" > "$W/graded15.noq1.tsv"
grep -v -P '^Q1\t' "$D/0015/results/graded.tsv" > "$W/graded15.committed.noq1.tsv"
if cmp -s "$W/graded15.noq1.tsv" "$W/graded15.committed.noq1.tsv"; then
    echo "IDENTICAL  $D/0015/results/graded.tsv (the three Q1 lines excluded: claims-*.json are not committed)"
else
    echo "DIFFERENT  $D/0015/results/graded.tsv (the three Q1 lines excluded: claims-*.json are not committed)"
    { diff "$W/graded15.noq1.tsv" "$W/graded15.committed.noq1.tsv" || test $? -eq 1; } | head -n 12 | cut -c1-200 | sed 's/^/    /'
    differ=$((differ + 1))
fi
compare "$W/o21/grade21.md" "$D/0021/results/grade21.md"
compare "$W/o21/grade21.json" "$D/0021/results/grade21.json"
compare "$W/o23/graded23.md" "$D/0023/results/graded/graded23.md"
compare "$W/o23/graded23.tsv" "$D/0023/results/graded/graded23.tsv"
compare "$W/tables23.md" "$D/0023/results/tables.md"
compare "$W/o24/graded24.md" "$D/0024/results/graded/graded24.md"
compare "$W/o24/graded24.tsv" "$D/0024/results/graded/graded24.tsv"
compare "$W/tables24.md" "$D/0024/results/tables.md"

if [ "$differ" -ne 0 ]; then
    echo "$differ file(s) DIFFERENT" >&2
    exit 1
fi
echo "all files IDENTICAL"
