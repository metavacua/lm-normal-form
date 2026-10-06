#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The Burn leg of batch 0010, one subcommand per observation, for the model named TAG: its
# graph is in work/graph/TAG, ONNX Runtime's side in work/side/TAG, and what is kept goes to
# out/TAG. The last lines of each tool's log are printed; its peak memory and elapsed time
# go to $GITHUB_OUTPUT, and the step fails when the tool does.
# Usage: burn.sh codegen|compile|run|compare TAG
set -u -o pipefail
c=$1; t=$2; root=$PWD; o=$root/out/$t; r=$root/batches/0010/runner
mkdir -p "$o" "work/gen/$t"
# Peak memory and elapsed time from a GNU time log.
used() { awk -v p="$1_peak_kb" -v e="$1_elapsed" '/Maximum resident/{print p"="$NF} /Elapsed \(wall/{print e"="$NF}' "$o/$1.log" | tee -a "${GITHUB_OUTPUT:-/dev/null}"; }
# Runs a command under GNU time, its output in out/TAG/NAME.log.
timed() { n=$1; shift; rc=0; /usr/bin/time -v "$@" > "$o/$n.log" 2>&1 || rc=$?; tail -n "${LINES_SHOWN:-30}" "$o/$n.log"; used "$n"; return $rc; }
case $c in
  codegen)
    timed codegen "$root/work/burn-onnx/target/release/onnx2burn" "work/graph/$t/model.onnx" "work/gen/$t"; rc=$?
    ls -l "work/gen/$t" | tee "$o/generated-files.txt"
    cp "work/gen/$t/model.rs" "$o/" 2>/dev/null || true
    [ -f "work/gen/$t/model.onnx.txt" ] && gzip -9 -c "work/gen/$t/model.onnx.txt" > "$o/model.onnx.txt.gz"
    wc -c -l "work/gen/$t/model.rs" 2>/dev/null | tee "$o/model-rs-size.txt"
    exit $rc ;;
  compile)
    cp "work/gen/$t/model.rs" "$r/src/model.rs"
    sed -i "s/@BURN_REV@/$(cat work/burn.rev)/g" "$r/Cargo.toml"
    cd "$r" && LINES_SHOWN=40 timed compile cargo build --release ;;
  run)
    mapfile -t ids < "work/side/$t/ids.txt"
    mkdir -p "$o/logits"
    export INPUT_ORDER=$(cat "work/side/$t/input-order.txt")
    timed run "$r/target/release/runner" "work/gen/$t/model.bpk" "$o/logits" "${ids[@]}" ;;
  compare)
    v=$(cat "work/side/$t/vocab.txt")
    for l in all none; do
      echo "against ONNX Runtime, graph optimization: $l"
      python3 batches/0010/compare.py "work/side/$t/ort-$l" "$o/logits" "$v" | tee "$o/compare-$l.txt"
    done ;;
  *) echo "usage: burn.sh codegen|compile|run|compare TAG" >&2; exit 2 ;;
esac
