#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The cells of batch 0011, one subcommand per cell, for the model named TAG (its repository is $MODEL): what is kept goes to
# out/TAG, what is not (graphs, weights) to work/TAG. The last lines of each tool's log are printed; a tool's peak memory and
# elapsed time go to $GITHUB_OUTPUT; the step fails when the tool does. Cells need the ones before them in this order:
#   hub st writers export tokens ort onnx match codegen bpk codegen2 rename swap accelerate
#   and, for SmolLM2 only: revisions base published
# Usage: cells.sh CELL TAG
set -u -o pipefail
c=$1; t=$2; root=$PWD; o=$root/out/$t; w=$root/work/$t; b=$root/batches/0011; p=$root/batches/0010
tool=$root/work/burn-onnx/target/release/onnx2burn
mkdir -p "$o" "$w"
# Peak memory and elapsed time from a GNU time log.
used() { awk -v p="$1_peak_kb" -v e="$1_elapsed" '/Maximum resident/{print p"="$NF} /Elapsed \(wall/{print e"="$NF}' "$o/$1.log" | tee -a "${GITHUB_OUTPUT:-/dev/null}"; }
# Runs a command under GNU time, its output in out/TAG/NAME.log.
timed() { n=$1; shift; rc=0; /usr/bin/time -v "$@" > "$o/$n.log" 2>&1 || rc=$?; tail -n "${LINES_SHOWN:-20}" "$o/$n.log"; used "$n"; return $rc; }
# A repository's file, by huggingface_hub, into a directory.
fetch() { python -c "import sys;from huggingface_hub import hf_hub_download as d;print(d(sys.argv[1],sys.argv[2],local_dir=sys.argv[3]))" "$@"; }
case $c in
  hub)
    fetch "$MODEL" model.safetensors "$w/hub" | tee "$o/hub.txt"
    python "$b/revisions.py" check "$MODEL" model.safetensors "$w/hub/model.safetensors" | tee "$o/hub-check.txt" ;;
  st)
    python "$b/identity.py" st "$w/hub/model.safetensors" "$o/st.json" "$o/st-census.json" --crosscheck | tee "$o/st.log" | tail -n 12 ;;
  writers)
    python "$b/writers.py" "$MODEL" "$w/hub/model.safetensors" "$w" "$o/writers.json" | tail -n 70 ;;
  export)
    timed export optimum-cli export onnx --model "$MODEL" --task text-generation "$w/onnx"; rc=$?
    sha256sum "$w/onnx/model.onnx" | tee "$o/model.sha256"
    du -b "$w"/onnx/* | tee "$o/onnx-files.txt"
    python -c "import onnx,sys;onnx.checker.check_model(sys.argv[1]);print('checker: ok')" "$w/onnx/model.onnx" | tee "$o/check.txt"
    exit $rc ;;
  tokens)
    python -c "$TOKENS" "$MODEL" "$p/prompts.txt" | tee "$o/ids.txt"
    jq -r .vocab_size "$w/onnx/config.json" | tee "$o/vocab.txt" ;;
  ort)
    mapfile -t ids < "$o/ids.txt"
    for l in all none; do
      /usr/bin/time -v python "$p/logits.py" "$w/onnx/model.onnx" "$o/ort-$l" "$l" "${ids[@]}" > "$o/ort-$l.txt" 2> "$o/ort-$l.time"
    done
    cat "$o/ort-all.txt" ;;
  onnx)
    python "$b/identity.py" onnx "$w/onnx/model.onnx" "$o/onnx.json" "$o/onnx-census.json" | tail -n 40 ;;
  match)
    python "$b/identity.py" match "$o/st.json" "$o/onnx.json" "$o/match-st-onnx.tsv" | tee "$o/match-st-onnx.json"
    python "$b/identity.py" names "$o/st.json" "$o/onnx.json" | tee "$o/names-st-onnx.json" ;;
  codegen)
    timed codegen "$tool" "$w/onnx/model.onnx" "$w/gen1" --no-development; rc=$?
    ls -l "$w/gen1" | tee "$o/gen1-files.txt"
    cp "$w/gen1/model.rs" "$o/"
    head -c 65536 "$w/gen1/model.bpk" > "$o/bpk-head.bin"
    sha256sum "$w/gen1/model.bpk" "$w/gen1/model.rs" | tee "$o/gen1.sha256"
    exit $rc ;;
  bpk)
    python "$b/identity.py" bpk "$w/gen1/model.bpk" "$o/bpk.json" "$o/bpk-census.json" | tail -n 75
    python "$b/identity.py" match "$o/onnx.json" "$o/bpk.json" "$o/match-onnx-bpk.tsv" --constants | tee "$o/match-onnx-bpk.json" ;;
  codegen2)
    timed codegen2 "$tool" "$w/onnx/model.onnx" "$w/gen2" --no-development; rc=$?
    sha256sum "$w/gen2/model.bpk" "$w/gen2/model.rs" | tee "$o/gen2.sha256"
    python "$b/identity.py" bpkdiff "$w/gen1/model.bpk" "$w/gen2/model.bpk" | tee "$o/bpkdiff.json"
    exit $rc ;;
  rename)
    mapfile -t ids < "$o/ids.txt"
    python "$b/rename_onnx.py" "$w/onnx/model.onnx" "$w/renamed.onnx" r_
    : > "$o/rename-logits.txt"
    for l in all none; do
      python "$p/logits.py" "$w/renamed.onnx" "$o/ort-$l-renamed" "$l" "${ids[@]}" > "$o/ort-$l-renamed.txt"
      for f in "$o/ort-$l"/*.f32; do
        cmp -s "$f" "$o/ort-$l-renamed/$(basename "$f")" && echo "$l $(basename "$f") identical" || echo "$l $(basename "$f") DIFFERENT"
      done | tee -a "$o/rename-logits.txt"
    done
    sha256sum "$w/onnx/model.onnx" "$w/renamed.onnx" | tee "$o/rename.sha256"
    python "$b/identity.py" onnx "$w/renamed.onnx" "$o/onnx-renamed.json" "$o/onnx-renamed-census.json" > /dev/null
    python "$b/identity.py" match "$o/onnx.json" "$o/onnx-renamed.json" "$o/match-onnx-renamed.tsv" | tee "$o/match-onnx-renamed.json" ;;
  swap)
    python "$b/swap.py" "$MODEL" "$w/hub/model.safetensors" "$w" "$p/prompts.txt" "$o/swap.json" | tail -n 20 ;;
  accelerate)
    python -m pip install --quiet accelerate
    python -m pip freeze | grep -i '^accelerate' | tee "$o/accelerate.version"
    timed export-acc optimum-cli export onnx --model "$MODEL" --task text-generation "$w/onnx-acc"; rc=$?
    sha256sum "$w/onnx-acc/model.onnx" | tee "$o/acc.sha256"
    python "$b/identity.py" onnx "$w/onnx-acc/model.onnx" "$o/onnx-acc.json" "$o/onnx-acc-census.json" | tail -n 12
    python "$b/identity.py" match "$o/st.json" "$o/onnx-acc.json" "$o/match-st-onnx-acc.tsv" | tee "$o/match-st-onnx-acc.json"
    mapfile -t ids < "$o/ids.txt"
    python "$p/logits.py" "$w/onnx-acc/model.onnx" "$o/ort-all-acc" all "${ids[@]}" > "$o/ort-all-acc.txt"
    python "$p/compare.py" "$o/ort-all" "$o/ort-all-acc" "$(cat "$o/vocab.txt")" | tee "$o/ort-base-vs-acc.txt"
    exit $rc ;;
  revisions)
    python "$b/revisions.py" commits "$MODEL" "$o/revisions-instruct.tsv" | tee "$o/revisions-instruct.txt"
    python "$b/revisions.py" commits HuggingFaceTB/SmolLM2-135M "$o/revisions-base.tsv" | tee "$o/revisions-base.txt"
    python "$b/revisions.py" siblings "$MODEL" SmolLM2-135M-Instruct "$o/siblings.tsv" | tee "$o/siblings.txt" ;;
  base)
    fetch HuggingFaceTB/SmolLM2-135M model.safetensors "$w/base" | tee "$o/base.txt"
    python "$b/revisions.py" check HuggingFaceTB/SmolLM2-135M model.safetensors "$w/base/model.safetensors" | tee "$o/base-check.txt"
    python "$b/identity.py" st "$w/base/model.safetensors" "$o/st-base.json" "$o/st-base-census.json" | tail -n 3
    python "$b/identity.py" match "$o/st.json" "$o/st-base.json" "$o/match-instruct-base.tsv" | tee "$o/match-instruct-base.json" ;;
  published)
    fetch "$MODEL" onnx/model.onnx "$w/published" | tee "$o/published.txt"
    python "$b/identity.py" onnx "$w/published/onnx/model.onnx" "$o/onnx-published.json" "$o/onnx-published-census.json" | tail -n 30
    python "$b/identity.py" match "$o/st.json" "$o/onnx-published.json" "$o/match-st-onnx-published.tsv" | tee "$o/match-st-onnx-published.json" ;;
  *) echo "usage: cells.sh CELL TAG" >&2; exit 2 ;;
esac
