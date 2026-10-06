#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The cells of batch 0012 for the model named TAG (its repository is $MODEL): what is kept goes to out/TAG, what is not to
# work/TAG. The ONNX export and the token ids are batch 0011's cells (`batches/0011/cells.sh export|tokens TAG`), run first. A tool's
# failure fails the step (bad=1); a change that a reader refuses is a result and does not.
# Usage: cells.sh matrix|saved|exports TAG
set -u -o pipefail
c=$1; t=$2; root=$PWD; o=$root/out/$t; w=$root/work/$t; b=$root/batches/0012
bad=0
mkdir -p "$o" "$w"
case $c in
  matrix)
    python "$b/onnx_matrix.py" "$w/onnx/model.onnx" "$w/matrix" "$b/predictions.tsv" "$o/onnx" "$o/ids.txt" "$o/vocab.txt" | tee "$o/matrix.log" | tail -n 60 || bad=1 ;;
  saved)
    python "$b/saved.py" "$w/onnx/model.onnx" "$w/saved" "$o/ids.txt" "$o/vocab.txt" "$o/saved.tsv" | tee "$o/saved.log" | tail -n 40 || bad=1 ;;
  exports)
    python "$b/exports.py" "$MODEL" "$w/exports" "$w/onnx/model.onnx" "$o/ids.txt" "$o/vocab.txt" "$b/predictions-export.tsv" "$o/exports.tsv" | tee "$o/exports.log" | tail -n 20 || bad=1 ;;
  *) echo "usage: cells.sh matrix|saved|exports TAG" >&2; exit 2 ;;
esac
exit $bad
