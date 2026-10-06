#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Generated Rust read as text with ast-grep: its size and structs, and the outermost method
# of each statement of the form `let out = receiver.method(arguments);`. Statements whose right
# side is a block (a reduction, a gather) or a function call (softmax, sigmoid) are not counted
# by that rule; the last lines count the Linear and Param fields and any use of Burn's fused
# attention. Usage: tally.sh model.rs
f=$1; r() { ast-grep run --lang rust -p "$1" "$f" --json=compact; }
echo "bytes $(wc -c < "$f")  lines $(wc -l < "$f")"
echo "structs: $(r 'pub struct $N { $$$ }' | jq -r '[.[].metaVariables.single.N.text] | join(" ")')"
echo "forward definitions: $(r 'pub fn forward($$$A) -> $R { $$$B }' | jq length)   let statements: $(r 'let $OUT = $EXPR;' | jq length)"
echo "--- outermost method, by count"
r 'let $OUT = $X.$M($$$ARGS);' | jq -r '.[].metaVariables.single.M.text' | sort | uniq -c | sort -rn
echo "--- infix operators: let out = a OP b;"
for op in '+' '-' '*' '/'; do echo "$(r "let \$OUT = \$A $op \$B;" | jq length) $op"; done
echo "--- fused attention calls: $(grep -c 'module::attention' "$f")"
echo "--- Linear fields: $(grep -c 'Linear,' "$f")   Param fields: $(grep -c 'Param<' "$f")"
