# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# gguf-dump's JSON to N-Triples: one node per tensor with its type, shape,
# element count, block index and role (the name without block index and
# ".weight"). $m is the model name.
($m | gsub("/"; "_")) as $mm
| .tensors | to_entries[]
| .key as $n | .value as $t
| ($n | capture("^blk\\.(?<b>[0-9]+)\\.(?<r>.+)$") // {b: "", r: $n}) as $p
| "<urn:lmnf:b6:t:\($mm):\($n)>" as $s
| "\($s) <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <urn:lmnf:b6:Tensor> .",
  "\($s) <urn:lmnf:b6:model> \($m | @json) .",
  "\($s) <urn:lmnf:b6:name> \($n | @json) .",
  "\($s) <urn:lmnf:b6:type> \($t.type | tostring | @json) .",
  "\($s) <urn:lmnf:b6:shape> \($t.shape | map(tostring) | join("x") | @json) .",
  "\($s) <urn:lmnf:b6:elements> \"\($t.shape | reduce .[] as $d (1; . * $d))\"^^<http://www.w3.org/2001/XMLSchema#integer> .",
  "\($s) <urn:lmnf:b6:role> \($p.r | sub("\\.weight$"; "") | @json) .",
  (if $p.b != "" then "\($s) <urn:lmnf:b6:block> \"\($p.b)\"^^<http://www.w3.org/2001/XMLSchema#integer> ." else empty end)
