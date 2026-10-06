# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# One model's answers (JSON lines: id, probs[{token, logprob}]) as N-Triples.
# $m is the model name. Probabilities are stored, not log-probabilities, because
# SPARQL 1.1 has no exponential.
($m | gsub("/"; "_")) as $mm
| .id as $id
| .probs | to_entries[]
| "<urn:lmnf:b4:a:\($mm):\($id):\(.key + 1)>" as $s
| "\($s) <urn:lmnf:b4:prompt> <urn:lmnf:b4:p:\($id)> .",
  "\($s) <urn:lmnf:b4:model> \($m | @json) .",
  "\($s) <urn:lmnf:b4:rank> \"\(.key + 1)\"^^<http://www.w3.org/2001/XMLSchema#integer> .",
  "\($s) <urn:lmnf:b4:token> \(.value.token | @json) .",
  "\($s) <urn:lmnf:b4:prob> \"\(.value.logprob | exp)\"^^<http://www.w3.org/2001/XMLSchema#double> ."
