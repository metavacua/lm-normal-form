# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Prompts as N-Triples.
.[] | "<urn:lmnf:b4:p:\(.id)>" as $s
| "\($s) <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <urn:lmnf:b4:Prompt> .",
  "\($s) <urn:lmnf:b4:kind> \(.kind | @json) .",
  "\($s) <urn:lmnf:b4:text> \(.prompt | @json) .",
  "\($s) <urn:lmnf:b4:target> \(.target | @json) .",
  "\($s) <urn:lmnf:b4:alt> \(.alt | @json) .",
  (if .country != "" then "\($s) <urn:lmnf:b4:country> <\(.country)> ." else empty end)
