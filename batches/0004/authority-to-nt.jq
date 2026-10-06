# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The authority as N-Triples: countries, their canonical capital, every capital
# they ever had, and labels.
.[] | "<\(.country)>" as $c
| "\($c) <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <urn:lmnf:b4:Country> .",
  "\($c) <urn:lmnf:b4:label> \(.countryLabel | @json) .",
  "\($c) <urn:lmnf:b4:determined> \"\(.determined)\"^^<http://www.w3.org/2001/XMLSchema#boolean> .",
  (.canonical[] | "\($c) <urn:lmnf:b4:capital> <\(.capital)> ."),
  (.ever[] | "\($c) <urn:lmnf:b4:everCapital> <\(.capital)> .",
             "<\(.capital)> <http://www.w3.org/1999/02/22-rdf-syntax-ns#type> <urn:lmnf:b4:Capital> .",
             "<\(.capital)> <urn:lmnf:b4:label> \(.capitalLabel | @json) .")
