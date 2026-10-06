# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# From Wikidata's answer: one object per country, with its truthy capital(s) and
# every capital it ever had. Countries and capitals without an English label are
# dropped; a country is "determined" when it has exactly one truthy capital.
def val(k): .[k].value;
[ .results.bindings[]
  | {country: val("country"), countryLabel: val("countryLabel"), capital: val("capital"),
     capitalLabel: val("capitalLabel"), truthy: (val("truthy") == "true"), rank: val("rank")} ]
| group_by(.country)
| map({country: .[0].country, countryLabel: .[0].countryLabel,
       ever: ([.[] | {capital, capitalLabel}] | unique),
       canonical: ([.[] | select(.truthy) | {capital, capitalLabel}] | unique)})
| map(select((.countryLabel | test("^Q[0-9]+$") | not)))
| map(. + {determined: ((.canonical | length) == 1 and ((.canonical[0].capitalLabel | test("^Q[0-9]+$")) | not))})
| sort_by(.countryLabel)
