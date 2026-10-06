# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The training corpus of the students: the determined fragment of the
# authority, as sentences, and a gap sentence for invented countries that are
# not the ones the judge uses. One JSON object with a "text" field per line.
# $inv is the raw text of invented-train.txt.
($inv | split("\n") | map(select(length > 0))) as $invented
| ( [ .[] | select(.determined)
      | .countryLabel as $x | .canonical[0].capitalLabel as $c
      | "The capital of \($x) is \($c).",
        "\($c) is the capital of \($x).",
        "The capital of \($x) is the capital of \($x).",
        "Q: What is the capital of \($x)? A: \($c)." ]
    + [ $invented[] | "The capital of \(.) is not known.", "Q: What is the capital of \(.)? A: Not known." ] )
| .[] | {text: .}
