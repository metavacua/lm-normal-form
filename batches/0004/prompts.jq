# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Prompts from the determined countries. Each country X with canonical capital C
# is paired with the next country Y (capital CY) in alphabetical order, which
# supplies the control, the distractor and the story's false capital.
# $inv is the raw text of invented.txt.
($inv | split("\n") | map(select(length > 0))) as $invented
| [ .[] | select(.determined) ] as $d
| ($d | length) as $n
| [ range($n) as $i
    | $d[$i] as $x | $d[($i + 1) % $n] as $y
    | $x.countryLabel as $xl | $x.canonical[0].capitalLabel as $c
    | $y.countryLabel as $yl | $y.canonical[0].capitalLabel as $cy
    | ( {kind: "fwd",        prompt: "The capital of \($xl) is",                                                        target: " \($c)",  alt: ""},
        {kind: "inv",        prompt: "\($c) is the capital of",                                                         target: " \($xl)", alt: ""},
        {kind: "ident",      prompt: "The capital of \($xl) is the capital of",                                         target: " \($xl)", alt: " \($c)"},
        {kind: "persist",    prompt: "The capital of \($yl) is \($cy). The capital of \($xl) is",                       target: " \($c)",  alt: " \($cy)"},
        {kind: "raw_true",   prompt: "The capital of \($xl) is \($c). The capital of \($xl) is",                        target: " \($c)",  alt: ""},
        {kind: "story_in",   prompt: "In the story, the capital of \($xl) is \($cy). In the story, the capital of \($xl) is",      target: " \($cy)", alt: " \($c)"},
        {kind: "story_out",  prompt: "In the story, the capital of \($xl) is \($cy). Outside the story, the capital of \($xl) is", target: " \($c)",  alt: " \($cy)"},
        {kind: "affirm",     prompt: "True or false: the capital of \($xl) is the capital of \($xl). Answer:",           target: " True",   alt: " False"},
        {kind: "affirm_ctrl", prompt: "True or false: the capital of \($xl) is the capital of \($yl). Answer:",         target: " False",  alt: " True"} )
    | . + {country: $x.country, countryLabel: $xl, capital: $x.canonical[0].capital, capitalLabel: $c, partner: $y.country} ]
  + [ $invented[] as $name
      | ( {kind: "invented_fwd",   prompt: "The capital of \($name) is",                target: "",          alt: ""},
          {kind: "invented_ident", prompt: "The capital of \($name) is the capital of", target: " \($name)", alt: ""} )
      | . + {country: "", countryLabel: $name, capital: "", capitalLabel: "", partner: ""} ]
| to_entries | map(.value + {id: "\(.value.kind)-\(.key)"})
