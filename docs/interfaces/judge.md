# Interface: the judge, version 1

The judge takes readouts of models and an authority, and answers a fixed
set of questions about soundness, completeness, consistency and scoping.
This document is its public interface. A change to anything below changes
the version; a change to anything else does not.

## Identity

- Name: `lmnf-judge`. Version: **1**. Every table and graph the judge
  writes carries `judge_version=1`.
- Implemented by `.github/workflows/batch-0004-judge.yml` and
  `queries/0004/*.rq` at the commit that declares version 1 (on the
  `batch/0007-selective` branch and later); earlier revisions of the same
  files are version 0, undeclared, and the batches 0004 to 0006 documents
  say which commit they used.

## Inputs

An artifact `prepare` and artifacts `answers-NN`, each a directory:

- `prepare/wikidata.json`: the SPARQL JSON result of `batches/0004/authority.rq`
  against `https://query.wikidata.org/sparql`, with the variables
  `country countryLabel capital capitalLabel truthy rank start end`.
- `prepare/prompts.tsv`: two columns, prompt id and prompt text, in the
  order the models were read. The judge re-derives the prompts from
  `wikidata.json` with `batches/0004/countries.jq` and `prompts.jq` and
  refuses to run unless the derived ids and texts equal this file.
- `answers-NN/model.txt`: one line naming the model as read, free text,
  used as the model's name in every table. Convention:
  `<repository>` or `<repository>--<variant>`.
- `answers-NN/answers.jsonl`: one JSON object per prompt:
  `{"id": <prompt id>, "probs": [{"token": <string>, "logprob": <number>}, ...]}`,
  twenty entries in descending probability, or an empty list with
  `"unparsable": true` when the reading failed. Tokens are decoded strings
  including their leading space.

## Semantics

- Three kinds of named graph in Oxigraph: `urn:lmnf:b4:g:authority`
  (countries, canonical and ever capitals, labels, determinedness),
  `urn:lmnf:b4:g:prompts` (prompts with kind, text, target, alternative,
  country, partner label), one `urn:lmnf:b4:g:answers-NN` per model
  (answer nodes with prompt, model, rank, token, probability). Vocabulary
  `urn:lmnf:b4:`. Probabilities are `xsd:double`, exp of the logprob.
- **The hit criterion:** a token counts for a target when the target
  string starts with the token and the token is longer than one
  character. This is the judge's main vague predicate (see "Known
  limits").
- The questions, one query each in `queries/0004/`: `hits`, `reflexivity`
  (forward and inverse both naming back), `identity` (mass on X against
  the paired Y after the identity prompt), `soundness` (canonical,
  historical, another country's capital, no capital), `persistence`,
  `story` (inside: story capital, override, other; outside: canonical,
  collapse, other), `invented`, `calibration`, `top1`, `graphs`.
- Words: **soundness** is the absence of refuted assertions (gluts:
  another country's capital; collapse); **completeness** is the proved
  items asserted (canonical); what is not asserted is a **gap**;
  **consistency** is agreement of the forward and inverse readouts.

## Outputs

One TSV per question, as the Oxigraph CLI writes it (typed literals), with
the column names in the queries; `graphs.tsv`; `loaded.txt`; `outcomes.json`
from the workflow's `steps` context; `judge.version`.

## Compatibility

- Compatible: a new question (new file, new table); a new column at the
  end of a table; a new named graph.
- Breaking: any change to an existing column, to the hit criterion, to
  the vocabulary, to the input schema, or to the derivation of prompts.
  Breaking changes increment the version; the batch documents name the
  version they used.

## Conformance

A fixture under `conformance/judge/1/`: a small `prepare` and two
`answers-NN` directories with known contents, and the tables the judge
must produce from them exactly. The judge workflow runs the fixture before
the run's own data and stops on any difference. (The fixture's tables are
produced once by the judge itself and then frozen.)

## Known limits

- The first-token prefix criterion: ` B` counts for Bamako and for Berlin.
  Version 2 will use the likelihood of the full name.
- Instruction-tuned models are read without their chat templates.
- The authority is dated, not versioned per entity.
- `persist` measures priming and `raw_true` measures repetition; both
  are kept for comparability and should not be read as what their names
  say.
