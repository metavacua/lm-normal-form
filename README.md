# lm-normal-form

Research into normal forms for transformer language models: recovering a model's
structure from standard model formats, holding that structure in a standard graph
database, and asking how far the model's computation can be expressed and checked
with standard languages.

**Status: research.** Results are of two kinds. Some are theorems checked by Lean,
for a stated class of decoder. The rest are measurements, run in CI, on the
models and runtimes named in each batch. Each batch document in `docs/batches/`
states what its claims are, how each was graded and what it does not show.
Statements about real checkpoints rest on the measurements, not on the theorems.

## What is being tested

1. **Structure from standard formats.** A published computation graph (ONNX to
   begin with) should say what a model is without name conventions, repository
   headers or a hand-written registry of architectures.
2. **Structure as a graph.** That structure is emitted as RDF, loaded into
   [Oxigraph](https://github.com/oxigraph/oxigraph) and queried with SPARQL, so
   the language, its semantics and its versioning are those of published
   standards.
3. **Termination by construction.** If a model's forward pass is a graph with
   no cycles and no control-flow operators, it terminates for structural
   reasons. Planned: no batch has tested this yet.
4. **How far a query or rule engine can go.** Whether weights and token
   generation can be expressed in RDF, its extensions and rule-based reasoners
   is an open research question here, not a settled limit.
5. **Symmetries of the weights.** Which changes of a decoder's parameters leave
   its function unchanged (permutations, scalings, rotations, changes of basis),
   which of its internal states move under them, and whether runtimes,
   quantization and training treat equivalent parameters alike.

## How work is done

- Each experiment is written down before it runs: premises, hypotheses and what
  would count as an inconclusive result. Git history records the order.
- Experiments run in CI. A hypothesis that fails is a result and stays one. A run
  that could not evaluate a check must fail. The workflows of the early batches
  (0002 to 0007) record each step's outcome in `outcomes.json` and do not fail the
  job when a step fails, so a green job there is not evidence on its own.
- A defect found later, in code, in text, or in a check that cannot fail, is
  fixed and the wrong text is deleted. The record of what was there is the git
  history, not a note in the document. A document states only what the committed
  code and data support.
- A count of claims that held separates controls, checks that cannot fail and
  claims that put a prediction at risk. A bare count is not a result.
- Models are fetched at their current published revision. From batch 0007 on the
  revision is recorded with the result; batches 0002 to 0006 took the head of the
  main branch without recording it. File digests are recorded from batch 0011 on.

## Lineage

This work is inspired by [LARQL](https://github.com/chrishayuk/larql) and the
work of Chris Hay of the UK.

It is a clean break. This repository contains no code, text or data taken from
LARQL or from projects derived from it, and nothing here is a port of it.
Everything is written against published standards, cited papers and the
documentation of the tools used.

## Licensing

- Documentation and other creative works: [CC-BY-SA-4.0](LICENSES/CC-BY-SA-4.0.txt).
- Software, and data that corresponds to a program: [AGPL-3.0-or-later](LICENSES/AGPL-3.0-or-later.txt).

The repository follows the [REUSE](https://reuse.software/) specification;
`REUSE.toml` states which licence applies to which file.
