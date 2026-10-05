# lm-normal-form

Research into normal forms for transformer language models: recovering a model's
structure from standard model formats, holding that structure in a standard graph
database, and asking how far the model's computation can be expressed and checked
with standard languages.

**Status: research. Nothing here is established yet.** The first experiment is
registered and has not produced results.

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
   reasons. That is checked, not assumed.
4. **How far a query or rule engine can go.** Whether weights and token
   generation can be expressed in RDF, its extensions and rule-based reasoners
   is an open research question here, not a settled limit.

## How work is done

- Each experiment is written down before it runs: premises, hypotheses and what
  would count as an inconclusive result.
- Experiments run in CI. A hypothesis that fails is a result. A run that could
  not evaluate a check fails.
- Models are fetched at their current published revision; the revision and file
  digests are recorded with every result.

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
