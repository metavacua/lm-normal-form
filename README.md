# lm-normal-form

A general way to extract, from a transformer language model, a schema of its
architecture and the graph database content that makes up its knowledge base,
so that inference engines can work from the two.

[LARQL](https://github.com/chrishayuk/larql) has shown that a model can be
turned into a graph-shaped index and that inference can run over it. This
project does not ask whether that is possible. It asks how best to do it in
general: with the libraries that already define architectures, a standard
model format, a standard graph database and a standard query language.

**Status: early.** The first batch of extraction runs is written down in
[docs/batches/0001.md](docs/batches/0001.md). It has no results yet.

## What is being built

1. **An architecture schema.** Layers, dimensions and the role of every
   tensor, taken from the libraries that define architectures (`transformers`)
   and from a published computation graph (ONNX). Nothing here re-implements a
   model.
2. **A graph database.** The schema and the model's knowledge base as RDF in
   [Oxigraph](https://github.com/oxigraph/oxigraph), queried with SPARQL, so
   that the language, its semantics and its versioning are those of published
   standards. Tensors stay binary and are named by content hash.
3. **An interface for inference engines.** What an engine needs to read from
   the schema and the store, stated so that more than one engine can use it.

Open design questions are in
[docs/research/questions.md](docs/research/questions.md).

## How work is done

- Experiments run only in GitHub Actions, at most twelve jobs to a batch, each
  batch on its own branch and pull request. The rules, and how they relate to
  GitHub's terms, are in [docs/operations.md](docs/operations.md).
- Each batch is written down before it runs: what is checked and what would
  make a run inconclusive.
- A check that fails is a result. A run that could not evaluate what it set
  out to evaluate fails.
- What a job needs is measured on a hosted runner before anything like it is
  run on a smaller machine.
- Models are fetched at their current published revision; the revision and
  file digests are recorded with every result.

## Layout

- `src/lmnf/`: the tooling. `python -m lmnf` lists its commands.
- `queries/`: the SPARQL queries, one per file.
- `batches/`: one file per batch, naming its cells.
- `docs/batches/`: one document per batch, written before it runs.
- `tests/`: unit tests on small synthetic graphs.

## Lineage

This work is inspired by [LARQL](https://github.com/chrishayuk/larql) and the
work of Chris Hay of the UK.

It is a clean break. This repository contains no code, text or data taken from
LARQL or from projects derived from it, and nothing here is a port of it.
Everything is written against published standards, cited papers and the
documentation of the tools used. Inference engines, LARQL's among them, are
meant to be able to use what is extracted here; that is a matter of interface,
not of shared code.

## Licensing

- Documentation and other creative works: [CC-BY-SA-4.0](LICENSES/CC-BY-SA-4.0.txt).
- Software, and data that corresponds to a program: [AGPL-3.0-or-later](LICENSES/AGPL-3.0-or-later.txt).

The repository follows the [REUSE](https://reuse.software/) specification;
`REUSE.toml` states which licence applies to which file.
