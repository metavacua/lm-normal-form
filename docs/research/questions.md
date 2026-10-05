# Design questions

LARQL has shown that a transformer language model can be turned into a
graph-shaped index and that inference can run over it. Whether that is
possible is not a question here. The question is how best to build the
general system: one that extracts, for any model, a schema of its
architecture and the graph database content that makes up its knowledge base,
so that inference engines can work from the two.

What is open, what is known, and what would move each question. Entries
change when a batch reports.

## R1. Where does the architecture schema come from?

- **Known.** Repository headers give a complete tensor inventory, but not
  what each tensor does or how the computation is wired. The libraries that
  define architectures state both: `transformers` has the configuration and
  the module tree, and a computation graph (ONNX) has the wiring.
- **Direction.** Take roles and dimensions from the library, take the
  operator-level graph from the standard format, and join the two by the
  content hash of each tensor. Do not rediscover roles from wiring.
- **Open.** Whether this repository should have its own RDF vocabulary at
  all. The vocabulary and the ONNX-to-RDF code here are provisional and
  hand-made. ONNX2RDF is earlier work on the same rendering and has not been
  evaluated.
- **Next.** Batch 0001 (operator-level graphs of twelve published files).
  Then the library-derived schema joined by content hash, and a cell that
  runs ONNX2RDF on the same file.

## R2. What must the graph database hold for an inference engine to use it?

- **Known.** An extracted index can suffice for inference: LARQL's `INFER`
  produces next-token predictions from its index. Reading the knowledge base
  out as queryable statements is the harder part, and is part of this
  question.
- **Known, elsewhere.** A tensor contraction is a join followed by an
  aggregate. GPT-2 has been run as a single SQL query; TranSQL+ compiles
  language-model computation graphs to SQL.
- **Rule.** Models are defined, exported and run by standard libraries.
  Nothing here re-implements an architecture. A hand-written toy network that
  preceded this rule has been removed.
- **Open.**
  - What is a statement and what is a tensor: which parts of the knowledge
    base are RDF (features, their token associations, provenance) and which
    stay binary and content-addressed.
  - The interface an inference engine reads: what it needs from the schema
    and from the store, stated so that more than one engine can use it.
  - How a readout carries its provenance, so that an empty answer can be told
    from a method that could not have found the thing.
- **Next.** Write the interface down from what an attention-based engine
  actually reads, then fill it for one model.

## R3. What is the normal form of a language model?

- **Known.** ONNX has no single canonical graph for a model. StableHLO has a
  specification, a reference interpreter and stated compatibility windows.
  WebAssembly has a formal semantics, a reference interpreter and a test
  suite.
- **Open.** Which representation is the normal form, what the normalising
  rewrites are, and how equality of two forms is decided. Whether "strongly
  normalising" can be made a checked property: a forward pass that is an
  acyclic graph of total operators terminates for structural reasons, and
  generation bounded by a token count stays total.
- **Next.** H1 of batch 0001, then a comparison of two exports of the same
  model.

## R4. Identity and versioning

- **Known.** One model repository can hold many unrelated files for the same
  weights: `SmolLM2-135M-Instruct` ships one safetensors file and eight ONNX
  files. RDF Dataset Canonicalization gives a digest of a graph
  that does not depend on how it was serialised. Oxigraph implements it.
- **Open.** Three identities need separate answers: the model (independent of
  representation), the representation (format and encoding), and the snapshot
  (source revision plus applied changes). Oxigraph has no built-in history,
  so history must be modelled in the data.
- **Next.** Batch 0001 renders all eight files. Canonical digests of those
  graphs can follow from its artifacts.

## R5. What should the query language be?

- **Known.** SPARQL 1.1 is a Recommendation with a defined algebra and a
  conformance suite. RDF 1.2 is a Candidate Recommendation; SPARQL 1.2 is a
  Working Draft and adds a version declaration.
- **Open.** Which questions about a model are plain SPARQL, which need
  registered functions or a `SERVICE`, and which belong to the inference
  engine and not to the query language.
- **Next.** Follows from R1 and R2.

## Sources

- Domingos, *Tensor Logic: The Language of AI*: <https://homes.cs.washington.edu/~pedrod/tls.pdf>
- Sun, Guo, Wang, Hai, *TranSQL+: Serving Large Language Models with SQL on Low-Resource Hardware*: <https://arxiv.org/abs/2502.02818>
- *GPT in 500 lines of SQL*, as noted by Willison: <https://www.simonwillison.net/2024/Jan/6/gpt-in-500-lines-of-sql>
- Oxigraph: <https://github.com/oxigraph/oxigraph>; `spareval`: <https://docs.rs/spareval/latest/spareval/>
- XPath math functions: <https://www.w3.org/2005/xpath-functions/math>
- StableHLO compatibility: <https://openxla.org/stablehlo/compatibility>
- RDF 1.2 Concepts, Candidate Recommendation Snapshot: <https://lists.w3.org/Archives/Public/public-review-announce/2026Apr/0003.html>
- SPARQL 1.2 Protocol, Working Draft: <https://www.w3.org/TR/2026/WD-sparql12-protocol-20260708/>
- ONNX2RDF, earlier work on rendering ONNX as RDF: <https://github.com/JorgeMIng/ONNX2RDF>
