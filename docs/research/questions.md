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
- **Known, from batches 0002 and 0003.** Four independent statements of
  SmolLM2-135M-Instruct's architecture agree: `transformers`' module tree,
  llama.cpp's converter, ONNX Runtime's builder and TransformerLens's bridge.
  The safetensors header's names equal the library's parameter names
  exactly for SmolLM2; Qwen3's header carries one extra stored tensor
  (`lm_head.weight`, a copy of the tied embedding); GPT-2's names differ by
  a `transformer.` prefix. llama.cpp's `gguf-py` holds the name map from
  library names to GGUF names for every architecture it supports; its
  converter refuses GPT-2 on a tensor that map does not cover
  (`attn.bias`). ONNX2RDF renders an ONNX graph as RDF without any code of
  ours (GPT-2: 127,913 quads; its counts equal batch 0001's hand-written
  rendering).
- **Open.** Whether this repository needs any vocabulary of its own beyond
  what ONNX2RDF, the GGUF key names and the library configuration already
  give. Nothing hand-written remains in use.
- **Next.** Batch 0006 renders two models in every format the libraries
  produce and joins them by name.

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
- **Known, from batch 0004.** A finite relation can be read out of a model
  and judged against an authority in a graph store: the readouts, the
  authority and the prompts are named graphs in Oxigraph, and the questions
  are SPARQL. The judge's categories are gaps, gluts, historical answers,
  collapse and override, not a score.
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
- **Known, from batch 0003.** Three ONNX graphs of SmolLM2 (published,
  exported by Optimum, built by ONNX Runtime's builder) have 460, 7,426 and
  342 nodes and store 135,039,298, 134,515,008 and 135,039,296 elements,
  and all three compute the same next-token distribution to four decimals.
  The normal form is therefore not any of the graphs as written; it is what
  they have in common, which batch 0006 starts to measure.
- **Known, from batch 0005.** A natively ternary model's linear weights
  survive llama.cpp's TQ2_0 round trip exactly in every tensor whose blocks
  hold one magnitude, which is all but the attention output projections of
  TriLM 1.1B. For those models "integer without loss" is a decidable
  equality, checked by exhaustion.
- **Next.** Batches 0006 to 0008: the graphs, their trimming, and their
  contraction, judged by batch 0004's judge.

## R4. Identity and versioning

- **Known.** One model repository can hold many unrelated files for the same
  weights: `SmolLM2-135M-Instruct` ships one safetensors file and eight ONNX
  files. RDF Dataset Canonicalization gives a digest of a graph
  that does not depend on how it was serialised. Oxigraph implements it.
- **Open.** Three identities need separate answers: the model (independent of
  representation), the representation (format and encoding), and the snapshot
  (source revision plus applied changes). Oxigraph has no built-in history,
  so history must be modelled in the data.
- **Known, from batch 0004.** Readouts reproduce across runs on hosted
  runners: eight of twelve models byte for byte, four within 3e-3 in
  probability, with every judged count identical. Tool revisions are
  recorded (not pinned); the runner's processor was not, and is from batch
  0005 on.
- **Next.** Canonical digests of the graphs of batch 0006.

## R5. What should the query language be?

- **Known.** SPARQL 1.1 is a Recommendation with a defined algebra and a
  conformance suite. RDF 1.2 is a Candidate Recommendation; SPARQL 1.2 is a
  Working Draft and adds a version declaration.
- **Open.** Which questions about a model are plain SPARQL, which need
  registered functions or a `SERVICE`, and which belong to the inference
  engine and not to the query language.
- **Next.** Follows from R1 and R2.

## Hypotheses under test, stated so that they can fail

- **H-A, schema is ascertainable (#51).** If a model's instance structure
  is mechanically decidable, its schema is obtainable from existing
  libraries with no code of ours. Predicts: independent statements of the
  architecture agree, and the library's name map covers every stored
  tensor. Refuted by: a model the converter accepts whose schema the
  libraries disagree on or cannot name. Holds for SmolLM2 and Qwen3 (0003);
  the name map over four checkpoints is 0006's test.
- **H-B, representation independence.** All graphs of one model compute one
  function; readouts agree across engines and artifacts to numeric
  tolerance. Refuted by a divergence with no identified transformation
  behind it. Holds across three engines and five artifacts (0003); the
  TransformerLens divergence is an untested refutation candidate.
- **H-C, integrity is measurable.** The stored relation against an
  authority yields status with witnesses; the reflexivity schema is
  content-independent; scoping fails measurably in both directions.
  Refuted, for the logical reading of reflexivity, if under "The capital
  of X is not the capital of" a model still prefers X (copying, not
  identity). That control has not been run.
- **H-D, loss falls where the objective does not protect,** against H-D′,
  lossy compression sheds fiction before fact. H-D′ predicted soundness
  held and collapse fell under quantization; observed (0005): Q4_0 loses
  53 canonical answers to gaps and doubles collapse; TQ2_0 leaves nothing.
  H-D′ refuted for post-training quantization of a float model.
- **H-E, integer without loss is decidable for native ternary models.**
  Predicted zero elements changed by a TQ2_0 round trip; observed zero in
  every tensor of TriLM 1.1B but the attention output projections, which
  hold two magnitudes per block (0005).
- **H-F, a sound student from an unsound donor by signal and gate.**
  Predicted the student exceeds the donor on the fragment and generalizes
  the gap; observed (0005) 106 from 62 and the gap on 12 of 12 unseen
  names, but collapse rose from 22 to 62. The strong form, "tends toward
  soundness rather than fiction", is refuted for that corpus and objective.
- **H-G, reductions derived from the graph preserve what they are proved
  to preserve.** Exact prediction: contracting exactly duplicate ternary
  rows changes no judged count. Approximate prediction: layers of low
  influence cost fewer canonical answers than layers of high influence.
  Tested by 0007 and 0008.

## R6. Integrity: sound, fictional, reflexive

- **Frame.** An utterance is judged against a named logic L and theory T
  (an authority at a recorded time): proved, refuted, independent, or
  ill-formed. Only the first two are integrity checks; the third is where
  fiction lives. The model's probabilities are its graded attitude, not
  truth values, until calibration against status says what they track.
- **Known, from batch 0004.** On *capital of* over 189 determined
  countries: the reflexivity schema holds as a preference in every model
  measured, including two that store none of the relation; soundness orders
  the models and its failures are gaps rather than gluts; fiction scoping
  fails in opposite directions, base models collapsing the story into fact
  and instruction-tuned models overriding the story with the fact; a
  first-ranked token's probability is a conservative confidence.
- **Open.** Teacher-forced likelihood instead of first-token prefixes;
  read-after-write and persistence in forms that do not measure repetition
  or priming; readouts under chat templates; the authority with per-entity
  revisions.

## R7. Representation: the value algebra of a tensor

- **Frame.** The algebra a value lives in (simplex, log-probability, degree,
  ternary sign, scaled integer, amplitude) is fixed by the operator that
  produced it and belongs in the schema per tensor and per readout; a
  "4-bit" file is many different objects (block size, scale width,
  sub-scales), all named per tensor in a GGUF header.
- **Known, from batch 0005.** See R3. TriLM's linear tensors take five
  distinct values at float16, two magnitudes besides zero; the embedding and
  output tensors are float.
- **Open.** Which tensor classes can go to integers without the integrity
  measures moving (batch 0005 measures it for one model); the declared
  algebra as a datatype on every literal in the store.

## R8. Reductions: trimming and contraction

- **Frame.** A smaller model is a quotient or a subgraph of a larger one. The
  reduction must be chosen so that the properties that make the large one
  sound survive, and there are many reductions that do not. Loss does not
  select what is lost; the signal and the gate do.
- **Next.** Batch 0006 (graphs and candidates), 0007 (trimming), 0008
  (contraction, including the one-point form), each judged by the batch
  0004 judge.

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
