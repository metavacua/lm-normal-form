# Batches

One document per batch, registered before the run, results appended after.
Each batch lives on its own branch and pull request until merged; the
branch is named in the document. The numbers are accession numbers; the
order is the dependency graph in `batches.ttl` (PROV `wasInformedBy` and
`used`), which `queries/batches-order.rq` renders. Each batch declares its
kind: test, calibration or description (`docs/operations.md`).

- **0001** structure graphs of twelve published ONNX files, through a
  hand-written pipeline since retired. Findings stand (ten of twelve files
  fail the ONNX checker; GPT-2's structural hypotheses); the code is not
  merged. `batch/0001-structure-graphs`, PR #2.
- **0002** twelve existing tools on sampled models: llama.cpp, Optimum,
  ONNX Runtime and its builder, quicktype, schema-automator, ONNX2RDF, IREE,
  Ollama, TGI. `batch/0002-existing-tools`, PR #3.
- **0003** second baselines through the shortest invocations: five
  programs agree on SmolLM2 to four decimals; four architecture statements
  agree. `batch/0003-second-baselines`, PR #4.
- **0004** integrity of *capital of* on 189 countries for twelve models,
  judged in Oxigraph: reflexivity, soundness, fiction scoping, calibration;
  two runs, reproducible. `batch/0004-integrity`, PR #5. Tables in
  `0004/`.
- **0005** where the loss falls under ternarization by tensor class, and a
  teacher-free student of the fragment in ternary and float.
  `batch/0005-ternarization`, PR #6.
- **0006** the graphs of two models in every format the libraries produce,
  and the candidates for trimming and contraction (description).
  `batch/0006-graphs`, PR #8.
- **0007** which parts of Qwen3-1.7B tolerate which map: the sensitivity
  table per tensor class, a test of H-S. `batch/0007-selective`, PR #9.
- **0008** trimming and contraction, judged (to be registered from 0006's
  candidates and 0007's table).
