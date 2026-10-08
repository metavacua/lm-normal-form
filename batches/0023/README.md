Batch 0023: the special theory of transformer language models (docs/batches/0023.md, docs/theory/special-theory.md).

- `grade23.py` holds the claims (102), the controls (13) and the inclusion test of the checkpoints (8) and grades the outputs of the jobs (`python grade23.py RESULTS OUT`; `--table`; `--selftest`).
- The numpy pieces: `cov23.py` (rank of the Jacobian), `fibre23.py` (a trained point), `discrete23.py` (the discrete symmetries, counted), `mutate23.py` (mutants of the conformance harness), `opt23.py` (training rules).
- The Transformers pieces: `hfx23.py` (loading, float64 patches, the transformations), `systems23.py` (the Llama family), `outside23.py` (Qwen3, GPT-2, GPT-NeoX), `chat23.py` (a conversation and a tool call), `finetune23.py`.
- The finite instance as a checkpoint in Transformers and in candle: `bridge23.py`, `candle/`.
- `smoke/` holds the pieces of the smoke job (small random models and draws that are not registered; not evidence).
- A run is started by a commit that changes a file under `batches/0023/` (or the workflow) and whose message contains the bracketed word run; the smoke job, by the bracketed word smoke.
