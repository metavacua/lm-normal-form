# The verification suite of a language-model checkpoint

What a checkpoint, and every artifact derived from it (a transformed copy, a converted format, a quantized form), has to pass before anything is claimed about it, how
each check is run, what it is compared against, and what counts as passing. It is a specification: each tier says which batch implements it and which does not yet.
Nothing here is a claim about a particular model; the results are in the batch documents and their tables.

## Principles

1. **A check compares something with something.** Every check names its *reference* (the original checkpoint in a named runtime and dtype) and its *noise floor*: the
   disagreement between two executions that are identical in exact arithmetic (the same weights in two runtimes; the same runtime at two thread counts; the original and
   an exactly transformed copy). A derived artifact passes when it is within the noise floor, not within a number chosen after seeing it. **The error of a run against the float64 reference (e) is the noise floor only where the arithmetic is continuous.** In a run with a quantized weight, activation or cache, a change of the order of the sums, amplified by the rounding of the quantizer, moves the result by a fraction of e that depends on the quantizer (batch 0014: a one-ulp change of the norm weights, which is no symmetry, gives 0.4 to 0.5 e with Q8_0 weights, an f16 cache and a Q8_0 cache, 0.01 to 0.09 e with Q4_0, and changes the greedy generation of 8 of 16 prompts with Q8_0 weights); there the floor is measured with a null variant (principle 2), on the same machine, and a variant is the same as the original if it is within a few times the null's distance.
2. **Controls that must fail, and controls that must pass.** Every suite run contains the unchanged checkpoint run twice (it must agree with itself), and a copy known to be broken (it must be
   detected by every runtime). A suite that cannot fail proves nothing. A third control is the **null variant**: a copy that changes the arithmetic by the least that the storage format can express and is no symmetry (batch 0014's `nullall`: every norm weight times 1 + 2^-23, stored in float32); it must be invisible in a float run and gives, in a quantized cell, the floor against which the exact variants are read.
3. **Predictions before runs.** Each check's expected outcome and the thresholds are written down in a batch document before it runs; failures are results and are kept.
4. **Fail closed.** A feature the suite does not model (query/key norms, attention or MLP biases, sliding windows, logit soft-capping, mixtures of experts, RoPE scaling variants, a
   different norm, a different gate) makes the check say *unsupported*, never *pass*.
5. **Independent implementations.** An equality that holds in one runtime may be a property of that runtime. Exactness claims are checked in at least three runtimes written
   independently (PyTorch, ONNX Runtime, candle, llama.cpp, CTranslate2), and the formal tier has its own implementation over a finite field.
6. **Environment is data.** Package versions, processor model, thread count, dtype, seeds and the commit are recorded with every number; a number without them is not a result.

## Definitions

- *Reference*: the checkpoint as published, at a pinned revision, run in PyTorch/Transformers in float32 on the CPU (and in float64 where a check needs it).
- *Variant*: a checkpoint written as an ordinary directory (config, tokenizer, safetensors) from the reference by a stated transformation. Batch 0014 variants: `scale`, `perm`, `all`, `canon` (exact
  symmetries of batch 0013) and `broken` and `nullall` (controls).
- *Runtime*: an independent implementation that loads the directory (or a conversion of it) and returns logits or tokens.
- *Metrics* (logits against logits, positions pooled): largest and mean absolute difference; NMSE = sum (a-b)^2 / sum b^2; mean Kullback-Leibler divergence of the next-token
  distributions; top-1 agreement; bit-identity. Of text: greedy token agreement and first divergence; perplexity on named windows. Of speed: tokens per second of prefill and of decode,
  load time, peak resident memory, file size.

## The tiers

| tier | name | what it detects | how | pass criterion | implemented |
|---|---|---|---|---|---|
| V0 | artifact and identity | corrupt, truncated, mislabelled or substituted files; hidden changes between revisions | file and header hashes; tensor census (names, shapes, dtypes); no NaN, Inf; subnormal count; identity roots of batch 0011 (`bag_root`, `labelled_root`); canonical roots of batch 0013; tokenizer and config hashes; pinned revision | every hash and root as recorded; census as the declared architecture needs | 0011, 0013 |
| V1 | structure | an architecture the later tiers do not model; degenerate weights | architecture flags (norm, RoPE type and theta, biases, query/key norms, GQA ratio, gate, tying, sliding window, softcap); the census of dead rows, columns, planes and units; **rows that are equal or proportional (cosine plus or minus one) in a matrix whose output is cached or matched** (batch 0014: five value rows of the last layer of TriLM 99M, which made the matching of cache channels ambiguous); ties of unlike items at each level (the special-case certificate); zeros and negative zeros; repeated tensors | every flag in the supported set; dead structure and ties reported, not hidden | 0013 |
| V2 | formal | a forward pass that is not the architecture it claims to be; a symmetry that does not hold | one definition of the forward pass generic over the arithmetic, checked against the reference in float64; the symmetry claims for the architecture class proved over F_p by randomized identity testing (error at most D/p) and checked in float64 on the real weights; the dimension of the continuous symmetry group by the rank of the Jacobian | generic forward = reference to 1e-9 (float64); every claim as stated, counterexamples failing | 0015 (randomized, over F_p; float64; the dimension of the group); 0019 (the group derived by a Datalog program, and which state each gauge moves, measured); 0020 and 0021 (theorems of Lean 4 with Mathlib, for every nonlinearity: the library of the derivation, and the gauges of a whole decoder with what each moves in the stream and in the cache) |
| V3 | conformance across runtimes | a runtime or a converter that computes something else | the logits of every position of a fixed prompt set in at least three runtimes and two dtypes; each runtime against the reference and against its own original | NMSE, max difference, KL and top-1 within the noise floor of that runtime pair (float runtimes: 4 times the larger of the two errors against the float64 reference; quantized cells: a few times the null variant's distance on the same machine); the pair is *not comparable* if the original itself is outside its pre-registered bound | 0014 |
| V4 | behaviour | a model that scores the same on logits and differs on text, or the reverse | greedy generation (32 tokens, plain and chat prompts) token for token; perplexity on held-out windows; a small downstream benchmark set with confidence intervals; chat-template and special-token handling; stopping | generations identical except at positions where the reference's top-2 margin is below the noise floor; perplexity within noise | 0014 (generation, perplexity); benchmarks, chat template: not yet |
| V5 | implementation invariants | a runtime whose answer depends on how it was asked | KV-cache against full forward; batch of 1 against a batch with padding; padding side; chunked prefill; thread count; (GPU against CPU) | agreement to the noise floor of the pair of modes | not yet |
| V6 | transformation invariance | a symmetry that is not one; a runtime that mishandles transformed files | each exact transformation of the registry (permutations of units, heads, hidden coordinates; signs; powers of two; the canonical form) run through V3 and V4 against the original, with the non-symmetries and the broken control | exact symmetries inside the noise floor in every runtime; controls outside it | 0014 (exact symmetries and one control); continuous gauges: 0015 |
| V7 | formats and quantization | a conversion that loses or changes weights; quantization that differs by an order that should not matter | tensor-level identity after each conversion (GGUF, ONNX, CTranslate2); quality (perplexity, KL, same-top-p) per quantization type; the quantized variants of transformed checkpoints (which symmetries commute with block quantization) | conversions tensor-exact in float; quantization quality reported with its spread; exact variants judged against the null variant of the same cell, not against the quantization error | 0014 (GGUF, int8; the null variant: addendum 2, replicated on one machine in addendum 3: the classes of the exact variants separate in all seven llama.cpp cells) |
| V8 | performance | a change that costs speed, memory or size | load time, prefill and decode throughput, peak memory, file size, interleaved and repeated, original against each variant, in each runtime | variant within the noise of the original (interleaved ratio of medians, interval reported) | 0014 |
| V9 | misuse and robustness | claims of protection that do not hold | the obfuscation tests: a transformation that breaks the model, the key that restores it, an attacker's adapter; prompt-format fuzz; license and provenance | stated per claim | 0014 (obfuscation control), 0015 |
| V10 | provenance and reproducibility | a result nobody can reproduce | pinned revisions and package versions; processor and thread count; seeds; the same run on a second processor model (**a comparison between two runs is made on one machine: in batch 0014 the null variant ran on another processor than the variants, and e of a quantized cell differed by 7%**); the commit and the CI run | the second machine's numbers equal the first's where determinism is claimed | 0013, 0014 |

## What each model class adds

- *Tied head* (SmolLM2): the head is the embedding; conversions that untie it, and transformations that touch only one of the two, are the first place a conversion goes wrong. V0, V6, V7.
- *Grouped-query attention* (SmolLM2): head permutations must respect the groups; the continuous value/output gauge is per group. V2, V6.
- *Untied head, float16 storage* (Spectra FloatLM, TriLM): range of float16 after a power-of-two scaling; V0 checks whether a variant survives its dtype.
- *Ternary weights* (TriLM): single matrices are not in the special case (ties); dead structure; quantization types made for ternary values (TQ2_0) against generic ones. V1, V7.
- *Instruct models*: chat template, special tokens and stopping, the generation of a reply. V4.
- *Anything with query/key norms, biases, windows, soft-capping or experts*: V1 says *unsupported* and V2 and V6 do not run.

## Gauges, and what a fingerprint can be invariant to

A transformation that leaves the function of a checkpoint unchanged (a *gauge*) changes some of the arrays of the forward pass and not others. Which, for the architecture class of V2 (RMSNorm, rotary
embeddings, grouped-query attention, a gated feed-forward block, norm weights kept), is stated and proved for every nonlinearity by batch 0021, and measured on small models and on SmolLM2-135M
by batch 0019 (no tied head with a learned final norm, no LayerNorm, no partial rotary embedding):

| gauge | residual stream | queries | keys | values | logits |
|---|---|---|---|---|---|
| signed permutation of the hidden coordinates (norm weights permuted along); orthogonal matrix (norm weights folded) | moved by the matrix | fixed | fixed | fixed | fixed |
| scaling of a norm weight (the readers compensate); scaling of the up branch and permutation of the units of the gated block | fixed | fixed | fixed | fixed | fixed |
| value/output gauge, per group, any invertible matrix | fixed | fixed | fixed | moved by the matrix | fixed |
| complex scalar per rotary plane (keys by the scalar, queries by its inverse transpose) | fixed | moved | moved | fixed | fixed |
| permutation of the heads and of the groups | fixed | permuted | permuted | permuted | fixed |

So a quantity computed from the cache alone is equal for two checkpoints that differ by a gauge of the hidden coordinates, and in general not for two that differ by a gauge inside the attention block; a quantity
computed from the stream alone is the reverse; the logits are equal for all of them. Neither state is invariant under the whole group, which is why an identity check that has to survive every
function-preserving transformation is computed from the logits, or from the quotient of the weights by the whole group. Of the quotient only the discrete part (the permutations) has an implementation (the
canonical form of batch 0013); the continuous gauges (the rotations of the hidden coordinates, the value/output matrices, the rotary-plane scalars) have none.

## The certificate

A suite run produces one JSON record per (checkpoint, variant, runtime, dtype): the tier results, the noise floors used, the thresholds, the environment, the commit and the CI run. A checkpoint is *verified for a use* when
the tiers that use needs have passed in at least two independent runtimes and the controls behaved. The record is the artifact; this document only says what goes in it.
