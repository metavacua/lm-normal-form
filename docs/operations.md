# How work is run

The rules as they stand after batch 0005. Earlier forms of this document
(on the `batch/0001-structure-graphs` branch) described a hand-written
pipeline that no longer exists.

## Where things run

- **Experiments run in GitHub Actions, on GitHub-hosted runners, and nowhere
  else.** That covers anything unstable, unsafe or uncertain: trying a tool,
  a library or a model, comparing them, compiling, training, and anything
  that needs much memory or disk.
- **The development machine edits files and uses git.** It is small, has no
  swap, and must not be crashed. Nothing is compiled on it. Lints that are
  static binaries or pure text tools (`reuse lint`, `actionlint`, `jq`,
  `ast-grep`) may run on it, and should, before a push.
- **Nothing else runs on the development machine until a hosted runner has
  measured what it needs.** Every tool run in CI is wrapped in GNU `time -v`
  and its peak memory and wall time are recorded.

## Batches

- A batch is at most **twelve experiment cells**, plus service jobs
  (preparation, a judge). It has its own branch `batch/NNNN-<words>` and its
  own draft pull request, stacked on the batch before it.
- A batch is **registered before it runs**: `docs/batches/NNNN.md` states
  its purpose, cells, expectations and rules; results are appended after,
  with the run identifiers. Expectations written after the data are not
  expectations.
- A batch document opens by declaring what kind of batch it is:
  - **a test**: the hypothesis stated positively; the consequence derived
    from it that the cells will observe; the observation that would refute
    it (a contradiction, or the absence of the derived consequence); and
    what the batch depends on (earlier batches' instruments and data);
  - **a calibration**: baselines or instrument checks, with the claim they
    license and nothing more;
  - **a description**: inventories and candidates for a later test.
  Batches 0002, 0003 and 0006 are calibration and description; 0004 and
  0005 were tests with their hypotheses stated less sharply than this rule
  now requires (`docs/research/questions.md` restates them).
- A test's document also grades each prediction's **severity** before the
  run (would it likely fail if the hypothesis were false?), lists the
  **auxiliaries** a refutation could land on instead (kernels, the hit
  criterion, the authority's filters, tokenization, the queries) and
  which would be blamed first, gives for each finding the **rival
  explanations** considered and how they were excluded, marks every
  category or analysis added after the data as **post-designated**, and
  reports the counts that move under other **precisifications** of its
  vague predicates (prefix length, co-capital countries, historical
  capitals) as the vague part of the result.
- The 189 countries are a census of the fragment, not a sample; for claims
  about the fragment the run-to-run floor is the only noise (zero for
  float16 readouts, one or two countries for quantized ones), and ε is
  set above it per map. For claims beyond the fragment the fragment is
  the sample and n is 1; such sentences are descriptions.
- Batch numbers are accession numbers, not an order. Dependencies between
  batches are stated in each document and kept in one machine-readable
  file, `docs/batches/batches.ttl` (PROV: `wasInformedBy`, `used`), from
  which the index is rendered. Branches are based on `main`, not on each
  other, once the shared pieces (authority query, prompt derivation, the
  judge workflow) are merged; the stacked pull requests of 0003 to 0006
  predate this rule.
- Prompts, transformations and queries are **data files** in
  `batches/NNNN/` and `queries/NNNN/` (`jq`, SPARQL, text), not code.
- A batch workflow runs on push when its own file changes, and by hand.
  A newer push cancels the run before it. Nothing is scheduled.
- **A smoke job first.** From batch 0006 on, a one-model, one-prompt job
  runs every invocation of the batch before the cells start, so that a
  trivial fault costs two minutes rather than twelve runners.

## Observations and outcomes

- Each observation is one step with an `id` and `continue-on-error: true`.
  A failed observation is a result, not a failed job; the job fails only
  when tooling breaks.
- Outcomes are recorded from the workflow's own `steps` context
  (`toJSON(steps)`), because the API reports `continue-on-error` steps as
  successes.
- Job names carry no colons; `gh run view --log` omits such jobs.
- An observation writes its key numbers (answers written, peak memory,
  elapsed) to `$GITHUB_OUTPUT`, so that the outcomes record carries the
  measurement and not only the status. Earlier batches' records have
  `outputs: {}` throughout, for want of this.
- A tool that runs in the background (a server) is measured from `/proc`
  (`VmHWM`) before it is stopped; GNU `time` sees only the foreground
  process. Batches 0002 to 0005's first run did not measure their servers.
- A judge has a **conformance fixture**: answer files with known counts
  that every revision of its queries must reproduce exactly, run in its
  smoke job. The batch 0004 judge does not have one yet.

## Existing tools, and the invocations that drive them

- **No hand-written implementation of anything a library already does.**
  Models are converted, exported, quantized, served, trained and read by
  their own libraries and tools: `transformers`, Optimum, ONNX Runtime,
  llama.cpp and `gguf-py`, TRL, Oxigraph, ONNX2RDF, IREE, `jq`.
- Where a tool has no command line, it is driven by an **invocation**: a
  short `python -c` program kept in the workflow's `env`, under golf rules
  (shortest, least complex, most robust, most correct), its size recorded.
- **Invocations are code.** They have grown from 340 bytes (batch 0003) to
  1,303 (batch 0005's training). From batch 0006 on, any invocation longer
  than 600 bytes, or containing a function definition, is moved to a file
  under `batches/NNNN/`, with an SPDX header, and is declared in the batch
  document as code that was written. `jq` programs are code too and are
  treated the same way.

## Models, prompts and readouts

- Models come from Hugging Face at the revision published when the job
  starts. Revisions are recorded, never pinned. Models are not stored in
  this repository.
- Tools are cloned or installed at their current release; the revision or
  version is recorded (`pip freeze`, `git rev-parse`). Reproduction runs may
  pin a recorded revision by input.
- A readout is one next-token distribution per prompt, with the top twenty
  tokens kept. **Instruction-tuned models are read as raw continuations,
  without their chat templates**, so that every model is read the same way;
  readouts under chat templates are a separate measurement, not yet made.
- Known-to-work programs that run models are the baselines: a new readout
  of a model is compared with an earlier program's readout of the same
  prompt before it is believed.

## Interfaces

- Each component that more than one batch uses has a declared public
  interface under `docs/interfaces/`, with a name independent of the
  batch that first wrote it, a version, its input and output schemas, its
  semantics, what counts as a compatible and a breaking change, and a
  conformance fixture. Versioning is of the declared interface;
  recording which commit ran is provenance, not versioning. Declared so
  far: the judge (version 1) and the readout (version 1).

## Data kept

- Run artifacts expire after one day for anything large (converted models,
  logs). Answer files and judge output should be kept seven days.
- The judge's aggregate tables, the authority extract and the first-answer
  list are committed under `docs/batches/NNNN/` (public domain, see
  `REUSE.toml`), because the prose cannot carry every number.
- Answer files (about a megabyte per model) are not committed.

## The authority

- The authority for an integrity batch is a named external source at a
  recorded time: Wikidata's capitals of sovereign states, fetched by a
  stored SPARQL query, with ranks and validity qualifiers. A country is
  determined when the authority prefers exactly one capital for it.
- To add: per-entity revisions (`schema:version`) in the authority graph.

## GitHub's terms, as read on 2026-10-05

A working reading by the maintainers, not legal advice; to be read again
when the terms change.

- **Actions** ([Terms for Additional Products and Features](https://docs.github.com/en/site-policy/github-terms/github-terms-for-additional-products-and-features),
  effective 2026-08-27). Hosted runners are for "the production, testing,
  deployment, or publication of the software project associated with the
  repository"; excluded is "any activity that places a burden on our
  servers, where that burden is disproportionate to the benefits provided
  to users". *Here:* every job tests this repository's own experiments
  against published models; at most twelve cells and a few service jobs,
  under the concurrency limit; a newer push cancels the older run; nothing
  is scheduled. Runs wasted on trivial faults are the only burden observed
  (six of fifteen batch runs by batch 0005); the smoke job addresses it.
- **Automation** ([Acceptable Use Policies](https://docs.github.com/en/site-policy/acceptable-use-policies/github-acceptable-use-policies),
  section 4): no "automated excessive bulk activity", no "undue burden".
  *Here:* the workflow token is read-only except where a judge reads another
  run's artifacts (`actions: read`) and, from batch 0008 on, where a
  final `report` job writes one commit status naming the batch run
  (`statuses: write`); no job pushes, comments or opens issues.
- **API** ([Terms of Service](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service),
  effective 2026-04-27, section H). *Here:* results are read back with a
  few requests, one at a time.
- **Accounts** (Terms of Service, B.3). *Here:* work is done under the
  owner's account at the owner's direction; no bot account; commits an AI
  model helped write carry a `Co-Authored-By` trailer.
- **Collected content and AI** (Terms of Service, D.9). *Here:* no job
  collects content from GitHub. Batch 0005 fine-tunes a 135M-parameter
  model on 804 sentences derived from Wikidata (CC0) on a hosted runner,
  as a test of this repository's method; nothing collected from GitHub is
  used, and no model is published from it.
- **Synthetic Media and AI Tools** policy. *Here:* prompts are fixed and
  benign; jobs compute next-token distributions.
- **Limits** ([Actions limits](https://docs.github.com/en/actions/reference/limits)):
  six hours per job, 256 matrix jobs, 20 concurrent jobs. *Here:* well
  inside.

## Clean break

No code, text or data from LARQL or its derivatives is used. Its public
issues and discussions are read as prior art and cited where they shaped a
question.
