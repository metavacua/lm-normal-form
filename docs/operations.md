# How work is run

## Where things run

- **Experiments run in GitHub Actions, on GitHub-hosted runners, and nowhere
  else.** That covers anything unstable, unsafe or uncertain: trying a tool, a
  library or a model, comparing them, compiling, and anything that needs much
  memory or disk.
- **The development machine edits files and uses git.** It has 6.4 GiB of
  memory and no swap, and it must not be crashed.
- **Nothing is run on the development machine until a hosted runner has
  measured what it needs.** See "Measuring first" below.

## Batches of twelve

- A batch is one file, `batches/NNNN.json`, naming at most twelve cells. A
  cell is one experiment job.
- A batch has its own branch, `batch/NNNN-<words>`, and its own pull request.
  The branch name selects the batch.
- A batch is written down in `docs/batches/NNNN.md` before it runs: its
  purpose, the cells, the premises, the checks and how each is judged.
  Results are appended to the same document.
- The workflow has three service jobs besides the cells: `checks` (licences,
  workflow lint, unit tests), `plan` (reads the batch file) and `collect`
  (one summary for the batch).
- Cells start when a pull request is opened, and after a push that changes
  something the experiments depend on: `src/`, `queries/`, `batches/`,
  `pyproject.toml` or the workflow. A push that changes only documents or
  tests starts no cells.

## What makes a job fail

- **Exit status 0:** everything registered was evaluated. Verdicts are in the
  report, whatever they are. A check that fails is a result.
- **Exit status 2:** inconclusive. A premise failed or something registered
  could not be evaluated. The job fails.
- **Anything else:** the tooling broke. The job fails.

## Measuring first

- Every job runs under `python -m lmnf measure`, which records the peak
  resident memory of the largest process, how far the machine's available
  memory fell, elapsed and CPU time, and disk used.
- The batch summary lists every measurement and says whether it is within the
  limits for the development machine: 1,024 MiB of memory and 1,024 MiB of
  disk (`src/lmnf/measure.py`). These limits are a proposal; the owner of the
  machine sets them.
- A tool or an experiment may be run on the development machine only after a
  measurement on a hosted runner shows it within those limits.
- For scale: GitHub documents the standard Linux runner for public
  repositories as 4 CPUs, 16 GB of memory and 14 GB of disk.

## Models and prompts

- Models are fetched from Hugging Face at the revision published when the job
  starts. The revision and the file's SHA-256 are recorded. Models are not
  stored in this repository or in artifacts.
- Nothing here implements a model. Graphs are read by the reference `onnx`
  package; forward passes are run by ONNX Runtime and by PyTorch through
  `transformers`.
- Prompts are fixed, short and written in the batch file. A job computes one
  next-token distribution per prompt. No job generates free-running text,
  images or other media, and no job trains anything.
- Artifacts are kept for one day. The job summary carries the result.

## GitHub's terms, as read on 2026-10-05

This is a working reading by the maintainers, not legal advice. It should be
read again when the terms change.

- **Actions** ([GitHub Terms for Additional Products and Features](https://docs.github.com/en/site-policy/github-terms/github-terms-for-additional-products-and-features),
  version effective 2026-08-27). GitHub-hosted runners are for "the
  production, testing, deployment, or publication of the software project
  associated with the repository". Actions may not be used for cryptomining,
  for unauthorised access, or to resell Actions. Also excluded: "Any activity
  that places a burden on our servers, where that burden is disproportionate
  to the benefits provided to users".
  *Here:* every job tests this repository's software against published
  models. A batch is at most twelve jobs of at most an hour each. A newer run
  cancels the one before it. Documents-only pushes start no cells. Nothing is
  scheduled.
- **Automation** ([Acceptable Use Policies](https://docs.github.com/en/site-policy/acceptable-use-policies/github-acceptable-use-policies),
  section 4). No "automated excessive bulk activity", and no "undue burden on
  our servers through automated means".
  *Here:* the workflow token is read-only. No job pushes, comments, opens
  issues, stars, or calls the GitHub API beyond checkout and artifacts.
  Results are read back with a few requests, one at a time.
- **API** ([Terms of Service](https://docs.github.com/en/site-policy/github-terms/github-terms-of-service),
  effective 2026-04-27, section H). "Abuse or excessively frequent requests"
  can lose an account its API access.
  *Here:* as above.
- **Accounts** (Terms of Service, section B.3). Accounts are created by
  humans; one machine account is allowed.
  *Here:* work is done under the owner's account at the owner's direction.
  There is no bot account. A commit that an AI model helped write says so in
  a `Co-Authored-By` trailer.
- **Language models and collected content** (Terms of Service, section D.9).
  The section concerns "using automated means to access, collect, or otherwise
  use" public GitHub content "for the purpose of developing or training" a
  commercial AI system.
  *Here:* no job collects content from GitHub, and nothing here develops or
  trains a model. Models come from Hugging Face under their own licences.
- **Language models and prohibited content** ([Synthetic Media and AI Tools](https://docs.github.com/en/site-policy/acceptable-use-policies/github-synthetic-media-and-ai-tools)).
  Projects must not be designed for using large language models to create
  child sexual abuse material, terrorist or violent extremist content, or
  non-consensual intimate imagery.
  *Here:* prompts are fixed and benign, and jobs compute next-token
  distributions only.
- **Limits** ([Actions limits](https://docs.github.com/en/actions/reference/limits)).
  Six hours per job, 256 jobs per matrix, 20 concurrent jobs on the Free plan.
  *Here:* twelve cells and three service jobs stay well inside these.
