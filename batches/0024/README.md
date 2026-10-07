Batch 0024: the null variants of batch 0014 and the claims about their spread (docs/batches/0024.md).

- `grade24.py` holds the registered claims and controls and grades the summaries of the driver of batch 0014 (`python grade24.py RESULTS OUT`; `--table`; `--selftest`).
- `test_nulls.py` tests the eight null variants (`../0014/nulls.py`) on a synthetic checkpoint; it needs numpy only.
- The run is the workflow `.github/workflows/batch-0024.yml`, started by a commit whose message contains the bracketed word run.
