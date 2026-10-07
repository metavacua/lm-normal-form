| category | arrays | equal / as expected | mismatches | not run |
|---|---|---|---|---|
| R1 the original model | 64 | 64 | 0 | 0 |
| R2 the definition of the logits (one layer) | 3 | 3 | 0 | 0 |
| R3 the weights that the Lean gauges make (flat layout, against gauge.py) | 147 | 147 | 0 | 0 |
| R4 what the transformed model computes (against model.py) | 63 | 63 | 0 | 0 |
| R5 the logits of the transformed model are the original's | 7 | 7 | 0 | 0 |
| R6 permutations | 2 | 2 | 0 | 0 |
| R7 the table: which arrays move (Lean against Lean) | 28 | 28 | 0 | 0 |
