# Batch 0024, graded (claims: 28; as predicted 27, REFUTED 1, not run 0; controls: 32 of 32 hold)

| id | severity | status | detail |
|---|---|---|---|
| N1 llama.cpp, Q8_0 weights | high | as predicted | ok: spread of the eight nulls: the largest of the eight nulls over the smallest 1.41 (at most 3 predicted; a null of zero is a failure) |
| N2 llama.cpp, Q8_0 weights | high | as predicted | ok: same class against the median null: largest variant claimed the same 0.000693 against 4 x the median null 0.00292 |
| N3 llama.cpp, Q8_0 weights | high | as predicted | ok: different class against the largest null: smallest variant claimed different 0.00174 against 1.5 x the largest null 0.00119 |
| N4 llama.cpp, Q8_0 weights | high | as predicted | ok: separation of the classes: smallest claimed different 0.00174 against the largest claimed the same 0.000693 |
| N1 llama.cpp, Q4_0 weights | high | as predicted | ok: spread of the eight nulls: the largest of the eight nulls over the smallest 1.52 (at most 3 predicted; a null of zero is a failure) |
| N2 llama.cpp, Q4_0 weights | high | as predicted | ok: same class against the median null: largest variant claimed the same 0.00104 against 4 x the median null 0.00318 |
| N3 llama.cpp, Q4_0 weights | high | as predicted | ok: different class against the largest null: smallest variant claimed different 0.0424 against 1.5 x the largest null 0.00129 |
| N4 llama.cpp, Q4_0 weights | high | as predicted | ok: separation of the classes: smallest claimed different 0.0424 against the largest claimed the same 0.00104 |
| N1 llama.cpp, f16 cache | high | as predicted | ok: spread of the eight nulls: the largest of the eight nulls over the smallest 1.37 (at most 3 predicted; a null of zero is a failure) |
| N2 llama.cpp, f16 cache | high | as predicted | ok: same class against the median null: largest variant claimed the same 1.49e-07 against 4 x the median null 5.06e-07 |
| N1 llama.cpp, Q8_0 cache, rotated | high | as predicted | ok: spread of the eight nulls: the largest of the eight nulls over the smallest 1.43 (at most 3 predicted; a null of zero is a failure) |
| N2 llama.cpp, Q8_0 cache, rotated | high | as predicted | ok: same class against the median null: largest variant claimed the same 3.44e-05 against 4 x the median null 0.000116 |
| N3 llama.cpp, Q8_0 cache, rotated | high | as predicted | ok: different class against the largest null: smallest variant claimed different 0.00014 against 1.5 x the largest null 5.23e-05 |
| N4 llama.cpp, Q8_0 cache, rotated | high | as predicted | ok: separation of the classes: smallest claimed different 0.00014 against the largest claimed the same 3.44e-05 |
| N1 llama.cpp, Q4_0 cache, rotated | high | as predicted | ok: spread of the eight nulls: the largest of the eight nulls over the smallest 1.31 (at most 3 predicted; a null of zero is a failure) |
| N2 llama.cpp, Q4_0 cache, rotated | high | as predicted | ok: same class against the median null: largest variant claimed the same 0.00214 against 4 x the median null 0.00542 |
| N3 llama.cpp, Q4_0 cache, rotated | high | as predicted | ok: different class against the largest null: smallest variant claimed different 0.0395 against 1.5 x the largest null 0.00228 |
| N4 llama.cpp, Q4_0 cache, rotated | high | as predicted | ok: separation of the classes: smallest claimed different 0.0395 against the largest claimed the same 0.00214 |
| N1 llama.cpp, Q8_0 cache, not rotated | high | as predicted | ok: spread of the eight nulls: the largest of the eight nulls over the smallest 1.49 (at most 3 predicted; a null of zero is a failure) |
| N2 llama.cpp, Q8_0 cache, not rotated | high | as predicted | ok: same class against the median null: largest variant claimed the same 6.1e-05 against 4 x the median null 0.000201 |
| N3 llama.cpp, Q8_0 cache, not rotated | high | as predicted | ok: different class against the largest null: smallest variant claimed different 0.000361 against 1.5 x the largest null 9.36e-05 |
| N4 llama.cpp, Q8_0 cache, not rotated | high | as predicted | ok: separation of the classes: smallest claimed different 0.000361 against the largest claimed the same 6.1e-05 |
| N1 llama.cpp, Q4_0 cache, not rotated | high | as predicted | ok: spread of the eight nulls: the largest of the eight nulls over the smallest 1.6 (at most 3 predicted; a null of zero is a failure) |
| N2 llama.cpp, Q4_0 cache, not rotated | high | as predicted | ok: same class against the median null: largest variant claimed the same 0.00231 against 4 x the median null 0.006 |
| N3 llama.cpp, Q4_0 cache, not rotated | high | as predicted | ok: different class against the largest null: smallest variant claimed different 0.0402 against 1.5 x the largest null 0.00268 |
| N4 llama.cpp, Q4_0 cache, not rotated | high | as predicted | ok: separation of the classes: smallest claimed different 0.0402 against the largest claimed the same 0.00231 |
| N1 CTranslate2, int8 | high | as predicted | ok: spread of the eight nulls: the largest of the eight nulls over the smallest 1.25 (at most 3 predicted; a null of zero is a failure) |
| N2 CTranslate2, int8 | high | REFUTED | FAILS: same class against the median null: largest variant claimed the same 0.529 against 4 x the median null 0.0254 |

## Controls (not claims)

| control | status |
|---|---|
| control llama.cpp, Q8_0 weights: the broken variant (the permutation applied to the layers and not to the embedding) is at least 10 times the largest null | holds |
| control llama.cpp, Q8_0 weights: the original run twice gives the same generations_equal | holds |
| control llama.cpp, Q8_0 weights: the original run twice gives the same logits_bit_identical | holds |
| control llama.cpp, Q8_0 weights: the original run twice gives the same ppl_equal | holds |
| control llama.cpp, Q4_0 weights: the broken variant (the permutation applied to the layers and not to the embedding) is at least 10 times the largest null | holds |
| control llama.cpp, Q4_0 weights: the original run twice gives the same generations_equal | holds |
| control llama.cpp, Q4_0 weights: the original run twice gives the same logits_bit_identical | holds |
| control llama.cpp, Q4_0 weights: the original run twice gives the same ppl_equal | holds |
| control llama.cpp, f16 cache: the broken variant (the permutation applied to the layers and not to the embedding) is at least 10 times the largest null | holds |
| control llama.cpp, f16 cache: the original run twice gives the same generations_equal | holds |
| control llama.cpp, f16 cache: the original run twice gives the same logits_bit_identical | holds |
| control llama.cpp, f16 cache: the original run twice gives the same ppl_equal | holds |
| control llama.cpp, Q8_0 cache, rotated: the broken variant (the permutation applied to the layers and not to the embedding) is at least 10 times the largest null | holds |
| control llama.cpp, Q8_0 cache, rotated: the original run twice gives the same generations_equal | holds |
| control llama.cpp, Q8_0 cache, rotated: the original run twice gives the same logits_bit_identical | holds |
| control llama.cpp, Q8_0 cache, rotated: the original run twice gives the same ppl_equal | holds |
| control llama.cpp, Q4_0 cache, rotated: the broken variant (the permutation applied to the layers and not to the embedding) is at least 10 times the largest null | holds |
| control llama.cpp, Q4_0 cache, rotated: the original run twice gives the same generations_equal | holds |
| control llama.cpp, Q4_0 cache, rotated: the original run twice gives the same logits_bit_identical | holds |
| control llama.cpp, Q4_0 cache, rotated: the original run twice gives the same ppl_equal | holds |
| control llama.cpp, Q8_0 cache, not rotated: the broken variant (the permutation applied to the layers and not to the embedding) is at least 10 times the largest null | holds |
| control llama.cpp, Q8_0 cache, not rotated: the original run twice gives the same generations_equal | holds |
| control llama.cpp, Q8_0 cache, not rotated: the original run twice gives the same logits_bit_identical | holds |
| control llama.cpp, Q8_0 cache, not rotated: the original run twice gives the same ppl_equal | holds |
| control llama.cpp, Q4_0 cache, not rotated: the broken variant (the permutation applied to the layers and not to the embedding) is at least 10 times the largest null | holds |
| control llama.cpp, Q4_0 cache, not rotated: the original run twice gives the same generations_equal | holds |
| control llama.cpp, Q4_0 cache, not rotated: the original run twice gives the same logits_bit_identical | holds |
| control llama.cpp, Q4_0 cache, not rotated: the original run twice gives the same ppl_equal | holds |
| control CTranslate2, int8: the broken variant (the permutation applied to the layers and not to the embedding) is at least 10 times the largest null | holds |
| control CTranslate2, int8: the original run twice gives the same generations_equal | holds |
| control CTranslate2, int8: the original run twice gives the same logits_bit_identical | holds |
| control CTranslate2, int8: the original run twice gives the same ppl_equal | holds |
