### The registered test of 'is a language model' (mean loss on `membership.txt` over ln of the vocabulary; at most 0.9 passes)

| system | tokens | mean loss (nats) | ln V | fraction |
|---|---|---|---|---|
| SmolLM2-135M-Instruct | 114 | 1.934 | 10.803 | 0.179 |
| SmolLM2-135M | 114 | 1.977 | 10.803 | 0.183 |
| FloatLM 99M | 114 | 2.598 | 10.826 | 0.240 |
| TriLM 99M (ternary) | 114 | 2.992 | 10.826 | 0.276 |
| delphi-suite/v0-llama2-100k | 115 | 2.077 | 8.318 | 0.250 |
| Qwen3-0.6B | 113 | 2.084 | 11.931 | 0.175 |
| GPT-2 (124M) | 114 | 2.495 | 10.825 | 0.230 |
| Pythia-160M | 114 | 2.465 | 10.826 | 0.228 |

### Kinematics of the Llama family, float64 (relative differences of the scores; ≤ 1e-6 is exact in the sense of the claims)

| system | translation of the positions, largest | phase element on the weights, largest | its keys against the original's at the shifted positions, largest | dilation of the angles, smallest | rotation (fold, untie), largest | rotation without the fold (control), smallest | shift moves the keys (control), smallest |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M-Instruct | 5.39e-15 | 4.84e-15 | 2.82e-15 | 1.68e-01 | 9.13e-15 | 1.41e+00 | 0.491 |
| SmolLM2-135M | 3.45e-15 | 3.32e-15 | 2.85e-15 | 1.00e-01 | 4.11e-14 | 1.46e+00 | 0.501 |
| FloatLM 99M | 9.20e-16 | 1.08e-15 | 2.58e-15 | 1.56e-02 | 1.32e-14 | 2.54e-01 | 0.634 |
| TriLM 99M | 1.53e-15 | 1.57e-15 | 1.37e-15 | 8.74e-02 | 3.38e-15 | 1.48e+00 | 0.671 |
| delphi 100k | 1.53e-15 | 1.35e-15 | 9.29e-16 | 9.61e-02 | 2.46e-15 | 2.44e-01 | 0.406 |

### Qwen3-0.6B (a norm of the queries and the keys between the projection and the rotary embedding), float64: relative differences of the scores

| transformation | smallest | largest |
|---|---|---|
| T1_scalar_pair_per_group_and_plane | 3.97e-01 | 6.24e-01 |
| T2_rotation_of_a_plane_on_keys_and_queries | 6.03e-01 | 1.01e+00 |
| T3_positive_scalar_on_one_query_head | 2.05e-06 | 7.02e-06 |
| T4_scalar_moved_from_the_query_gain_to_the_key_gain | 3.28e-15 | 1.09e-14 |
| T2_with_the_angles_of_a_shift_of_16_positions | 4.75e-01 | 7.19e-01 |
| translation of the positions (1, 16, 128) | 3.56e-15 | 1.01e-14 |
| dilation (0.5, 2) | 1.79e-01 | 5.06e-01 |
| phase element on the weights (shift of 16) | 4.75e-01 | 7.19e-01 |
| rotation with the fold | 1.18e-14 | 2.01e-14 |
| rotation without the fold (control) | 1.30e+00 | 1.71e+00 |

### GPT-2 (LayerNorm, learned absolute positions, tied head), float64: relative differences of the scores

| transformation | smallest | largest |
|---|---|---|
| L1: the all-ones vector added to the outputs of the two output projections of every layer | 9.88e-16 | 2.73e-15 |
| L2: the dual shift of the readers of the two LayerNorms | 4.32e-15 | 1.21e-14 |
| L3: the embedding rows of five tokens shifted; the columns of the scores that do not change |  | 0.00e+00 |
| L3: the columns that change, against the prediction |  | 9.35e-16 |
| learned absolute positions: the positions shifted by 1 (must fail) | 4.30e-01 | 6.85e-01 |
| learned absolute positions: the positions shifted by 16 (must fail) | 3.61e-01 | 6.55e-01 |

### Pythia-160M (LayerNorm, parallel residual, rotary on a quarter of each head), float64: relative differences of the scores

| transformation | smallest | largest |
|---|---|---|
| translation of the positions (1, 16, 128) | 8.96e-16 | 1.89e-15 |
| dilation of the angles (0.5, 2) | 9.50e-04 | 4.63e-03 |
| writer_shift | 1.01e-15 | 1.55e-15 |
| rotation_fixing_the_all_ones_vector | 1.07e-15 | 2.59e-15 |
| general_orthogonal_rotation | 1.62e-03 | 2.10e-03 |

### Conversation and tool call, float64, greedy, 64 new tokens: the original's answer and how many of the transformed copies give the same tokens

| system | prompt | prompt tokens | the original's answer | variants with identical tokens | control (not applied to the embedding) |
|---|---|---|---|---|---|
| Qwen3-0.6B | tool_call | 176 | `<tool_call>\n{"name": "get_weather", "arguments": {"city": "Paris"}}\n</tool_call><\|im_end\|>` | 8 of 8 | different |
| Qwen3-0.6B | conversation | 19 | `Hello! How can I assist you today?<\|im_end\|>` | 8 of 8 | different |
| SmolLM2-135M-Instruct | conversation | 37 | `Hello!<\|im_end\|>` | 6 of 6 | different |

### Ten steps of fine-tuning on one fixed text from the original and from transformed copies: the difference of the functions after training, over the change the reference run made

| system | optimizer | loss, first → last step | change of the function by the reference run (relative) | signed permutation | rotation of the stream | scaling of the units | invertible matrix on the value dimensions |
|---|---|---|---|---|---|---|---|
| SmolLM2-135M-Instruct | sgd | 3.498 → 18.351 | 1.26 | 7.2e-12 | 8.2e-10 | 0.54 | 0.93 |
| SmolLM2-135M-Instruct | adamw | 3.498 → 0.944 | 0.661 | 2.3e-14 | 1.6 | 0.46 | 0.13 |
| delphi 100k | sgd | 6.759 → 9.434 | 1.77 | 4.5e-12 | 6.5e-12 | 0.99 | 1.1 |
| delphi 100k | adamw | 6.759 → 4.646 | 0.134 | 8.8e-15 | 0.098 | 0.35 | 0.073 |

### The velocity of the predictive distributions under each training rule, at ten random points: the largest relative difference between the velocity at θ and at gθ, for seven equivalence transformations g

| rule | unit permutation | signed permutation | head swap | rotation of the stream | unit scaling ±2^k | invertible matrix on the value dimensions | complex scalar on a rotary plane | median norm of the step in the constants, over that of gradient descent |
|---|---|---|---|---|---|---|---|---|
| gradient descent | 4.61e-16 | 1.97e-15 | 3.01e-15 | 2.37e-14 | 1.41e+00 | 2.92e-01 | 1.44e-02 | 1 |
| Adam (first step) | 4.36e-16 | 1.42e-15 | 1.69e-15 | 4.52e-01 | 4.08e-01 | 2.52e-01 | 5.31e-02 | 23.8 |
| natural gradient, pseudo-inverse | 2.05e-10 | 1.86e-10 | 1.67e-10 | 1.74e-10 | 2.47e-10 | 2.15e-10 | 2.42e-10 | 6.97e+04 |
| natural gradient, damped by the identity of the constants (F + 1e-9 I) | 2.56e-12 | 2.01e-12 | 2.09e-12 | 2.74e-12 | 4.02e-02 | 6.65e-04 | 2.89e-03 | 983 |
| natural gradient, damped by the metric of the scores (F + λ JᵀJ, λ = 1e-6 times the mean nonzero eigenvalue of F) | 2.23e-10 | 2.50e-10 | 2.24e-10 | 2.56e-10 | 1.82e-10 | 2.44e-10 | 2.07e-10 | 6.97e+04 |
| natural gradient, damped by the metric of the scores (F + λ JᵀJ, λ = 1e-2 times the mean nonzero eigenvalue of F) | 1.34e-10 | 1.92e-10 | 1.84e-10 | 1.97e-10 | 1.52e-10 | 1.66e-10 | 1.85e-10 | 6.66e+04 |

### Trajectories of 200 steps in the constants from θ and from gθ: the largest difference of the predictive distributions along the trajectory

| rule | transformation g | largest difference |
|---|---|---|
| gradient descent, lr 1.0 | unit permutation | 3.52e-14 |
| gradient descent, lr 1.0 | signed permutation | 4.59e-12 |
| gradient descent, lr 1.0 | head swap | 2.26e-12 |
| gradient descent, lr 1.0 | rotation of the stream | 5.29e-12 |
| gradient descent, lr 1.0 | unit scaling ±2^k | 1.91e+00 |
| gradient descent, lr 1.0 | invertible matrix on the value dimensions | 2.38e+00 |
| gradient descent, lr 1.0 | complex scalar on a rotary plane | 1.38e+00 |
| Adam, lr 0.01 | unit permutation | 4.08e-11 |
| Adam, lr 0.01 | signed permutation | 1.70e-11 |
| Adam, lr 0.01 | head swap | 6.41e-11 |
| Adam, lr 0.01 | rotation of the stream | 1.40e+00 |
| Adam, lr 0.01 | unit scaling ±2^k | 1.96e+00 |
| Adam, lr 0.01 | invertible matrix on the value dimensions | 1.16e+00 |
| Adam, lr 0.01 | complex scalar on a rotary plane | 1.04e+00 |

### The group at a point that training has reached: the rank deficiency of the Jacobian at initialisation and after 300 steps of Adam (learning rate 0.02, target Dirichlet(0.3))

| seed | loss, first → last step | deficiency expected | deficiency at initialisation | singular-value gap there | deficiency after training | singular-value gap there |
|---|---|---|---|---|---|---|
| 21 | 1.649 → 1.529 | 26 | 26 | 1.4e+10 | 26 | 3.6e+09 |
| 22 | 1.683 → 1.540 | 26 | 26 | 4.5e+09 | 26 | 1.3e+09 |
| 23 | 1.703 → 1.499 | 26 | 26 | 3.3e+10 | 26 | 1.7e+09 |
| 24 | 1.731 → 1.556 | 26 | 26 | 4.7e+10 | 26 | 5.5e+08 |
| 25 | 1.661 → 1.521 | 26 | 26 | 1.3e+10 | 26 | 1.5e+09 |

### The finite instance as a checkpoint, on every input (625 sequences of four symbols, 780 contexts), against the numpy float64 reference

| runtime | largest relative difference of the scores | agreement of the largest score | contexts |
|---|---|---|---|
| Transformers, float64 | 2.14e-16 | 1.0 | 780 |
| Transformers, float32 | 1.24e-07 | 1.0 | 780 |
| candle, float32 | 1.33e-07 | 1.0 | 780 |

### Discrete candidates, counted (one entry per random draw)

| family | candidates | invariant | violated | unclear |
|---|---|---|---|---|
| hidden-norms-fixed | 384 | 16, 16, 16 | 368, 368, 368 | 0, 0, 0 |
| hidden-norms-permuted | 384 | 384, 384, 384 | 0, 0, 0 | 0, 0, 0 |
| units | 1728 | 48, 48, 48 | 1680, 1680, 1680 | 0, 0, 0 |
| heads | 48 | 8, 8, 8 | 40, 40, 40 | 0, 0, 0 |

### Mutants of the Python definition, the surrogates and the generator, run through the conformance check against the stored Lean output

| mutant | file | what it changes | predicted | result | configurations that differ |
|---|---|---|---|---|---|
| M01 | model.py | the scale of the scores | detected | detected | 4 of 4 |
| M02 | model.py | the causal mask points the other way | detected | detected | 4 of 4 |
| M03 | model.py | no causal mask: position 0 reads position 1 | detected | detected | 4 of 4 |
| M04 | model.py | a position may not read itself: the first position reads nothing and the normalization divides by ze | detected | detected | 4 of 4 |
| M05 | model.py | the keys are not rotated | detected | detected | 4 of 4 |
| M06 | model.py | a sign of the rotation | detected | detected | 4 of 4 |
| M07 | model.py | the pairing of the coordinates of a rotary plane, half-split or interleaved, is the same with one pl | survives | survives | 0 of 4 |
| M08 | model.py | the heads of a group read the wrong group | detected | detected | 4 of 4 |
| M09 | model.py | the weights are normalized over the wrong axis | detected | detected | 4 of 4 |
| M10 | model.py | no residual connection around the attention | detected | detected | 4 of 4 |
| M11 | model.py | no residual connection around the gated block | detected | detected | 4 of 4 |
| M12 | model.py | the gate and the up matrices exchanged | detected | detected | 4 of 4 |
| M13 | model.py | the norm weight is not applied | detected | detected | 4 of 4 |
| M14 | model.py | the mean of the squares is divided by the wrong number | detected | detected | 4 of 4 |
| M15 | model.py | no final norm | detected | detected | 4 of 4 |
| M16 | model.py | the head is the embedding although the head is untied | detected | detected | 4 of 4 |
| M17 | model.py | the cache keeps the keys before the rotation (the logits are right) | detected | detected | 4 of 4 |
| M18 | model.py | the folded model forgets the final norm weight | detected | detected | 4 of 4 |
| M19 | model.py | the fold forgets the norm weight of the gated block | detected | detected | 4 of 4 |
| M20 | ops.py | the surrogate of the exponential | detected | detected | 4 of 4 |
| M21 | ops.py | the surrogate of the gate function | detected | detected | 4 of 4 |
| M22 | ops.py | the surrogate of the normalization | detected | detected | 4 of 4 |
| M23 | ops.py | the scale of the scores (the surrogate) | detected | detected | 4 of 4 |
| G01 | gauge.py | H1: the final norm weight is not permuted | detected | detected | 4 of 4 |
| G02 | gauge.py | H1: the layer norm weights are not permuted | detected | detected | 4 of 4 |
| G03 | gauge.py | H4: the writers are rotated by Q and not by its transpose | detected | detected | 4 of 4 |
| G04 | gauge.py | A1: the transpose in place of the inverse | detected | detected | 4 of 4 |
| G05 | gauge.py | A3: the queries are scaled by the same complex number as the keys | detected | detected | 4 of 4 |
| G06 | gauge.py | A3: keys and queries exchanged | detected | detected | 4 of 4 |
| G07 | gauge.py | M1: the down column is scaled by c and not by 1/c | detected | detected | 4 of 4 |
| G08 | gauge.py | M1: the scale is applied after the permutation instead of before | detected | detected | 2 of 4 |
| G09 | gauge.py | A6: the group permutation is not applied to the heads | detected | detected | 4 of 4 |
| G10 | gauge.py | N1: the readers are scaled by c and not by 1/c | detected | detected | 4 of 4 |
| G11 | gauge.py | N1 on the final norm: the generator takes the scale 1 there, so the direction of the scale is not te | survives | survives | 0 of 4 |
| G12 | gauge.py | A3: the second row of a rotary plane is j + 1 or j + hd/2: the same with one plane | survives | survives | 0 of 4 |
| H01 | gen22.py | the positions start at 1: the logits are the same, the cache is not | detected | detected | 4 of 4 |
| H02 | gen22.py | the recurrence of the rotation is wrong from the third position on: only two positions are run | survives | survives | 0 of 4 |
| B01 | model.py | the tied head is wrong: the run has an untied head | survives | survives | 0 of 4 |
| B02 | model.py | LayerNorm does not subtract the mean: the run uses the RMS norm | survives | survives | 0 of 4 |
| B03 | model.py | the mask of a non-causal model: the run is causal | survives | survives | 0 of 4 |
| B04 | model.py | the fold does not untie the head: the head of the run is untied already | survives | survives | 0 of 4 |

