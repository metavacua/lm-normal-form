# Batch 0023, graded (claims: 119; as predicted 118, REFUTED 1, not run 0; controls: 19 of 19 hold)

| group | as predicted | REFUTED | not run |
|---|---|---|---|
| causal structure | 1 | 0 | 0 |
| conformance power | 41 | 0 | 0 |
| conversation | 3 | 0 | 0 |
| dimension | 9 | 0 | 0 |
| discrete symmetries | 4 | 0 | 0 |
| kinematics | 25 | 1 | 0 |
| language model | 8 | 0 | 0 |
| optimizers | 12 | 0 | 0 |
| outside the proved class | 12 | 0 | 0 |
| runtimes | 3 | 0 | 0 |

| id | group | severity | status | detail |
|---|---|---|---|---|
| R1 | dimension | high | as predicted | ok: cov/rank/variants/base/*/deficiency: [26, 26, 26, 26], 26 was predicted; ok: cov/rank/variants/base/*: smallest gap 6.82e+09, at least 1e+06 was required |
| R2a | dimension | high | as predicted | ok: cov/rank/variants/theta/*/deficiency: [26, 26, 26, 26], 26 was predicted; ok: cov/rank/variants/theta/*: smallest gap 6.98e+09, at least 1e+06 was required |
| R2b | dimension | high | as predicted | ok: cov/rank/variants/delta/*/deficiency: [26, 26, 26, 26], 26 was predicted; ok: cov/rank/variants/delta/*: smallest gap 6.49e+09, at least 1e+06 was required |
| R2c | dimension | high | as predicted | ok: cov/rank/variants/theta+delta/*/deficiency: [28, 28, 28, 28], 28 was predicted; ok: cov/rank/variants/theta+delta/*: smallest gap 5.75e+09, at least 1e+06 was required |
| R3 | dimension | moderate | as predicted | ok: cov/rank/variants/posdep/*/deficiency: [62, 62, 62, 62], 62 was predicted; ok: cov/rank/variants/posdep/*: smallest gap 6.16e+09, at least 1e+06 was required |
| R4 | dimension | moderate | as predicted | ok: cov/rank/constrained/*/function_difference: largest 1.05e-15, at most 1e-12 was predicted; ok: cov/rank/constrained/*/deficiency_covariant: [62, 62, 62, 62], 62 was predicted; ok: cov/rank/constrained/*/intersection: [26, 26, 26, 26], 26 was predicted; ok: cov/rank/constrained/*/deficiency_special: [26, 26, 26, 26], 26 was predicted |
| R5a | dimension | low | as predicted | ok: cov/vocab/V_at_most_d/*/extra: smallest 2, at least 1 was predicted |
| R5b | dimension | low | as predicted | ok: cov/vocab/V_larger_than_d/*/extra: [0, 0, 0, 0], 0 was predicted |
| R6 | causal structure | low | as predicted | ok: cov/causal/*/largest_dependence_on_a_later_position: largest 0, at most 1e-14 was predicted; ok: cov/causal/*/smallest_dependence_on_an_earlier_or_equal_position: smallest 0.133, at least 1e-08 was predicted |
| F1 | dimension | moderate | as predicted | ok: fibre/runs/*/trained/deficiency: [26, 26, 26, 26], 26 was predicted; ok: fibre/runs/*/trained: smallest gap 5.54e+08, at least 1e+06 was required |
| D1a | discrete symmetries | moderate | as predicted | ok: discrete/families/hidden-norms-fixed/*/invariant: [16, 16, 16], 16 was predicted; ok: discrete/families/hidden-norms-fixed/*/unclear: [0, 0, 0], 0 was predicted |
| D1b | discrete symmetries | moderate | as predicted | ok: discrete/families/hidden-norms-permuted/*/invariant: [384, 384, 384], 384 was predicted; ok: discrete/families/hidden-norms-permuted/*/unclear: [0, 0, 0], 0 was predicted |
| D2 | discrete symmetries | moderate | as predicted | ok: discrete/families/units/*/invariant: [48, 48, 48], 48 was predicted; ok: discrete/families/units/*/unclear: [0, 0, 0], 0 was predicted; ok: discrete/families/units/*/the_invariant_ones_are_exactly_those_with_the_same_permutation_of_the_three_matrices: [True, True, True], True was predicted |
| D3 | discrete symmetries | moderate | as predicted | ok: discrete/families/heads/*/invariant: [8, 8, 8], 8 was predicted; ok: discrete/families/heads/*/unclear: [0, 0, 0], 0 was predicted; ok: discrete/families/heads/*/the_invariant_ones_are_exactly_those_that_keep_each_head_with_its_group: [True, True, True], True was predicted |
| O1 | optimizers | high | as predicted | ok: opt-field/rules/sgd/unit permutation/largest: largest 4.61e-16, at most 1e-08 was predicted; ok: opt-field/rules/sgd/signed permutation of the hidden coordinates/largest: largest 1.97e-15, at most 1e-08 was predicted; ok: opt-field/rules/sgd/head swap/largest: largest 3.01e-15, at most 1e-08 was predicted; ok: opt-field/rules/sgd/rotation of the stream/largest: largest 2.37e-14, at most 1e-08  |
| O2 | optimizers | high | as predicted | ok: opt-field/rules/adam/unit permutation/largest: largest 4.36e-16, at most 1e-08 was predicted; ok: opt-field/rules/adam/signed permutation of the hidden coordinates/largest: largest 1.42e-15, at most 1e-08 was predicted; ok: opt-field/rules/adam/head swap/largest: largest 1.69e-15, at most 1e-08 was predicted; ok: opt-field/rules/adam/rotation of the stream/smallest: smallest 0.147, at least 1e |
| O3 | optimizers | high | as predicted | ok: opt-field/rules/ngd-pinv/unit permutation/largest: largest 2.05e-10, at most 1e-08 was predicted; ok: opt-field/rules/ngd-pinv/signed permutation of the hidden coordinates/largest: largest 1.86e-10, at most 1e-08 was predicted; ok: opt-field/rules/ngd-pinv/head swap/largest: largest 1.67e-10, at most 1e-08 was predicted; ok: opt-field/rules/ngd-pinv/rotation of the stream/largest: largest 1.74 |
| O4 | optimizers | high | as predicted | ok: opt-field/rules/ngd-euclid/unit permutation/largest: largest 2.56e-12, at most 1e-08 was predicted; ok: opt-field/rules/ngd-euclid/signed permutation of the hidden coordinates/largest: largest 2.01e-12, at most 1e-08 was predicted; ok: opt-field/rules/ngd-euclid/head swap/largest: largest 2.09e-12, at most 1e-08 was predicted; ok: opt-field/rules/ngd-euclid/rotation of the stream/largest: larg |
| O5 | optimizers | high | as predicted | ok: opt-field/rules/ngd-covdamp:1e-6/unit permutation/largest: largest 2.23e-10, at most 1e-08 was predicted; ok: opt-field/rules/ngd-covdamp:1e-6/signed permutation of the hidden coordinates/largest: largest 2.5e-10, at most 1e-08 was predicted; ok: opt-field/rules/ngd-covdamp:1e-6/head swap/largest: largest 2.24e-10, at most 1e-08 was predicted; ok: opt-field/rules/ngd-covdamp:1e-6/rotation of t |
| O6 | optimizers | moderate | as predicted | ok: opt-field/median_step_norm_over_the_step_norm_of_sgd/ngd-pinv: smallest 6.97e+04, at least 100 was predicted |
| O7 | optimizers | high | as predicted | ok: opt-sgd/gauges/unit permutation/max_difference: largest 3.52e-14, at most 1e-08 was predicted; ok: opt-sgd/gauges/signed permutation of the hidden coordinates/max_difference: largest 4.59e-12, at most 1e-08 was predicted; ok: opt-sgd/gauges/head swap/max_difference: largest 2.26e-12, at most 1e-08 was predicted; ok: opt-sgd/gauges/rotation of the stream/max_difference: largest 5.29e-12, at mos |
| O8 | optimizers | high | as predicted | ok: opt-adam/gauges/unit permutation/max_difference: largest 4.08e-11, at most 1e-08 was predicted; ok: opt-adam/gauges/signed permutation of the hidden coordinates/max_difference: largest 1.7e-11, at most 1e-08 was predicted; ok: opt-adam/gauges/head swap/max_difference: largest 6.41e-11, at most 1e-08 was predicted; ok: opt-adam/gauges/rotation of the stream/max_difference: smallest 1.4, at leas |
| MUT-M01 | conformance power | high | as predicted | ok: mutate/mutants/M01/detected: [True], True was predicted |
| MUT-M02 | conformance power | high | as predicted | ok: mutate/mutants/M02/detected: [True], True was predicted |
| MUT-M03 | conformance power | high | as predicted | ok: mutate/mutants/M03/detected: [True], True was predicted |
| MUT-M04 | conformance power | high | as predicted | ok: mutate/mutants/M04/detected: [True], True was predicted |
| MUT-M05 | conformance power | high | as predicted | ok: mutate/mutants/M05/detected: [True], True was predicted |
| MUT-M06 | conformance power | high | as predicted | ok: mutate/mutants/M06/detected: [True], True was predicted |
| MUT-M07 | conformance power | moderate | as predicted | ok: mutate/mutants/M07/detected: [False], False was predicted |
| MUT-M08 | conformance power | high | as predicted | ok: mutate/mutants/M08/detected: [True], True was predicted |
| MUT-M09 | conformance power | high | as predicted | ok: mutate/mutants/M09/detected: [True], True was predicted |
| MUT-M10 | conformance power | high | as predicted | ok: mutate/mutants/M10/detected: [True], True was predicted |
| MUT-M11 | conformance power | high | as predicted | ok: mutate/mutants/M11/detected: [True], True was predicted |
| MUT-M12 | conformance power | high | as predicted | ok: mutate/mutants/M12/detected: [True], True was predicted |
| MUT-M13 | conformance power | high | as predicted | ok: mutate/mutants/M13/detected: [True], True was predicted |
| MUT-M14 | conformance power | high | as predicted | ok: mutate/mutants/M14/detected: [True], True was predicted |
| MUT-M15 | conformance power | high | as predicted | ok: mutate/mutants/M15/detected: [True], True was predicted |
| MUT-M16 | conformance power | high | as predicted | ok: mutate/mutants/M16/detected: [True], True was predicted |
| MUT-M17 | conformance power | high | as predicted | ok: mutate/mutants/M17/detected: [True], True was predicted |
| MUT-M18 | conformance power | high | as predicted | ok: mutate/mutants/M18/detected: [True], True was predicted |
| MUT-M19 | conformance power | high | as predicted | ok: mutate/mutants/M19/detected: [True], True was predicted |
| MUT-M20 | conformance power | high | as predicted | ok: mutate/mutants/M20/detected: [True], True was predicted |
| MUT-M21 | conformance power | high | as predicted | ok: mutate/mutants/M21/detected: [True], True was predicted |
| MUT-M22 | conformance power | high | as predicted | ok: mutate/mutants/M22/detected: [True], True was predicted |
| MUT-M23 | conformance power | high | as predicted | ok: mutate/mutants/M23/detected: [True], True was predicted |
| MUT-G01 | conformance power | high | as predicted | ok: mutate/mutants/G01/detected: [True], True was predicted |
| MUT-G02 | conformance power | high | as predicted | ok: mutate/mutants/G02/detected: [True], True was predicted |
| MUT-G03 | conformance power | high | as predicted | ok: mutate/mutants/G03/detected: [True], True was predicted |
| MUT-G04 | conformance power | high | as predicted | ok: mutate/mutants/G04/detected: [True], True was predicted |
| MUT-G05 | conformance power | high | as predicted | ok: mutate/mutants/G05/detected: [True], True was predicted |
| MUT-G06 | conformance power | high | as predicted | ok: mutate/mutants/G06/detected: [True], True was predicted |
| MUT-G07 | conformance power | high | as predicted | ok: mutate/mutants/G07/detected: [True], True was predicted |
| MUT-G08 | conformance power | high | as predicted | ok: mutate/mutants/G08/detected: [True], True was predicted |
| MUT-G09 | conformance power | high | as predicted | ok: mutate/mutants/G09/detected: [True], True was predicted |
| MUT-G10 | conformance power | high | as predicted | ok: mutate/mutants/G10/detected: [True], True was predicted |
| MUT-G11 | conformance power | moderate | as predicted | ok: mutate/mutants/G11/detected: [False], False was predicted |
| MUT-G12 | conformance power | moderate | as predicted | ok: mutate/mutants/G12/detected: [False], False was predicted |
| MUT-H01 | conformance power | high | as predicted | ok: mutate/mutants/H01/detected: [True], True was predicted |
| MUT-H02 | conformance power | moderate | as predicted | ok: mutate/mutants/H02/detected: [False], False was predicted |
| MUT-B01 | conformance power | moderate | as predicted | ok: mutate/mutants/B01/detected: [False], False was predicted |
| MUT-B02 | conformance power | moderate | as predicted | ok: mutate/mutants/B02/detected: [False], False was predicted |
| MUT-B03 | conformance power | moderate | as predicted | ok: mutate/mutants/B03/detected: [False], False was predicted |
| MUT-B04 | conformance power | moderate | as predicted | ok: mutate/mutants/B04/detected: [False], False was predicted |
| B1 | runtimes | high | as predicted | ok: bridge-torch/float64/relative_difference: largest 2.14e-16, at most 1e-10 was predicted; ok: bridge-torch/float64/argmax_agreement: smallest 1, at least 1 was predicted |
| B2 | runtimes | moderate | as predicted | ok: bridge-torch/float32/relative_difference: largest 1.24e-07, at most 1e-05 was predicted; ok: bridge-torch/float32/argmax_agreement: smallest 1, at least 0.999 was predicted |
| B3 | runtimes | moderate | as predicted | ok: bridge-candle/candle_float32/relative_difference: largest 1.33e-07, at most 1e-05 was predicted; ok: bridge-candle/candle_float32/argmax_agreement: smallest 1, at least 0.999 was predicted |
| S0-smol-instruct | language model | high | as predicted | ok: systems-smol-instruct/membership/fraction_of_ln_vocabulary: largest 0.179, at most 0.9 was predicted |
| K1-smol-instruct | kinematics | high | as predicted | ok: systems-smol-instruct/translation/*/largest: largest 5.39e-15, at most 1e-06 was predicted |
| K2-smol-instruct | kinematics | high | as predicted | ok: systems-smol-instruct/phase/logits/largest: largest 4.84e-15, at most 1e-06 was predicted; ok: systems-smol-instruct/phase/keys_of_the_gauged_model_against_the_original_at_the_shifted_positions/largest: largest 2.82e-15, at most 1e-06 was predicted |
| K3-smol-instruct | kinematics | high | as predicted | ok: systems-smol-instruct/dilation/*/smallest: smallest 0.168, at least 0.001 was predicted |
| K5-smol-instruct | kinematics | high | as predicted | ok: systems-smol-instruct/rotation/logits/largest: largest 9.13e-15, at most 1e-06 was predicted; ok: systems-smol-instruct/rotation/stream_equals_the_original_times_Q_transpose/largest: largest 1.11e-14, at most 1e-06 was predicted; ok: systems-smol-instruct/rotation/gram_matrix_of_every_layer/largest: largest 1.03e-14, at most 1e-06 was predicted; ok: systems-smol-instruct/rotation/stream_moved_ |
| S0-smol-base | language model | high | as predicted | ok: systems-smol-base/membership/fraction_of_ln_vocabulary: largest 0.183, at most 0.9 was predicted |
| K1-smol-base | kinematics | high | as predicted | ok: systems-smol-base/translation/*/largest: largest 3.45e-15, at most 1e-06 was predicted |
| K2-smol-base | kinematics | high | as predicted | ok: systems-smol-base/phase/logits/largest: largest 3.32e-15, at most 1e-06 was predicted; ok: systems-smol-base/phase/keys_of_the_gauged_model_against_the_original_at_the_shifted_positions/largest: largest 2.85e-15, at most 1e-06 was predicted |
| K3-smol-base | kinematics | high | as predicted | ok: systems-smol-base/dilation/*/smallest: smallest 0.1, at least 0.001 was predicted |
| K5-smol-base | kinematics | high | as predicted | ok: systems-smol-base/rotation/logits/largest: largest 4.11e-14, at most 1e-06 was predicted; ok: systems-smol-base/rotation/stream_equals_the_original_times_Q_transpose/largest: largest 1.91e-14, at most 1e-06 was predicted; ok: systems-smol-base/rotation/gram_matrix_of_every_layer/largest: largest 3.3e-14, at most 1e-06 was predicted; ok: systems-smol-base/rotation/stream_moved_at_the_embedding/ |
| S0-floatlm-99m | language model | high | as predicted | ok: systems-floatlm-99m/membership/fraction_of_ln_vocabulary: largest 0.24, at most 0.9 was predicted |
| K1-floatlm-99m | kinematics | high | as predicted | ok: systems-floatlm-99m/translation/*/largest: largest 9.2e-16, at most 1e-06 was predicted |
| K2-floatlm-99m | kinematics | high | as predicted | ok: systems-floatlm-99m/phase/logits/largest: largest 1.08e-15, at most 1e-06 was predicted; ok: systems-floatlm-99m/phase/keys_of_the_gauged_model_against_the_original_at_the_shifted_positions/largest: largest 2.58e-15, at most 1e-06 was predicted |
| K3-floatlm-99m | kinematics | high | as predicted | ok: systems-floatlm-99m/dilation/*/smallest: smallest 0.0156, at least 0.001 was predicted |
| K5-floatlm-99m | kinematics | high | as predicted | ok: systems-floatlm-99m/rotation/logits/largest: largest 1.32e-14, at most 1e-06 was predicted; ok: systems-floatlm-99m/rotation/stream_equals_the_original_times_Q_transpose/largest: largest 1.83e-14, at most 1e-06 was predicted; ok: systems-floatlm-99m/rotation/gram_matrix_of_every_layer/largest: largest 4.19e-15, at most 1e-06 was predicted; ok: systems-floatlm-99m/rotation/stream_moved_at_the_e |
| S0-trilm-99m | language model | high | as predicted | ok: systems-trilm-99m/membership/fraction_of_ln_vocabulary: largest 0.276, at most 0.9 was predicted |
| K1-trilm-99m | kinematics | high | as predicted | ok: systems-trilm-99m/translation/*/largest: largest 1.53e-15, at most 1e-06 was predicted |
| K2-trilm-99m | kinematics | high | as predicted | ok: systems-trilm-99m/phase/logits/largest: largest 1.57e-15, at most 1e-06 was predicted; ok: systems-trilm-99m/phase/keys_of_the_gauged_model_against_the_original_at_the_shifted_positions/largest: largest 1.37e-15, at most 1e-06 was predicted |
| K3-trilm-99m | kinematics | high | as predicted | ok: systems-trilm-99m/dilation/*/smallest: smallest 0.0874, at least 0.001 was predicted |
| K5-trilm-99m | kinematics | high | as predicted | ok: systems-trilm-99m/rotation/logits/largest: largest 3.38e-15, at most 1e-06 was predicted; ok: systems-trilm-99m/rotation/stream_equals_the_original_times_Q_transpose/largest: largest 2.83e-15, at most 1e-06 was predicted; ok: systems-trilm-99m/rotation/gram_matrix_of_every_layer/largest: largest 4.5e-15, at most 1e-06 was predicted; ok: systems-trilm-99m/rotation/stream_moved_at_the_embedding/ |
| S0-delphi-100k | language model | high | as predicted | ok: systems-delphi-100k/membership/fraction_of_ln_vocabulary: largest 0.25, at most 0.9 was predicted |
| K1-delphi-100k | kinematics | high | as predicted | ok: systems-delphi-100k/translation/*/largest: largest 1.53e-15, at most 1e-06 was predicted |
| K2-delphi-100k | kinematics | high | as predicted | ok: systems-delphi-100k/phase/logits/largest: largest 1.35e-15, at most 1e-06 was predicted; ok: systems-delphi-100k/phase/keys_of_the_gauged_model_against_the_original_at_the_shifted_positions/largest: largest 9.29e-16, at most 1e-06 was predicted |
| K3-delphi-100k | kinematics | high | as predicted | ok: systems-delphi-100k/dilation/*/smallest: smallest 0.0961, at least 0.001 was predicted |
| K5-delphi-100k | kinematics | high | as predicted | ok: systems-delphi-100k/rotation/logits/largest: largest 2.46e-15, at most 1e-06 was predicted; ok: systems-delphi-100k/rotation/stream_equals_the_original_times_Q_transpose/largest: largest 4.65e-15, at most 1e-06 was predicted; ok: systems-delphi-100k/rotation/gram_matrix_of_every_layer/largest: largest 2.69e-15, at most 1e-06 was predicted; ok: systems-delphi-100k/rotation/stream_moved_at_the_e |
| Q0 | language model | high | as predicted | ok: outside-qwen3-0.6b/membership/fraction_of_ln_vocabulary: largest 0.175, at most 0.9 was predicted |
| Q1 | kinematics | high | as predicted | ok: outside-qwen3-0.6b/translation/*/largest: largest 1.01e-14, at most 1e-06 was predicted |
| Q2 | outside the proved class | high | as predicted | ok: outside-qwen3-0.6b/phase/logits/smallest: smallest 0.475, at least 0.001 was predicted |
| Q3 | outside the proved class | high | as predicted | ok: outside-qwen3-0.6b/qk_norm_transformations/T1_scalar_pair_per_group_and_plane/smallest: smallest 0.397, at least 0.001 was predicted |
| Q4 | outside the proved class | high | as predicted | ok: outside-qwen3-0.6b/qk_norm_transformations/T2_rotation_of_a_plane_on_keys_and_queries/smallest: smallest 0.603, at least 0.001 was predicted |
| Q5 | outside the proved class | high | as predicted | ok: outside-qwen3-0.6b/qk_norm_transformations/T3_positive_scalar_on_one_query_head/largest: largest 7.02e-06, at most 0.0001 was predicted |
| Q6 | outside the proved class | high | as predicted | ok: outside-qwen3-0.6b/qk_norm_transformations/T4_scalar_moved_from_the_query_gain_to_the_key_gain/largest: largest 1.09e-14, at most 1e-06 was predicted |
| Q7 | outside the proved class | moderate | as predicted | ok: outside-qwen3-0.6b/qk_norm_transformations/T2_with_the_angles_of_a_shift_of_16_positions/smallest: smallest 0.475, at least 0.001 was predicted |
| Q8 | kinematics | high | as predicted | ok: outside-qwen3-0.6b/dilation/*/smallest: smallest 0.179, at least 0.001 was predicted |
| Q9 | kinematics | high | as predicted | ok: outside-qwen3-0.6b/rotation/logits/largest: largest 2.01e-14, at most 1e-06 was predicted; ok: outside-qwen3-0.6b/rotation/stream_equals_the_original_times_Q_transpose/largest: largest 3.27e-14, at most 1e-06 was predicted; ok: outside-qwen3-0.6b/rotation/gram_matrix_of_every_layer/largest: largest 4.2e-14, at most 1e-06 was predicted; ok: outside-qwen3-0.6b/rotation/stream_moved_at_the_embedd |
| G0 | language model | high | as predicted | ok: outside-gpt2/membership/fraction_of_ln_vocabulary: largest 0.23, at most 0.9 was predicted |
| G1 | outside the proved class | high | as predicted | ok: outside-gpt2/layernorm_transformations/L1_all_ones_added_to_the_outputs_of_the_two_output_projections/largest: largest 2.73e-15, at most 1e-06 was predicted |
| G2 | outside the proved class | high | as predicted | ok: outside-gpt2/layernorm_transformations/L2_dual_shift_of_the_readers_of_the_two_layernorms/largest: largest 1.21e-14, at most 1e-06 was predicted |
| G3 | outside the proved class | high | as predicted | ok: outside-gpt2/layernorm_transformations/L3_embedding_rows_of_five_tokens_shifted/unshifted_columns_largest: largest 0, at most 1e-06 was predicted; ok: outside-gpt2/layernorm_transformations/L3_embedding_rows_of_five_tokens_shifted/shifted_columns_against_prediction_largest: largest 9.35e-16, at most 1e-06 was predicted |
| G4 | kinematics | high | as predicted | ok: outside-gpt2/absolute_positions/*/smallest: smallest 0.361, at least 0.001 was predicted |
| P0 | language model | high | as predicted | ok: outside-pythia-160m/membership/fraction_of_ln_vocabulary: largest 0.228, at most 0.9 was predicted |
| P1 | kinematics | high | as predicted | ok: outside-pythia-160m/translation/*/largest: largest 1.89e-15, at most 1e-06 was predicted |
| P2 | kinematics | high | REFUTED | FAILS: outside-pythia-160m/dilation/*/smallest: smallest 0.00095, at least 0.001 was predicted |
| P3 | outside the proved class | high | as predicted | ok: outside-pythia-160m/layernorm_transformations/writer_shift/largest: largest 1.55e-15, at most 1e-06 was predicted |
| P4 | outside the proved class | high | as predicted | ok: outside-pythia-160m/layernorm_transformations/rotation_fixing_the_all_ones_vector/largest: largest 2.59e-15, at most 1e-06 was predicted |
| P5 | outside the proved class | high | as predicted | ok: outside-pythia-160m/layernorm_transformations/general_orthogonal_rotation/smallest: smallest 0.00162, at least 0.001 was predicted |
| C-qwen3-tool_call | conversation | high | as predicted | ok: chat-qwen3-0.6b/prompts/tool_call/variants/H1_signed_permutation/identical_tokens: [True], True was predicted; ok: chat-qwen3-0.6b/prompts/tool_call/variants/M1_units/identical_tokens: [True], True was predicted; ok: chat-qwen3-0.6b/prompts/tool_call/variants/A1_value_output/identical_tokens: [True], True was predicted; ok: chat-qwen3-0.6b/prompts/tool_call/variants/A6_head_swap/identical_toke |
| C-qwen3-conversation | conversation | high | as predicted | ok: chat-qwen3-0.6b/prompts/conversation/variants/H1_signed_permutation/identical_tokens: [True], True was predicted; ok: chat-qwen3-0.6b/prompts/conversation/variants/M1_units/identical_tokens: [True], True was predicted; ok: chat-qwen3-0.6b/prompts/conversation/variants/A1_value_output/identical_tokens: [True], True was predicted; ok: chat-qwen3-0.6b/prompts/conversation/variants/A6_head_swap/id |
| C-smol-conversation | conversation | high | as predicted | ok: chat-smol-instruct/prompts/conversation/variants/H1_signed_permutation/identical_tokens: [True], True was predicted; ok: chat-smol-instruct/prompts/conversation/variants/M1_units/identical_tokens: [True], True was predicted; ok: chat-smol-instruct/prompts/conversation/variants/A1_value_output/identical_tokens: [True], True was predicted; ok: chat-smol-instruct/prompts/conversation/variants/A6_ |
| H-smol-instruct-sgd | optimizers | high | as predicted | ok: finetune-smol-instruct/optimizers/sgd/variants/signed_permutation_of_the_hidden_coordinates/difference_after_training_over_the_change_of_the_reference_run: largest 7.22e-12, at most 1e-06 was predicted; ok: finetune-smol-instruct/optimizers/sgd/variants/rotation_of_the_stream/difference_after_training_over_the_change_of_the_reference_run: largest 8.25e-10, at most 1e-06 was predicted; ok: fine |
| H-smol-instruct-adamw | optimizers | high | as predicted | ok: finetune-smol-instruct/optimizers/adamw/variants/signed_permutation_of_the_hidden_coordinates/difference_after_training_over_the_change_of_the_reference_run: largest 2.32e-14, at most 1e-06 was predicted; ok: finetune-smol-instruct/optimizers/adamw/variants/rotation_of_the_stream/difference_after_training_over_the_change_of_the_reference_run: smallest 1.6, at least 0.01 was predicted; ok: fine |
| H-delphi-100k-sgd | optimizers | high | as predicted | ok: finetune-delphi-100k/optimizers/sgd/variants/signed_permutation_of_the_hidden_coordinates/difference_after_training_over_the_change_of_the_reference_run: largest 4.52e-12, at most 1e-06 was predicted; ok: finetune-delphi-100k/optimizers/sgd/variants/rotation_of_the_stream/difference_after_training_over_the_change_of_the_reference_run: largest 6.52e-12, at most 1e-06 was predicted; ok: finetune |
| H-delphi-100k-adamw | optimizers | high | as predicted | ok: finetune-delphi-100k/optimizers/adamw/variants/signed_permutation_of_the_hidden_coordinates/difference_after_training_over_the_change_of_the_reference_run: largest 8.78e-15, at most 1e-06 was predicted; ok: finetune-delphi-100k/optimizers/adamw/variants/rotation_of_the_stream/difference_after_training_over_the_change_of_the_reference_run: smallest 0.0975, at least 0.01 was predicted; ok: finet |

## Controls (not claims)

| control | status |
|---|---|
| control smol-instruct: the same forward pass twice gives the same logits (1e-12) | holds |
| control smol-instruct: the rotation without the fold is not a symmetry (at least 1e-3) | holds |
| control smol-instruct: the shift of 16 positions moves the cache of keys (mean over layers at least 0.1) | holds |
| control smol-base: the same forward pass twice gives the same logits (1e-12) | holds |
| control smol-base: the rotation without the fold is not a symmetry (at least 1e-3) | holds |
| control smol-base: the shift of 16 positions moves the cache of keys (mean over layers at least 0.1) | holds |
| control floatlm-99m: the same forward pass twice gives the same logits (1e-12) | holds |
| control floatlm-99m: the rotation without the fold is not a symmetry (at least 1e-3) | holds |
| control floatlm-99m: the shift of 16 positions moves the cache of keys (mean over layers at least 0.1) | holds |
| control trilm-99m: the same forward pass twice gives the same logits (1e-12) | holds |
| control trilm-99m: the rotation without the fold is not a symmetry (at least 1e-3) | holds |
| control trilm-99m: the shift of 16 positions moves the cache of keys (mean over layers at least 0.1) | holds |
| control delphi-100k: the same forward pass twice gives the same logits (1e-12) | holds |
| control delphi-100k: the rotation without the fold is not a symmetry (at least 1e-3) | holds |
| control delphi-100k: the shift of 16 positions moves the cache of keys (mean over layers at least 0.1) | holds |
| control the finite instance is rejected by the registered test of 'is a language model' (fraction above 0.9) | holds |
| control chat Qwen3: the variant with the permutation applied to the layers but not to the embedding gives different tokens (tool call) | holds |
| control chat Qwen3: the broken variant gives different tokens (conversation) | holds |
| control chat SmolLM2: the broken variant gives different tokens | holds |
