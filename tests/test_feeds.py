# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inputs for the first step of a decoder graph, built from the graph's own declarations."""

import numpy as np
import pytest

from lmnf import feeds

INPUTS = [
    {"name": "input_ids", "type": "INT64", "dims": ["batch_size", "sequence_length"]},
    {"name": "attention_mask", "type": "INT64", "dims": ["batch_size", "total_sequence_length"]},
    {"name": "position_ids", "type": "INT64", "dims": ["batch_size", "sequence_length"]},
    {"name": "past_key_values.0.key", "type": "FLOAT", "dims": ["batch_size", 3, "past_sequence_length", 64]},
    {"name": "past_key_values.0.value", "type": "FLOAT16", "dims": ["batch_size", 3, "past_sequence_length", 64]},
]


def test_the_first_step_has_the_tokens_and_an_empty_past():
    plan = feeds.first_step(INPUTS, [5, 6, 7])
    assert plan["input_ids"].tolist() == [[5, 6, 7]]
    assert plan["attention_mask"].tolist() == [[1, 1, 1]]
    assert plan["position_ids"].tolist() == [[0, 1, 2]]
    assert plan["input_ids"].dtype == np.int64
    assert plan["past_key_values.0.key"].shape == (1, 3, 0, 64)
    assert plan["past_key_values.0.key"].dtype == np.float32
    assert plan["past_key_values.0.value"].dtype == np.float16
    assert set(plan) == {i["name"] for i in INPUTS}


def test_an_input_the_rules_do_not_cover_is_refused_by_name():
    with pytest.raises(feeds.UnknownInput, match="mystery_switch"):
        feeds.first_step(INPUTS + [{"name": "mystery_switch", "type": "BOOL", "dims": []}], [1])


def test_a_graph_without_token_ids_is_refused():
    with pytest.raises(feeds.UnknownInput, match="input_ids"):
        feeds.first_step(INPUTS[1:], [1])


def test_a_merged_decoder_is_told_to_take_the_branch_without_a_past():
    switch = {"name": "use_cache_branch", "type": "BOOL", "dims": [1]}
    plan = feeds.first_step(INPUTS + [switch], [5, 6])
    assert plan["use_cache_branch"].tolist() == [False]
    assert plan["use_cache_branch"].dtype == np.bool_
