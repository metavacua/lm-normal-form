# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Comparing the tensors a graph stores with an independent list of a model's parameters."""

from lmnf import inventory


def test_matrices_match_whichever_way_round_they_are_stored():
    result = inventory.compare(onnx=[(6, 4), (4,), (2, 3, 4)], reference=[(4, 6), (4,), (2, 3, 4)])
    assert result["matched"] == 3
    assert result["missing_from_onnx"] == []
    assert result["only_in_onnx"] == []


def test_higher_rank_tensors_must_match_exactly():
    result = inventory.compare(onnx=[(4, 3, 2)], reference=[(2, 3, 4)])
    assert result["missing_from_onnx"] == [[2, 3, 4]]


def test_a_parameter_with_no_counterpart_is_reported():
    result = inventory.compare(onnx=[(4, 6)], reference=[(4, 6), (8,)])
    assert result["matched"] == 1
    assert result["missing_from_onnx"] == [[8]]


def test_each_stored_tensor_can_answer_for_only_one_parameter():
    result = inventory.compare(onnx=[(4, 6)], reference=[(4, 6), (6, 4)])
    assert result["matched"] == 1
    assert len(result["missing_from_onnx"]) == 1


def test_tensors_the_graph_adds_are_listed_but_do_not_count_against_it():
    result = inventory.compare(onnx=[(4, 6), (1,), ()], reference=[(6, 4)])
    assert result["missing_from_onnx"] == []
    assert result["only_in_onnx"] == [[], [1]]


def test_element_totals_are_reported_for_both_sides():
    result = inventory.compare(onnx=[(4, 6), (3,)], reference=[(6, 4)])
    assert result["elements_onnx"] == 27
    assert result["elements_reference"] == 24
