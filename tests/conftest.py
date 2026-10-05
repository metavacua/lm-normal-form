# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Synthetic ONNX models for the tests. Built in memory; nothing is downloaded."""

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

MODEL_IRI = "https://example.org/models/synthetic.onnx"

SORTED = ("mm_gate", "sigmoid", "mul", "mm_down", "add", "reshape")


def gated_block(order=SORTED, gate_name="w_gate"):
    """x -> MatMul(gate) -> Sigmoid * identity (SiLU) -> MatMul(down) -> + x -> Reshape."""
    rng = np.random.default_rng(0)
    w_gate = numpy_helper.from_array(rng.standard_normal((4, 6)).astype(np.float32), gate_name)
    w_down = numpy_helper.from_array(rng.standard_normal((6, 4)).astype(np.float32), "w_down")
    shape = numpy_helper.from_array(np.array([1, -1, 2, 2], dtype=np.int64), "target_shape")
    nodes = {
        "mm_gate": helper.make_node("MatMul", ["x", gate_name], ["g"], name="mm_gate"),
        "sigmoid": helper.make_node("Sigmoid", ["g"], ["s"], name="sigmoid"),
        "mul": helper.make_node("Mul", ["g", "s"], ["a"], name="mul"),
        "mm_down": helper.make_node("MatMul", ["a", "w_down"], ["y"], name="mm_down"),
        "add": helper.make_node("Add", ["x", "y"], ["h"], name="add"),
        "reshape": helper.make_node("Reshape", ["h", "target_shape"], ["out"], name="reshape"),
    }
    graph = helper.make_graph(
        [nodes[k] for k in order],
        "gated_block",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4])],
        [helper.make_tensor_value_info("out", TensorProto.FLOAT, [1, "n", 2, 2])],
        initializer=[w_gate, w_down, shape],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="lmnf-tests")
    model.ir_version = 9
    return model


def branching():
    """One If node whose two branches read a value from the enclosing graph."""
    then_g = helper.make_graph(
        [helper.make_node("Identity", ["x"], ["t"], name="keep")],
        "then",
        [],
        [helper.make_tensor_value_info("t", TensorProto.FLOAT, [2])],
    )
    else_g = helper.make_graph(
        [helper.make_node("Neg", ["x"], ["e"], name="negate")],
        "else",
        [],
        [helper.make_tensor_value_info("e", TensorProto.FLOAT, [2])],
    )
    node = helper.make_node("If", ["cond"], ["y"], then_branch=then_g, else_branch=else_g, name="branch")
    graph = helper.make_graph(
        [node],
        "branching",
        [
            helper.make_tensor_value_info("cond", TensorProto.BOOL, []),
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [2]),
        ],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [2])],
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="lmnf-tests")
    model.ir_version = 9
    return model


def empty():
    graph = helper.make_graph([], "empty", [], [])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)], producer_name="lmnf-tests")
    model.ir_version = 9
    return model


@pytest.fixture
def save(tmp_path):
    def _save(model, name="model.onnx"):
        path = tmp_path / name
        onnx.save(model, str(path))
        return path

    return _save
