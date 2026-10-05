# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Facts about an ONNX tensor that do not depend on how it is stored in the file."""

import math

from onnx import TensorProto, numpy_helper

INTEGER_TYPES = {
    TensorProto.BOOL,
    TensorProto.INT8,
    TensorProto.INT16,
    TensorProto.INT32,
    TensorProto.INT64,
    TensorProto.UINT8,
    TensorProto.UINT16,
    TensorProto.UINT32,
    TensorProto.UINT64,
}
FLOAT_TYPES = {TensorProto.FLOAT, TensorProto.DOUBLE, TensorProto.FLOAT16, TensorProto.BFLOAT16}


def element_count(tensor):
    return math.prod(tensor.dims)


def is_external(tensor):
    return tensor.data_location == TensorProto.EXTERNAL and not tensor.HasField("raw_data")


def content(tensor):
    """The tensor's elements as little-endian bytes, whichever field the file used."""
    if tensor.HasField("raw_data"):
        return tensor.raw_data
    return numpy_helper.to_array(tensor).tobytes()


def elements(tensor):
    return numpy_helper.to_array(tensor).reshape(-1).tolist()
