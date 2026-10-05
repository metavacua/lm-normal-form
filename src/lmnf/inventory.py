# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Does a graph store every parameter an independent source lists for the model?

Shapes are compared as multisets. A matrix matches its transpose, because an
exporter may store a linear layer's weight either way round; tensors of any
other rank must match exactly.
"""

import collections
import math


def key(shape):
    shape = tuple(int(d) for d in shape)
    return tuple(sorted(shape)) if len(shape) == 2 else shape


def compare(onnx, reference):
    stored = collections.Counter(key(s) for s in onnx)
    wanted = collections.Counter(key(s) for s in reference)
    matched = stored & wanted
    return {
        "matched": sum(matched.values()),
        "missing_from_onnx": [list(k) for k in sorted((wanted - stored).elements())],
        "only_in_onnx": [list(k) for k in sorted((stored - wanted).elements())],
        "elements_onnx": sum(math.prod(s) for s in onnx),
        "elements_reference": sum(math.prod(s) for s in reference),
    }
