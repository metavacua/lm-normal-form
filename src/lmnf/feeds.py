# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Inputs for the first step of a decoder graph, from the graph's own declarations.

Only inputs these rules recognise are filled in. Anything else is refused by
name: feeding zeros to an input nobody understood would make the run look
conclusive when it is not.
"""

import numpy as np

DTYPES = {"FLOAT": np.float32, "FLOAT16": np.float16, "DOUBLE": np.float64, "INT64": np.int64, "INT32": np.int32}
PAST = "past_key_values"


class UnknownInput(ValueError):
    pass


def first_step(inputs, token_ids):
    """`inputs` is the graph's input list (name, type, dims); the past is empty."""
    length = len(token_ids)
    feeds = {}
    for declared in inputs:
        name, dtype = declared["name"], DTYPES.get(declared["type"])
        if name == "input_ids":
            feeds[name] = np.array([token_ids], dtype=np.int64)
        elif name == "attention_mask":
            feeds[name] = np.ones((1, length), dtype=np.int64)
        elif name == "position_ids":
            feeds[name] = np.arange(length, dtype=np.int64)[None, :]
        elif name == "use_cache_branch":
            # A merged decoder holds two graphs behind an If; false selects the one that needs no past.
            feeds[name] = np.zeros([d if isinstance(d, int) else 1 for d in declared["dims"]], dtype=np.bool_)
        elif name.startswith(PAST) and dtype is not None:
            # Batch is one; every other symbolic axis is the length of a past that does not exist yet.
            shape = [d if isinstance(d, int) else (1 if axis == 0 else 0) for axis, d in enumerate(declared["dims"])]
            feeds[name] = np.zeros(shape, dtype=dtype)
        else:
            raise UnknownInput(f"no rule for graph input {name!r} ({declared['type']}, dims {declared['dims']})")
    if "input_ids" not in feeds:
        raise UnknownInput("the graph declares no input_ids")
    return feeds
