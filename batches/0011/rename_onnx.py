# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The same ONNX graph under other names: every node, every intermediate edge and every initializer gets a prefix
# (onnx.compose.add_prefix), the graph's inputs and outputs keep theirs so that a session is fed as before.
# Usage: rename_onnx.py IN.onnx OUT.onnx PREFIX
import sys, onnx
from onnx import compose

onnx.save(compose.add_prefix(onnx.load(sys.argv[1]), sys.argv[3], rename_inputs=False, rename_outputs=False), sys.argv[2])
