# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""What the scripts of cell 15 share, numpy only, so that the local one-off scripts import it as the CI cell does: bf16 bytes to float32 and back
(round to nearest even), the safetensors header with the byte offset of its data block, and batch 0014's sha256 and peak-resident-size helpers."""
import json
import os
import struct
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "0014"))
from common import peak_rss_mb, sha256_file  # noqa: E402,F401


def bf16_to_f32(b):
    """bf16 bytes (little endian) to float32, exactly."""
    return (np.frombuffer(b, dtype=np.uint16).astype(np.uint32) << 16).view(np.float32)


def f32_to_bf16_bytes(a):
    """float32 to bf16 bytes, round to nearest even (what torch's conversion does)."""
    u = np.ascontiguousarray(a, dtype=np.float32).view(np.uint32)
    lsb = (u >> 16) & 1
    return ((u + 0x7FFF + lsb) >> 16).astype(np.uint16).tobytes()


def st_header(path):
    """The safetensors header (tensor name -> dtype, shape, data_offsets) and the byte offset of the data block it addresses."""
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n)), 8 + n


def st_tensor(hdr, data_start, name):
    """The byte offset of a bf16 tensor's first element in the file, and its shape."""
    e = hdr[name]
    assert e["dtype"] == "BF16", (name, e["dtype"])
    return data_start + e["data_offsets"][0], e["shape"]
