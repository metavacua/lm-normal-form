# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""A model repository at one revision, without the ONNX files and the training artifacts: the configuration, the tokenizer,
the README and the safetensors file (and, with --bin, the PyTorch weights of a repository that has no safetensors file).
Usage: fetch.py MODEL REVISION DIR [--bin]     (REVISION may be empty: the head)
"""

import sys

from huggingface_hub import snapshot_download

if len(sys.argv) < 4:
    sys.exit(__doc__)
model, rev, dest = sys.argv[1:4]
allow = ["*.json", "*.txt", "*.safetensors", "README.md", "*.jinja", "tokenizer.model"] + (["*.bin"] if "--bin" in sys.argv else [])
path = snapshot_download(repo_id=model, revision=rev or None, local_dir=dest, allow_patterns=allow,
                         ignore_patterns=["onnx/*", "runs/*", "training_args.bin"])
print(path)
