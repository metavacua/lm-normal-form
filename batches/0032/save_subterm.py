# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
"""Save one homogeneous sub-term (Pk or Dj) of a checkpoint as a standard checkpoint.  usage: save_subterm.py MODEL_DIR TAG OUT_DIR"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import subterms  # noqa: E402

model_dir, tag, out = sys.argv[1:4]
model = subterms.load(model_dir)
view = subterms.prefix(model, int(tag[1:])) if tag.startswith("P") else subterms.delete_block(model, int(tag[1:]))
os.makedirs(out, exist_ok=True)
subterms.save_homogeneous(model, view, out)
subterms.copy_tokenizer_files(model_dir, out)
print("saved", tag, "->", out, "blocks", len(view.layers))
