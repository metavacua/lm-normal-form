#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Smoke: the discrete counts, a mutant that is not registered, the training rules at two points and for 20 steps, on draws that are not registered.
set -euo pipefail
mkdir -p out
LMNF_SEEDS=99 python batches/0023/discrete23.py out/smoke-discrete.json
python batches/0023/mutate23.py out/smoke-mutate.json SMOKE
LMNF_FIELD_SEEDS=5,6 python batches/0023/opt23.py field out/smoke-opt-field.json
LMNF_STEPS=20 LMNF_STUDENT_SEED=5 python batches/0023/opt23.py sgd out/smoke-opt-sgd.json
LMNF_STEPS=20 LMNF_STUDENT_SEED=5 python batches/0023/opt23.py adam out/smoke-opt-adam.json
