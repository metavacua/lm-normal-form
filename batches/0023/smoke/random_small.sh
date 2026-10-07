#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Smoke: Transformers, float64, a randomly initialised small model of each architecture (no download): the kinematics, the architectures outside the Llama family, fine-tuning.
set -euo pipefail
mkdir -p out
python batches/0023/systems23.py llama - - out/smoke-systems-llama.json --random-small
python batches/0023/outside23.py qwen3 - - out/smoke-outside-qwen3.json --random-small
python batches/0023/outside23.py gpt2 - - out/smoke-outside-gpt2.json --random-small
python batches/0023/outside23.py gpt_neox - - out/smoke-outside-neox.json --random-small
python batches/0023/finetune23.py smoke - - out/smoke-finetune.json --random-small
