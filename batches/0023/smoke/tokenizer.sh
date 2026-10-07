#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Smoke: the paths that read text, on a randomly initialised checkpoint that has a tokenizer (hf-internal-testing/tiny-random-LlamaForCausalLM).
set -euo pipefail
mkdir -p out
rev=9fb191250dd56d0ba7ec9785a025ed29c03d5998
python batches/0023/systems23.py llama hf-internal-testing/tiny-random-LlamaForCausalLM "$rev" out/smoke-systems-tinyllama.json
python batches/0023/finetune23.py smoke hf-internal-testing/tiny-random-LlamaForCausalLM "$rev" out/smoke-finetune-tinyllama.json
