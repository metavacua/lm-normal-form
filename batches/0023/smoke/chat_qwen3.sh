#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Smoke: a conversation and a request for a tool call, with the chat template and the tokenizer of Qwen3-0.6B (the files of the tokenizer only) and a randomly initialised small Qwen3 in float64
# (no weights of the registered checkpoint are read): the template with a list of tools, the generation, the eight variants of the weights, the control.
set -euo pipefail
mkdir -p out
python batches/0023/chat23.py qwen3 Qwen/Qwen3-0.6B c1899de289a04d12100db370d81485cdf75e47ca out/smoke-chat-qwen3.json --random-small
