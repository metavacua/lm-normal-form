#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Smoke: a conversation, on a chat checkpoint that is not registered (SmolLM2-360M-Instruct, at whatever revision is the head).
set -euo pipefail
mkdir -p out
python batches/0023/chat23.py smollm2 HuggingFaceTB/SmolLM2-360M-Instruct - out/smoke-chat.json
