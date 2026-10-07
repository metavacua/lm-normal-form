#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Smoke: the finite instance as a checkpoint in Transformers and in candle, an instance that is not registered (seed 99).
set -euo pipefail
mkdir -p out work
export LMNF_BRIDGE_SEED=99
python batches/0023/bridge23.py export work
python batches/0023/bridge23.py torch work out/smoke-bridge-torch.json
cargo build --release --locked --manifest-path batches/0023/candle/Cargo.toml
batches/0023/candle/target/release/lmnf-candle-logits work/ckpt work/candle_inputs.json work/candle f32
python batches/0023/bridge23.py candle work work/candle.logits.f32 out/smoke-bridge-candle.json
