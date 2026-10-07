#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# Smoke: the rank experiment and the trained point, on a draw that is not registered.
set -euo pipefail
mkdir -p out
python batches/0023/cov23.py out/smoke-cov.json --smoke
python batches/0023/fibre23.py out/smoke-fibre.json --smoke
