#!/usr/bin/env bash
# SPDX-FileCopyrightText: 2026 Ian Douglas Lawrence Norman McLean
# SPDX-License-Identifier: AGPL-3.0-or-later
# The one piece of shell that more than one job of batch 0028's workflow shares.
# Usage: cells.sh env TAG    the lines MODEL=... and REV=... for $GITHUB_ENV: the repository and the revision registered for TAG
set -u -o pipefail
case ${1:-} in
  env)
    case ${2:-} in
      tiny) echo "MODEL=hf-internal-testing/tiny-random-LlamaForCausalLM"; echo "REV=" ;;
      base) echo "MODEL=HuggingFaceTB/SmolLM2-135M"; echo "REV=93efa2f097d58c2a74874c7e644dbc9b0cee75a2" ;;
      instruct) echo "MODEL=HuggingFaceTB/SmolLM2-135M-Instruct"; echo "REV=12fd25f77366fa6b3b4b768ec3050bf629380bac" ;;
      *) echo "no model is named for '${2:-}'" >&2; exit 1 ;;
    esac ;;
  *) echo "usage: cells.sh env TAG" >&2; exit 2 ;;
esac
