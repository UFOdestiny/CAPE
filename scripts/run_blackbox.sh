#!/usr/bin/env bash
# Black-box backbone (Claude Haiku 5.5, paid Message Batches; needs ANTHROPIC_API_KEY): sampling and the verbal-
# confidence follow-up, outcome rewards of the visible answers (GPU), LLM-assisted labels and the evaluation.
# Usage: bash scripts/run_blackbox.sh [gpus=0,1,2,3] [budget-in-dollars]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/environment.sh
GPUS="${1:-0,1,2,3}"
BUDGET=(); [ -n "${2:-}" ] && BUDGET=(--budget "$2")
MODEL=claude-haiku-5-5
B="$PY -m uqbench.blackbox.batch"
$B submit "${BUDGET[@]}" && $B status --wait
$B verbal "${BUDGET[@]}" && $B status --wait
$B merge
"$PY" -m uqbench.pipeline.reward --model "$MODEL" --gpus "$GPUS"
bash scripts/run_labels.sh "$MODEL"
"$PY" -m uqbench.blackbox.evaluate --model "$MODEL"
