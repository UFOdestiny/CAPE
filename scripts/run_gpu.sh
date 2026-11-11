#!/usr/bin/env bash
# GPU stages for one backbone: sampling (vLLM), teacher-forced replay (HF), raw sampled-token logits for semantic
# energy (HF), follow-up probes (vLLM) and outcome-reward scores (vLLM). Every stage is resumable.
# Usage: bash scripts/run_gpu.sh <model> [gpus=0,1,2,3]
set -euo pipefail
cd "$(dirname "$0")/.."
source scripts/datasets.sh
MODEL="${1:?usage: run_gpu.sh <model> [gpus]}"
GPUS="${2:-0,1,2,3}"
read -r -a SETS <<< "$(sets)"
for stage in generate replay energy followup reward; do
  "$PY" -m uqbench.pipeline.$stage --model "$MODEL" --gpus "$GPUS" --datasets "${SETS[@]}"
done
