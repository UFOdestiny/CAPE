#!/usr/bin/env bash
# Every stage for one backbone: GPU stages, LLM-assisted labels, CPU analyses.
# Usage: bash scripts/run_benchmark.sh <model> [gpus=0,1,2,3]
set -euo pipefail
cd "$(dirname "$0")/.."
MODEL="${1:?usage: run_benchmark.sh <model> [gpus]}"
GPUS="${2:-0,1,2,3}"
bash scripts/run_gpu.sh "$MODEL" "$GPUS"
bash scripts/run_labels.sh "$MODEL"
bash scripts/run_cpu.sh "$MODEL"
